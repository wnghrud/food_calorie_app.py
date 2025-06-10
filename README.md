pip install streamlit
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st
import io

idx_to_class = {
    0: "pizza",
    1: "bibimbap",
    2: "sushi",
    3: "burger",
    4: "ramen"
}

calorie_table = {
    "pizza": 266,
    "bibimbap": 500,
    "sushi": 300,
    "burger": 295,
    "ramen": 436
}

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load("food_classifier.pt", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_calorie(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    food_name = idx_to_class.get(predicted_idx, "알 수 없음")
    calorie = calorie_table.get(food_name, "알 수 없음")
    return food_name, calorie

st.title("🍱 음식 사진 칼로리 추정기")
st.write("이미지를 업로드하면 음식 종류와 예상 칼로리를 보여줍니다.")

uploaded_file = st.file_uploader("음식 사진 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    with st.spinner("분석 중..."):
        food_name, calorie = predict_calorie(image)

    st.success(f"예상 음식: {food_name}")
    st.info(f"예상 칼로리: {calorie} kcal")
