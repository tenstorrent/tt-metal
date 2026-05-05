from transformers import AutoModel, AutoTokenizer

model_name = "mistralai/Devstral-Small-2-24B-Instruct-2512"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print(model)
