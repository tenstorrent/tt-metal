"""Quick check: does HF transformers produce Paris with these weights?"""
import os

import torch

os.environ.setdefault("HF_MODEL", os.path.expanduser("~/models/Qwen3.5-27B-FP8"))

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ["HF_MODEL"]
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"Prompt: '{prompt}', tokens: {inputs.input_ids[0].tolist()}")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=1.0)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nHF output: '{text}'")
if "paris" in text.lower():
    print("PASS: HF model produces Paris")
else:
    print("FAIL: HF model does NOT produce Paris")
