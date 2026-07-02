from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
text = "八百标兵奔北坡，炮兵并排北边跑。"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# Check prompt text
prompt_text = "希望你以后能够做的更好。"
prompt_tokens = tokenizer.encode(prompt_text)
print(f"Prompt Text: {prompt_text}")
print(f"Prompt Tokens: {prompt_tokens}")
