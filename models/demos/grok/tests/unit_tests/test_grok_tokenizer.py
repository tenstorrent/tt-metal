import torch
from transformers import AutoTokenizer


def test_grok_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("alvarobartt/grok-2-tokenizer")
    prompts = ["What is your favorite condiment? "] * 32
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    encoded_prompts_tensor = torch.tensor(encoded_prompts)
    breakpoint()
