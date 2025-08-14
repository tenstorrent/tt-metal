# write a code which downloads gpt2s weights from huggingface and prints path
# This script is used to download GPT-2 model weights from Hugging Face and print the path to the downloaded file.
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os


def load_model(model_name, cache_dir="/tmp/.huggingface/"):
    print(f"Loading model {model_name} from Hugging Face...")

    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer


def main():
    model_name = "gpt2"
    model, tokenizer = load_model(model_name)
    state_dict = model.state_dict()
    print("GPT2 state_dict keys:")
    print(state_dict.keys())


if __name__ == "__main__":
    main()
