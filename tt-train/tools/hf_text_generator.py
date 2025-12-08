from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate text data using a Hugging Face tokenizer and model, given an input"
    )

    parser.add_argument(
        "--hf_model",
        type=str,
        required=True,
        help="Hugging Face model identifier (e.g., gpt2, tinyllama, bert, etc).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input text to be tokenized and processed by the model.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for text generation.",
    )

    parser.add_argument(
        "--sample",
        type=bool,
        default=0,
        help="Whether to use sampling for text generation.",
    )

    args = parser.parse_args()
    hf_model = args.hf_model
    prompt = args.prompt
    max_tokens = args.max_tokens
    temperature = args.temperature
    sample = args.sample

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(hf_model)
    inputs = tokenizer(prompt, return_tensors="pt")

    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs, max_length=max_tokens, temperature=temperature, do_sample=sample
    )
    print("Generated tokenized output:")
    print(outputs)
    decoded_output = tokenizer.decode(outputs[0][prompt_len:])
    print("Decoded output:")
    print(decoded_output)


if __name__ == "__main__":
    main()
