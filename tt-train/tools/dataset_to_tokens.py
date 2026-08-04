# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizes a text dataset using a pre-trained tokenizer
and saves the tokenized data as a flat list in MessagePack or CSV format using space delimiter.
This helps to avoid a pretty long starting times for the nano_gpt example.
Tokenizing is done in 128 splits to avoid memory issues.
"""

import os
import argparse
import yaml
from transformers import AutoTokenizer


def load_text_data(text_file):
    """
    Reads a text file line by line and returns the content as a single string.
    """
    with open(text_file, "r", encoding="utf-8") as file:
        return " ".join([line.strip() for line in file.readlines()])


def tokenize_text_data(tokenizer_file, text_data):
    """
    Tokenizes the text data using a pre-trained tokenizer and returns a flat list of tokens (integers).
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_file)
    tokenized_data = tokenizer.encode(text_data).ids
    return tokenized_data


def save_to_yaml(data_list, vocab_size, output_file):
    """
    Saves the tokenized data as a single space-separated line + data length in a YAML file.
    """

    yaml_data = {
        "tokenizer_vocab_size": vocab_size,
        "data_length": len(data_list),
        "tokens": data_list,
    }

    with open(output_file, "w", encoding="utf-8") as file:
        yaml.dump(yaml_data, file, default_flow_style=True)

    print(f"Saved tokenized data as YAML to {output_file}")


def tokenize_string(hf_tokenizer, text):
    """
    Tokenizes a single string and returns comma-separated token IDs.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    tokenized_data = tokenizer.encode(text)
    return ",".join(map(str, tokenized_data))


def decode_tokens(hf_tokenizer, tokens_str):
    """
    Decodes comma-separated token IDs back to text.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    # Parse comma-separated token IDs
    token_ids = [int(token.strip()) for token in tokens_str.split(",")]
    decoded_text = tokenizer.decode(token_ids)
    return decoded_text


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a text dataset using a tokenizer and save tokenized data with metadata (vocab size, data length) in YAML format, or tokenize/decode strings."
    )
    parser.add_argument(
        "--text_file",
        type=str,
        help="Path to the input text dataset (e.g., merged.txt).",
    )
    parser.add_argument(
        "--string",
        type=str,
        help="String to tokenize (outputs comma-separated token IDs).",
    )
    parser.add_argument(
        "--decode",
        type=str,
        help="Comma-separated token IDs to decode back to text.",
    )
    parser.add_argument(
        "--hf_tokenizer",
        type=str,
        required=True,
        help="Hugging Face tokenizer identifier (e.g., gpt2, distilgpt2, meta-llama/Llama-3.2-1B).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"{os.environ.get('TT_METAL_HOME', '~/tt-metal')}/tt-train/data/tokenized_data",
        help="Base path to save the tokenized data (extension will be added based on format).",
    )

    args = parser.parse_args()

    # Mode 1: Tokenize a single string
    if args.string:
        print(f"Tokenizing string using {args.hf_tokenizer}...")
        tokens = tokenize_string(args.hf_tokenizer, args.string)
        print(f"\nTokenized output:")
        print(tokens)
        return

    # Mode 2: Decode tokens to text
    if args.decode:
        print(f"Decoding tokens using {args.hf_tokenizer}...")
        decoded_text = decode_tokens(args.hf_tokenizer, args.decode)
        print(f"\nDecoded text:")
        print(decoded_text)
        return

    # Mode 3: Tokenize a file
    if not args.text_file:
        parser.error("Either --text_file, --string, or --decode must be provided")

    # Load text data
    print(f"Loading text data from {args.text_file}...")
    text_data = load_text_data(args.text_file)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    print(f"Tokenizing data using Hugging Face tokenizer {args.hf_tokenizer}...")
    splits_num = 128
    tokenized_data = []
    for i in range(splits_num):
        text_data_split = text_data[
            i * len(text_data) // splits_num : (i + 1) * len(text_data) // splits_num
        ]
        tokenized_data_split = tokenizer.encode(text_data_split)
        tokenized_data.extend(tokenized_data_split)

    # Save tokenized data
    save_to_yaml(tokenized_data, tokenizer.vocab_size, f"{args.output_file}.yaml")


if __name__ == "__main__":
    main()
