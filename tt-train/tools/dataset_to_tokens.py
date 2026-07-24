# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizes a UTF-8 text corpus with a Hugging Face fast tokenizer into a flat uint32 token
stream, framed with the tokenizer's BOS/EOS special tokens, and writes it as YAML
(`tokens`, `tokenizer_vocab_size`, `data_length`) for the C++ examples/nano_gpt and the Python examples/train.
Pre-tokenizing here avoids long start-up times for those examples.
"""

import os
import argparse
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_text_data(text_file) -> str:
    """Reads a UTF-8 text file verbatim (newlines and structure preserved)."""
    with open(text_file, "r", encoding="utf-8") as f:
        return f.read()


def load_tokenizer(spec):
    """
    Resolve a fast tokenizer from a directory, a single *.json tokenizer file, or a
    Hugging Face hub id (mirrors ttml/common/data.py). The hub-id branch preserves the
    existing behaviour for automation (e.g. TinyLlama/Qwen are downloaded as before).
    """
    if os.path.isdir(spec):
        return AutoTokenizer.from_pretrained(spec, local_files_only=True)
    if os.path.isfile(spec) and spec.endswith(".json"):
        return PreTrainedTokenizerFast(tokenizer_file=spec)
    return AutoTokenizer.from_pretrained(spec)


def frame_with_special_tokens(tokenizer, ids):
    """
    Prepend BOS and append EOS when the tokenizer defines them.
    No-op for ids the tokenizer leaves as None -- note a bare
    tokenizer.json carries no special-token metadata, so use the hub id or a tokenizer
    directory if you need BOS/EOS.
    """
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    framed = list(ids)
    if eos is not None:
        framed.append(eos)
    if bos is not None:
        framed.insert(0, bos)
    return framed


def save_to_yaml(data_list, vocab_size, output_file):
    """
    Writes the tokens plus metadata (vocab size, data length) as flow-style YAML.
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
    tokenizer = load_tokenizer(hf_tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokenized_data = frame_with_special_tokens(tokenizer, ids)
    return ",".join(map(str, tokenized_data))


def decode_tokens(hf_tokenizer, tokens_str):
    """
    Decodes comma-separated token IDs back to text.
    """
    tokenizer = load_tokenizer(hf_tokenizer)
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
        help="Path to save the tokenized data (.yaml is appended unless the path already ends in .yaml/.yml).",
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

    tokenizer = load_tokenizer(args.hf_tokenizer)
    print(f"Tokenizing data using {args.hf_tokenizer}...")
    # Encode without the tokenizer's implicit post-processor so special tokens are fully
    # under our control (a single call avoids broken BPE merges and spurious BOS tokens).
    ids = tokenizer.encode(text_data, add_special_tokens=False)
    tokenized_data = frame_with_special_tokens(tokenizer, ids)

    # Save tokenized data
    output_file = args.output_file
    if os.path.splitext(output_file)[1].lower() not in (".yaml", ".yml"):
        output_file += ".yaml"
    save_to_yaml(tokenized_data, tokenizer.vocab_size, output_file)


if __name__ == "__main__":
    main()
