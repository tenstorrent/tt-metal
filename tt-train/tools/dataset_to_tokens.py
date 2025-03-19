# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
import numpy as np
import msgpack
import msgpack_numpy as m
import csv

# Enable numpy serialization for msgpack
m.patch()


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


def save_to_msgpack(data, output_file):
    """
    Saves the tokenized data as a flat NumPy array in MessagePack format.
    """
    np_data = np.array(data, dtype=np.int32)
    with open(output_file, "wb") as file:
        msgpack.pack(np_data, file, default=m.encode)
    print(f"Saved tokenized data as MessagePack to {output_file}")


def save_to_csv(data, output_file):
    """
    Saves the tokenized data as a single space-separated line in a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)
    print(f"Saved tokenized data as CSV to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a text dataset using a tokenizer and save all tokens as a single flat list in MessagePack or CSV format."
    )
    parser.add_argument(
        "--text_file", type=str, required=True, help="Path to the input text dataset (e.g., merged.txt)."
    )
    parser.add_argument(
        "--tokenizer_file", type=str, required=True, help="Path to the pre-trained tokenizer.json file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="tokenized_data",
        help="Base path to save the tokenized data (extension will be added based on format).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["msgpack", "csv"],
        default="csv",
        help="Output format for tokenized data (default: msgpack).",
    )
    args = parser.parse_args()

    # Load text data
    print(f"Loading text data from {args.text_file}...")
    text_data = load_text_data(args.text_file)

    # Tokenize the text data
    print(f"Tokenizing data using tokenizer {args.tokenizer_file}...")
    splits_num = 128
    tokenized_data = []
    for i in range(splits_num):
        text_data_split = text_data[i * len(text_data) // splits_num : (i + 1) * len(text_data) // splits_num]
        tokenized_data_split = tokenize_text_data(args.tokenizer_file, text_data_split)
        tokenized_data.extend(tokenized_data_split)

    # Save tokenized data in the specified format
    if args.format == "msgpack":
        output_file = f"{args.output_file}.msgpack"
        save_to_msgpack(tokenized_data, output_file)
    elif args.format == "csv":
        output_file = f"{args.output_file}"
        save_to_csv(tokenized_data, output_file)


if __name__ == "__main__":
    main()
