# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Creates one large file with all the data from valid_exts and trains a BPE tokenizer on it.
The tokenizer is saved as a JSON file.
Dataset is saved to the txt file.
"""

import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def gather_source_files(folder):
    """
    Recursively walk `folder`, yielding paths to files with
    extensions in ('.hpp', '.cpp', '.h', '.c').
    """
    valid_exts = {".hpp", ".cpp", ".h", ".c", ".cxx", ".hxx", ".py", ".md"}
    for root, _, files in os.walk(folder):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() in valid_exts:
                print(f"Found file: {fname}")
                yield os.path.join(root, fname)


def merge_into_one_file(file_paths, merged_file_path="merged.txt"):
    """
    Merges the content of all files in `file_paths` into
    a single text file `merged_file_path`, removing trailing
    whitespace from each line.
    """
    with open(merged_file_path, "w", encoding="utf-8") as writer:
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as reader:
                    for line in reader:
                        # Remove trailing whitespace from each line
                        line = " ".join(line.split())
                        writer.write(line + "\n")
            except Exception as e:
                print(f"Warning: Could not read file {path}: {e}")
    return merged_file_path


def train_bpe_tokenizer(text_file, output_tokenizer="tokenizer.json", vocab_size=32000):
    """
    Trains a BPE tokenizer on the text file, saves it to `output_tokenizer`.
    """
    tokenizer = Tokenizer(BPE(unk_token=None))
    # 2. No normalizer; GPT-2 works byte-level, so we skip lowercasing or accent-stripping
    # tokenizer.normalizer = None

    # 3. Byte-level pre-tokenizer + Byte-level decoder
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.pre_tokenizer.add_prefix_space = False
    tokenizer.decoder.add_prefix_space = False
    # tokenizer.pre_tokenizer = Whitespace()

    # 3. Setup BPE trainer with desired vocab size + special tokens
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "\n"])

    # 4. Train on the merged text file
    tokenizer.train([text_file], trainer)

    # 5. Save the tokenizer as a single JSON
    tokenizer.save(output_tokenizer)
    print(f"Saved tokenizer to {output_tokenizer}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively gather .hpp/.cpp/.h/.c files, merge them, then train a BPE tokenizer."
    )
    parser.add_argument("--folder", type=str, required=True, help="Root folder to recursively find source files.")
    parser.add_argument(
        "--merged_txt",
        type=str,
        default="merged.txt",
        help="Path to the merged output text file (default: merged.txt).",
    )
    parser.add_argument(
        "--tokenizer_output",
        type=str,
        default="tokenizer.json",
        help="Path to save the trained tokenizer JSON (default: tokenizer.json).",
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Desired vocabulary size (default: 32000).")
    args = parser.parse_args()

    # 1. Gather source files
    file_paths = list(gather_source_files(args.folder))
    if not file_paths:
        print("No .hpp, .cpp, .h, or .c files found. Exiting.")
        return

    # 2. Merge into one file
    merged_path = merge_into_one_file(file_paths, args.merged_txt)

    # 3. Train tokenizer
    train_bpe_tokenizer(merged_path, args.tokenizer_output, args.vocab_size)


if __name__ == "__main__":
    main()
