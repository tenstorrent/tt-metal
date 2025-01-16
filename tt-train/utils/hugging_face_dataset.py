#!/usr/bin/env python3

import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def download_wikipedia(output_folder="data/wiki", language="en"):
    """
    Downloads the Wikipedia dataset from Hugging Face, filters for English articles,
    and saves the content to a single text file.
    """
    print("Downloading Wikipedia dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    print("Processing articles...")

    for item in dataset:
        with open(output_folder + "/" + item["id"], "w", encoding="utf-8") as file:
            if "text" in item:
                file.write(item["text"] + "\n")

    print(f"English articles saved to {output_file}")


def train_tokenizer(text_file, output_tokenizer="tokenizer.json", vocab_size=8190):
    """
    Trains a BPE tokenizer on the text file, saves it to `output_tokenizer`.
    """
    tokenizer = Tokenizer(BPE(unk_token=None))
    # 2. No normalizer; GPT-2 works byte-level, so we skip lowercasing or accent-stripping
    # tokenizer.normalizer = None

    # 3. Byte-level pre-tokenizer + Byte-level decoder
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.pre_tokenizer.add_prefix_space = False
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.decoder.add_prefix_space = False
    # tokenizer.pre_tokenizer = Whitespace()

    # 3. Setup BPE trainer with desired vocab size + special tokens
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    # 4. Train on the merged text file
    tokenizer.train([text_file], trainer)

    # 5. Save the tokenizer as a single JSON
    tokenizer.save(output_tokenizer)
    print(f"Saved tokenizer to {output_tokenizer}")


def main():
    # Step 1: Download and preprocess Wikipedia dataset
    data_file = "data.txt"
    download_wikipedia(language="en")

    # Step 2: Train a tokenizer with 8,000 tokens
    # train_tokenizer(input_file=data_file, output_tokenizer="tokenizer.json", vocab_size=8000)


if __name__ == "__main__":
    main()
