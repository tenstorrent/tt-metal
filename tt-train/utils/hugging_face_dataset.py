#!/usr/bin/env python3

import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def download_wikipedia(output_file="data.txt", language="en"):
    """
    Downloads the Wikipedia dataset from Hugging Face, filters for English articles,
    and saves the content to a single text file.
    """
    print("Downloading Wikipedia dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train")

    print("Processing articles...")
    with open(output_file, "w", encoding="utf-8") as file:
        for item in dataset:
            if "text" in item:
                file.write(item["text"] + "\n")

    print(f"English articles saved to {output_file}")


def train_tokenizer(input_file, output_tokenizer="tokenizer.json", vocab_size=8000):
    """
    Trains a tokenizer with a specified vocabulary size on the given text file.
    """
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    print("Training tokenizer...")
    tokenizer.train([input_file], trainer)

    tokenizer.save(output_tokenizer)
    print(f"Tokenizer saved to {output_tokenizer}")


def main():
    # Step 1: Download and preprocess Wikipedia dataset
    data_file = "data.txt"
    download_wikipedia(output_file=data_file, language="en")

    # Step 2: Train a tokenizer with 8,000 tokens
    train_tokenizer(input_file=data_file, output_tokenizer="tokenizer.json", vocab_size=8000)


if __name__ == "__main__":
    main()
