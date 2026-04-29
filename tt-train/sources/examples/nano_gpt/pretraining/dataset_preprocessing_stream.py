"""Stream the DKYoon/SlimPajama-6B dataset and tokenize it with the TinyLlama tokenizer.

For each split (train/validation/test) the script counts the total number of
tokens produced by the tokenizer and prints the totals.

Usage:
    python dataset_preprocessing.py
"""

from __future__ import annotations

import argparse
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "DKYoon/SlimPajama-6B"
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SPLITS = ("train", "validation", "test")


def iter_texts(dataset: Iterable[dict], text_field: str = "text") -> Iterable[str]:
    for example in dataset:
        text = example.get(text_field)
        if text:
            yield text


def count_tokens(
    split: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    log_every: int,
    max_examples: int | None,
) -> int:
    print(f"\n=== Streaming split: {split} ===")
    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)

    total_tokens = 0
    num_examples = 0
    batch: list[str] = []

    def flush(batch: list[str]) -> int:
        if not batch:
            return 0
        encodings = tokenizer(batch, add_special_tokens=False)
        return sum(len(ids) for ids in encodings["input_ids"])

    for text in iter_texts(dataset):
        batch.append(text)
        num_examples += 1

        if len(batch) >= batch_size:
            total_tokens += flush(batch)
            batch.clear()

        if log_every and num_examples % log_every == 0:
            print(f"  [{split}] processed {num_examples:,} examples, " f"{total_tokens:,} tokens so far")

        if max_examples is not None and num_examples >= max_examples:
            break

    total_tokens += flush(batch)
    print(f"  [{split}] done: {num_examples:,} examples, {total_tokens:,} tokens")
    return total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents to tokenize at a time.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Print progress every N processed examples (0 disables).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on examples per split (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Which splits to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    totals: dict[str, int] = {}
    for split in args.splits:
        totals[split] = count_tokens(
            split=split,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            log_every=args.log_every,
            max_examples=args.max_examples,
        )

    print("\n=== Token counts ===")
    for split in args.splits:
        print(f"  {split:<10s}: {totals[split]:,} tokens")
    print(f"  {'total':<10s}: {sum(totals.values()):,} tokens")


if __name__ == "__main__":
    main()
