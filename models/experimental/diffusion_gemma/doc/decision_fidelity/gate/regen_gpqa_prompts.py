#!/usr/bin/env python3
"""Regenerate the 198 GPQA-Diamond prompts after the box wiped /tmp on reboot.

The rendering is copied from `benchmarks/gpqa_run.py` on the A100 -- the script that produced the
CUDA reference arm, and the same rendering the TT prompts came from -- rather than guessed. A first
attempt guessed `rng.sample(range(4), 4)` and matched only 46 of 198 gold letters; the real convention
is an in-place `random.Random(idx).shuffle(perm)` with the gold letter read off `perm.index(0)`.

Getting the shuffle wrong misaligns every gold letter silently, so this does not trust the
reconstruction: `gpu_reference.jsonl` survived on the laptop and carries the gold letter for all 198
indices, and the script REQUIRES all 198 to match before it writes anything.

That makes the recovery self-checking rather than hopeful -- if the shuffle convention were different,
the run would abort instead of quietly producing a set whose answers no longer line up with every
result recorded so far.
"""
import argparse
import json
import os
import random
import sys

TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}""".strip()

LETTERS = "ABCD"


def build(rows):
    """Render every row exactly as benchmarks/gpqa_run.py:build does."""
    out = []
    for index, row in enumerate(rows):
        correct = row["Correct Answer"].strip()
        choices = [
            correct,
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]
        perm = list(range(4))
        random.Random(index).shuffle(perm)
        shuffled = [choices[p] for p in perm]
        gold = LETTERS[perm.index(0)]
        text = TEMPLATE.format(
            question=row["Question"].strip(), a=shuffled[0], b=shuffled[1], c=shuffled[2], d=shuffled[3]
        )
        out.append((text, gold))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="/tmp/gpu_reference.jsonl", help="the surviving gold oracle")
    parser.add_argument("--out-dir", default="/tmp/gpqa198")
    parser.add_argument(
        "--csv",
        default="/home/zni/.cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/"
        "633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_diamond.csv",
        help="the raw CSV gpqa_run.py reads, so row ORDER matches too",
    )
    args = parser.parse_args()

    import csv

    expected = {}
    for line in open(args.reference, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("gold"):
                expected[row["index"]] = row["gold"]
    print(f"oracle: {len(expected)} gold letters from {args.reference}")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"dataset: {len(rows)} rows from {args.csv}")

    built = build(rows)
    matches = sum(1 for i, (_, gold) in enumerate(built) if expected.get(i) == gold)
    print(f"  gold letters matching the surviving oracle: {matches}/{len(expected)}")
    if matches != len(expected):
        print(f"\nABORT: best reconstruction matches only {matches}/{len(expected)} gold letters.")
        print("Writing a mismatched set would silently invalidate every comparison against the")
        print("existing reference and arm results, so nothing is written.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    gold = {}
    for index, (text, letter) in enumerate(built):
        open(f"{args.out_dir}/q{index:03d}.txt", "w", encoding="utf-8").write(text)
        gold[str(index)] = letter
    json.dump(gold, open(f"{args.out_dir}/gold.json", "w"))
    print(f"\nVERIFIED: {matches}/{len(expected)} gold letters match the surviving oracle")
    print(f"wrote {len(built)} prompts + gold.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
