# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Download the task-eval datasets used by task_eval.py into a local directory.

    python prepare_datasets.py --out /path/to/data            # MMLU-Pro + AIME 2026 (ungated)
    HF_TOKEN=... python prepare_datasets.py --out ... --gpqa  # also GPQA-Diamond (gated: accept terms on HF first)

Writes mmlu_pro_test.jsonl (TIGER-Lab/MMLU-Pro, test split, rows as-is), aime_2026.jsonl (MathArena/aime_2026)
and, with --gpqa, gpqa_diamond.csv (Idavidrein/gpqa). Pass the files to task_eval.py with --data / --csv.
Do not commit GPQA data anywhere public (the dataset asks not to reveal examples).
"""
import argparse
import json
import os


def dump_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpqa", action="store_true", help="also fetch the gated GPQA-Diamond csv (needs HF_TOKEN)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    dump_jsonl([dict(r) for r in mmlu], os.path.join(args.out, "mmlu_pro_test.jsonl"))
    aime = load_dataset("MathArena/aime_2026", split="train")
    dump_jsonl([dict(r) for r in aime], os.path.join(args.out, "aime_2026.jsonl"))

    if args.gpqa:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(
            "Idavidrein/gpqa",
            "gpqa_diamond.csv",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            local_dir=args.out,
        )
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
