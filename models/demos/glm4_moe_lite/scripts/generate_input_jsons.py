#!/usr/bin/env python3
"""Generate pre-tokenized JSON input files for GLM-4.7-Flash ISL sweeps.

Tokenizes source .txt files and produces one JSON per target ISL in
sample_prompts/input_data_prefill_{isl}.json.  For ISLs longer than the
source text, the token sequence is repeated to reach the target length.

Usage (from tt-metal repo root with venv activated):
  python models/demos/glm4_moe_lite/scripts/generate_input_jsons.py [--model-id ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from models.demos.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir

ISL_VALUES = [128, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

SOURCE_TXT_FILES = [
    "wiki_history_us_summarize.txt",
    "wiki_artificial_intelligence_summarize.txt",
]

SAMPLE_PROMPTS_DIR = Path("models/demos/glm4_moe_lite/sample_prompts")


def tokenize_sources(tokenizer, source_dir: Path) -> list[int]:
    """Concatenate and tokenize all source .txt files into a single token pool."""
    all_text = []
    for fname in SOURCE_TXT_FILES:
        path = source_dir / fname
        if not path.is_file():
            print(f"  Warning: {path} not found, skipping")
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        all_text.append(text)
        print(f"  Loaded {fname}: {len(text)} chars")

    combined = "\n\n".join(all_text)
    enc = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    token_ids = enc["input_ids"][0].tolist()
    print(f"  Combined token pool: {len(token_ids)} tokens")
    return token_ids


def make_token_sequence(pool: list[int], target_len: int) -> list[int]:
    """Produce a token sequence of exactly target_len by truncating or repeating the pool."""
    if len(pool) >= target_len:
        return pool[:target_len]
    repeats = (target_len + len(pool) - 1) // len(pool)
    extended = (pool * repeats)[:target_len]
    return extended


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate pre-tokenized JSON inputs for GLM-4.7-Flash sweeps")
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument(
        "--isl",
        type=int,
        nargs="+",
        default=ISL_VALUES,
        help=f"ISL values to generate (default: {ISL_VALUES})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SAMPLE_PROMPTS_DIR,
        help="Output directory for JSON files",
    )
    args = ap.parse_args()

    model_id = str(args.model_id)
    snap = Path(resolve_best_effort_snapshot_dir(model_id))
    print(f"Loading tokenizer from {snap}")
    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)

    print("Tokenizing source texts...")
    source_dir = SAMPLE_PROMPTS_DIR
    pool = tokenize_sources(tokenizer, source_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for isl in sorted(args.isl):
        tokens = make_token_sequence(pool, isl)
        out_path = args.out_dir / f"input_data_prefill_{isl}.json"
        data = {
            "isl": isl,
            "input_ids": tokens,
            "source_pool_size": len(pool),
            "description": (
                f"Pre-tokenized {isl} tokens from GLM-4.7-Flash wiki sources. "
                f"{'Truncated' if len(pool) >= isl else 'Repeated'} from {len(pool)}-token pool."
            ),
        }
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"  Wrote {out_path} ({isl} tokens, {out_path.stat().st_size / 1024:.1f} KB)")

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
