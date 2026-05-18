"""Reconstruct meta.json from existing .bin shards on disk.

The parallel preprocessor writes meta.json only at the *end* of process_split,
after all workers finish. If the run was killed mid-flight, the on-disk shards
are still valid (uint16 raw token arrays with no headers), but meta.json is
missing.

This script regenerates meta.json by walking the shard directory and reading
each shard's size. Token counts come from file_size / 2 (uint16 = 2 bytes).
Tokenizer / vocab metadata is loaded from the same HF tokenizer the original
run used. Shards are sorted by (worker_id, shard_idx) parsed from the filename
suffix to match what the live writer would have produced.

The only field that can NOT be recovered exactly is ``num_documents`` — that
counter only existed in worker process memory. Pass --num-documents-estimate
to record an estimate (recommended: leave at 0; only used for logging).

Usage:
    python derive_meta_json.py
    python derive_meta_json.py --dir /data/awliu/datasets/SlimPajama-627B-tokenized/train
    python derive_meta_json.py --split validation --shard-tokens 33570816
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DEFAULT_DIR = "/data/awliu/datasets/SlimPajama-627B-tokenized/train"
DEFAULT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEFAULT_SHARD_TOKENS = 2049 * 16384  # 33,570,816 — matches dataset_preprocessing_parallel.py default
DTYPE = np.uint16
BYTES_PER_TOKEN = np.dtype(DTYPE).itemsize

SHARD_RE = re.compile(r"^(?P<split>train|validation|test)_w(?P<wid>\d+)_s(?P<sidx>\d+)\.bin$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(DEFAULT_DIR),
        help=f"Split directory containing .bin shards (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split name (train/validation/test). Inferred from shard filenames if omitted.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER, help=f"HF tokenizer name (default: {DEFAULT_TOKENIZER})"
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help=f"Tokens per shard from the original run (default: {DEFAULT_SHARD_TOKENS}).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="num_workers field; inferred from filenames if omitted."
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=0,
        help="num_documents field. Cannot be recovered from .bin shards; "
        "defaults to 0 (loader uses this only for log output).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output meta.json path (default: <dir>/meta.json)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute meta but don't write anything; print summary to stdout."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dir.is_dir():
        raise SystemExit(f"Directory does not exist: {args.dir}")

    files = list(args.dir.glob("*_w*_s*.bin"))
    if not files:
        raise SystemExit(f"No shard files matching '*_w*_s*.bin' in {args.dir}")

    parsed: list[tuple[int, int, str, str, int]] = []  # (wid, sidx, split, filename, tokens)
    splits_seen: set[str] = set()
    workers_seen: set[int] = set()
    for f in files:
        m = SHARD_RE.match(f.name)
        if not m:
            print(f"  WARN: skipping unrecognised shard filename: {f.name}")
            continue
        size = f.stat().st_size
        if size % BYTES_PER_TOKEN != 0:
            raise SystemExit(f"Shard {f.name} has odd byte count {size}; possibly truncated.")
        tokens = size // BYTES_PER_TOKEN
        wid = int(m.group("wid"))
        sidx = int(m.group("sidx"))
        split = m.group("split")
        parsed.append((wid, sidx, split, f.name, tokens))
        splits_seen.add(split)
        workers_seen.add(wid)

    if len(splits_seen) > 1:
        raise SystemExit(f"Multiple splits found in {args.dir}: {splits_seen}. Run separately per split.")
    inferred_split = next(iter(splits_seen))
    split = args.split or inferred_split
    if args.split and args.split != inferred_split:
        print(f"  WARN: --split={args.split!r} but shards say {inferred_split!r}; using --split.")

    parsed.sort(key=lambda r: (r[0], r[1]))  # (worker_id, shard_idx)

    total_tokens = sum(r[4] for r in parsed)
    num_workers = args.num_workers if args.num_workers is not None else len(workers_seen)

    print(f"Loading tokenizer to recover vocab/bos/eos: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.bos_token_id is None or tok.eos_token_id is None:
        raise SystemExit(f"Tokenizer {args.tokenizer!r} missing BOS/EOS ids.")

    meta = {
        "split": split,
        "dtype": np.dtype(DTYPE).name,
        "tokenizer": args.tokenizer,
        "vocab_size": int(tok.vocab_size),
        "bos_id": int(tok.bos_token_id),
        "eos_id": int(tok.eos_token_id),
        "shard_tokens": int(args.shard_tokens),
        "total_tokens": int(total_tokens),
        "num_documents": int(args.num_documents),
        "num_workers": int(num_workers),
        "shards": [{"file": fname, "tokens": int(tokens)} for _, _, _, fname, tokens in parsed],
    }

    print(f"\n=== Reconstructed meta for split={split!r} ===")
    print(f"  shards:         {len(meta['shards']):,}")
    print(f"  total_tokens:   {meta['total_tokens']:,} ({meta['total_tokens']/1e9:.3f} B)")
    print(f"  num_workers:    {meta['num_workers']}")
    print(f"  shard_tokens:   {meta['shard_tokens']:,}")
    print(f"  vocab_size:     {meta['vocab_size']}")
    print(f"  bos_id={meta['bos_id']}, eos_id={meta['eos_id']}")
    print(f"  num_documents:  {meta['num_documents']:,}  (NOT recoverable from .bin alone)")

    out_path = args.output or (args.dir / "meta.json")
    if args.dry_run:
        print(f"\n[dry-run] would write to {out_path}")
        return

    if out_path.exists():
        backup = out_path.with_suffix(".json.bak")
        print(f"\n  {out_path} exists; backing up to {backup}")
        out_path.rename(backup)
    with out_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
