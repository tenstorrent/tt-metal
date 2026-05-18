"""Count total tokens across all .bin shards in a tokenized split directory.

Each shard is a flat uint16 array (2 bytes per token), so total tokens for the
split is just sum(file_size) / 2. This script walks the directory, sums shard
sizes, and prints the total along with a per-worker breakdown.

Usage:
    python count_bin_tokens.py
    python count_bin_tokens.py --dir /data/awliu/datasets/SlimPajama-627B-tokenized/train
    python count_bin_tokens.py --dir /path/to/train --pattern 'train_w*.bin'
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DIR = "/data/awliu/datasets/SlimPajama-627B-tokenized/train"
DEFAULT_PATTERN = "*_w*_s*.bin"
DTYPE = np.uint16
BYTES_PER_TOKEN = np.dtype(DTYPE).itemsize

WORKER_RE = re.compile(r"_w(\d+)_s\d+\.bin$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path(DEFAULT_DIR), help=f"Directory containing .bin shards (default: {DEFAULT_DIR})"
    )
    parser.add_argument(
        "--pattern", type=str, default=DEFAULT_PATTERN, help=f"Glob pattern for shards (default: {DEFAULT_PATTERN})"
    )
    parser.add_argument("--per-worker", action="store_true", help="Print a per-worker breakdown.")
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        metavar="N",
        help="Open each shard and print its first N tokens (0 disables, default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dir.is_dir():
        raise SystemExit(f"Directory does not exist: {args.dir}")

    files = sorted(args.dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} in {args.dir}")

    total_bytes = 0
    per_worker_bytes: dict[int, int] = defaultdict(int)
    per_worker_count: dict[int, int] = defaultdict(int)

    if args.peek > 0:
        print(f"First {args.peek} tokens of each shard (dtype={np.dtype(DTYPE).name}):")

    for f in files:
        size = f.stat().st_size
        total_bytes += size
        m = WORKER_RE.search(f.name)
        if m:
            wid = int(m.group(1))
            per_worker_bytes[wid] += size
            per_worker_count[wid] += 1

        if args.peek > 0:
            n = min(args.peek, size // BYTES_PER_TOKEN)
            head = np.fromfile(f, dtype=DTYPE, count=n) if n > 0 else np.empty(0, dtype=DTYPE)
            print(f"  {f.name}: {head.tolist()}")

    if total_bytes % BYTES_PER_TOKEN != 0:
        print(
            f"WARNING: total bytes {total_bytes} is not divisible by {BYTES_PER_TOKEN} " f"— a shard may be truncated."
        )

    total_tokens = total_bytes // BYTES_PER_TOKEN
    print(f"Directory:       {args.dir}")
    print(f"Pattern:         {args.pattern}")
    print(f"Shards found:    {len(files):,}")
    print(f"Total bytes:     {total_bytes:,}")
    print(f"Total tokens:    {total_tokens:,}")
    print(f"                 {total_tokens / 1e9:.3f} B tokens " f"({total_tokens / 1e6:.1f} M tokens)")

    if args.per_worker and per_worker_bytes:
        print("\nPer-worker breakdown:")
        print(f"{'wid':>4} {'shards':>7} {'tokens':>18} {'B tokens':>10}")
        for wid in sorted(per_worker_bytes):
            tokens = per_worker_bytes[wid] // BYTES_PER_TOKEN
            print(f"{wid:>4} {per_worker_count[wid]:>7} {tokens:>18,} {tokens / 1e9:>10.3f}")


if __name__ == "__main__":
    main()
