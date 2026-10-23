# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Read the PLE n-gram table files through the page cache and report their residency.

The 128 table parts of checkpoint layer 1 live in 33 safetensors files
(104,298,732,704 B).  A decode token reads sixteen 320 B rows from them; on a
cold cache each row is an NVMe page-in that the runner's WILLNEED batch only
overlaps.  One sequential pass over the files makes every later row a
page-cache hit, and the residency report labels a run cold or warm.

Measured on a host with 755 GB RAM (707 GB available, ext4 on LVM over two Samsung
PM9A3 NVMe drives): about 35-60 s for the 104 GB at 2-3 GB/s on a cold cache, a
few seconds when the files are already resident; the table stays resident and
evictable.  Nothing is written; no model tensor is decoded.  Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint

EXPECTED_TABLE_PARTS = 128
EXPECTED_TABLE_FILES = 33


def table_files(checkpoint: Qwen38Checkpoint) -> list[Path]:
    """The safetensors files holding the 128 PLE table parts, in name order."""

    prefix = (
        f"model.language_model.layers.{checkpoint.config.ple_checkpoint_layer}.ple.ple_embedding.ngram_embedding.shard_"
    )
    names = checkpoint.names_with_prefix(prefix)
    if len(names) != EXPECTED_TABLE_PARTS:
        raise RuntimeError(f"expected {EXPECTED_TABLE_PARTS} PLE table parts, found {len(names)}")
    shards = sorted({checkpoint.metadata(name).shard for name in names})
    if len(shards) != EXPECTED_TABLE_FILES:
        raise RuntimeError(f"expected the table in {EXPECTED_TABLE_FILES} files, found {len(shards)}")
    return [checkpoint.root / shard for shard in shards]


def residency(files: list[Path]) -> dict[str, int] | None:
    """Resident bytes per file from util-linux fincore; None when fincore is absent."""

    if shutil.which("fincore") is None:
        return None
    completed = subprocess.run(
        ["fincore", "--bytes", "--noheadings", "--output", "RES,SIZE,FILE", *map(str, files)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    resident: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            resident[parts[-1]] = int(parts[0])
    return resident


def read_through_page_cache(path: Path, buffer: bytearray) -> tuple[int, float]:
    """Sequential WILLNEED + preadv pass over one file; returns bytes read and seconds."""

    started = time.perf_counter()
    size = path.stat().st_size
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        fadvise = getattr(os, "posix_fadvise", None)
        if fadvise is not None:
            fadvise(descriptor, 0, 0, os.POSIX_FADV_WILLNEED)
        view = memoryview(buffer)
        offset = 0
        while offset < size:
            count = os.preadv(descriptor, [view[: min(len(buffer), size - offset)]], offset)
            if count <= 0:
                raise OSError(f"short read at offset {offset} of {path}")
            offset += count
    finally:
        os.close(descriptor)
    return size, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="pinned checkpoint root")
    parser.add_argument("--chunk-mib", type=int, default=64, help="read size per preadv call")
    parser.add_argument("--report-only", action="store_true", help="only print the residency report")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary as JSON")
    args = parser.parse_args()
    if args.chunk_mib < 1:
        raise SystemExit("--chunk-mib must be positive")

    files = table_files(Qwen38Checkpoint(args.checkpoint))
    total_bytes = sum(path.stat().st_size for path in files)
    before = residency(files)
    summary = {
        "checkpoint": str(args.checkpoint),
        "files": len(files),
        "total_bytes": total_bytes,
        "resident_bytes_before": None if before is None else sum(before.values()),
        "fincore_available": before is not None,
        "read": None,
    }
    if before is not None:
        print(
            f"resident before: {sum(before.values()):,} / {total_bytes:,} B ({100 * sum(before.values()) / total_bytes:.2f}%)"
        )
    else:
        print("fincore is not on PATH: residency cannot be reported")

    if not args.report_only:
        buffer = bytearray(args.chunk_mib << 20)
        read_bytes = 0
        started = time.perf_counter()
        for index, path in enumerate(files, start=1):
            size, seconds = read_through_page_cache(path, buffer)
            read_bytes += size
            print(
                f"[{index:2d}/{len(files)}] {path.name} {size / 1e9:7.3f} GB in {seconds:6.2f} s "
                f"({size / 1e9 / max(seconds, 1e-9):5.2f} GB/s); total {read_bytes / 1e9:7.3f} GB",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        summary["read"] = {
            "bytes": read_bytes,
            "seconds": elapsed,
            "gb_per_second": read_bytes / 1e9 / max(elapsed, 1e-9),
        }
        print(f"read {read_bytes:,} B in {elapsed:.1f} s ({read_bytes / 1e9 / max(elapsed, 1e-9):.2f} GB/s)")

    after = residency(files)
    summary["resident_bytes_after"] = None if after is None else sum(after.values())
    summary["resident_fraction_after"] = None if after is None else sum(after.values()) / total_bytes
    if after is not None:
        print(
            f"resident after: {sum(after.values()):,} / {total_bytes:,} B ({100 * sum(after.values()) / total_bytes:.2f}%)"
        )
        summary["files_fully_resident"] = sum(after[str(path)] >= path.stat().st_size for path in files)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PREWARM_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
