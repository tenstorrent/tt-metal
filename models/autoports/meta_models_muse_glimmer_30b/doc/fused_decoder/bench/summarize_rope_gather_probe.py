# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize ``decode_rope_gather_probe.py``'s ops CSV.

``summarize_device_probe.py`` slices on ``GROUP`` markers and assumes one
device op per announced call.  This probe does not fit that shape: each "call"
is three or four ops, and it runs one unannounced correctness call of each form
first.  So the slicing is by op kind and emission order instead, which is what
this script does -- committed so the numbers in README limitation 4 and work
log 4.1 are regenerable rather than hand-derived.

    python summarize_rope_gather_probe.py [ops.csv]
"""

from __future__ import annotations

import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "tracy/probes/decode_rope_gather_ops.csv"
REPS = 16
#: ops per call: shipped = 2x(embedding, transpose, i2s); merged = embedding,
#: transpose, 2x(slice, i2s)
SHIPPED_OPS, MERGED_OPS = 6, 6


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rows = [r for r in csv.DictReader(open(path)) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    print("# decode RoPE cos/sin gather: the shipped two gathers vs one packed table")
    print("# python -m tracy -r -p -v bench/decode_rope_gather_probe.py; 16 reps each,")
    print("# preceded by one correctness call of each form (bit-identical outputs).")
    print("# Regenerate with: python bench/summarize_rope_gather_probe.py")
    print(f"# {len(rows)} profiled ops\n")

    # drop the two unannounced correctness calls: 3 embeddings, 3 transposes,
    # 2 slices, 4 reshards
    pool = list(rows)

    def drop(code: str, n: int) -> None:
        taken = 0
        for r in list(pool):
            if r["OP CODE"] == code and taken < n:
                pool.remove(r)
                taken += 1

    drop("EmbeddingsDeviceOperation", 3)
    drop("TransposeDeviceOperation", 3)
    drop("SliceDeviceOperation", 2)
    drop("InterleavedToShardedDeviceOperation", 4)

    for label, chunk in (
        ("shipped (2 gathers)", pool[: REPS * SHIPPED_OPS]),
        ("merged (1 packed table)", pool[REPS * SHIPPED_OPS :]),
    ):
        per: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
        for r in chunk:
            entry = per[r["OP CODE"]]
            entry[0] += 1
            entry[1] += float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 / REPS
        total = sum(t for _, t in per.values())
        print(f"{label:26s} {total:6.2f} us/call   ({len(chunk) // REPS} ops)")
        for code, (count, t) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            print(f"    {code:44s} x{count // REPS}  {t:6.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
