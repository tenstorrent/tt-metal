# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score test_qwen3_tts_prefill_gaps_sweep.py against its manifest.

Each group emits exactly ONE op code, so rows are matched by op code and then chunked in
run order — never by absolute CSV position, which desynchronises the moment one arm is
refused or launches a different number of ops (the bug that corrupted the SE tap report).
Each arm launches 1 correctness call + REPS timed calls; the first is dropped as cold.
"""

from __future__ import annotations

import csv
import glob
import json

MANIFEST = "generated/prefill_gaps_manifest.json"
OP_CODE = {
    "silumul": "BinaryNgDeviceOperation",
    "oproj": "MatmulDeviceOperation",
    "sdpa": "SDPAOperation",
}
TITLE = {
    "silumul": "SiLU-mul output grid  (win => down's InterleavedToSharded, op 21, disappears)",
    "oproj": "o_proj in0 layout     (win => NLPConcatHeads' ShardedToInterleaved, op 13, disappears)",
    "sdpa": "masked-prefill SDPA k_chunk  (PERF_NOTES 6.5)",
}


def main():
    man = json.load(open(MANIFEST))
    reps = man["reps"]
    csv_path = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
    rows = [r for r in csv.DictReader(open(csv_path)) if r["OP TYPE"] == "tt_dnn_device"]
    print(f"CSV:      {csv_path}")
    print(f"manifest: {MANIFEST}  ({len(man['arms'])} arms x {reps} timed reps)\n")

    by_code: dict[str, list] = {}
    for r in rows:
        by_code.setdefault(r["OP CODE"], []).append(r)

    for group in ("silumul", "oproj", "sdpa"):
        arms = [a for a in man["arms"] if a["group"] == group]
        if not arms:
            continue
        pool = by_code.get(OP_CODE[group], [])
        need = len(arms) * (reps + 1)
        print(f"===== {TITLE[group]} =====")
        if len(pool) < need:
            print(f"  !! {OP_CODE[group]}: {len(pool)} rows, need {need} — cannot align, skipping\n")
            continue
        if len(pool) != need:
            print(f"  note: {len(pool)} rows for {need} expected; using the first {need} in run order")
        base: dict = {}
        print(f"  {'arm':46s} {'us':>8} {'cores':>6} {'vs shipped':>11}  maxdiff")
        for i, arm in enumerate(arms):
            chunk = pool[i * (reps + 1) : (i + 1) * (reps + 1)][1:]  # drop the cold launch
            us = sum(int(c["DEVICE FW DURATION [ns]"]) for c in chunk) / len(chunk) / 1000
            cores = chunk[0]["CORE COUNT"]
            key = arm.get("m"), arm.get("kv")
            if "shipped" in arm["tag"] or key not in base:
                base[key] = us
                delta = "reference"
            else:
                d = us - base[key]
                delta = f"{d:+.1f} ({100 * d / base[key]:+.0f}%)"
            md = arm.get("max_abs_diff")
            md_s = "reference" if md is None else ("BIT-EXACT" if md == 0 else f"{md:.2e}")
            print(f"  {arm['tag'][:46]:46s} {us:8.1f} {cores:>6} {delta:>11}  {md_s}")
        print()


if __name__ == "__main__":
    main()
