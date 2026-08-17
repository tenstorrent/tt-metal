# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Reduce the tt-perf-report filtered CSVs into one compact per-case table.

Reads `doc/functional_decoder/tracy/<kind>_<mode>/<mode>_perf_report.csv` (the
signpost-filtered output of `tt-perf-report --csv`) and sums device kernel time over the
measured window, dividing by the iteration count recorded in `perf_host_summary.jsonl`.

Units: this `tt-perf-report` (1.2.8) writes **`Device Time` in microseconds** and
`Op-to-Op Gap` in microseconds — not the raw Tracy `DEVICE KERNEL DURATION [ns]`. Both are
summed here and reported separately so device-busy time and dispatch gap are never conflated.

Also rewrites `perf_host_summary.jsonl` keeping only the newest row per (mode, kind), so the
host wall-clock rows always correspond to the artifacts currently in `tracy/`.

Each case additionally gets a three-way `blocks` split — token mixer (attention or gated delta
net), expert matmuls, and the MoE's dense-over-experts elementwise/layout work — because the
mixer/MoE boundary is what an optimization stage needs and it is not visible in a per-op-code
total. The boundary is found structurally, not hard-coded: within one iteration the MoE begins at
the last `LayerNormDeviceOperation` before the first `SparseMatmulDeviceOperation`, which is
`post_attention_layernorm`.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/summarize_perf.py
"""

import csv
import json
from collections import defaultdict

from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import ARTIFACT_DIR

CASES = [(kind, mode) for mode in ("prefill", "decode") for kind in ("linear", "full")]


def dedupe_host_summary():
    path = ARTIFACT_DIR / "perf_host_summary.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    newest = {}
    for row in rows:
        newest[(row["mode"], row["kind"])] = row  # later rows win
    path.write_text("".join(json.dumps(newest[k]) + "\n" for k in sorted(newest)))
    return newest


def block_split(rows, iters):
    """(mixer_ms, expert_matmul_ms, moe_other_ms) per iteration.

    `rows` is one signposted window holding `iters` identical iterations, so the boundary is found
    on the first iteration and the totals are divided at the end.
    """
    per = len(rows) // iters if iters else len(rows)
    first = rows[:per] if per else rows
    first_sparse = next((i for i, r in enumerate(first) if "Sparse" in r["OP Code"]), None)
    if first_sparse is None:
        return None
    norms = [i for i, r in enumerate(first[:first_sparse]) if r["OP Code"] == "LayerNormDeviceOperation"]
    boundary = norms[-1] if norms else first_sparse
    mixer = expert = other = 0.0
    for i, r in enumerate(rows):
        dt = float(r["Device Time"] or 0)
        if i % per < boundary:
            mixer += dt
        elif "Sparse" in r["OP Code"]:
            expert += dt
        else:
            other += dt
    return {
        "boundary_op_index_in_iteration": boundary,
        "mixer_ms_per_iter": round(mixer / 1e3 / iters, 3),
        "expert_matmul_ms_per_iter": round(expert / 1e3 / iters, 3),
        "moe_elementwise_ms_per_iter": round(other / 1e3 / iters, 3),
    }


def main():
    host = dedupe_host_summary()
    out = []
    for kind, mode in CASES:
        csv_path = ARTIFACT_DIR / "tracy" / f"{kind}_{mode}" / f"{mode}_perf_report.csv"
        if not csv_path.exists():
            print(f"MISSING {csv_path}")
            continue
        rows = list(csv.DictReader(csv_path.open()))
        device_us = sum(float(r["Device Time"] or 0) for r in rows)
        gap_us = sum(float(r["Op-to-Op Gap"] or 0) for r in rows)
        by_op = defaultdict(float)
        for r in rows:
            by_op[r["OP Code"]] += float(r["Device Time"] or 0)
        top = sorted(by_op.items(), key=lambda kv: -kv[1])[:5]
        meta = host.get((mode, kind), {})
        iters = meta.get("iters", 1)
        out.append(
            {
                "kind": kind,
                "mode": mode,
                "ops_in_window": len(rows),
                "iters": iters,
                "device_kernel_ms_per_iter": round(device_us / 1e3 / iters, 3),
                "op_to_op_gap_ms_per_iter": round(gap_us / 1e3 / iters, 3),
                "host_wall_ms_per_iter": meta.get("host_wall_ms_per_iter"),
                "seq_len": meta.get("seq_len"),
                "batch": meta.get("batch"),
                "current_pos": meta.get("current_pos"),
                "traced": meta.get("traced", False),
                "supported_context": meta.get("supported_context"),
                "top_ops_device_ms_per_iter": {k: round(v / 1e3 / iters, 3) for k, v in top},
                "blocks": block_split(rows, iters),
            }
        )
    path = ARTIFACT_DIR / "perf_summary.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
