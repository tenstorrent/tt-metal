# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print validated paired experiment summaries without modifying raw files."""

import json
from pathlib import Path

here = Path(__file__).resolve().parent
for path in sorted(here.glob("final-v1-*-candidate-*.jsonl")):
    baseline = Path(str(path).replace("-candidate-", "-baseline-").replace("-b2.jsonl", "-b1.jsonl"))
    if not baseline.exists():
        continue
    a, b = [json.loads(p.read_text()) for p in (baseline, path)]
    n = b["kv_len"]
    flops = 4 * b["heads"] * n * n * b["dim"]
    assert a["full_output_sha256"] == b["full_output_sha256"]
    assert a["l2_pct"] == b["l2_pct"] and a["pcc"] == b["pcc"]
    assert a["full_trace_equality"] and b["full_trace_equality"]
    r = dict(
        label=b["label"],
        n=n,
        heads=b["heads"],
        seed=b["seed"],
        distribution=b["distribution"],
        l2_pct=b["l2_pct"],
        pcc=b["pcc"],
        max_relative_pct=b["max_relative_error_pct"],
        worst_head_l2_pct=max(h["l2_pct"] for h in b["per_head"]),
        bit_exact=True,
        baseline_ms=a["trace_median_ms"],
        candidate_ms=b["trace_median_ms"],
        baseline_tflops=flops / (a["trace_median_ms"] * 1e9),
        candidate_tflops=flops / (b["trace_median_ms"] * 1e9),
        time_reduction_pct=100 * (1 - b["trace_median_ms"] / a["trace_median_ms"]),
        sustained_timing=b["benchmark_warmup"] == 40,
    )
    print(json.dumps(r))
