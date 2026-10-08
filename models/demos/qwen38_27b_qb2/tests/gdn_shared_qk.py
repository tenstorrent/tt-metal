# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qualification rules for sharing FP32 Q/K normalization across GDN work."""

import math
import statistics

BATCHES = (32, 16, 8, 64, 1)


def compare(cases):
    if len(cases) != 3 or [case["shared_qk"] for case in cases] != [False, True, False]:
        raise ValueError("Require fused/shared/fused controls")
    before, candidate, after = cases
    if before["batch"] not in BATCHES or any(
        case[key] != before[key] for case in cases[1:] for key in ("batch", "input_sha256")
    ):
        raise ValueError("Mismatched GDN operands or geometry")
    if any(not case["passed"] for case in cases):
        raise ValueError("GDN numerical gate failed")
    for key in ("state_sha256_per_rank", "output_sha256_per_rank"):
        if any(len(case[key]) != 4 or case[key] != before[key] for case in cases):
            raise ValueError("GDN output or state differs across controls")
    samples = [case["traced_call_us"] for case in cases]
    if any(len(values) != 5 for values in samples) or any(
        not math.isfinite(value) or value <= 0 for values in samples for value in values
    ):
        raise ValueError("Incomplete or invalid GDN timing")
    medians = [statistics.median(values) for values in samples]
    drift = abs(medians[2] / medians[0] - 1)
    baseline = statistics.median(samples[0] + samples[2])
    return dict(
        batch=before["batch"],
        fused_normalization_us=baseline,
        shared_normalization_us=medians[1],
        output_and_state_bit_identical=True,
        baseline_drift_fraction=drift,
        timing_comparison_qualified=drift <= 0.03,
        qualified_speedup=baseline / medians[1] if drift <= 0.03 else None,
        promoted_to_model=False,
        scope="GDN model adapter including preparation, extra program launch and output layout; no full-model claim",
    )
