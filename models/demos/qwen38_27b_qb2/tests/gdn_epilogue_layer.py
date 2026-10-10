# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight epilogue acceptance, separate from full-model qualification."""

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings

BATCHES = (32, 16, 8, 1)
VARIANTS = ("native", "fused", "native")
POLICIES = dict(native="single_step_shared_qk", fused="single_step_shared_qk_epilogue")
CANDIDATES = {
    "epilogue": "single_step_shared_qk_epilogue",
    "flat_prepare": "single_step_flat_prepare_epilogue",
}


def compare(cases):
    if len(cases) != 3 or tuple(case["variant"] for case in cases) != VARIANTS:
        raise ValueError("Require native/fused/native real-weight epilogue controls")
    before = cases[0]
    if before["batch"] not in BATCHES or any(case["batch"] != before["batch"] for case in cases):
        raise ValueError("Mismatched or unsupported batch")
    if not all(case["passed"] for case in cases):
        raise ValueError("Real-weight FP32 reference check failed")
    for key, length in (
        ("input_sha256", 20),
        ("state_sha256_per_rank", 4),
        ("output_sha256_per_rank", 4),
        ("projected_output_sha256_per_rank", 4),
    ):
        if len(before[key]) != length or any(case[key] != before[key] for case in cases):
            raise ValueError("Real-weight operands, state or projected output changed: " + key)
    return dict(
        compare_timings(cases),
        batch=before["batch"],
        projected_output_and_state_bit_identical=True,
        scope="Real-weight GDN block including projection, convolution, recurrence and epilogue; MLP excluded",
        fused_epilogue_selected=before["batch"] in (16, 32),
    )
