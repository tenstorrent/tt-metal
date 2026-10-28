# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Floors, drift margins and golden paths. Floors change only through a decision in DECISIONS.md."""
from pathlib import Path

GOLDENS = Path("/home/andy/bringup/goldens/kairos")
TEST_GOLDENS = GOLDENS / "test"
BENCH_GOLDENS = GOLDENS / "bench"

# CONTRACT v1.1 §3 and the D0 amendments. Band / row / inactive floors are proposed at G1 (None = not set yet).
FLOORS = {
    "pcc_normalized": 0.999,  # minimum (D0 item 5); every suite and bench case (PLAN v1.20)
    "pcc_normalized_full": 0.999,  # both decoder tokens, 128 steps (D0 amendment 4)
    "pcc_final": 0.99,
    "route_mismatch_rows": 0,  # exact routing (D0 amendment 1)
    "bias_max_abs_diff": 1e-6,  # D0 amendment 2
    "module_pcc": 0.999,
    "pcc_spread": None,
    "pcc_row_p01": None,
    "pcc_row_min": None,
    "spread_ratio_p50": None,
    "abs_err_max": None,
}
FLOORS_PERFORMANCE = {**FLOORS, "pcc_normalized": 0.995, "pcc_normalized_full": 0.995, "pcc_final": 0.98}
LONG_HORIZON_FREE_PCC_FINAL = 0.98  # provisional until G1

# Lower-is-better keys (v1.17): every one needs a unit check in test_check_gates.py.
LOWER_IS_BETTER = {"route_mismatch_rows", "bias_max_abs_diff", "abs_err_max"}
# Report-only for free-running rollout steps >= 2 and e2e_stream (D0 amendment 1).
REPORT_ONLY_ROUTING_KEYS = {"route_mismatch_rows", "bias_max_abs_diff", "min_top3_margin"}

# Device module tests (tests/pcc/test_modules.py, v1.24: asserted in both precisions). Proposed at the G1 re-raise
# from the measured minimum over the test goldens; provisional until the decision.
MODULE_FLOORS = {
    m: {"accuracy": 0.999, "performance": 0.99}
    for m in ("embed", "rope_qk", "enc_block", "enc_chain", "enc_out", "cross_attn", "dec_self")
}

DRIFT = {}  # set from the fixed round-0 record at G1

# Padded lengths and SDPA chunks the code can reach (PLAN v1.20). Each needs an F1/F9 test and a v1.13
# positive control (mask dropped on one key block) plus the negative control (chunk forced to 256).
SDPA_CASES = [
    # (name, q_len, k_len, chunk, note)
    ("enc_self_96", 96, 96, 96, "encoder L=65 padded to 96 (layout a/b)"),
    ("enc_self_96_c32", 96, 96, 32, "chunk 96 vs 32 (D0 amendment 3)"),
    ("cross_q32_k96", 32, 96, 32, "decoder 2 queries tile-padded to 32, encoder keys 96"),
    ("dec_self_q32_k32", 32, 32, 32, "causal 2x2, block >= 1 (block 0 has v0 = v1)"),
    ("enc_self_256", 256, 256, 128, "contract §7: capped chunk"),
    # ("enc_self_160", 160, 160, ..., "only if row layout (c) is chosen"),
]
