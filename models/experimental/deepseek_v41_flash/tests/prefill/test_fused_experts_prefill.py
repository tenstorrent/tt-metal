# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek_prefill.fused_experts_prefill`` at V4.1's shapes, on random weights.

V4.1 has H = 5120 and a per-rank I of 576 (TP4, the production split: 18 I tiles), 1152 (TP2) or 2304
(TP1). They differ from V4's H = 4096, I = 512 in what the op had to change: the I tiles split evenly over 6
cores but not 8, so the op runs 3x2 groups of 6 (20 groups instead of 15) and the 80 down shards split
unevenly over a group (14 or 13 per core, two kernel variants); and at I = 2304 a core's gate_up run may not
fit L1 beside the rest, so its shards are streamed through the weight ring once per M-block.

The checks are V4's (``_run_case`` of V4's ``test_fused_experts_prefill.py``: torch reference per checked
token through the same bf4 / bf8 round trips), at V4.1's hidden and intermediate sizes. At most 256 experts:
the scores that make the top-k unambiguous are distinct bf16-exact values.

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_fused_experts_prefill.py
"""

import pytest

from models.experimental.deepseek_v4_flash.tests.prefill.test_fused_experts_prefill import _run_case

HIDDEN = 5120

# (T, E, top_k, hot experts, path): one tile row; several experts per group and a hot expert taking every
# token (T = 480: 15 tile rows, so two chunks and an odd number of M-blocks); the decode index format.
_CASES = {
    "t32_e16_k2": (32, 16, 2, [], "ranking"),
    "t96_e32_k6_hot": (96, 32, 6, [0, 17], "ranking"),
    "t64_e40_k4_indices_bf16": (64, 40, 4, [], "indices_bf16"),
    "t480_e64_k6_hot": (480, 64, 6, [3, 21], "ranking"),
}


@pytest.mark.parametrize("inter", (576, 1152, 2304), ids=("tp4", "tp2", "tp1"))
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_fused_experts_prefill_v41(device, reset_seeds, case, inter):
    num_tokens, num_experts, top_k, hot, path = _CASES[case]
    _run_case(
        device,
        num_tokens,
        num_experts,
        top_k,
        hot,
        path,
        check_every=1 if num_tokens <= 96 else 13,
        hidden=HIDDEN,
        inter=inter,
    )


def test_fused_experts_prefill_v41_256_experts_480_tokens(device, reset_seeds):
    """Full scale at TP4: 480 tokens, 256 experts, top-6."""
    _run_case(device, 480, 256, 6, [], "ranking", check_every=29, hidden=HIDDEN, inter=576)
