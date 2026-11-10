# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek_prefill.fused_experts_prefill`` at V4.1's shapes, on random weights.

V4.1 has H = 5120 and a per-rank I of 576 (TP4, the production split: 18 I tiles), 1152 (TP2) or 2304
(TP1). They differ from V4's H = 4096, I = 512 in what the op had to change: the I tiles split evenly over 6
cores but not 8, so the op runs 3x2 groups of 6 (20 groups instead of 15) and the 80 down shards split
unevenly over a group (14 or 13 per core, two kernel variants); and at I = 2304 a core's gate_up run may not
fit L1 beside the rest, so its shards are streamed through the weight ring once per M-block.

``banked`` stores gate_up the way the model's experts do (``balance_dram_banks=True``, the
:class:`~...tt.model.DeepSeekV41Model` default): each [H, 64] column as K-row pieces, enough of them that every
DRAM bank holds the same number of shards. The bank count is the model's
``system_config.moe.fused_dram_banks``, not the device's DRAM grid -- the model shards over the configured
count, so a test that used the grid size would be exercising a layout production never builds. They agree on
Blackhole today; ``test_fused_experts_prefill_dram_banks_match_model`` pins that.

The checks are V4's (``_run_case`` of V4's ``test_fused_experts_prefill.py``: torch reference per checked
token through the same bf4 / bf8 round trips), at V4.1's hidden and intermediate sizes. ``_make_routing``
builds the top-k out of distinct bf16-exact values, which caps it at 256 experts -- above that the bf16
scores tie, and ``test_fused_experts_prefill_v41_384_experts_1024_tokens`` relies on the ties being
harmless: they are all among the selected experts, and every selected score still rounds above the largest
unselected one, so the *set* of top-6 is the same as in fp32.

The token cap is what makes a prefill chunk one op call: the model runs chunks of 1024 tokens and splits
``_routed`` at ``MAX_OP_TOKENS``, so a cap of 1024 halves the weight traffic of every layer (see
``test_moe_chunk.py`` for the end-to-end version).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_fused_experts_prefill.py
"""

import pytest

from models.experimental.deepseek_v4_flash.tests.prefill.test_fused_experts_prefill import _run_case
from models.experimental.deepseek_v4_flash.tt.decode.moe import _gate_up_k_shards
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config

HIDDEN = 5120
# The model's TP4 per-rank intermediate size (I = 2304 over 4 ranks).
INTER_TP4 = 576


def _model_dram_banks() -> int:
    """The bank count the model shards its experts over (``system_config.moe.fused_dram_banks``)."""
    return active_system_config().moe.fused_dram_banks


def _k_shards(inter: int, banked: bool) -> int:
    """The model's gate_up K-split for this ``inter``, or 1 for whole-K (``balance_dram_banks=False``)."""
    return _gate_up_k_shards(inter // 32, HIDDEN // 32, _model_dram_banks()) if banked else 1


def test_fused_experts_prefill_dram_banks_match_model(device):
    """The model shards over ``fused_dram_banks``; the device's DRAM grid is a different number.

    They agree on today's hardware, which is why the tests below can use either. Pinning it means a change
    to the configured count (or a board whose grid differs) fails here instead of silently making every
    ``banked`` case exercise a layout production never builds.
    """
    grid = device.dram_grid_size().x
    configured = _model_dram_banks()
    assert configured == grid, (
        f"the model shards its experts over system_config.moe.fused_dram_banks={configured}, but this device's "
        f"DRAM grid is {grid} wide: the banked cases build the configured layout, so check they still match the "
        f"one the model builds"
    )


# (T, E, top_k, hot experts, path): one tile row; several experts per group and a hot expert taking every
# token (T = 480: 15 tile rows, so two chunks and an odd number of M-blocks); the decode index format.
_CASES = {
    "t32_e16_k2": (32, 16, 2, [], "ranking"),
    "t96_e32_k6_hot": (96, 32, 6, [0, 17], "ranking"),
    "t64_e40_k4_indices_bf16": (64, 40, 4, [], "indices_bf16"),
    "t480_e64_k6_hot": (480, 64, 6, [3, 21], "ranking"),
    "t1024_e64_k6_hot": (1024, 64, 6, [3, 21], "ranking"),
}


@pytest.mark.parametrize("banked", (False, True), ids=("whole_k", "banked"))
@pytest.mark.parametrize("inter", (576, 1152, 2304), ids=("tp4", "tp2", "tp1"))
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_fused_experts_prefill_v41(device, reset_seeds, case, inter, banked):
    num_tokens, num_experts, top_k, hot, path = _CASES[case]
    k_shards = _k_shards(inter, banked)
    if banked and k_shards == 1:
        pytest.skip(f"I = {inter}'s gate_up columns already cover every DRAM bank")
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
        gate_up_k_shards=k_shards,
        dram_banks=_model_dram_banks(),
    )


@pytest.mark.parametrize("banked", (False, True), ids=("whole_k", "banked"))
def test_fused_experts_prefill_v41_256_experts_480_tokens(device, reset_seeds, banked):
    """Full scale at TP4: 480 tokens, 256 experts, top-6."""
    _run_case(
        device,
        480,
        256,
        6,
        [],
        "ranking",
        check_every=29,
        hidden=HIDDEN,
        inter=INTER_TP4,
        gate_up_k_shards=_k_shards(INTER_TP4, banked),
        dram_banks=_model_dram_banks(),
    )


@pytest.mark.parametrize("banked", (False, True), ids=("whole_k", "banked"))
def test_fused_experts_prefill_v41_384_experts_1024_tokens(device, reset_seeds, banked):
    """The production prefill shape: 1024 tokens (one whole chunk), 384 experts, top-6, TP4, balanced DRAM.

    6144 (token, expert) pairs over 384 experts is 16 tokens per expert -- still one 32-row tile row, so the
    op reads the same 1.7 GB of weights as the 512-token call it replaces, for twice the tokens. Only one in
    127 tokens is checked against torch: the reference is host fp32 over 384 experts.
    """
    _run_case(
        device,
        1024,
        384,
        6,
        [],
        "ranking",
        check_every=127,
        hidden=HIDDEN,
        inter=INTER_TP4,
        gate_up_k_shards=_k_shards(INTER_TP4, banked),
        dram_banks=_model_dram_banks(),
    )
