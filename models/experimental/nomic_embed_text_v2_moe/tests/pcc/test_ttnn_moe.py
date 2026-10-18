# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicMoELayer, router and experts composed. Bring-up gate 7.

Routing is live here rather than injected, so a token whose routing disagrees with torch has a
legitimately different output and would drag the PCC down for a reason that is already measured
by gate 5. PCC is therefore taken over the tokens whose routing agrees, with the disagreeing
count reported and bounded.

Both negative controls are at this level because both are properties of how the router's output
meets the experts, which neither module sees on its own.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicMoELayer
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    MOE_LAYER,
    TOKEN_SHAPES,
    assert_bias_added_once,
    from_block_layout,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import HELD_MAX_TILES, StackedBuffers
from models.experimental.nomic_embed_text_v2_moe.tt.moe import TtNomicMoELayer
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.998

# Gate 5 measures routing agreement properly. This is the loose ceiling that keeps a routing
# regression from hiding inside a PCC taken over agreeing tokens only.
MAX_DISAGREEING_FRACTION = 0.01

PREFIX = f"encoder.layers.{MOE_LAYER}.mlp."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(lambda: NomicMoELayer(config), state_dict, PREFIX)


@pytest.fixture
def tt_moe(device, config, tt_config, state_dict):
    return TtNomicMoELayer(device, config, tt_config, state_dict, PREFIX)


def agreeing_tokens(config, reference, tt_moe, x: torch.Tensor, x_tt: ttnn.Tensor) -> torch.Tensor:
    """(B, S) bool mask of tokens that both sides routed to the same pair of experts.

    The pair is compared as a set: each selected weight multiplies its own expert's output, so
    the order of the two positions does not reach the result.
    """
    batch, seqlen, _ = x.shape
    _, values, indices = tt_moe.router.select(flatten_tokens(x_tt))
    selected = ttnn.to_torch(indices).long().reshape(batch * seqlen, config.moe_top_k)

    with torch.no_grad():
        _, _, ref_indices = reference.router(x)

    agreeing = torch.tensor(
        [set(selected[token].tolist()) == set(ref_indices[token].tolist()) for token in range(batch * seqlen)]
    )
    return agreeing.reshape(batch, seqlen)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_moe_layer(device, config, reference, tt_moe, batch, seqlen):
    """Route each token and combine its experts, against the reference layer.

    The reference's own forward takes the ragged path; dense_forward is the equivalent the port
    shares, and test_dense_forward_matches_the_ragged_loop holds the two together.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    out = tt_moe(x_tt)

    with torch.no_grad():
        ref = reference(x)
    got = from_block_layout(out)
    agreeing = agreeing_tokens(config, reference, tt_moe, x, x_tt)

    disagreeing = float((~agreeing).float().mean())
    assert disagreeing <= MAX_DISAGREEING_FRACTION, f"{disagreeing:.4f} of tokens routed differently"
    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert_with_pcc(ref[agreeing], got[agreeing], MODULE_PCC)


def test_renormalizing_the_routed_weights_is_measurably_wrong(device, config, reference, tt_moe):
    """Negative control: moe_normalize_expert_weights is false in this checkpoint.

    Dividing the top-2 weights by their sum is what Mixtral and Switch do and the reflex to copy.
    It runs, it produces plausible output, and it scores around 0.993 PCC, which sits right on a
    typical 0.99 gate. Asserted below this module's own gate so it cannot pass as precision loss.
    """
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)
    flat = flatten_tokens(x_tt)

    with torch.no_grad():
        ref = reference(x)
    agreeing = agreeing_tokens(config, reference, tt_moe, x, x_tt)

    dense = tt_moe.router(flat)
    renormalized = ttnn.divide(dense, ttnn.sum(dense, dim=-1, keepdim=True))

    correct = from_block_layout(tt_moe(x_tt))
    wrong = ttnn.to_torch(tt_moe.experts(flat, renormalized)).float().reshape(batch, seqlen, config.hidden_size)

    assert compute_pcc(correct[agreeing], ref[agreeing]) > MODULE_PCC
    assert compute_pcc(wrong[agreeing], ref[agreeing]) < MODULE_PCC


@pytest.mark.parametrize("batch, seqlen", [(2, 128), (2, 512)])
def test_the_shared_bias_lands_outside_the_reduce(device, config, reference, tt_moe, tt_config, batch, seqlen):
    """Negative control: the bias placement PCC cannot see.

    Adding the shared expert bias inside the per-expert loop scales it by the routed-weight sum,
    leaving a nearly constant offset of (sum(w) - 1) * bias.

    As in test_ttnn_experts.py, the placement is read off the module's own bias-zeroed output
    rather than the reference: at real weights the bfloat16 noise is 0.22 against an offset far
    below it, so comparing against the reference would measure the dtype rather than the
    placement. The output's difference from the bias-zeroed run is projected onto the bias over every
    token, which averages away the rounding the two runs do differently when the bias is added inside
    a matmul. 2x128 runs a stacked expert pass, 2x512 a transposed one.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    with torch.no_grad():
        _, top_weights, _ = reference.router(x)

    got = from_block_layout(tt_moe(x_tt))
    tt_moe.experts.bias = to_device(torch.zeros(1, 1, 1, config.hidden_size), device, dtype=tt_config.weight_dtype)
    weighted_sum = from_block_layout(tt_moe(x_tt))

    assert_bias_added_once(
        got, weighted_sum, reference.experts.bias.detach(), top_weights.sum(-1).reshape(batch, seqlen, 1)
    )


@pytest.mark.parametrize("batch, seqlen", [(1, 128), (2, 64), (5, 1), (120, 1)])
def test_buffers_are_held_only_for_a_small_block_input(device, config, tt_config, state_dict, batch, seqlen):
    """A stacked pass's buffers are held only for at most HELD_MAX_TILES tile rows as the blocks count them.

    The blocks pad each sequence to the tile, so the flat token axis undercounts a batch of short
    sequences: 120 one-token sequences are 4 tile rows to the experts and 120 to every dense layer.
    Held at 120x1, the buffers overflowed L1 beside the next layer's fc1.
    """
    buffers = StackedBuffers()
    moe = TtNomicMoELayer(device, config, tt_config, state_dict, PREFIX, buffers=buffers)
    x_tt = to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device)
    try:
        moe(x_tt)
        held = buffers.get(batch * seqlen) is not None
    finally:
        buffers.release()

    assert held == (batch * ttnn.core.divup(seqlen, ttnn.TILE_SIZE) <= HELD_MAX_TILES)
