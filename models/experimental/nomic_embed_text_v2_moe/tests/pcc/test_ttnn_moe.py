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

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicMoELayer
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    MOE_LAYER,
    TOKEN_SHAPES,
    from_block_layout,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.moe import TtNomicMoELayer
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.998

# Gate 5 measures routing agreement properly. This is the loose ceiling that keeps a routing
# regression from hiding inside a PCC taken over agreeing tokens only.
MAX_DISAGREEING_FRACTION = 0.01

BIAS_MISPLACEMENT_MAX_ABS = 1e-3

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


def test_the_shared_bias_lands_outside_the_reduce(device, config, reference, tt_moe, tt_config):
    """Negative control: the bias placement PCC cannot see.

    Adding the shared expert bias inside the per-expert loop scales it by the routed-weight sum,
    leaving a nearly constant offset of (sum(w) - 1) * bias.

    As in test_ttnn_experts.py, the two oracles are built from the module's own bias-zeroed
    output rather than from the reference, so the bfloat16 noise is common to both and cancels.
    At real weights that noise is 0.22 against an offset far below it, so comparing against the
    reference instead would measure the dtype rather than the placement.
    """
    batch, seqlen = 2, 128
    x = hidden_states(batch, seqlen, config.hidden_size)
    x_tt = to_device(to_block_layout(x), device)

    with torch.no_grad():
        _, top_weights, _ = reference.router(x)

    got = from_block_layout(tt_moe(x_tt))
    tt_moe.experts.bias = to_device(torch.zeros(1, 1, 1, config.hidden_size), device, dtype=tt_config.weight_dtype)
    weighted_sum = from_block_layout(tt_moe(x_tt))

    bias = reference.experts.bias.detach()
    routed_weight_sum = top_weights.sum(-1).reshape(batch, seqlen, 1)
    correct = weighted_sum + bias
    inside_the_loop = weighted_sum + bias * routed_weight_sum

    assert (routed_weight_sum < 1.0).all(), "top-k weights are not renormalized, so they must sum below 1"
    assert compute_max_abs_error(correct, inside_the_loop) > BIAS_MISPLACEMENT_MAX_ABS
    assert compute_pcc(inside_the_loop, correct) > 0.9999, "if PCC caught this, the docstring is stale"
    assert compute_max_abs_error(got, correct) < compute_max_abs_error(got, inside_the_loop)
