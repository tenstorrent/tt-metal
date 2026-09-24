# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicRouter. Bring-up gate 5, the hardest gate in the port.

Routing is a discrete decision, so PCC is the wrong instrument: a token sent to the wrong pair of
experts has a completely different output, and one token in a thousand doing that barely moves a
PCC computed over 768 features. The gate is therefore index agreement, with every disagreement
shown to be a near-tie that the softmax's own error budget explains.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicRouter
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    MOE_LAYER,
    TOKEN_SHAPES,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.router import TtNomicRouter
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

MODULE_PCC = 0.999

# Gate 5: at least this fraction of tokens must reach the same two experts as torch.
MIN_AGREEMENT = 0.99

# The softmax's measured max-abs budget under the port's kernel config is 1.4e-3 to 1.9e-3, and
# the router matmul adds its own. A disagreement is only acceptable if the two probabilities at
# the selection boundary sit closer together than this, which is what makes it a tie rather than
# a wrong answer.
NEAR_TIE_MARGIN = 5e-3

PREFIX = f"encoder.layers.{MOE_LAYER}.mlp.router."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(
        lambda: NomicRouter(config.hidden_size, config.num_experts, config.moe_top_k), state_dict, PREFIX
    )


@pytest.fixture
def tt_router(device, config, tt_config, state_dict):
    return TtNomicRouter(device, config, tt_config, state_dict, PREFIX)


def flat_input(x: torch.Tensor, device) -> ttnn.Tensor:
    """(B, S, H) -> the flat (1, 1, T, H) device tensor the router consumes."""
    return flatten_tokens(to_device(to_block_layout(x), device))


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_router_probabilities(device, config, reference, tt_router, batch, seqlen):
    """The fp32 softmax over all eight experts, before any selection."""
    x = hidden_states(batch, seqlen, config.hidden_size)

    probabilities, _, _ = tt_router.select(flat_input(x, device))

    with torch.no_grad():
        ref, _, _ = reference(x)
    assert probabilities.dtype == ttnn.float32
    assert_with_pcc(ref.reshape(1, 1, batch * seqlen, config.num_experts), probabilities, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_selection_agrees_with_torch(device, config, reference, tt_router, batch, seqlen):
    """Gate 5: which experts each token visits, and what every disagreement costs.

    Agreement is measured on the expert set rather than the ordered index pair, because the two
    are used as an unordered pair: each selected weight multiplies its own expert's output, so
    swapping the two positions leaves the token's output unchanged.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)
    tokens = batch * seqlen

    _, values, indices = tt_router.select(flat_input(x, device))

    with torch.no_grad():
        probabilities, ref_values, ref_indices = reference(x)
    selected = ttnn.to_torch(indices).long().reshape(tokens, config.moe_top_k)

    assert int(selected.max()) < config.num_experts
    agreeing = torch.zeros(tokens, dtype=torch.bool)
    for token in range(tokens):
        agreeing[token] = set(selected[token].tolist()) == set(ref_indices[token].tolist())

    agreement = float(agreeing.float().mean())
    assert agreement >= MIN_AGREEMENT, f"only {agreement:.4f} of tokens reached the same experts"

    # Every disagreement has to be a near-tie at the selection boundary, not a wrong answer.
    sorted_probabilities = probabilities.sort(dim=-1, descending=True).values
    margins = sorted_probabilities[:, config.moe_top_k - 1] - sorted_probabilities[:, config.moe_top_k]
    disagreeing_margins = margins[~agreeing]
    if disagreeing_margins.numel():
        worst = float(disagreeing_margins.max())
        assert worst < NEAR_TIE_MARGIN, (
            f"{int((~agreeing).sum())} tokens disagreed and the widest boundary margin was "
            f"{worst:.3e}, above the {NEAR_TIE_MARGIN:.0e} the softmax error can explain"
        )

    # On the tokens that agree, the weights themselves must match.
    agreeing_values = ttnn.to_torch(values).float().reshape(tokens, config.moe_top_k)[agreeing]
    assert_with_pcc(ref_values[agreeing], agreeing_values, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_dense_routing_weights(device, config, tt_config, reference, tt_router, batch, seqlen):
    """forward's dense (1, 1, T, E) output, which is what the expert gate multiply consumes."""
    x = hidden_states(batch, seqlen, config.hidden_size)
    tokens = batch * seqlen

    dense = tt_router(flat_input(x, device))

    with torch.no_grad():
        _, ref_values, ref_indices = reference(x)
    ref = reference.dense_weights(ref_values, ref_indices)
    got = ttnn.to_torch(dense).float().reshape(tokens, config.num_experts)

    # bfloat16, not fp32: ttnn.scatter rejects fp32, and the cast lands after the selection so
    # the routing decision itself is still made in fp32.
    assert dense.dtype == tt_config.activation_dtype
    assert torch.equal((got != 0).sum(-1), torch.full((tokens,), config.moe_top_k))
    assert_with_pcc(ref.reshape(1, 1, tokens, config.num_experts), dense, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_weights_reach_the_gate_unrenormalized(device, config, tt_router, batch, seqlen):
    """moe_normalize_expert_weights is false, so the routed weights must sum below 1.

    Dividing by the top-k sum is what Mixtral and Switch do and the reflex to copy here; it
    still scores about 0.993 PCC end to end, which sits right on a typical gate.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)

    dense = tt_router(flat_input(x, device))

    row_sums = ttnn.to_torch(dense).float().reshape(batch * seqlen, config.num_experts).sum(-1)
    assert (row_sums < 1.0).all(), f"a routed row summed to {float(row_sums.max()):.4f}"
