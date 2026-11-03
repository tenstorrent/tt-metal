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
    selection_of,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import routed_windows
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
    """The fp32 softmax over the eight experts, before any selection, and exact zeros past them."""
    x = hidden_states(batch, seqlen, config.hidden_size)

    routing, rows = tt_router(flat_input(x, device))
    ttnn.deallocate(rows)

    with torch.no_grad():
        ref, _, _ = reference(x)
    assert routing.scores.dtype == ttnn.float32
    # The experts' width: the softmax masks the tile's padded columns over it (TtNomicRouter.forward).
    assert routing.scores.shape[-1] == config.num_experts
    scored = ttnn.to_torch(routing.scores)
    routing.deallocate()
    assert_with_pcc(ref.reshape(1, 1, batch * seqlen, config.num_experts), scored, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_selection_agrees_with_torch(device, config, reference, tt_router, batch, seqlen):
    """Gate 5: which experts each token visits, and what every disagreement costs.

    Agreement is measured on the expert set rather than the ordered index pair, because the two
    are used as an unordered pair: each selected weight multiplies its own expert's output, so
    swapping the two positions leaves the token's output unchanged.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)
    tokens = batch * seqlen

    routing, rows = tt_router(flat_input(x, device))
    ttnn.deallocate(rows)

    with torch.no_grad():
        probabilities, ref_values, ref_indices = reference(x)
    selected_values, selected = selection_of(routing, config.moe_top_k, config.num_experts)

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
    agreeing_values = selected_values[agreeing]
    routing.deallocate()
    assert_with_pcc(ref_values[agreeing], agreeing_values, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_routed_weights_are_not_renormalized(device, config, tt_router, batch, seqlen):
    """moe_normalize_expert_weights is false, so each token's top-k weights must sum below 1.

    Dividing by the top-k sum is what Mixtral and Switch do and the reflex to copy here; it
    still scores about 0.993 PCC end to end, which sits right on a typical gate.
    """
    x = hidden_states(batch, seqlen, config.hidden_size)

    routing, rows = tt_router(flat_input(x, device))
    ttnn.deallocate(rows)

    row_sums = selection_of(routing, config.moe_top_k, config.num_experts)[0].sum(-1)
    routing.deallocate()
    assert (row_sums < 1.0).all(), f"a token's weights summed to {float(row_sums.max()):.4f}"


@pytest.mark.parametrize("batch, seqlen", [*TOKEN_SHAPES, (16, 512)])
def test_routing_metadata_from_the_router(device, config, tt_router, batch, seqlen):
    """The routed pass's metadata, selected on device from the router's own scores, in its passes:
    the oracle's for the scores' top-k, and bit for bit what ttnn.topk's selection of them gives.

    16x512 puts the scores in DRAM and splits them into two passes; 2x37 leaves the scores' tile
    padding rows inside the pass's last tile row.
    """
    # Imported here, so a moved op test tree fails this test alone rather than the module.
    from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
        assert_matches_reference,
    )

    tokens = batch * seqlen
    routing, rows = tt_router(flat_input(hidden_states(batch, seqlen, config.hidden_size), device))
    ttnn.deallocate(rows)
    values, indices = selection_of(routing, config.moe_top_k, config.num_experts)
    topk_values, topk_indices = ttnn.topk(routing.scores, k=config.moe_top_k, dim=-1)

    for token_start, num_tokens in routed_windows(tokens):
        window = dict(num_experts=config.num_experts, token_start=token_start, num_tokens=num_tokens)
        routing_metadata = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata
        outputs = routing_metadata(scores=routing.scores, top_k=config.moe_top_k, **window)
        assert_matches_reference(outputs, indices, values, config.num_experts, token_start, num_tokens)
        for got, want in zip(outputs, routing_metadata(topk_indices, topk_values, **window)):
            assert torch.equal(ttnn.to_torch(got).view(torch.int32), ttnn.to_torch(want).view(torch.int32))


def test_one_call_covers_the_largest_batches(device, config, reference, tt_router):
    """The router takes a whole batch in one call, so a core's 32-token blocks grow without bound with it.

    The experts split the token axis into passes; the router does not. router_front_end runs a core's blocks one
    after another through fixed buffers, so 131,072 tokens, 256 texts of 512 and about 38 blocks a core, fit L1
    as one block does.
    """
    x = hidden_states(256, 512, config.hidden_size)

    routing, rows = tt_router(flat_input(x, device))
    ttnn.deallocate(rows)
    scores = ttnn.to_torch(routing.scores).float()
    routing.deallocate()

    with torch.no_grad():
        ref, _, _ = reference(x)
    assert_with_pcc(ref.reshape(1, 1, -1, config.num_experts), scores, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_rows_are_the_input_untilized(device, config, tt_router, batch, seqlen):
    """The dispatch table the router writes beside its scores: the input's rows bit for bit, its tile padding left
    out."""
    x_tt = flat_input(hidden_states(batch, seqlen, config.hidden_size), device)

    routing, rows = tt_router(x_tt)
    routing.deallocate()

    assert list(rows.shape) == [1, 1, batch * seqlen, config.hidden_size]
    assert rows.layout == ttnn.ROW_MAJOR_LAYOUT
    assert torch.equal(ttnn.to_torch(rows), ttnn.to_torch(x_tt))
