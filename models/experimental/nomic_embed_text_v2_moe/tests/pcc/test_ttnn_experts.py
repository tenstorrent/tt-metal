# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC for TtNomicExperts. Bring-up gate 6.

Routing is injected rather than routed, so this measures the expert arithmetic alone: a routing
flip would otherwise land here as an expert error. The router has its own gate.

The oracle is NomicExperts.dense_forward rather than its ragged forward. The two are proved
equivalent by test_dense_forward_matches_the_ragged_loop in test_reference_vs_hf_e2e.py, so this
compares against the formulation the port shares with it and does not restate that proof.
"""

import pytest
import torch

import ttnn

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicExperts
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    MOE_LAYER,
    TOKEN_SHAPES,
    dense_routing,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import (
    MAX_TILE_ROWS_MEASURED_SAFE,
    MAX_TILE_ROWS_PER_CORE,
    TtNomicExperts,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

# Two chained matmuls, one 768-deep and one 3072-deep, in bfloat16.
MODULE_PCC = 0.998

# The shared-bias offset the wrong placement leaves behind. Measured well above this; PCC cannot
# see it at all.
BIAS_MISPLACEMENT_MAX_ABS = 1e-3

PREFIX = f"encoder.layers.{MOE_LAYER}.mlp.experts."


@pytest.fixture
def reference(config, state_dict):
    return load_reference(lambda: NomicExperts(config), state_dict, PREFIX)


@pytest.fixture
def tt_experts(device, config, tt_config, state_dict):
    return TtNomicExperts(device, config, tt_config, state_dict, PREFIX)


def run_both(device, config, reference, tt_experts, batch, seqlen):
    """Drive the reference and the module with the same input and the same injected routing.

    Returns:
        tuple: (reference output (B, S, H), module output (B, S, H), dense routing (T, E)).
    """
    tokens = batch * seqlen
    x = hidden_states(batch, seqlen, config.hidden_size)
    dense = dense_routing(tokens, config.num_experts, config.moe_top_k)

    out = tt_experts(
        flatten_tokens(to_device(to_block_layout(x), device)),
        to_device(dense.reshape(1, 1, tokens, config.num_experts), device),
    )

    with torch.no_grad():
        ref = reference.dense_forward(x, dense)
    return ref, ttnn.to_torch(out).float().reshape(batch, seqlen, config.hidden_size), dense


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_experts(device, config, reference, tt_experts, batch, seqlen):
    """Every token through every expert, gated by the routing and summed."""
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert_with_pcc(ref, got, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_the_expert_reduce_covers_every_token(device, config, reference, tt_experts, batch, seqlen):
    """fast_reduce_nc reports the tile-padded row count, so the module has to slice back to T.

    At T=74 it returns 96 rows with the trailing 22 zero. Left unsliced the logical shape is
    wrong even though the data is right, and the error would surface as a shape mismatch several
    modules later. Off-tile token counts are in TOKEN_SHAPES for exactly this.
    """
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert got.shape == ref.shape
    # No all-zero token rows: a token dropped by a bad slice would leave one behind.
    assert (got.abs().sum(dim=-1) > 0).all()


def test_the_pass_size_keeps_one_output_tile_row_per_core(device, config, tt_config, tt_experts):
    """The pass size has to hold per_core_M at 1, which is what avoids the matmul hang.

    A regression here hangs the board rather than failing an assert, so the derivation is pinned
    rather than probed; tt/experts.py carries the mechanism. The 110-tile cap is pinned apart
    from the per-core rule because the boundary did not move with the requested core grid, so
    deriving from the core count alone would silently raise the limit on a wider board. The
    operand widths are pinned because what must stay 1 is h_dim * w_dim, making the bound joint
    in M and N: at 32 tile rows, N=3072 passes and N=4096 hangs.
    """
    cores = tt_config.core_grid.x * tt_config.core_grid.y

    assert tt_experts.max_tokens_per_pass == min(cores, MAX_TILE_ROWS_MEASURED_SAFE) * ttnn.TILE_SIZE
    assert tt_experts.max_tokens_per_pass <= MAX_TILE_ROWS_MEASURED_SAFE * ttnn.TILE_SIZE
    assert MAX_TILE_ROWS_PER_CORE == 1
    assert (config.hidden_size, config.intermediate_size) == (768, 3072)


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_chunking_the_token_axis_does_not_change_the_answer(device, config, reference, tt_experts, batch, seqlen):
    """Splitting the token axis has to be exact, not merely close.

    Each pass runs the same weights over a disjoint slice and the shared bias is added once to
    the assembled result, so the split is arithmetically a no-op. Forcing a small pass size on a
    shape that fits in one is the only way to compare both paths on the same input: the shapes
    that chunk for real cannot be run unchunked, that being the geometry that hangs. The pass
    sizes cover tile-aligned and unaligned, and the smallest puts a boundary inside a sequence.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    dense = to_device(
        dense_routing(tokens, config.num_experts, config.moe_top_k).reshape(1, 1, tokens, config.num_experts), device
    )

    whole = ttnn.to_torch(tt_experts(x, dense)).float()

    for pass_size in (tokens, tokens // 2 + 1, 64, 30):
        tt_experts.max_tokens_per_pass = pass_size
        chunked = ttnn.to_torch(tt_experts(x, dense)).float()
        assert chunked.shape == whole.shape, f"pass size {pass_size} changed the shape"
        assert torch.equal(chunked, whole), (
            f"pass size {pass_size} ({-(-tokens // pass_size)} passes) changed the result: "
            f"max abs {float((chunked - whole).abs().max()):.3e}"
        )


def test_shared_bias_is_added_after_the_weighted_sum(device, config, tt_config, state_dict, reference):
    """The eight experts share one bias, added once outside the reduce.

    Adding it inside the per-expert loop scales it by the routed-weight sum, leaving an offset of
    (sum(w) - 1) * bias, which is real because the weights are deliberately not renormalized.

    Neither PCC nor a plain max-abs against the reference can see that offset here. PCC
    mean-centres it away, and at real weights the expert output reaches a magnitude of about 5,
    where the bfloat16 noise floor is 0.22 and the offset is far below it. So the noise is
    cancelled instead of tolerated: the module is run once with its bias zeroed to get the
    weighted sum on device, and both oracles are built from that same tensor. What remains
    between them is the bias placement alone.
    """
    batch, seqlen = 2, 128
    tokens = batch * seqlen
    x = hidden_states(batch, seqlen, config.hidden_size)
    dense = dense_routing(tokens, config.num_experts, config.moe_top_k)
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)

    def run():
        out = experts(
            flatten_tokens(to_device(to_block_layout(x), device)),
            to_device(dense.reshape(1, 1, tokens, config.num_experts), device),
        )
        return ttnn.to_torch(out).float().reshape(batch, seqlen, config.hidden_size)

    got = run()
    experts.bias = to_device(torch.zeros(1, 1, 1, config.hidden_size), device, dtype=tt_config.weight_dtype)
    weighted_sum = run()

    bias = reference.bias.detach()
    routed_weight_sum = dense.sum(-1).reshape(batch, seqlen, 1)
    assert (routed_weight_sum < 1.0).all(), "top-k weights are not renormalized, so they must sum below 1"

    correct = weighted_sum + bias
    inside_the_loop = weighted_sum + bias * routed_weight_sum

    assert (
        compute_max_abs_error(correct, inside_the_loop) > BIAS_MISPLACEMENT_MAX_ABS
    ), "the two placements agreed; the offset this test exists to catch is not present"
    assert compute_pcc(inside_the_loop, correct) > 0.9999, "if PCC caught this, the docstring is stale"
    assert compute_max_abs_error(got, correct) < compute_max_abs_error(got, inside_the_loop)


def test_transposed_expert_weights_are_a_shape_error(device, config, tt_config, state_dict, expect_error):
    """Negative control: pack_expert_weights is what makes the w2 misorientation loud.

    Viewing w2 as (E, H, F) rather than (E, F, H) is an equally legal reshape, since E*F*H is
    symmetric in those two, and in torch the wrong slab plus a .T typechecks and returns noise.
    As a 4D operand there is no .T to paper over it and the matmul's inner dimensions disagree.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)
    experts.w2 = to_device(
        state_dict[PREFIX + "mlp.w2"]
        .view(config.num_experts, config.hidden_size, config.intermediate_size)
        .unsqueeze(0)
        .contiguous(),
        device,
        dtype=tt_config.weight_dtype,
    )
    tokens = 128

    with expect_error(RuntimeError, "width of the first tensor must be equal to the height"):
        experts(
            to_device(torch.randn(1, 1, tokens, config.hidden_size), device),
            to_device(dense_routing(tokens, config.num_experts, config.moe_top_k).reshape(1, 1, tokens, -1), device),
        )
