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
    DECORRELATED_PCC,
    MOE_LAYER,
    TOKEN_SHAPES,
    dense_routing,
    hidden_states,
    load_reference,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import (
    MAX_TOKENS_PER_PASS,
    TOKEN_MAJOR_MAX_TOKENS,
    TtNomicExperts,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import expert_w1_transposed_config
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device, pytest.mark.needs_weights]

# Two chained matmuls, one 768-deep and one 3072-deep, with bfloat8_b weights and intermediate.
MODULE_PCC = 0.998

# The shared-bias offset the wrong placement leaves behind. Measured well above this; PCC cannot
# see it at all.
BIAS_MISPLACEMENT_MAX_ABS = 1e-3

# Splits that change the arithmetic: the two pass layouts run different programs over the same
# products, the transposed ones pick K blocks by pass size, and only they write the w2 output in
# bfloat8_b. Over seeds 0..7 at TOKEN_SHAPES and (3, 100), a token-major split scores 0.99976 at
# worst against the whole transposed pass, 0.99980 with that output in bfloat16.
LAYOUT_SWITCH_PCC = 0.9995

# The same splits, token by token: the error norm of a token's output relative to its norm in the
# whole pass. Measured at 5.4e-02 at worst over the same seeds and shapes. A token zeroed or
# crushed at a pass boundary scores 1.0, and PCC over the whole output hardly moves for it.
LAYOUT_SWITCH_TOKEN_ERROR = 0.15

# TOKEN_SHAPES plus a transposed pass off the tile grid: 300 tokens, padded to 320.
EXPERT_SHAPES = [*TOKEN_SHAPES, (3, 100)]

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


@pytest.mark.parametrize("batch, seqlen", EXPERT_SHAPES)
def test_experts(device, config, reference, tt_experts, batch, seqlen):
    """Every token through every expert, gated by the routing and summed."""
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert_with_pcc(ref, got, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", EXPERT_SHAPES)
def test_the_expert_reduce_covers_every_token(device, config, reference, tt_experts, batch, seqlen):
    """The expert sum keeps the logical token count in both layouts.

    Left to allocate its output, fast_reduce_nc reports the tile-padded count, 96 rows at T=74
    with the trailing 22 zero, so the module hands it an output of the logical shape. A wrong
    logical shape would surface as a shape mismatch several modules later. 2x37 and 3x100 are
    off-tile token counts in each layout for exactly this.
    """
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert got.shape == ref.shape
    # No all-zero token rows: a dropped token would leave one behind.
    assert (got.abs().sum(dim=-1) > 0).all()


def test_one_pass_covers_the_largest_perf_shape(config, tt_config, tt_experts):
    """8x512 runs as one pass, and the transposed w1 keeps its measured blocking at that size.

    The pass size caps the w1 intermediate; past it the token axis is split. At 4096 tokens a core
    holds 13 token tiles, taken in N blocks of 2 beside a 14-tile M block. The config halves the N
    block until its buffers fit, so a pass size L1 cannot hold shows up here as a block of 1.
    """
    assert tt_experts.max_tokens_per_pass == MAX_TOKENS_PER_PASS >= 8 * 512
    w1 = expert_w1_transposed_config(
        config.hidden_size // ttnn.TILE_SIZE,
        MAX_TOKENS_PER_PASS,
        tt_config.matmul_weight_dtype(OpGroup.EXPERT_W1),
        tt_config.activation_dtype,
        tt_config.expert_intermediate_dtype,
        tt_config.core_grid,
        tt_config.l1_cb_bytes,
        tt_config.compute_kernel_config(OpGroup.EXPERT_W1),
    )
    assert w1.N_block_size > 1


def pass_layouts(tokens: int, pass_size: int) -> set[str]:
    """The layouts the passes of a split take: TOKEN_MAJOR_MAX_TOKENS decides each by its size."""
    sizes = [min(pass_size, tokens - begin) for begin in range(0, tokens, pass_size)]
    return {"token-major" if size <= TOKEN_MAJOR_MAX_TOKENS else "transposed" for size in sizes}


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_chunking_the_token_axis_does_not_change_the_answer(device, config, reference, tt_experts, batch, seqlen):
    """Splitting the token axis has to be exact where the programs allow it, and close elsewhere.

    Each pass runs the same weights over a disjoint slice and the shared bias is added once to
    the assembled result, so the split is arithmetically a no-op. Token-major passes take K in
    one block whatever their size, so splits that stay token-major must agree bit for bit. The
    transposed programs pick their K blocks by pass size, which reorders the accumulation, so a
    split with a transposed pass agrees to LAYOUT_SWITCH_PCC, and each token to
    LAYOUT_SWITCH_TOKEN_ERROR, the bound that sees a defect at one boundary. Forcing a small pass
    size on a shape that fits in one is the only way to compare them on the same input. The pass
    sizes cover tile-aligned and unaligned, and the smallest puts a boundary inside a sequence.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    dense = to_device(
        dense_routing(tokens, config.num_experts, config.moe_top_k).reshape(1, 1, tokens, config.num_experts), device
    )

    results = {}
    for pass_size in (tokens, tokens // 2 + 1, 64, 30):
        tt_experts.max_tokens_per_pass = pass_size
        results[pass_size] = ttnn.to_torch(tt_experts(x, dense)).float()

    whole = results[tokens]
    token_major = [(size, got) for size, got in results.items() if pass_layouts(tokens, size) == {"token-major"}]
    for pass_size, chunked in results.items():
        assert chunked.shape == whole.shape, f"pass size {pass_size} changed the shape"
        assert compute_pcc(chunked, whole) > LAYOUT_SWITCH_PCC, f"pass size {pass_size} moved the result"
        token_error = (chunked - whole).norm(dim=-1) / whole.norm(dim=-1)
        assert token_error.max() < LAYOUT_SWITCH_TOKEN_ERROR, (
            f"pass size {pass_size} moved token {int(token_error.argmax())} "
            f"by {float(token_error.max()):.3e} of its norm"
        )
    for pass_size, chunked in token_major[1:]:
        first_size, first = token_major[0]
        assert torch.equal(chunked, first), (
            f"token-major pass sizes {first_size} and {pass_size} differ: "
            f"max abs {float((chunked - first).abs().max()):.3e}"
        )


@pytest.mark.parametrize("batch, seqlen", [(3, 100), (5, 37)])
def test_the_padding_of_x_does_not_reach_the_output(device, config, tt_experts, batch, seqlen):
    """Whatever the tile padding of x holds, a transposed pass returns the same result.

    The transpose turns the padding rows of x into padding token columns, and 16 columns of the
    bfloat8_b w1 output, GELU and w2 output share one exponent, so the last real tokens share
    theirs with that padding. Nothing keeps it zero: flattening an off-tile batch or slicing a
    pass leaves it unset, so the module zeroes it. Without that fill the 1e4 here crushes real
    tokens of the last tile toward zero, an error of 1.0 of their norm, and module PCC is 0.984
    at 3x100.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    dense = to_device(
        dense_routing(tokens, config.num_experts, config.moe_top_k).reshape(1, 1, tokens, config.num_experts), device
    )

    clean = ttnn.to_torch(tt_experts(x, dense))
    # In place: the fill writes the padding of x itself.
    poisoned = ttnn.to_torch(tt_experts(ttnn.fill_implicit_tile_padding(x, 1e4), dense))

    assert torch.equal(clean, poisoned)


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


@pytest.mark.parametrize("batch, seqlen", [(1, 128), (2, 512)])
def test_misoriented_w2_decorrelates(device, config, tt_config, state_dict, reference, batch, seqlen):
    """Negative control: in this module the w2 misorientation is silent, so PCC is the guard.

    TtNomicExperts keeps w2 transposed per expert, (E, H, F). The checkpoint's (E*F, H) block
    viewed as (E, H, F) rather than (E, F, H) has that same shape, since E*F*H is symmetric in F
    and H, so every matmul typechecks with it. It has to decorrelate instead, in both layouts:
    1x128 runs token-major, 2x512 transposed.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)
    experts.w2 = to_device(
        state_dict[PREFIX + "mlp.w2"]
        .view(config.num_experts, config.hidden_size, config.intermediate_size)
        .unsqueeze(0)
        .contiguous(),
        device,
        dtype=tt_config.matmul_weight_dtype(OpGroup.EXPERT_W2),
    )

    ref, got, _ = run_both(device, config, reference, experts, batch, seqlen)

    assert compute_pcc(ref, got) < DECORRELATED_PCC
