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

from models.common.metrics import compute_pcc
from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicExperts
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.module_common import (
    DECORRELATED_PCC,
    MOE_LAYER,
    TOKEN_SHAPES,
    assert_bias_added_once,
    dense_routing,
    device_routing,
    device_selection,
    hidden_states,
    load_reference,
    routing_selection,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import (
    HELD_MAX_TILES,
    MAX_TOKENS_PER_PASS,
    STACKED_MAX_TOKENS,
    StackedBuffers,
    TtNomicExperts,
    routed_windows,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import expert_w1_transposed_config
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.router import Routing
from tests.ttnn.utils_for_testing import assert_with_pcc

# The trace region holds test_routed_pass_replays_in_a_trace's commands: a few hundred kilobytes.
pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"trace_region_size": 8 << 20}),
    pytest.mark.needs_weights,
]

# Two chained matmuls, one 768-deep and one 3072-deep, with bfloat8_b weights and intermediate.
MODULE_PCC = 0.998

# Splits that change the arithmetic: the two pass layouts run different programs over the same
# products, the transposed ones pick K blocks by pass size, and only they write the w2 output in
# bfloat8_b. Measured on the token-major passes the stacked ones replaced: over seeds 0..7 at
# TOKEN_SHAPES and (3, 100), a token-major split scored 0.99976 at worst against the whole
# transposed pass.
LAYOUT_SWITCH_PCC = 0.9995

# The same splits, token by token: the error norm of a token's output relative to its norm in the
# whole pass. Measured at 5.4e-02 at worst over the same seeds and shapes. A token zeroed or
# crushed at a pass boundary scores 1.0, and PCC over the whole output hardly moves for it.
LAYOUT_SWITCH_TOKEN_ERROR = 0.15

# TOKEN_SHAPES plus a pass off the tile grid on each side of STACKED_MAX_TOKENS: 300 tokens, 10 tile
# rows, stacked, and 330 transposed.
EXPERT_SHAPES = [*TOKEN_SHAPES, (3, 100), (3, 110)]

# The routed pass's steps: one pass off the tile grid, one on it, and 5000 tokens, which run as two
# windows over one dispatch table.
ROUTED_SHAPES = [(3, 110), (2, 512), (10, 500)]

# The routed pass against the reference's loop with the experts in torch, so only the bfloat16
# expert rows and sum differ from it. Measured over seeds 0..7 at ROUTED_SHAPES: PCC 0.9999976 at
# worst, and a token's error norm relative to its norm 4.97e-03. A token summed from a wrong row
# scores about 1.
ROUTED_PCC = 0.9999
ROUTED_TOKEN_ERROR = 0.015

# The routed pass with the experts on the device: grouped_matmul's bfloat8_b weights and
# intermediate, HiFi2 and LoFi, and the tanh GELU where the reference uses exact erf. Measured over
# seeds 0..7 at ROUTED_SHAPES: PCC 0.9998944 at worst and a token error of 2.96e-02; the bounds are
# about three times as loose.
ROUTED_DEVICE_PCC = 0.9995
ROUTED_DEVICE_TOKEN_ERROR = 0.09

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

    out = tt_experts(flatten_tokens(to_device(to_block_layout(x), device)), device_routing(dense, device))

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

    Left to allocate its output, fast_reduce_nc, the transposed pass's sum, reports the tile-padded
    count, 352 columns at T=330 with the trailing 22 zero, so the module hands it an output of the
    logical shape. A wrong logical shape would surface as a shape mismatch several modules later.
    2x37 and 3x100 are off-tile token counts on a stacked pass, 3x110 on a transposed one, for
    exactly this.
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
    """The layouts the passes of a split take: STACKED_MAX_TOKENS decides each by its size."""
    sizes = [min(pass_size, tokens - begin) for begin in range(0, tokens, pass_size)]
    return {"stacked" if size <= STACKED_MAX_TOKENS else "transposed" for size in sizes}


@pytest.mark.parametrize("batch, seqlen", TOKEN_SHAPES)
def test_chunking_the_token_axis_does_not_change_the_answer(device, config, reference, tt_experts, batch, seqlen):
    """Splitting the token axis has to be exact where the programs allow it, and close elsewhere.

    Each pass runs the same weights over a disjoint slice and adds the shared bias to its own
    tokens, so the split is arithmetically a no-op. Up to 8 tile rows a stacked pass reads w2 in
    K blocks of 32 tiles whatever its size, and every stacked pass here is that small, so splits
    that stay stacked must agree bit for bit; above 8 rows the blocks are 24 tiles, which reorders
    the accumulation. The transposed programs pick their K blocks by pass size too, so a
    split with a transposed pass agrees to LAYOUT_SWITCH_PCC, and each token to
    LAYOUT_SWITCH_TOKEN_ERROR, the bound that sees a defect at one boundary. Forcing a small pass
    size on a shape that fits in one is the only way to compare them on the same input. The pass
    sizes cover tile-aligned and unaligned, and the smallest puts a boundary inside a sequence.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    routing = device_routing(dense_routing(tokens, config.num_experts, config.moe_top_k), device)

    results = {}
    for pass_size in (tokens, tokens // 2 + 1, 64, 30):
        tt_experts.max_tokens_per_pass = pass_size
        results[pass_size] = ttnn.to_torch(tt_experts(x, routing)).float()

    whole = results[tokens]
    stacked = [(size, got) for size, got in results.items() if pass_layouts(tokens, size) == {"stacked"}]
    for pass_size, chunked in results.items():
        assert chunked.shape == whole.shape, f"pass size {pass_size} changed the shape"
        assert compute_pcc(chunked, whole) > LAYOUT_SWITCH_PCC, f"pass size {pass_size} moved the result"
        token_error = (chunked - whole).norm(dim=-1) / whole.norm(dim=-1)
        assert token_error.max() < LAYOUT_SWITCH_TOKEN_ERROR, (
            f"pass size {pass_size} moved token {int(token_error.argmax())} "
            f"by {float(token_error.max()):.3e} of its norm"
        )
    for pass_size, chunked in stacked[1:]:
        first_size, first = stacked[0]
        assert torch.equal(chunked, first), (
            f"stacked pass sizes {first_size} and {pass_size} differ: "
            f"max abs {float((chunked - first).abs().max()):.3e}"
        )


@pytest.mark.parametrize("batch, seqlen", [(1, 128), (2, 37), (1, 160)])
def test_held_buffers_match_a_pass_that_allocates_its_own(device, config, tt_experts, batch, seqlen):
    """A stacked pass writing into StackedBuffers an earlier pass holds is bit-identical to a fresh one.

    The encoder gives one StackedBuffers to its six MoE layers: the first stacked pass allocates the
    three width-sharded tensors and holds them, the later ones write into the same buffers, which
    is what saves their allocations. Up to HELD_MAX_TILES tile rows of M only; 1x160 holds nothing.
    """
    buffers = StackedBuffers()
    tokens = batch * seqlen
    holds = ttnn.core.divup(tokens, ttnn.TILE_SIZE) <= HELD_MAX_TILES
    addresses = []
    try:
        for _ in range(2):
            x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
            routing = device_routing(dense_routing(tokens, config.num_experts, config.moe_top_k), device)
            got = ttnn.to_torch(tt_experts(x, routing, buffers=buffers))
            held = buffers.get(tokens)

            assert (held is not None) == holds
            assert torch.equal(got, ttnn.to_torch(tt_experts(x, routing)))
            addresses.append([tensor.buffer_address() for tensor in held or ()])
    finally:
        buffers.release()
    assert addresses[0] == addresses[1], "the second pass allocated its own buffers rather than writing the held ones"


@pytest.mark.parametrize("batch, seqlen", [(3, 110), (5, 67)])
def test_the_padding_of_x_does_not_reach_the_output(device, config, tt_experts, batch, seqlen):
    """Whatever the tile padding of x holds, a transposed pass returns the same result.

    The transpose turns the padding rows of x into padding token columns, and 16 columns of the
    bfloat8_b w1 output, GELU and w2 output share one exponent, so the last real tokens share
    theirs with that padding. Nothing keeps it zero: flattening an off-tile batch or slicing a
    pass leaves it unset, so the module zeroes it. Without that fill the 1e4 here crushes real
    tokens of the last tile toward zero, an error of 1.0 of their norm, and module PCC is 0.991
    at 3x110 and 0.980 at 5x67. Both are above STACKED_MAX_TOKENS: a stacked pass keeps the tokens
    on the rows, where padding shares no exponent with a real token, and returns the same result
    without any fill.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    routing = device_routing(dense_routing(tokens, config.num_experts, config.moe_top_k), device)

    clean = ttnn.to_torch(tt_experts(x, routing))
    # In place: the fill writes the padding of x itself.
    poisoned = ttnn.to_torch(tt_experts(ttnn.fill_implicit_tile_padding(x, 1e4), routing))

    assert torch.equal(clean, poisoned)


@pytest.mark.parametrize("batch, seqlen", [(2, 128), (2, 512)])
def test_shared_bias_is_added_after_the_weighted_sum(device, config, tt_config, state_dict, reference, batch, seqlen):
    """The eight experts share one bias, added once outside the reduce.

    Adding it inside the per-expert loop scales it by the routed-weight sum, leaving an offset of
    (sum(w) - 1) * bias, which is real because the weights are deliberately not renormalized.

    Neither PCC nor a plain max-abs against the reference can see that offset here. PCC
    mean-centres it away, and at real weights the expert output reaches a magnitude of about 5,
    where the bfloat16 noise floor is 0.22 and the offset is far below it. So the noise is
    cancelled instead of tolerated: the module is run once with its bias zeroed to get the
    weighted sum on device, and the number of times the output holds the bias is read off by
    projecting its difference from that run onto the bias: 1 for the right placement, the mean
    routed-weight sum, about 0.55, for the wrong one. 2x128 runs stacked, where w2 adds the bias in
    its accumulator, so the two runs round differently by a bfloat16 step, as large as the small
    bias itself: one token's projection scatters from 0.5 to 1.1, the mean over all of them by
    0.01. 2x512 runs transposed, the bias its own add.
    """
    tokens = batch * seqlen
    x = hidden_states(batch, seqlen, config.hidden_size)
    dense = dense_routing(tokens, config.num_experts, config.moe_top_k)
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)

    def run():
        out = experts(flatten_tokens(to_device(to_block_layout(x), device)), device_routing(dense, device))
        return ttnn.to_torch(out).float().reshape(batch, seqlen, config.hidden_size)

    got = run()
    experts.bias = to_device(torch.zeros(1, 1, 1, config.hidden_size), device, dtype=tt_config.weight_dtype)
    weighted_sum = run()

    assert_bias_added_once(got, weighted_sum, reference.bias.detach(), dense.sum(-1).reshape(batch, seqlen, 1))


@pytest.mark.parametrize("batch, seqlen", [(1, 128), (2, 512)])
def test_misoriented_w2_decorrelates(device, config, tt_config, state_dict, reference, batch, seqlen):
    """Negative control: in this module the w2 misorientation is silent, so PCC is the guard.

    TtNomicExperts keeps w2 transposed per expert, (E, H, F). The checkpoint's (E*F, H) block
    viewed as (E, H, F) rather than (E, F, H) has that same shape, since E*F*H is symmetric in F
    and H, so every matmul typechecks with it. It has to decorrelate instead, in both layouts:
    1x128 runs stacked, 2x512 transposed.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)
    experts.set_w2(
        state_dict[PREFIX + "mlp.w2"]
        .view(config.num_experts, config.hidden_size, config.intermediate_size)
        .unsqueeze(0)
        .contiguous(),
        device,
    )

    ref, got, _ = run_both(device, config, reference, experts, batch, seqlen)

    assert compute_pcc(ref, got) < DECORRELATED_PCC


def window_metadata(routing: Routing, num_experts: int, token_start: int, num_tokens: int):
    """routing_metadata's four outputs for one routed pass, on the device and on the host."""
    # Imported here, so a moved op test tree fails the routed tests alone rather than the module.
    from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
        to_host,
    )

    outputs = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
        routing.indices, routing.values, num_experts=num_experts, token_start=token_start, num_tokens=num_tokens
    )
    return outputs, to_host(outputs)


def bfloat16_ties_away(values: torch.Tensor) -> torch.Tensor:
    """float32 to bfloat16 rounding to nearest, ties away from zero, as the packer rounds."""
    bits = values.float().view(torch.int32)
    return ((bits + 0x8000) & ~0xFFFF).view(torch.float32).bfloat16()


def device_rows(rows: torch.Tensor, device):
    """(R, H) expert output rows as grouped_w2 writes them: bfloat16, row-major."""
    return to_device(rows.reshape(1, 1, *rows.shape), device, layout=ttnn.ROW_MAJOR_LAYOUT)


def routed_pass(tt_experts, x, routing: Routing, num_experts: int) -> list:
    """Each window's (1, 1, t, H) result of the routed pass: metadata, dispatch, both grouped
    matmuls and combine, every step on the device. Each intermediate is freed once its consumer has
    run, which the routed passes' L1 budget (ROUTED_L1_MAX_BANK_BYTES) relies on."""
    x_rows = tt_experts.dispatch_table(x)
    windows = routed_windows(x.shape[-2])
    sums = []
    for window, (token_start, num_tokens) in enumerate(windows):
        (
            gather_index,
            combine_index,
            combine_weights,
            expert_counts,
        ) = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
            routing.indices, routing.values, num_experts=num_experts, token_start=token_start, num_tokens=num_tokens
        )
        dispatched = tt_experts.dispatch(x_rows, gather_index)
        ttnn.deallocate(gather_index)
        if window == len(windows) - 1:
            ttnn.deallocate(x_rows)
        activated = tt_experts.grouped_w1(dispatched, expert_counts)
        ttnn.deallocate(dispatched)
        expert_rows = tt_experts.grouped_w2(activated, expert_counts)
        ttnn.deallocate(activated)
        sums.append(tt_experts.combine(expert_rows, combine_index, combine_weights, num_tokens))
        for tensor in (combine_index, combine_weights, expert_counts, expert_rows):
            ttnn.deallocate(tensor)
    return sums


def torch_experts(reference, dispatched: torch.Tensor, expert_counts: torch.Tensor) -> torch.Tensor:
    """The routed experts on the host: each block of dispatched rows through its expert, in bfloat16.

    The expert of each block follows from the counts alone, as the device's grouped matmul derives it.
    """
    from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
        BLOCK_ROWS,
        block_experts,
    )

    experts, blocks = torch.unique_consecutive(
        block_experts(expert_counts, dispatched.shape[0] // BLOCK_ROWS), return_counts=True
    )
    with torch.no_grad():
        outputs = [
            reference.mlp(rows, int(expert))
            for rows, expert in zip(dispatched.split((blocks * BLOCK_ROWS).tolist()), experts)
        ]
    return torch.cat(outputs).bfloat16()


@pytest.mark.parametrize("batch, seqlen", ROUTED_SHAPES)
def test_dispatch_copies_each_slot_row(device, config, tt_experts, batch, seqlen):
    """Every row the routed experts read is a row of x, bit for bit.

    Rows that hold no slot copy the pass's first token, so they are finite too. The ids are global:
    every pass of a split batch gathers from the one table.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    routing = device_selection(*routing_selection(tokens, config.num_experts, config.moe_top_k), device)
    x_host = ttnn.to_torch(x).reshape(tokens, config.hidden_size)

    x_rows = tt_experts.dispatch_table(x)
    for token_start, num_tokens in routed_windows(tokens):
        (gather_index, *_), metadata = window_metadata(routing, config.num_experts, token_start, num_tokens)
        dispatched = ttnn.to_torch(tt_experts.dispatch(x_rows, gather_index))
        ids = metadata.gather_index

        assert dispatched.shape == (1, 1, ids.numel(), config.hidden_size)
        assert torch.equal(dispatched.reshape(ids.numel(), -1).view(torch.int16), x_host[ids].view(torch.int16))


@pytest.mark.parametrize("batch, seqlen", ROUTED_SHAPES)
def test_combine_returns_each_slot_row(device, config, tt_experts, batch, seqlen):
    """At weights (1, 0) each token's result is its first slot's row plus the shared bias, at (0, 1)
    its second's, rounded to bfloat16 once, bit for bit.

    Every expert output row that holds no slot is NaN, so a token reading one would show it. Padding
    tokens read row 0, a real slot's row, at weight 0, so the tile padding of the result is the bias.
    """
    tokens, hidden, top_k = batch * seqlen, config.hidden_size, config.moe_top_k
    _, indices = routing_selection(tokens, config.num_experts, top_k)
    bias = ttnn.to_torch(tt_experts.bias).float().reshape(1, hidden)

    for k in range(top_k):
        values = torch.zeros(tokens, top_k)
        values[:, k] = 1.0
        routing = device_selection(values, indices, device)
        for token_start, num_tokens in routed_windows(tokens):
            (_, combine_index, combine_weights, _), metadata = window_metadata(
                routing, config.num_experts, token_start, num_tokens
            )
            rows = metadata.combine_index
            padded_tokens = rows.shape[-1]
            expert_outputs = torch.randn(metadata.gather_index.numel(), hidden).bfloat16()
            used = torch.zeros(expert_outputs.shape[0], dtype=torch.bool)
            used[rows[:, :num_tokens].reshape(-1)] = True
            expert_outputs[~used] = float("nan")

            out = tt_experts.combine(device_rows(expert_outputs, device), combine_index, combine_weights, num_tokens)
            padded = ttnn.to_torch(ttnn.reshape(out, (1, 1, padded_tokens, hidden), (1, 1, padded_tokens, hidden)))
            padded = padded.reshape(padded_tokens, hidden)
            want = bfloat16_ties_away(expert_outputs[rows[k, :num_tokens]].float() + bias)

            assert list(out.shape) == [1, 1, num_tokens, hidden]
            assert torch.equal(
                padded[:num_tokens].view(torch.int16), want.view(torch.int16)
            ), f"window at {token_start}: a token's result at k={k} is not its slot's row plus the bias"
            assert torch.equal(
                padded[num_tokens:].float(), bias.expand(padded_tokens - num_tokens, hidden)
            ), f"window at {token_start}: the padding of the result is not the bias"


@pytest.mark.parametrize("batch, seqlen", ROUTED_SHAPES)
def test_routed_pass_with_torch_experts(device, config, reference, tt_experts, batch, seqlen):
    """Dispatch, the experts in torch and combine reproduce the reference's own per-expert loop.

    reference.forward gathers each expert's tokens by value and scatter-adds the weighted results;
    the routed pass does the same with routing_metadata's indices on the device. The torch experts
    write bfloat16 rows, as a device pass does, so this measures the data movement and the weighted
    sum: each token's two rows, from the right experts, at the right weights, back in its place.
    """
    tokens, hidden = batch * seqlen, config.hidden_size
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, hidden)), device))
    values, indices = routing_selection(tokens, config.num_experts, config.moe_top_k)
    routing = device_selection(values, indices, device)

    x_rows = tt_experts.dispatch_table(x)
    sums = []
    for token_start, num_tokens in routed_windows(tokens):
        (gather_index, combine_index, combine_weights, _), metadata = window_metadata(
            routing, config.num_experts, token_start, num_tokens
        )
        dispatched = ttnn.to_torch(tt_experts.dispatch(x_rows, gather_index)).float().reshape(-1, hidden)
        expert_outputs = torch_experts(reference, dispatched, metadata.expert_counts)
        out = tt_experts.combine(device_rows(expert_outputs, device), combine_index, combine_weights, num_tokens)
        sums.append(ttnn.to_torch(out).float().reshape(num_tokens, hidden))
    got = torch.cat(sums)

    with torch.no_grad():
        x_host = ttnn.to_torch(x).float().reshape(1, tokens, hidden)
        ref = reference(x_host, values, indices).reshape(tokens, hidden)
    token_error = (got - ref).norm(dim=-1) / ref.norm(dim=-1)

    assert_with_pcc(ref, got, ROUTED_PCC)
    assert (
        token_error.max() < ROUTED_TOKEN_ERROR
    ), f"token {int(token_error.argmax())} is off by {float(token_error.max()):.3e} of its norm"


@pytest.mark.parametrize("batch, seqlen", ROUTED_SHAPES)
def test_routed_pass_with_device_experts(device, config, reference, tt_experts, batch, seqlen):
    """The whole routed pass on the device reproduces the reference's per-expert loop.

    Routing metadata, dispatch, grouped_matmul for w1 and its GELU and for w2, and routed_combine
    with the shared bias, window by window. The token-wise bound catches a token routed to a wrong
    expert or summed from a wrong row, which whole-output PCC barely sees.
    """
    tokens, hidden = batch * seqlen, config.hidden_size
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, hidden)), device))
    values, indices = routing_selection(tokens, config.num_experts, config.moe_top_k)
    routing = device_selection(values, indices, device)

    got = torch.cat(
        [
            ttnn.to_torch(window).float().reshape(-1, hidden)
            for window in routed_pass(tt_experts, x, routing, config.num_experts)
        ]
    )

    with torch.no_grad():
        x_host = ttnn.to_torch(x).float().reshape(1, tokens, hidden)
        ref = reference(x_host, values, indices).reshape(tokens, hidden)
    token_error = (got - ref).norm(dim=-1) / ref.norm(dim=-1)

    assert_with_pcc(ref, got, ROUTED_DEVICE_PCC)
    assert (
        token_error.max() < ROUTED_DEVICE_TOKEN_ERROR
    ), f"token {int(token_error.argmax())} is off by {float(token_error.max()):.3e} of its norm"


def test_routed_pass_replays_in_a_trace(device, config, tt_experts):
    """The routed pass reads its routing on the device only, so a captured trace replays it for a
    new routing: a replay after copying in a new selection equals running that selection directly.
    """
    batch, seqlen = 2, 512
    tokens, hidden = batch * seqlen, config.hidden_size
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, hidden)), device))
    first = device_selection(*routing_selection(tokens, config.num_experts, config.moe_top_k), device)
    second_values, second_indices = routing_selection(tokens, config.num_experts, config.moe_top_k)
    second = device_selection(second_values, second_indices, device)

    # Compiles every program first: a trace captures commands, not compilation.
    (direct,) = routed_pass(tt_experts, x, second, config.num_experts)
    want = ttnn.to_torch(direct)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    (traced,) = routed_pass(tt_experts, x, first, config.num_experts)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for source, target in ((second.indices, first.indices), (second.values, first.values)):
        ttnn.copy_host_to_device_tensor(ttnn.from_device(source), target)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    got = ttnn.to_torch(traced)
    ttnn.release_trace(device, trace_id)

    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
