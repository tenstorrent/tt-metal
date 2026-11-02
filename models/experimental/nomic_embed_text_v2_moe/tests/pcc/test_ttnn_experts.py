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
    device_selection,
    hidden_states,
    injected_routing,
    load_reference,
    routing_selection,
    to_block_layout,
)
from models.experimental.nomic_embed_text_v2_moe.tt.common import flatten_tokens, to_device
from models.experimental.nomic_embed_text_v2_moe.tt.experts import ROUTED_MAX_TOKENS, TtNomicExperts, routed_windows
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

# TOKEN_SHAPES plus a second token count off the tile grid, 330, over several tile rows.
EXPERT_SHAPES = [*TOKEN_SHAPES, (3, 110)]

# The routed pass's steps: one pass off the tile grid, one on it, and 5000 tokens, which the step
# tests run as two windows of STEP_WINDOW_TOKENS at most over one dispatch table.
ROUTED_SHAPES = [(3, 110), (2, 512), (10, 500)]
STEP_WINDOW_TOKENS = 4096

# The routed pass against the reference's loop with the experts in torch, so only the bfloat16
# expert rows and sum differ from it. Measured over seeds 0..7 at ROUTED_SHAPES: PCC 0.9999976 at
# worst, and a token's error norm relative to its norm 4.97e-03. A token summed from a wrong row
# scores about 1.
ROUTED_PCC = 0.9999
ROUTED_TOKEN_ERROR = 0.015

# The routed pass with the experts on the device: grouped_matmul's bfloat8_b weights and
# intermediate, w1 at HiFi3 and w2 at LoFi, and the tanh GELU where the reference uses exact erf.
# Measured over seeds 0..7 at ROUTED_SHAPES: PCC 0.9998974 at worst and a token error of 2.59e-02;
# the bounds are about three times as loose. Most of the token error is w2's LoFi, which drops low
# bits of its bfloat8_b activations and so scales the outputs by 0.9885 to 0.9887 on every seed and
# shape. Neither bound can see a scale, so it has its own.
ROUTED_DEVICE_PCC = 0.9997
ROUTED_DEVICE_TOKEN_ERROR = 0.08
ROUTED_DEVICE_SCALE = (0.98, 1.01)

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
    routing, dense = injected_routing(tokens, config.num_experts, config.moe_top_k, device)

    out = tt_experts(flatten_tokens(to_device(to_block_layout(x), device)), routing)

    with torch.no_grad():
        ref = reference.dense_forward(x, dense)
    return ref, ttnn.to_torch(out).float().reshape(batch, seqlen, config.hidden_size), dense


@pytest.mark.parametrize("batch, seqlen", EXPERT_SHAPES)
def test_experts(device, config, reference, tt_experts, batch, seqlen):
    """Each token's two experts weighted by the routing and summed."""
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert_with_pcc(ref, got, MODULE_PCC)


@pytest.mark.parametrize("batch, seqlen", EXPERT_SHAPES)
def test_the_expert_reduce_covers_every_token(device, config, reference, tt_experts, batch, seqlen):
    """The expert sum keeps the logical token count, and every token in it.

    A wrong logical shape would surface as a shape mismatch several modules later, a dropped token
    as an all-zero row, which PCC over the whole output hardly sees. 2x37 and 3x110 are off the tile
    grid.
    """
    ref, got, _ = run_both(device, config, reference, tt_experts, batch, seqlen)

    assert got.shape == ref.shape
    # No all-zero token rows: a dropped token would leave one behind.
    assert (got.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("batch, seqlen", [(3, 110), (5, 67)])
def test_the_padding_of_x_does_not_reach_the_output(device, config, tt_experts, batch, seqlen):
    """Whatever the tile padding of x holds, the routed pass returns the same result.

    Flattening an off-tile batch leaves that padding unset, and in a bfloat8_b tile that mixed it
    with real tokens the 1e4 here would crush them: a dense pass that put the tokens on the columns
    measured module PCC 0.991 at 3x110 and 0.980 at 5x67 without zeroing it. The routed pass gathers
    the real token rows only, so the padding never enters it.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    routing, _ = injected_routing(tokens, config.num_experts, config.moe_top_k, device)

    clean = ttnn.to_torch(tt_experts(x, routing))
    # In place: the fill writes the padding of x itself.
    poisoned = ttnn.to_torch(tt_experts(ttnn.fill_implicit_tile_padding(x, 1e4), routing))

    assert torch.equal(clean, poisoned)


def test_shared_bias_is_added_after_the_weighted_sum(device, config, tt_config, state_dict, reference):
    """The eight experts share one bias, added once outside the reduce.

    Adding it inside the per-expert loop scales it by the routed-weight sum, leaving an offset of
    (sum(w) - 1) * bias, which is real because the weights are deliberately not renormalized.

    Neither PCC nor a plain max-abs against the reference can see that offset here. PCC
    mean-centres it away, and at real weights the expert output reaches a magnitude of about 5,
    where the bfloat16 noise floor is 0.22 and the offset is far below it. So the noise is
    cancelled instead of tolerated: the module is run once with its bias zeroed to get the
    weighted sum on device, and the number of times the output holds the bias is read off by
    projecting its difference from that run onto the bias: 1 for the right placement, the mean
    routed-weight sum, about 0.55, for the wrong one.
    """
    batch, seqlen = 2, 512
    tokens = batch * seqlen
    x = hidden_states(batch, seqlen, config.hidden_size)
    routing, dense = injected_routing(tokens, config.num_experts, config.moe_top_k, device)
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)

    def run():
        out = experts(flatten_tokens(to_device(to_block_layout(x), device)), routing)
        return ttnn.to_torch(out).float().reshape(batch, seqlen, config.hidden_size)

    got = run()
    experts.bias = to_device(torch.zeros(1, 1, 1, config.hidden_size), device, dtype=tt_config.weight_dtype)
    weighted_sum = run()

    assert_bias_added_once(got, weighted_sum, reference.bias.detach(), dense.sum(-1).reshape(batch, seqlen, 1))


def test_misoriented_w2_decorrelates(device, config, tt_config, state_dict, reference):
    """Negative control: in this module the w2 misorientation is silent, so PCC is the guard.

    TtNomicExperts keeps w2 transposed per expert, (E, H, F). The checkpoint's (E*F, H) block
    viewed as (E, H, F) rather than (E, F, H) has that same shape, since E*F*H is symmetric in F
    and H, so every matmul typechecks with it. It has to decorrelate instead.
    """
    experts = TtNomicExperts(device, config, tt_config, state_dict, PREFIX)
    experts.set_w2(
        state_dict[PREFIX + "mlp.w2"]
        .view(config.num_experts, config.hidden_size, config.intermediate_size)
        .unsqueeze(0)
        .contiguous(),
        device,
    )

    ref, got, _ = run_both(device, config, reference, experts, 2, 512)

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
    for token_start, num_tokens in routed_windows(tokens, STEP_WINDOW_TOKENS):
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
        for token_start, num_tokens in routed_windows(tokens, STEP_WINDOW_TOKENS):
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
    for token_start, num_tokens in routed_windows(tokens, STEP_WINDOW_TOKENS):
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

    got = ttnn.to_torch(tt_experts(x, routing)).float().reshape(tokens, hidden)

    with torch.no_grad():
        x_host = ttnn.to_torch(x).float().reshape(1, tokens, hidden)
        ref = reference(x_host, values, indices).reshape(tokens, hidden)
    token_error = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
    scale = float((got * ref).sum() / (ref * ref).sum())

    assert_with_pcc(ref, got, ROUTED_DEVICE_PCC)
    assert (
        token_error.max() < ROUTED_DEVICE_TOKEN_ERROR
    ), f"token {int(token_error.argmax())} is off by {float(token_error.max()):.3e} of its norm"
    assert ROUTED_DEVICE_SCALE[0] < scale < ROUTED_DEVICE_SCALE[1], f"the output is {scale:.4f} times the reference"


@pytest.mark.parametrize("batch, seqlen, windows", [(2, 512, 1), (10, 500, 2)])
def test_routed_pass_replays_in_a_trace(device, config, tt_experts, batch, seqlen, windows):
    """The routed pass reads its routing on the device only, so a captured trace replays it for a
    new routing: a replay after copying in a new selection equals running that selection directly.
    The two windows of 10x500, 2528 and 2472 tokens, are joined off the tile grid.
    """
    tokens, hidden = batch * seqlen, config.hidden_size
    tt_experts.max_tokens_per_routed_pass = ttnn.core.divup(tokens, windows)
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, hidden)), device))
    first = device_selection(*routing_selection(tokens, config.num_experts, config.moe_top_k), device)
    second_values, second_indices = routing_selection(tokens, config.num_experts, config.moe_top_k)
    second = device_selection(second_values, second_indices, device)

    # Compiles every program first: a trace captures commands, not compilation.
    want = ttnn.to_torch(tt_experts(x, second))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_experts(x, first)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for source, target in ((second.indices, first.indices), (second.values, first.values)):
        ttnn.copy_host_to_device_tensor(ttnn.from_device(source), target)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    got = ttnn.to_torch(traced)
    ttnn.release_trace(device, trace_id)

    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("batch, seqlen", [*ROUTED_SHAPES, (18, 500), (64, 512)])
def test_routed_windows_do_not_change_the_answer(device, config, tt_experts, batch, seqlen):
    """However the routed pass splits a layer into windows, every token's result is the same, bit
    for bit.

    A token's rows are computed on their own: grouped_matmul takes K in one block, its bfloat8_b
    output shares exponents only within a token's row, and routed_combine sums each token's rows by
    themselves, its chunk width only grouping tiles. So a window changes which rows share a block,
    never a result. The default windows are compared against windows of a third and a fifth of the
    layer, three and five of them, four at 3x110 where the tile rounding leaves the last one empty.
    3x110 and 10x500 are off the tile grid, where ttnn.concat joins the windows through an untilize.
    18x500 is the largest window whose gathered rows and expert rows both take L1, the tightest L1
    budget, and 64x512 the largest single window, ROUTED_MAX_TOKENS.
    """
    tokens = batch * seqlen
    x = flatten_tokens(to_device(to_block_layout(hidden_states(batch, seqlen, config.hidden_size)), device))
    routing = device_selection(*routing_selection(tokens, config.num_experts, config.moe_top_k), device)

    results = []
    for window_tokens in (ROUTED_MAX_TOKENS, tokens // 3 + 1, tokens // 5 + 1):
        tt_experts.max_tokens_per_routed_pass = window_tokens
        results.append(ttnn.to_torch(tt_experts(x, routing)))

    for result in results[1:]:
        assert torch.equal(result.view(torch.int16), results[0].view(torch.int16))
