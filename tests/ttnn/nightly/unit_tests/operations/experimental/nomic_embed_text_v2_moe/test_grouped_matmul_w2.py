# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2 against grouped_matmul_w1's w2 path and ttnn.matmul.

Both run each block of rows through the same multiply-accumulate sequence (K in one block, the weight tiles read
transposed, fp32 accumulation, LoFi) and untilize the same way, so every row of every expert's region must agree bit
for bit; a wrong expert, block, share, K order or lane shows as a mismatch.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
    BLOCK_ROWS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.test_grouped_matmul_w1 import (
    HIDDEN,
    INTERMEDIATE,
    NUM_EXPERTS,
    PROJECTIONS,
    ROUTINGS,
    assert_rows_match_stock,
    compute_config,
    device_counts,
    expert_counts,
    gathered_rows,
    make_weight,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

module = ttnn.experimental.nomic_embed_text_v2_moe
CHUNK_TILES = 48


def written_rows(counts: torch.Tensor, rows: int) -> int:
    """The rows the op writes: every expert's region, each its count in whole blocks, cut at the input."""
    return min(rows, sum(-(-int(count) // BLOCK_ROWS) * BLOCK_ROWS for count in counts.tolist()))


def make_inputs(device, rows: int, generator: torch.Generator):
    """The same values in grouped_matmul_w2's layout and interleaved in DRAM (for the oracles).

    Quantized to bfloat8_b once, interleaved, and laid out again from the exact values: from_torch into the ND-sharded
    layout rounds unquantized values differently, so the two tensors would not hold the same tiles.
    """
    values = torch.randn(1, 1, rows, INTERMEDIATE, generator=generator) * 0.5
    interleaved = ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    laid_out = ttnn.from_torch(
        ttnn.to_torch(interleaved),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=module.grouped_matmul_w2_input_memory_config(rows, INTERMEDIATE, device=device),
    )
    return laid_out, interleaved


def w2(device, inputs, prepared, counts, memory_config=ttnn.DRAM_MEMORY_CONFIG, **slots):
    return module.grouped_matmul_w2(
        inputs,
        prepared,
        counts,
        weight_chunk_tiles=CHUNK_TILES,
        weight_n=HIDDEN,
        memory_config=memory_config,
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.LoFi),
        **slots,
    )


def todays_w2(device, inputs, weight, counts):
    """grouped_matmul_w1's transposed, row-major path: the call the model made before this op."""
    prepared = module.prepare_grouped_matmul_w1_weight(
        weight, NUM_EXPERTS, transpose_weight=True, chunk_tiles=CHUNK_TILES, device=device
    )
    output = module.grouped_matmul_w1(
        inputs,
        prepared,
        counts,
        dtype=ttnn.bfloat16,
        transpose_weight=True,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.LoFi),
        weight_chunk_tiles=CHUNK_TILES,
        weight_n=HIDDEN,
    )
    ttnn.deallocate(prepared)
    return output


def assert_same_rows(expected, actual, rows: int):
    a = ttnn.to_torch(expected).reshape(-1, HIDDEN)[:rows]
    b = ttnn.to_torch(actual).reshape(-1, HIDDEN)[:rows]
    mismatched = (a.view(torch.int16) != b.view(torch.int16)).any(-1).nonzero().reshape(-1)
    assert mismatched.numel() == 0, f"rows {mismatched.tolist()[:8]} of {rows} differ"


# 352 and 1000 tokens leave lanes with no block of some experts; 352 to 2048 tokens (up to 135 blocks) run 10 lanes,
# 2400 (157 blocks, just past the 156 that switch) and 4096 (the model's largest window in L1) 13, the column lanes
# fed by forwarders.
CASES = [(tokens, routing) for tokens in (352, 1000, 2048, 2400, 4096) for routing in ROUTINGS]


@pytest.mark.parametrize("tokens, routing", CASES)
def test_rows_match_grouped_matmul_w1(device, tokens, routing):
    """Every written row equals grouped_matmul_w1's w2 path on the same values, bit for bit."""
    generator = torch.Generator().manual_seed(tokens)
    counts = expert_counts(routing, tokens, generator)
    weight = make_weight(device, generator)
    inputs, interleaved = make_inputs(device, gathered_rows(tokens), generator)
    counts_tensor = device_counts(device, counts)
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)

    output = w2(device, inputs, prepared, counts_tensor)
    expected = todays_w2(device, interleaved, weight, counts_tensor)

    assert list(output.shape) == [1, 1, gathered_rows(tokens), HIDDEN]
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT and output.dtype == ttnn.bfloat16
    assert_same_rows(expected, output, written_rows(counts, gathered_rows(tokens)))


@pytest.mark.parametrize("tokens", [1000, 4096])
def test_rows_match_a_stock_matmul_per_expert(device, tokens):
    """Every written row equals ttnn.matmul of that expert's rows and transposed weight, bit for bit."""
    generator = torch.Generator().manual_seed(tokens + 1)
    counts = expert_counts("uniform", tokens, generator)
    weight = make_weight(device, generator)
    inputs, interleaved = make_inputs(device, gathered_rows(tokens), generator)
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)

    output = w2(device, inputs, prepared, device_counts(device, counts))

    assert_rows_match_stock(device, PROJECTIONS["w2_row_major"], output, interleaved, weight, counts)


@pytest.mark.parametrize("first_slots, second_slots", [(1, 1), (3, 2), (3, 3)])
def test_weight_slots_leave_the_rows_unchanged(device, first_slots, second_slots):
    """The weight rings' depth changes when slices stream, never what is computed."""
    tokens = 4096
    generator = torch.Generator().manual_seed(tokens + 2)
    counts = expert_counts("one_mod_32", tokens, generator)
    weight = make_weight(device, generator)
    inputs, interleaved = make_inputs(device, gathered_rows(tokens), generator)
    counts_tensor = device_counts(device, counts)
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)

    output = w2(device, inputs, prepared, counts_tensor, first_slots=first_slots, second_slots=second_slots)
    expected = todays_w2(device, interleaved, weight, counts_tensor)

    assert_same_rows(expected, output, written_rows(counts, gathered_rows(tokens)))


@pytest.mark.parametrize("tokens", [1000, 4096])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_every_used_row_is_written(device, tokens, buffer_type):
    """Every row of every region comes from the call itself, not from what an earlier call left.

    Freed memory keeps its contents and the allocator hands the same addresses back first, so the output is placed on
    memory filled with NaN. The call before it runs a different routing through the same program, so its circular
    buffers hold other rows where the checked call must write.
    """
    generator = torch.Generator().manual_seed(tokens + 3)
    weight = make_weight(device, generator)
    inputs, interleaved = make_inputs(device, gathered_rows(tokens), generator)
    counts = expert_counts("one_mod_32", tokens, generator)
    checked_counts = device_counts(device, counts)
    stale_counts = device_counts(device, expert_counts("uniform", tokens, generator))
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)
    expected = todays_w2(device, interleaved, weight, checked_counts)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

    ttnn.deallocate(w2(device, inputs, prepared, checked_counts, memory_config))
    ttnn.deallocate(w2(device, inputs, prepared, stale_counts, memory_config))
    poison = ttnn.from_torch(
        torch.full((1, 1, gathered_rows(tokens), HIDDEN), float("nan")),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    poisoned_address = poison.buffer_address()
    ttnn.synchronize_device(device)
    ttnn.deallocate(poison)

    output = w2(device, inputs, prepared, checked_counts, memory_config)

    assert output.buffer_address() == poisoned_address
    assert_same_rows(expected, output, written_rows(counts, gathered_rows(tokens)))


def test_program_cache_reuse(device):
    """A second call with new routing at new addresses reuses the program and follows the new buffers."""
    tokens = 4096
    generator = torch.Generator().manual_seed(tokens + 4)
    weight = make_weight(device, generator)
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)
    first_counts = expert_counts("uniform", tokens, generator)
    first_inputs, first_interleaved = make_inputs(device, gathered_rows(tokens), generator)
    first_counts_tensor = device_counts(device, first_counts)
    first_output = w2(device, first_inputs, prepared, first_counts_tensor)
    entries = device.num_program_cache_entries()

    second_counts = expert_counts("empty_between", tokens, generator)
    second_inputs, second_interleaved = make_inputs(device, gathered_rows(tokens), generator)
    second_counts_tensor = device_counts(device, second_counts)
    second_output = w2(device, second_inputs, prepared, second_counts_tensor)

    assert device.num_program_cache_entries() == entries
    assert second_inputs.buffer_address() != first_inputs.buffer_address()
    assert second_output.buffer_address() != first_output.buffer_address()
    rows = gathered_rows(tokens)
    assert_same_rows(
        todays_w2(device, first_interleaved, weight, first_counts_tensor),
        first_output,
        written_rows(first_counts, rows),
    )
    assert_same_rows(
        todays_w2(device, second_interleaved, weight, second_counts_tensor),
        second_output,
        written_rows(second_counts, rows),
    )


def test_an_input_outside_its_layout_raises(device, expect_error):
    """The op reads its input where grouped_matmul_w2_input_memory_config puts it, so any other layout is refused."""
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens + 5)
    weight = make_weight(device, generator)
    _, interleaved = make_inputs(device, gathered_rows(tokens), generator)
    prepared = module.prepare_grouped_matmul_w2_weight(weight, NUM_EXPERTS, chunk_tiles=CHUNK_TILES, device=device)
    counts = device_counts(device, expert_counts("uniform", tokens, generator))

    with expect_error(RuntimeError, "grouped_matmul_w2_input_memory_config"):
        w2(device, interleaved, prepared, counts)


def test_circular_buffer_bytes_follow_the_slots(device):
    """The footprint the planners read grows by a half slice (144 bfloat8_b tiles) a slot."""
    two = module.grouped_matmul_w2_circular_buffer_bytes(INTERMEDIATE, HIDDEN, 2, 2, NUM_EXPERTS, device=device)
    five = module.grouped_matmul_w2_circular_buffer_bytes(INTERMEDIATE, HIDDEN, 3, 2, NUM_EXPERTS, device=device)
    assert five - two == 144 * ttnn.tile_size(ttnn.bfloat8_b)
