# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_moe_dispatch (immutable spec: the implementer must not modify this file).

out[0, 0, r, :] = table[0, 0, gather_index[0, r], :], tilized, bit for bit. The shapes cover every value of the
column extent the design selects per R (pieces_per_row k in {1, 2, 3, 4, 6}, op_design.md "Selection rule"), the
three perf shapes (R = 2272, 4320, 8416), ragged windows (T % 32 != 0) and the model's spare tile rows.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.nomic_moe_dispatch import nomic_moe_dispatch
from ttnn.operations.nomic_moe_dispatch import (
    circular_buffer_bytes,
    default_compute_kernel_config,
    output_memory_config,
)

HIDDEN = 768
TILE = 32
EXPERTS, TOP_K = 8, 2
NUM_CORES = 110
PCC_BY_DTYPE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
# A real text's expert popularity (the first MoE layer at 8x512).
POPULARITY = (2409, 539, 1518, 273, 314, 336, 1250, 1553)


def rows_for_tokens(tokens: int) -> int:
    return TILE * (-(-TOP_K * tokens // TILE) + EXPERTS - 1)


def routing_index(tokens: int, spare_tile_rows: int = 0, seed: int = 42) -> list[int]:
    """gather_index as routing_metadata lays it out: experts' regions in expert order, each whole 32-row blocks, each
    region's slots in token order; a row holding no slot copies token r mod T; then the model's spare tile rows
    (tokens 0 to 31 each)."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.tensor(POPULARITY, dtype=torch.float64).expand(tokens, EXPERTS).contiguous()
    experts = torch.multinomial(weights, TOP_K, replacement=False, generator=generator).tolist()
    slots = [[] for _ in range(EXPERTS)]
    for token in range(tokens):
        for expert in experts[token]:
            slots[expert].append(token)
    rows = rows_for_tokens(tokens)
    gather = [row % tokens for row in range(rows)]
    start = 0
    for expert in range(EXPERTS):
        for offset, token in enumerate(slots[expert]):
            gather[start + offset] = token
        start += TILE * -(-len(slots[expert]) // TILE)
    assert start <= rows
    return gather + [row % TILE for row in range(TILE * spare_tile_rows)]


def random_index(tokens: int, rows: int, seed: int = 42) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, tokens, (rows,), generator=generator).tolist()


def torch_reference(table: torch.Tensor, gather: list[int]) -> torch.Tensor:
    return table.reshape(-1, HIDDEN)[torch.tensor(gather, dtype=torch.int64)].reshape(1, 1, len(gather), HIDDEN)


def make_inputs(device, tokens: int, gather: list[int], table_memory=ttnn.L1_MEMORY_CONFIG):
    torch.manual_seed(42)
    table = torch.randn(1, 1, tokens, HIDDEN).to(torch.bfloat16)
    tt_table = ttnn.from_torch(
        table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=table_memory
    )
    tt_index = ttnn.from_torch(
        torch.tensor(gather, dtype=torch.int32).reshape(1, len(gather)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return table, tt_table, tt_index


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return 1.0 if denominator == 0 else float((a @ b) / denominator)


def check_output(output, table: torch.Tensor, gather: list[int], device):
    rows = len(gather)
    assert tuple(output.shape) == (1, 1, rows, HIDDEN)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config().buffer_type == ttnn.BufferType.L1
    assert output.memory_config() == output_memory_config(rows, device, ttnn.BufferType.L1)

    got = ttnn.to_torch(output).reshape(1, 1, rows, HIDDEN)
    want = torch_reference(table, gather)
    assert pcc(got.float(), want.float()) >= PCC_BY_DTYPE[ttnn.bfloat16]
    same = got.view(torch.int16) == want.view(torch.int16)
    if not bool(same.all()):
        bad = (~same).reshape(rows, HIDDEN).any(dim=1).nonzero().flatten()
        raise AssertionError(f"{bad.numel()} of {rows} rows differ bit-wise; first: {bad[:8].tolist()}")


# (tokens, spare tile rows): R and the selected pieces_per_row k in the comment.
SHAPES = [
    (1000, 0),  # R = 2240, k = 3, ragged window, the smallest R
    (1024, 0),  # R = 2272, k = 3, perf shape 2x512
    (1152, 0),  # R = 2528, k = 4
    (1216, 0),  # R = 2656, k = 6
    (1536, 0),  # R = 3296, k = 1 (whole rows; 7 cores own no shard)
    (2048, 0),  # R = 4320, k = 3, perf shape 4x512
    (2500, 0),  # R = 5248, k = 2, ragged window
    (3904, 1),  # R = 8064, k = 3, one spare tile row
    (4096, 0),  # R = 8416, k = 2, perf shape 8x512 (the perf focus)
    (3968, 3),  # R = 8256, k = 2, three spare tile rows
]


def _id(shape):
    tokens, spare = shape
    return f"T{tokens}_R{rows_for_tokens(tokens) + TILE * spare}"


@pytest.mark.parametrize("shape", SHAPES, ids=[_id(s) for s in SHAPES])
def test_gather_routing_index(shape, device):
    tokens, spare = shape
    gather = routing_index(tokens, spare)
    table, tt_table, tt_index = make_inputs(device, tokens, gather)
    output = nomic_moe_dispatch(tt_table, tt_index, memory_config=ttnn.L1_MEMORY_CONFIG)
    check_output(output, table, gather, device)
    for tensor in (tt_table, tt_index, output):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("tokens", [1024, 4096])
def test_gather_random_index(tokens, device):
    gather = random_index(tokens, rows_for_tokens(tokens))
    table, tt_table, tt_index = make_inputs(device, tokens, gather)
    output = nomic_moe_dispatch(tt_table, tt_index)  # memory_config defaults to L1
    check_output(output, table, gather, device)
    for tensor in (tt_table, tt_index, output):
        ttnn.deallocate(tensor)


def test_new_inputs_each_call(device):
    """One cached program, new tables and indices at new addresses: each output matches its own inputs."""
    tokens = 2048
    rows = rows_for_tokens(tokens)
    keep = []
    for call in range(3):
        gather = routing_index(tokens, seed=42 + call) if call != 1 else random_index(tokens, rows, seed=call)
        table, tt_table, tt_index = make_inputs(device, tokens, gather)
        table = table.roll(call, dims=-1)  # a different table each call
        ttnn.deallocate(tt_table)
        tt_table = ttnn.from_torch(
            table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        keep.append(
            ttnn.from_torch(
                torch.zeros(1, 1, TILE * (call + 1), HIDDEN, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
        output = nomic_moe_dispatch(
            tt_table,
            tt_index,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=default_compute_kernel_config(),
        )
        check_output(output, table, gather, device)
        for tensor in (tt_table, tt_index, output):
            ttnn.deallocate(tensor)
    for tensor in keep:
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("rows", [2240, 2272, 2656, 3296, 4320, 7168, 8416, 8512])
def test_output_layout_fits_the_l1_plan(rows, device):
    """The output's largest per-core share exceeds the L1-interleaved share by at most 8 KB (4 tiles), and the
    circular buffers take at most 256 KB a core."""
    config = output_memory_config(rows, device, ttnn.BufferType.L1)
    assert config.buffer_type == ttnn.BufferType.L1
    if config.nd_shard_spec is not None:
        shard = list(config.nd_shard_spec.shard_shape)
    else:
        shard = list(config.shard_spec.shape)
    shard_rows, shard_cols = shard[-2], shard[-1]
    assert shard_rows % TILE == 0 and shard_cols % TILE == 0 and HIDDEN % shard_cols == 0
    shard_tiles = (shard_rows // TILE) * (shard_cols // TILE)
    num_shards = math.ceil(rows / shard_rows) * (HIDDEN // shard_cols)
    largest_share = math.ceil(num_shards / NUM_CORES) * shard_tiles
    interleaved_share = math.ceil((rows // TILE) * (HIDDEN // TILE) / NUM_CORES)
    assert largest_share <= interleaved_share + 4, (rows, shard, largest_share, interleaved_share)
    assert 0 < circular_buffer_bytes(rows) <= 256 * 1024
    assert output_memory_config(rows, device, ttnn.BufferType.DRAM) == ttnn.DRAM_MEMORY_CONFIG


def _refused(device, expect_error, tokens, rows, *, table_memory=ttnn.L1_MEMORY_CONFIG, memory_config=None):
    gather = [r % tokens for r in range(rows)]
    _, tt_table, tt_index = make_inputs(device, tokens, gather, table_memory)
    try:
        with expect_error(SupportRefusal, ""):
            nomic_moe_dispatch(tt_table, tt_index, memory_config=memory_config)
    finally:
        ttnn.deallocate(tt_table)
        ttnn.deallocate(tt_index)


@pytest.mark.parametrize(
    "tokens,rows",
    [
        (999, 2240),  # T < 1000
        (4128, 8512),  # T > 4096
        (1024, 2250),  # R not a multiple of 32
        (1024, 2208),  # R < 2240
        (4096, 8544),  # R > 8512
    ],
)
def test_refuses_out_of_range(tokens, rows, device, expect_error):
    _refused(device, expect_error, tokens, rows)


def test_refuses_dram_output(device, expect_error):
    _refused(device, expect_error, 1024, rows_for_tokens(1024), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_refuses_dram_table(device, expect_error):
    _refused(device, expect_error, 1024, rows_for_tokens(1024), table_memory=ttnn.DRAM_MEMORY_CONFIG)
