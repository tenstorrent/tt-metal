# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_embedding (Phase 0: ROW_MAJOR uint32 index, bfloat16 table).

Every case is a production call of the spec (BB-22) or a case one of its rules
requires. The op copies table values and computes none, so every check is bit
equality of the logical result (helpers.TOLERANCES holds ("exact", None)).
"""

import functools

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.g6_embedding import g6_embedding
from ttnn.operations._op_contract import UnsupportedAxisValue

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)
FLUX_LARGE = (131072, 2560)

# bfloat16 bit patterns the spec names: -0.0, two subnormals, +inf, -inf, quiet NaN,
# NaN with sign and payload set, signalling NaN.
SPECIAL_BITS = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81)


def reference(index, weight):
    """torch.nn.functional.embedding in the table's own dtype; a [1, V, D] weight is weight[0]."""
    table = weight[0] if weight.dim() == 3 else weight
    return F.embedding(index.to(torch.int64), table)


@functools.lru_cache(maxsize=1)
def make_table(weight_shape):
    # One cached table at a time: the Qwen and FLUX tables are 0.6-0.7 GB each, and the
    # parametrizations below are grouped by table so each one is drawn once.
    generator = torch.Generator().manual_seed(42)
    return torch.randn(weight_shape, generator=generator, dtype=torch.bfloat16)


def make_index(index_shape, vocab):
    return torch.randint(0, vocab, index_shape, dtype=torch.int32)


def to_device_table(table, device):
    return ttnn.from_torch(
        table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def to_device_index(index, device):
    return ttnn.from_torch(
        index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def call_kwargs(pass_memory_config):
    kwargs = {"layout": ttnn.TILE_LAYOUT}
    if pass_memory_config:
        kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    return kwargs


def assert_bit_exact(output, expected):
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.dtype == ttnn.bfloat16
    placement = output.memory_config()
    assert placement.buffer_type == ttnn.BufferType.DRAM
    assert placement.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = ttnn.to_torch(output)
    assert tuple(actual.shape) == tuple(expected.shape)
    got = actual.contiguous().view(torch.int16)
    want = expected.contiguous().view(torch.int16)
    differs = got != want
    if bool(differs.any()):
        first = tuple(int(i) for i in differs.nonzero()[0])
        raise AssertionError(
            f"{int(differs.sum())} of {differs.numel()} values differ; first at {first}: "
            f"got {int(got[first]) & 0xFFFF:#06x} expected {int(want[first]) & 0xFFFF:#06x}"
        )


# (index shape, weight shape, pass an explicit DRAM memory_config); grouped by table.
CASES = [
    ((1, 1), QWEN_SMALL, True),  # Qwen 231: one token, a partial tile row
    ((2, 128), QWEN_SMALL, True),  # Qwen 203: two images, tile-aligned
    ((1, 4096), QWEN_SMALL, True),  # Qwen 225: the most frequent call
    ((1, 8), QWEN_LARGE, True),  # Qwen 222: decode, one tile row split across cores by width
    ((1, 77), QWEN_LARGE, True),  # Qwen 214: three tile rows, the last one partial
    ((16, 128), QWEN_LARGE, True),  # Qwen 202: sixteen images, several width blocks per core
    ((1, 3844), QWEN_LARGE, True),  # Qwen 196 (p4): long, partial last tile row
    ((16, 1024), QWEN_LARGE, True),  # Qwen 198 (p3): several full-width tile rows per core
    ((16, 4096), QWEN_LARGE, True),  # Qwen 192 (p1): the largest call, the perf focus
    ((1, 512), FLUX_LARGE, False),  # FLUX 41: D = 2560, default placement
    ((1, 1), (32, 5120), False),  # FLUX 43: D past the per-block width cap
    ((1, 1), (1, 256, 128), False),  # FLUX 44: a [1, V, D] weight read as it arrives
]


def _case_id(case):
    index_shape, weight_shape, pass_memory_config = case
    parts = ["idx" + "x".join(map(str, index_shape)), "w" + "x".join(map(str, weight_shape))]
    if pass_memory_config:
        parts.append("dram")
    return "-".join(parts)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_g6_embedding_matches_reference(case, device):
    index_shape, weight_shape, pass_memory_config = case
    torch.manual_seed(42)
    table = make_table(weight_shape)
    index = make_index(index_shape, weight_shape[-2])

    tt_weight = to_device_table(table, device)
    tt_index = to_device_index(index, device)
    output = g6_embedding(tt_index, tt_weight, **call_kwargs(pass_memory_config))

    expected = reference(index, table)
    assert tuple(output.shape) == tuple(index_shape) + (weight_shape[-1],)
    assert_bit_exact(output, expected)

    ttnn.deallocate(output)
    ttnn.deallocate(tt_index)
    ttnn.deallocate(tt_weight)


def place_special_rows(table, index):
    """A copy of `table` whose rows named by `index` hold SPECIAL_BITS, shifted by row number.

    Written through an int16 view so no float conversion touches the patterns.
    """
    special = table.clone()
    rows = torch.unique(index.to(torch.int64))
    bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    width = special.shape[-1]
    pattern = bits[(torch.arange(width).unsqueeze(0) + rows.unsqueeze(1)) % len(bits)]
    special.view(-1, width).view(torch.int16)[rows] = pattern
    return special


@pytest.mark.parametrize("weight_shape", [QWEN_SMALL, QWEN_LARGE], ids=["w262144x64", "w248320x1280"])
def test_g6_embedding_special_bit_patterns(weight_shape, device):
    """-0.0, subnormals, infinities and NaN payloads come back bit for bit (spec rule 2)."""
    torch.manual_seed(42)
    index = make_index((1, 128), weight_shape[-2])
    table = place_special_rows(make_table(weight_shape), index)

    tt_weight = to_device_table(table, device)
    back = ttnn.to_torch(tt_weight)
    assert torch.equal(back.contiguous().view(torch.int16), table.contiguous().view(torch.int16))
    del back

    tt_index = to_device_index(index, device)
    output = g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert_bit_exact(output, reference(index, table))

    ttnn.deallocate(output)
    ttnn.deallocate(tt_index)
    ttnn.deallocate(tt_weight)


def test_g6_embedding_new_indices_reuse_program(device):
    """Rows follow the index values of each call; repeats reuse the cached program (rule 1, BB-06, BB-07)."""
    torch.manual_seed(42)
    weight_shape = QWEN_SMALL
    table = make_table(weight_shape)
    tt_weight = to_device_table(table, device)
    first = make_index((1, 4096), weight_shape[-2])
    second = make_index((1, 4096), weight_shape[-2])
    tt_first = to_device_index(first, device)
    tt_second = to_device_index(second, device)

    kwargs = dict(layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, runtime_rows=False)
    out_first = g6_embedding(tt_first, tt_weight, **kwargs)
    entries_after_first = device.num_program_cache_entries()
    out_second = g6_embedding(tt_second, tt_weight, **kwargs)
    out_again = g6_embedding(tt_first, tt_weight, **kwargs)
    assert device.num_program_cache_entries() == entries_after_first

    assert_bit_exact(out_first, reference(first, table))
    assert_bit_exact(out_second, reference(second, table))
    assert_bit_exact(out_again, reference(first, table))

    for tensor in (out_first, out_second, out_again, tt_first, tt_second, tt_weight):
        ttnn.deallocate(tensor)


def test_g6_embedding_rejects_invalid_inputs(device, expect_error):
    """The input errors the spec's Validation section lists raise ValueError or RuntimeError.

    The spec fixes the error types, not their wording, so any message matches.
    """
    torch.manual_seed(42)
    vocab, width = 64, 64
    table = torch.randn((vocab, width), dtype=torch.bfloat16)
    index = make_index((1, 8), vocab)
    tt_weight = to_device_table(table, device)
    tt_index = to_device_index(index, device)
    invalid = (ValueError, RuntimeError)

    rank3_index = to_device_index(make_index((1, 1, 8), vocab), device)
    with expect_error(invalid, ""):
        g6_embedding(rank3_index, tt_weight)

    float_index = ttnn.from_torch(
        index.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(invalid, ""):
        g6_embedding(float_index, tt_weight)

    tiled_weight = ttnn.from_torch(
        table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error(invalid, ""):
        g6_embedding(tt_index, tiled_weight)

    narrow_weight = to_device_table(torch.randn((vocab, 48), dtype=torch.bfloat16), device)
    with expect_error(invalid, ""):
        g6_embedding(tt_index, narrow_weight)

    stacked_weight = to_device_table(torch.randn((2, vocab, width), dtype=torch.bfloat16), device)
    with expect_error(invalid, ""):
        g6_embedding(tt_index, stacked_weight)

    host_weight = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(invalid, ""):
        g6_embedding(tt_index, host_weight)

    with expect_error(invalid, ""):
        g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    with expect_error(UnsupportedAxisValue, "runtime_rows"):
        g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT, runtime_rows=True)

    for tensor in (rank3_index, float_index, tiled_weight, narrow_weight, stacked_weight, tt_index, tt_weight):
        ttnn.deallocate(tensor)
