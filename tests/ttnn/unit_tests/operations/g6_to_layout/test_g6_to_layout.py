# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_layout, Phase 0 (immutable spec; do not modify).

Phase 0 is bfloat16 TILE -> ROW_MAJOR, interleaved in DRAM or L1, every alignment and rank 2 to 4.
The shapes are production cases of that class (BB-22), with each case id in the test id. The gate
is the golden suite's gate for a bfloat16 output, helpers.TOLERANCES[bfloat16] == ("exact", None):
every logical value must be bit-identical to the input, NaN payloads included.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# (production case id, shape, input placement). Each one exercises a different part of the blocking:
PHASE0_CASES = [
    pytest.param(2003, (1, 32, 2560), DRAM, id="c2003_single_tile_row_wide"),  # width chunks for occupancy
    pytest.param(1992, (1, 64, 512), DRAM, id="c1992_multi_tile"),
    pytest.param(1993, (16, 1, 2560), DRAM, id="c1993_h1_batch"),  # H=1: one live row per tile row
    pytest.param(269, (1, 512), DRAM, id="c269_rank2_h1"),
    pytest.param(309, (1, 185), DRAM, id="c309_w185"),  # W tail with 25 live columns
    pytest.param(290, (1, 196), DRAM, id="c290_w196"),  # W tail with 4 live columns
    pytest.param(311, (1, 207), DRAM, id="c311_w207"),  # W tail with 15 live columns
    pytest.param(292, (32, 5120), DRAM, id="c292_rank2_wide"),
    pytest.param(271, (1, 2048, 128), DRAM, id="c271_tall"),
    pytest.param(273, (1, 64, 128, 32), DRAM, id="c273_rank4_one_tile_col"),
    pytest.param(293, (1, 1, 131072), DRAM, id="c293_row_wider_than_l1"),
    pytest.param(286, (1, 1024, 1024, 3), DRAM, id="c286_p4_w3"),
    pytest.param(2038, (1, 32, 2560), L1, id="c2038_l1_in_l1_out"),
]


def reference(x):
    """PyTorch reference: the logical values unchanged (x.clone())."""
    return x.clone()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _to_device(x, device, memory_config):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        preserve_nan_values=True,
    )


def _check(output, expected, shape, buffer_type):
    assert list(output.shape) == list(shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT
    mem = output.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mem.buffer_type == buffer_type
    actual = ttnn.to_torch(output).reshape(shape)
    assert actual.dtype == torch.bfloat16
    differ = _bits(actual) != _bits(expected)
    assert int(differ.sum()) == 0, (
        f"{int(differ.sum())} of {differ.numel()} values differ; first at "
        f"{tuple(int(i) for i in differ.nonzero()[0])}"
    )


@pytest.mark.parametrize("case, shape, memory_config", PHASE0_CASES)
def test_tile_to_row_major(device, case, shape, memory_config):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = _to_device(x, device, memory_config)
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    _check(out, reference(x), shape, memory_config.buffer_type)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 64, 1536), id="c1990"),  # 98304 values; holds every pattern at least once
        pytest.param((1, 1024, 1024, 3), id="c286_p4_w3"),  # narrow rows, live-face reads
    ],
)
def test_all_bit_patterns(device, shape):
    """Every one of the 65536 bfloat16 bit patterns (signed zeros, subnormals, inf, NaN payloads)
    must come back unchanged."""
    torch.manual_seed(42)
    numel = 1
    for d in shape:
        numel *= d
    patterns = torch.arange(65536, dtype=torch.int32)
    reps = -(-numel // 65536)
    flat = patterns.repeat(reps)[:numel]
    flat = flat[torch.randperm(numel)] if numel > 65536 else flat
    bits = torch.where(flat >= 0x8000, flat - 0x10000, flat).to(torch.int16)
    x = bits.view(torch.bfloat16).reshape(shape)
    tt_x = _to_device(x, device, DRAM)
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    _check(out, reference(x), shape, ttnn.BufferType.DRAM)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 185), id="c309_w185"),
        pytest.param((1, 196), id="c290_w196"),
        pytest.param((16, 1, 2560), id="c1993_h1"),
        pytest.param((1, 1024, 1024, 3), id="c286_p4_w3"),
    ],
)
@pytest.mark.parametrize("poison", [1000.0, float("nan")], ids=["poison1000", "poisonNaN"])
def test_poisoned_padding(device, shape, poison):
    """The content of the input's tile padding must never reach a logical output value."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.fill_implicit_tile_padding(_to_device(x, device, DRAM), poison)
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    _check(out, reference(x), shape, ttnn.BufferType.DRAM)


def test_repeat_call_bit_identical(device):
    """BB-06 / BB-07: repeated calls with the same inputs return identical bits."""
    torch.manual_seed(42)
    shape = (1, 64, 1536)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = _to_device(x, device, DRAM)
    first = ttnn.to_torch(g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT))
    second = ttnn.to_torch(g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT))
    assert torch.equal(_bits(first), _bits(second))
    assert torch.equal(_bits(first.reshape(shape)), _bits(x))


# --- Validation (spec ## Validation) -------------------------------------------------------------


def test_rejects_same_layout(device, expect_error):
    x = _to_device(torch.randn((1, 32, 64), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_layout(x, ttnn.TILE_LAYOUT)


def test_rejects_rank_5(device, expect_error):
    x = _to_device(torch.randn((1, 1, 1, 32, 64), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


def test_rejects_bfloat8_b_row_major(device, expect_error):
    x = _to_device(torch.randn((1, 32, 64), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_layout(x, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat8_b)


def test_rejects_sharded_memory_config(device, expect_error):
    x = _to_device(torch.randn((1, 32, 64), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
