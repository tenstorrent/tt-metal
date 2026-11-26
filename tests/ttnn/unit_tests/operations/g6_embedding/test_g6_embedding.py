# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_embedding (Phase 0: bfloat16 table, uint32 ROW_MAJOR index, TILE result).

Every value of the result is a copy of a table value, so the gate is bit equality of the logical
values (helpers.TOLERANCES for this op is ("exact", None)), never PCC. Shapes are production cases
of eval/golden_tests/g6_embedding/feature_spec.py only (BB-22).
"""

import functools

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.g6_embedding import describe_plan, g6_embedding

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)
FLUX_LARGE = (131072, 2560)
FLUX_NARROW = (32, 5120)
FLUX_RANK3 = (1, 256, 128)

L1_BOUND = 1347776

# -0.0, two subnormals, +inf, -inf, quiet NaN, NaN with sign and payload set, signalling NaN.
SPECIAL_BITS = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81)

DRAM_KW = {"layout": ttnn.TILE_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
DEFAULT_KW = {"layout": ttnn.TILE_LAYOUT}
FLAG_KW = {"layout": ttnn.TILE_LAYOUT, "runtime_rows": False}


@functools.lru_cache(maxsize=3)
def _torch_table(shape):
    # A table of up to 335M values is drawn once per shape; its own generator keeps it independent
    # of test order.
    generator = torch.Generator().manual_seed(42)
    return torch.randn(shape, generator=generator, dtype=torch.bfloat16)


_device_tables = {}


def _device_table(shape, device):
    key = (id(device), shape)
    if key not in _device_tables:
        _device_tables.clear()
        _device_tables[key] = _to_device_weight(_torch_table(shape), device)
    return _device_tables[key]


def _to_device_weight(torch_weight, device):
    return ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_device_index(torch_index, device):
    return ttnn.from_torch(
        torch_index,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _table_2d(weight):
    return weight[0] if weight.dim() == 3 else weight


def pytorch_reference(torch_index, torch_weight):
    """torch.nn.functional.embedding in the table's own dtype (a float32 detour rewrites NaN payloads)."""
    return F.embedding(torch_index.to(torch.int64), _table_2d(torch_weight))


def _random_index(index_shape, vocab):
    torch.manual_seed(42)
    return torch.randint(0, vocab, index_shape, dtype=torch.int32)


def _assert_bit_identical(output, expected):
    assert output.layout == ttnn.TILE_LAYOUT, f"layout {output.layout}"
    assert output.dtype == ttnn.bfloat16, f"dtype {output.dtype}"
    placement = output.memory_config()
    assert placement.buffer_type == ttnn.BufferType.DRAM, f"placement {placement}"
    assert placement.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"placement {placement}"
    actual = ttnn.to_torch(output)
    assert tuple(actual.shape) == tuple(expected.shape), f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    differs = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    if bool(differs.any()):
        first = tuple(int(i) for i in differs.nonzero()[0])
        pytest.fail(f"{int(differs.sum())} of {differs.numel()} values differ bitwise; first at {first}")


CASES = [
    pytest.param((1, 1), QWEN_SMALL, DRAM_KW, id="qwen_1x1_d64"),
    pytest.param((1, 31), QWEN_SMALL, DRAM_KW, id="qwen_1x31_d64"),
    pytest.param((2, 128), QWEN_SMALL, DRAM_KW, id="qwen_2x128_d64_multibatch"),
    pytest.param((16, 4096), QWEN_SMALL, DRAM_KW, id="qwen_16x4096_d64"),
    pytest.param((1, 8), QWEN_LARGE, DRAM_KW, id="qwen_1x8_d1280"),
    pytest.param((1, 33), QWEN_LARGE, DRAM_KW, id="qwen_1x33_d1280"),
    pytest.param((1, 128), QWEN_LARGE, DRAM_KW, id="qwen_1x128_d1280"),
    pytest.param((16, 128), QWEN_LARGE, DRAM_KW, id="qwen_16x128_d1280_multibatch"),
    pytest.param((1, 1000), QWEN_LARGE, DRAM_KW, id="qwen_1x1000_d1280"),
    pytest.param((1, 3844), QWEN_LARGE, DRAM_KW, id="qwen_1x3844_d1280_p4"),
    pytest.param((16, 4096), QWEN_LARGE, DRAM_KW, id="qwen_16x4096_d1280_p1"),
    pytest.param((1, 512), FLUX_LARGE, DEFAULT_KW, id="flux_1x512_d2560"),
    pytest.param((1, 1), FLUX_NARROW, FLAG_KW, id="flux_1x1_d5120"),
    pytest.param((1, 1), FLUX_RANK3, FLAG_KW, id="flux_1x1_rank3_weight"),
]


@pytest.mark.parametrize("index_shape, weight_shape, call_kwargs", CASES)
def test_g6_embedding(device, index_shape, weight_shape, call_kwargs):
    torch_weight = _torch_table(weight_shape)
    torch_index = _random_index(index_shape, weight_shape[-2])
    tt_weight = _device_table(weight_shape, device)
    tt_index = _to_device_index(torch_index, device)

    plan = describe_plan(tt_index, tt_weight, memory_config=call_kwargs.get("memory_config"))
    assert plan["l1_bytes_per_core"] <= L1_BOUND, plan

    output = g6_embedding(tt_index, tt_weight, **call_kwargs)
    _assert_bit_identical(output, pytorch_reference(torch_index, torch_weight))


@pytest.mark.parametrize("weight_shape", [pytest.param(QWEN_SMALL, id="d64"), pytest.param(QWEN_LARGE, id="d1280")])
def test_g6_embedding_special_values(device, weight_shape):
    """Rows named by the indices hold -0.0, subnormals, infinities and NaN patterns; moved bit for bit."""
    index_shape = (1, 128)
    torch_index = _random_index(index_shape, weight_shape[-2])
    torch_weight = _torch_table(weight_shape).clone()
    rows = torch.unique(torch_index.to(torch.int64))
    bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    width = weight_shape[-1]
    pattern = bits[(torch.arange(width).unsqueeze(0) + rows.unsqueeze(1)) % len(bits)]
    torch_weight.view(torch.int16)[rows] = pattern

    tt_weight = _to_device_weight(torch_weight, device)
    tt_index = _to_device_index(torch_index, device)
    output = g6_embedding(tt_index, tt_weight, **DRAM_KW)
    _assert_bit_identical(output, pytorch_reference(torch_index, torch_weight))


def test_g6_embedding_new_indices_same_program(device):
    """A repeat call with new index values on the same shapes returns the rows of the new indices."""
    weight_shape = QWEN_SMALL
    tt_weight = _device_table(weight_shape, device)
    torch_weight = _torch_table(weight_shape)
    first = _random_index((1, 4096), weight_shape[-2])
    second = torch.randint(0, weight_shape[-2], (1, 4096), dtype=torch.int32)
    for torch_index in (first, second, first):
        output = g6_embedding(_to_device_index(torch_index, device), tt_weight, **DRAM_KW)
        _assert_bit_identical(output, pytorch_reference(torch_index, torch_weight))


@pytest.mark.parametrize(
    "index_shape, regime",
    [
        pytest.param((16, 4096), "token_split", id="p1_token_split"),
        pytest.param((1, 1), "token_column_split", id="one_token_column_split"),
    ],
)
def test_g6_embedding_regime_pinned(device, index_shape, regime):
    """Both built regimes, on shapes whose regime does not depend on the grid size."""
    weight_shape = QWEN_LARGE
    torch_index = _random_index(index_shape, weight_shape[-2])
    tt_weight = _device_table(weight_shape, device)
    tt_index = _to_device_index(torch_index, device)

    plan = describe_plan(tt_index, tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert plan["regime"] == regime, plan
    assert plan["num_cores"] > 1, plan
    assert plan["l1_bytes_per_core"] <= L1_BOUND, plan

    output = g6_embedding(tt_index, tt_weight, **DRAM_KW)
    _assert_bit_identical(output, pytorch_reference(torch_index, _torch_table(weight_shape)))


def test_g6_embedding_runtime_rows_true_refused(device, expect_error):
    tt_weight = _to_device_weight(_torch_table(FLUX_RANK3), device)
    tt_index = _to_device_index(_random_index((1, 1), FLUX_RANK3[-2]), device)
    with expect_error(UnsupportedAxisValue, "runtime_rows"):
        g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT, runtime_rows=True)


def test_g6_embedding_int32_index_refused(device, expect_error):
    tt_weight = _to_device_weight(_torch_table(FLUX_RANK3), device)
    tt_index = ttnn.from_torch(
        _random_index((1, 1), FLUX_RANK3[-2]),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), "uint32"):
        g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT)


def test_g6_embedding_l1_memory_config_refused(device, expect_error):
    tt_weight = _to_device_weight(_torch_table(FLUX_RANK3), device)
    tt_index = _to_device_index(_random_index((1, 1), FLUX_RANK3[-2]), device)
    with expect_error((ValueError, RuntimeError), "DRAM"):
        g6_embedding(tt_index, tt_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
