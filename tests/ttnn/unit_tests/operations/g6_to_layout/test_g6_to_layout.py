# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_layout, Phase 0: TILE -> ROW_MAJOR, bfloat16, interleaved.

The op computes nothing, so the gate is bit identity of every logical value
(helpers.TOLERANCES[bfloat16] == ("exact", None)): signed zeros, subnormals,
infinities and NaN payloads included. Every shape is a production case of the
Phase 0 class (case id in the test id); the remaining tests are the cases the
spec's rules require.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout

# -0.0, subnormals, +inf, -inf, NaN with a payload.
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)

PHASE0_SHAPES = [
    pytest.param((1, 185), id="c309-1x185-one-tile-row-w-unaligned"),
    pytest.param((1, 512), id="c269-1x512-h1"),
    pytest.param((32, 5120), id="c292-32x5120-one-wide-tile-row"),
    pytest.param((1, 131072), id="c294-1x131072-one-long-row"),
    pytest.param((1, 1, 131072), id="c293-1x1x131072-rank3-long-row"),
    pytest.param((16, 1, 2560), id="c1993-16x1x2560-multi-batch-h1"),
    pytest.param((1, 64, 512), id="c1992-1x64x512-multi-tile"),
    pytest.param((1, 256, 128), id="c287-1x256x128-non-square"),
    pytest.param((1, 64, 128, 32), id="c273-1x64x128x32-one-tile-wide"),
    pytest.param((1, 64, 128, 256), id="c274-1x64x128x256-multi-batch"),
    pytest.param((1, 1024, 1024, 3), id="c286-1x1024x1024x3-narrow-rows"),
]

PADDED_SHAPES = [
    pytest.param((1, 185), id="c309-w-unaligned"),
    pytest.param((16, 1, 2560), id="c1993-h-unaligned-multi-image"),
    pytest.param((1, 1024, 1024, 3), id="c286-w3"),
]


def pytorch_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor.clone()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def assert_bit_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert list(actual.shape) == list(expected.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    assert actual.dtype == torch.bfloat16, f"torch dtype {actual.dtype}"
    if torch.equal(_bits(actual), _bits(expected)):
        return
    differ = _bits(actual) != _bits(expected)
    first = tuple(int(i) for i in differ.nonzero()[0])
    got = int(_bits(actual)[first]) & 0xFFFF
    want = int(_bits(expected)[first]) & 0xFFFF
    pytest.fail(
        f"{int(differ.sum())} of {differ.numel()} values differ; first at {first}: got 0x{got:04X} expected 0x{want:04X}"
    )


def to_device_tile(x: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        preserve_nan_values=True,
    )


def assert_output_contract(output: ttnn.Tensor, shape, buffer_type) -> None:
    assert list(output.shape) == list(shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT
    memory = output.memory_config()
    assert memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory.buffer_type == buffer_type


def with_special_bits(shape) -> torch.Tensor:
    x = torch.randn(shape, dtype=torch.bfloat16)
    flat = x.view(-1).view(torch.int16)
    count = max(len(SPECIAL_BITS), flat.numel() // 100)
    positions = torch.randperm(flat.numel())[:count]
    table = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    flat[positions] = table[torch.arange(count) % len(SPECIAL_BITS)]
    return x


@pytest.mark.parametrize("shape", PHASE0_SHAPES)
def test_tile_to_row_major_dram(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    output = g6_to_layout(to_device_tile(torch_input, device), ttnn.ROW_MAJOR_LAYOUT)

    assert_output_contract(output, shape, ttnn.BufferType.DRAM)
    assert_bit_identical(ttnn.to_torch(output).reshape(shape), pytorch_reference(torch_input))


def test_tile_to_row_major_l1_keeps_placement(device):
    # Production case 2038: an L1-interleaved input with no memory_config keeps L1.
    shape = (1, 32, 2560)
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    output = g6_to_layout(to_device_tile(torch_input, device, ttnn.L1_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)

    assert_output_contract(output, shape, ttnn.BufferType.L1)
    assert_bit_identical(ttnn.to_torch(output).reshape(shape), pytorch_reference(torch_input))


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 1, 2560), id="c1993-16x1x2560"),
        pytest.param((1, 64, 128, 256), id="c274-1x64x128x256"),
        pytest.param((1, 1024, 1024, 3), id="c286-1x1024x1024x3"),
    ],
)
def test_special_bit_patterns_survive(device, shape):
    torch.manual_seed(42)
    torch_input = with_special_bits(shape)

    output = g6_to_layout(to_device_tile(torch_input, device), ttnn.ROW_MAJOR_LAYOUT)

    assert_bit_identical(ttnn.to_torch(output).reshape(shape), pytorch_reference(torch_input))


@pytest.mark.parametrize("pad_value", [float("nan"), 1000.0], ids=["nan", "1000"])
@pytest.mark.parametrize("shape", PADDED_SHAPES)
def test_input_tile_padding_never_reaches_output(device, shape, pad_value):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.fill_implicit_tile_padding(to_device_tile(torch_input, device), pad_value)

    output = g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)

    assert_bit_identical(ttnn.to_torch(output).reshape(shape), pytorch_reference(torch_input))


def test_repeated_call_is_bit_identical(device):
    shape = (1, 64, 128, 256)
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = to_device_tile(torch_input, device)

    first = ttnn.to_torch(g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)).reshape(shape)
    second = ttnn.to_torch(g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)).reshape(shape)

    assert_bit_identical(second, first)
    assert_bit_identical(first, pytorch_reference(torch_input))


def test_rejects_layout_equal_to_input_layout(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = to_device_tile(torch.randn((1, 32, 512), dtype=torch.bfloat16), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, ttnn.TILE_LAYOUT)


def test_rejects_rank_outside_2_to_4(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = to_device_tile(torch.randn((1, 1, 1, 32, 32), dtype=torch.bfloat16), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)


def test_rejects_bfloat8_b_row_major_output(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = to_device_tile(torch.randn((1, 32, 512), dtype=torch.bfloat16), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat8_b)


def test_rejects_sharded_output_memory_config(device, expect_error):
    torch.manual_seed(42)
    ttnn_input = to_device_tile(torch.randn((1, 32, 512), dtype=torch.bfloat16), device)
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded)
