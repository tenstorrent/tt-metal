# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_unsqueeze: the production cases, graded bit for bit.

The spec limits tests to the production cases and the cases its rules require (BB-22), so
every case here is p1 [1, 512] or p2 [1, 256], float32 TILE in DRAM interleaved:

- p1, dim 2: randn with special float32 bit patterns mixed in (NaN payloads, +-inf, -0.0,
  subnormals); a converting path changes them (rule 1).
- p2, dim 2: randn with NaN in every padding position of the input tiles; padding must not
  reach a logical output (rule 2).
- p1, dim=-1 passed as a keyword: the same position named from the end.

A moved value is graded by bit equality, never by a tolerance (helpers.TOLERANCES holds
("exact", None) for float32).
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze

# float32 patterns a converting path changes: NaN payloads (quiet, negative quiet, signaling),
# +-inf, -0.0, subnormals, largest and smallest normal magnitudes.
SPECIAL_BITS = (
    0x7FC12345,
    0xFFFC5433,
    0x7F800001,
    0x7F800000,
    0xFF800000,
    0x80000000,
    0x000116C2,
    0x80000001,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x00800000,
    0x80800000,
)
# A prime stride puts the patterns in every tile and in both 16-column faces of a tile.
SPECIAL_STRIDE = 7


def pytorch_g6_unsqueeze(input_tensor, dim):
    return torch.unsqueeze(input_tensor, dim)


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def _as_int32(pattern):
    return pattern - (1 << 32) if pattern >= (1 << 31) else pattern


def _make_input(shape, special_values):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)
    if special_values:
        flat = x.view(-1).view(torch.int32)
        for i, position in enumerate(range(0, flat.numel(), SPECIAL_STRIDE)):
            flat[position] = _as_int32(SPECIAL_BITS[i % len(SPECIAL_BITS)])
    return x


def _to_device(torch_input, device, pad_value):
    host = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, pad_value=pad_value)
    return ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _assert_input_as_built(ttnn_input, torch_input, pad_value):
    """The case tests what it claims: the device input holds the torch bits, and the poison
    value fills the tile padding."""
    back = ttnn.to_torch(ttnn_input)
    assert torch.equal(_bits(back), _bits(torch_input)), "harness: host conversion changed input bits"
    if pad_value is None:
        return
    padded = ttnn_input.cpu().to_torch_with_padded_shape()
    logical = torch.zeros(padded.shape, dtype=torch.bool)
    logical[tuple(slice(0, n) for n in ([1] * (padded.dim() - torch_input.dim()) + list(torch_input.shape)))] = True
    padding = padded[~logical]
    assert padding.numel() > 0 and bool(torch.isnan(padding).all()), "harness: tile padding does not hold NaN"


@pytest.mark.parametrize(
    "shape, dim, pass_dim_as, special_values, pad_value",
    [
        pytest.param((1, 512), 2, "positional", True, None, id="p1-1x512-dim2-special_values"),
        pytest.param((1, 256), 2, "positional", False, float("nan"), id="p2-1x256-dim2-nan_padding"),
        pytest.param((1, 512), -1, "keyword", False, None, id="p1-1x512-dim-1-keyword"),
    ],
)
def test_g6_unsqueeze(device, shape, dim, pass_dim_as, special_values, pad_value):
    torch_input = _make_input(shape, special_values)
    expected = pytorch_g6_unsqueeze(torch_input, dim)

    ttnn_input = _to_device(torch_input, device, pad_value)
    _assert_input_as_built(ttnn_input, torch_input, pad_value)

    if pass_dim_as == "keyword":
        output = g6_unsqueeze(ttnn_input, dim=dim)
    else:
        output = g6_unsqueeze(ttnn_input, dim)

    assert output.storage_type() == ttnn.StorageType.DEVICE
    assert output.dtype == ttnn.float32
    assert output.layout == ttnn.TILE_LAYOUT
    memory_config = output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert output.buffer_address() != ttnn_input.buffer_address(), "output must not alias the input buffer"

    actual = ttnn.to_torch(output)
    assert tuple(actual.shape) == tuple(expected.shape)
    differ = _bits(actual) != _bits(expected)
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        got = int(_bits(actual)[first]) & 0xFFFFFFFF
        want = int(_bits(expected)[first]) & 0xFFFFFFFF
        pytest.fail(
            f"{int(differ.sum())} of {differ.numel()} logical values differ; "
            f"first at {list(first)}: got 0x{got:08X} expected 0x{want:08X}"
        )
