# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_unsqueeze: the production calls, graded bit for bit.

The spec limits tests to the production cases and the cases its rules require (BB-22): p1
[1, 512] and p2 [1, 256], float32 TILE in DRAM interleaved. The four cases cross both shapes
with both names of the new axis (2 positional, -1 keyword) and with both stressors the rules
name: float32 special bit patterns (rule 1, a converting path changes them) and NaN in every
input tile-padding position (rule 2, padding must never reach a logical output).

The op moves values, so there is no tolerance: logical values are compared as int32 bit
patterns, NaN payloads included.
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze

# NaN payloads (quiet, negative quiet, signaling), +-inf, -0.0, subnormals, largest and smallest
# normal magnitudes.
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
# Coprime to 32 and to 16: the patterns reach every tile, both face halves of a tile row, and
# every lane position.
SPECIAL_STRIDE = 7


def pytorch_g6_unsqueeze(input_tensor, dim):
    return torch.unsqueeze(input_tensor, dim)


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def _make_input(shape, special_values):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)
    if special_values:
        # Written as int32 so no float operation can quiet a NaN on the way in.
        patterns = torch.tensor([p - (1 << 32) if p >= (1 << 31) else p for p in SPECIAL_BITS], dtype=torch.int32)
        flat = x.view(-1).view(torch.int32)
        positions = torch.arange(0, flat.numel(), SPECIAL_STRIDE)
        flat[positions] = patterns[torch.arange(positions.numel()) % patterns.numel()]
    return x


def _padding_values(ttnn_input, logical_shape):
    padded = ttnn_input.cpu().to_torch_with_padded_shape()
    logical = torch.zeros(padded.shape, dtype=torch.bool)
    leading = [1] * (padded.dim() - len(logical_shape))
    logical[tuple(slice(0, n) for n in leading + list(logical_shape))] = True
    return padded[~logical]


def _to_device(torch_input, device, nan_padding):
    """Build the DRAM interleaved TILE input and confirm it holds what the case claims.

    A harness that cannot place the input bits or the NaN padding skips with a reason starting
    "harness:", so its failure is never charged to the op.
    """
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=float("nan") if nan_padding else None,
        preserve_nan_values=True,
    )
    back = ttnn.to_torch(ttnn_input)
    if not torch.equal(_bits(back), _bits(torch_input)):
        pytest.skip("harness: the host round trip changed input bit patterns")
    if nan_padding:
        padding = _padding_values(ttnn_input, tuple(torch_input.shape))
        if not bool(torch.isnan(padding).all()):
            ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"))
            padding = _padding_values(ttnn_input, tuple(torch_input.shape))
        if padding.numel() == 0 or not bool(torch.isnan(padding).all()):
            pytest.skip("harness: NaN did not reach the input tile padding")
    return ttnn_input


@pytest.mark.parametrize(
    "shape, dim, pass_dim_as, special_values, nan_padding",
    [
        pytest.param((1, 512), 2, "positional", True, False, id="p1-1x512-dim2-special_values"),
        pytest.param((1, 256), 2, "positional", False, True, id="p2-1x256-dim2-nan_padding"),
        pytest.param((1, 512), -1, "keyword", False, True, id="p1-1x512-dim-1-keyword-nan_padding"),
        pytest.param((1, 256), -1, "keyword", True, False, id="p2-1x256-dim-1-keyword-special_values"),
    ],
)
def test_g6_unsqueeze(device, shape, dim, pass_dim_as, special_values, nan_padding):
    torch_input = _make_input(shape, special_values)
    expected = pytorch_g6_unsqueeze(torch_input, dim)
    ttnn_input = _to_device(torch_input, device, nan_padding)

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
    assert output.buffer_address() != ttnn_input.buffer_address(), "the output must not alias the input buffer"
    assert tuple(output.shape) == tuple(expected.shape)

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
