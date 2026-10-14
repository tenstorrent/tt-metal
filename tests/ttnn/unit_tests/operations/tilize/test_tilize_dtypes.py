# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5: dtype surface — bit-exact diagonals, casts, 1-byte DRAM bounce path."""

import pytest
import torch
import ttnn

from ttnn.operations.tilize.tilize import tilize


def _make(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 60000, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-100000, 100000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape) * 1e30
    if dtype == ttnn.fp8_e4m3:
        return torch.randn(shape).to(torch.float8_e4m3fn).to(torch.float32)
    return torch.randn(shape).bfloat16()


# (1,1,32,32) / (1,1,32,1056) with 1-byte dtypes exercise the segment bounce (32 B rows on 64 B DRAM).
SHAPES = [(1, 1, 32, 32), (1, 1, 64, 128), (1, 1, 32, 1056), (2, 3, 96, 1056)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "dtype, out_dtype",
    [
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.uint32, ttnn.uint32),
        (ttnn.int32, ttnn.int32),
        (ttnn.int32, ttnn.uint32),
        (ttnn.uint16, ttnn.uint16),
        (ttnn.uint8, ttnn.uint8),
    ],
    ids=str,
)
def test_tilize_dtype_exact(device, shape, dtype, out_dtype):
    torch.manual_seed(0)
    x = _make(dtype, shape)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = tilize(t, dtype=out_dtype)
    assert out.layout == ttnn.TILE_LAYOUT and out.dtype == out_dtype
    y = ttnn.to_torch(out)
    if x.is_floating_point():
        assert torch.equal(y.to(torch.float64), x.to(torch.float64))
    else:  # bit pattern (int32 <-> uint32 is a bit_cast)
        assert torch.equal(y.to(torch.int64) & 0xFFFFFFFF, x.to(torch.int64) & 0xFFFFFFFF)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "dtype, out_dtype, pcc",
    [
        (ttnn.float32, ttnn.bfloat16, 0.999),
        (ttnn.float32, ttnn.bfloat8_b, 0.999),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.999),
        (ttnn.fp8_e4m3, ttnn.float32, 0.9999),
        (ttnn.fp8_e4m3, ttnn.bfloat16, 0.9999),
    ],
    ids=str,
)
def test_tilize_dtype_cast(device, shape, dtype, out_dtype, pcc):
    torch.manual_seed(0)
    x = _make(dtype, shape).float()
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = ttnn.to_torch(tilize(t, dtype=out_dtype)).double().flatten()
    r = torch.corrcoef(torch.stack([x.double().flatten(), y]))[0, 1].item()
    assert r >= pcc, r


@pytest.mark.parametrize("dtype, fill", [(ttnn.int32, -7), (ttnn.uint32, -7), (ttnn.uint8, 9), (ttnn.uint16, 5)])
def test_tilize_integer_fill(device, dtype, fill):
    torch.manual_seed(0)
    x = _make(dtype, (1, 1, 50, 50))
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = tilize(t, pad_value=fill).cpu().to_torch_with_padded_shape()
    exp = torch.nn.functional.pad(x.to(torch.int64), (0, 14, 0, 14), value=fill)
    # bit pattern: a negative fill is bit_cast (two's complement), never truncated through a float
    assert torch.equal(y.to(torch.int64) & 0xFFFFFFFF, exp & 0xFFFFFFFF)
