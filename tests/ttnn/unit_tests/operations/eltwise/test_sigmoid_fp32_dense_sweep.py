# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32-accurate ttnn.sigmoid, ttnn.silu and ttnn.exp over a dense sweep of FP32 bit patterns (not only the bfloat16
lattice).

Inputs: every FP32 encoding at a stride of 251 (both signs, every exponent, zeros, subnormals, infinities and NaNs),
every encoding with |x| in [1/16, 128) at a stride of 11 (the range where sigmoid is neither 1/2, 1 nor flushed),
every bfloat16 value, and the encodings next to the saturation points. About 38 M values.

Checks, against a float64 golden:
- where the golden is an FP32 normal, the error is at most MAX_ULP units in the last place of the rounded golden;
- where the golden is below the smallest FP32 normal, the device result is 0 or within MAX_ULP of 2**-126, and where
  sigmoid(x) is (x below -87.3), both sigmoid and silu are 0: the device flushes the sigmoid;
- exp is 0 where exp(x) is below the smallest FP32 normal and +inf from 2**128 up;
- sigmoid maps +-inf to 1 and 0, silu and exp +inf to +inf, exp -inf to 0, and a NaN input never yields a finite
  value other than 0 or 1.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

# Measured maxima: sigmoid 3.12 ULP at x = -16.654, silu 3.53 ULP at x = -16.686, exp 0.86 ULP at x = 26.699.
MAX_ULP = {"sigmoid": 3.5, "silu": 4.0, "exp": 1.0}
_CHUNK_ROWS = 4096  # 4096 x 1024 values per device call
_SMALLEST_NORMAL = 2.0**-126


def _dense_inputs() -> torch.Tensor:
    every = torch.arange(0, 2**32, 251, dtype=torch.int64)
    lo, hi = 0x3D800000, 0x43000000  # 1/16 and 128
    band = torch.arange(lo, hi, 11, dtype=torch.int64)
    band = torch.cat([band, band | 0x80000000])
    bf16 = torch.arange(0, 2**16, dtype=torch.int64) << 16
    edges = []
    for value in (87.0, 87.5, 88.0, 88.5, 88.75, 89.0, 103.0, 104.0, 16.0, 17.0, 17.5, 24.0):
        for sign in (1.0, -1.0):
            center = torch.tensor([sign * value], dtype=torch.float32).view(torch.int32).to(torch.int64)
            edges.append(center + torch.arange(-4096, 4097, dtype=torch.int64))
    bits = torch.cat([every, band, bf16, *edges]) & 0xFFFFFFFF
    bits = torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
    pad = (-bits.numel()) % (_CHUNK_ROWS * 1024)
    bits = torch.cat([bits, torch.zeros(pad, dtype=torch.int32)])
    return bits.view(torch.float32)


def _run_device(device, op, x: torch.Tensor) -> torch.Tensor:
    out = []
    for chunk in x.view(-1, _CHUNK_ROWS, 1024):
        tt_in = ttnn.from_torch(chunk, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = op(tt_in)
        out.append(ttnn.to_torch(tt_out).reshape(-1))
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_out)
    return torch.cat(out)


def _ulp_error(result: torch.Tensor, golden64: torch.Tensor) -> torch.Tensor:
    golden32 = golden64.float()
    ulp = (torch.nextafter(golden32.abs(), torch.tensor(float("inf"))) - golden32.abs()).double()
    return (result.double() - golden64).abs() / ulp


@pytest.mark.parametrize("name", ["sigmoid", "silu", "exp"])
def test_sigmoid_fp32_dense_sweep(device, name):
    op = {"sigmoid": ttnn.sigmoid, "silu": ttnn.silu, "exp": ttnn.exp}[name]
    x = _dense_inputs()
    result = _run_device(device, op, x)
    nan_in = torch.isnan(x)
    finite_in = torch.isfinite(x)
    # The device flushes subnormal inputs to zero.
    x_ftz = torch.where(x.abs() < _SMALLEST_NORMAL, torch.zeros_like(x), x).double()
    sigmoid = torch.sigmoid(x_ftz)
    if name == "exp":
        golden = torch.exp(x_ftz)
        # Below the smallest normal the device flushes exp(x) to 0; from 2**128 up it is +inf.
        flushed = finite_in & (golden < _SMALLEST_NORMAL)
        overflow = finite_in & (golden >= 2.0**128)
        assert (result[overflow] == float("inf")).all()
        finite_in = finite_in & ~overflow
    else:
        golden = sigmoid if name == "sigmoid" else x_ftz * sigmoid
        # Where sigmoid(x) is below the smallest normal the device flushes it, and silu(x) = x * 0 with it.
        flushed = finite_in & (sigmoid < _SMALLEST_NORMAL)
    assert (result[flushed] == 0).all()

    normal = finite_in & ~flushed & (golden.abs() >= _SMALLEST_NORMAL)
    err = _ulp_error(result[normal], golden[normal])
    worst = int(err.argmax())
    print(
        f"{name} sweep: {x.numel()} inputs, {int(normal.sum())} with a normal golden, max {err.max().item():.3f} ULP "
        f"at x={x[normal][worst].item()!r}, mean {err.mean().item():.4f} ULP, "
        f"{int((err > 0.5).sum())} not correctly rounded"
    )
    assert err.max().item() <= MAX_ULP[name]

    tiny = finite_in & ~flushed & (golden.abs() < _SMALLEST_NORMAL)
    assert (result[tiny].abs() <= _SMALLEST_NORMAL * (1 + MAX_ULP[name] * 2.0**-23)).all()
    if name == "sigmoid":
        assert (result[x == float("inf")] == 1).all()
        assert (result[x == float("-inf")] == 0).all()
    else:
        assert (result[x == float("inf")] == float("inf")).all()
    if name == "exp":
        assert (result[x == float("-inf")] == 0).all()
    nan_out = result[nan_in]
    print(f"{name} sweep: NaN inputs -> {torch.unique(nan_out.view(torch.int32))[:8].tolist()} (bit patterns)")
    assert (torch.isnan(nan_out) | (nan_out == 0) | (nan_out == 1)).all()
