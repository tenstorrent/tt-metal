# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32-accurate ttnn.sigmoid over a dense sweep of FP32 bit patterns (not only the bfloat16 lattice).

Inputs: every FP32 encoding at a stride of 251 (both signs, every exponent, zeros, subnormals, infinities and NaNs),
every encoding with |x| in [1/16, 128) at a stride of 11 (the range where sigmoid is neither 1/2, 1 nor flushed),
every bfloat16 value, and the encodings next to the saturation points. About 34 M values.

Checks, against a float64 golden:
- where the golden is an FP32 normal, the error is at most MAX_ULP units in the last place of the rounded golden;
- where the golden is below the smallest FP32 normal, the device result is 0 or within MAX_ULP of 2**-126;
- +-inf map to 1 and 0, and a NaN input never yields a finite value other than 0 or 1.

SIGMOID_SWEEP_SAVE=<file> saves the device results (int32 bit patterns); SIGMOID_SWEEP_REF=<file> compares them bit by
bit against a saved run, so a kernel change can be shown bit-identical over the whole sweep.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

MAX_ULP = 3.5  # the accurate FP32 sigmoid measures 3.12 ULP at x = -16.654
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


def _run_device(device, x: torch.Tensor) -> torch.Tensor:
    out = []
    for chunk in x.view(-1, _CHUNK_ROWS, 1024):
        tt_in = ttnn.from_torch(chunk, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = ttnn.sigmoid(tt_in)
        out.append(ttnn.to_torch(tt_out).reshape(-1))
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_out)
    return torch.cat(out)


def _ulp_error(result: torch.Tensor, golden64: torch.Tensor) -> torch.Tensor:
    golden32 = golden64.float()
    ulp = (torch.nextafter(golden32.abs(), torch.tensor(float("inf"))) - golden32.abs()).double()
    return (result.double() - golden64).abs() / ulp


def test_sigmoid_fp32_dense_sweep(device):
    x = _dense_inputs()
    result = _run_device(device, x)
    bits = result.view(torch.int32)

    save = os.environ.get("SIGMOID_SWEEP_SAVE")
    if save:
        torch.save(bits.clone(), save)
    ref = os.environ.get("SIGMOID_SWEEP_REF")
    if ref:
        ref_bits = torch.load(ref)
        both_nan = torch.isnan(result) & torch.isnan(ref_bits.view(torch.float32))
        differ = (bits != ref_bits) & ~both_nan
        print(f"sigmoid sweep: {int(differ.sum())} of {bits.numel()} results differ from {ref}")
        if differ.any():
            idx = differ.nonzero()[:8, 0]
            for i in idx.tolist():
                print(
                    f"  x={x[i].item()!r} ({x.view(torch.int32)[i].item() & 0xFFFFFFFF:#010x}) "
                    f"new={result[i].item()!r} ref={ref_bits.view(torch.float32)[i].item()!r}"
                )
        assert not differ.any()

    nan_in = torch.isnan(x)
    finite_in = torch.isfinite(x)
    # The device flushes subnormal inputs to zero.
    x_ftz = torch.where(x.abs() < _SMALLEST_NORMAL, torch.zeros_like(x), x)
    golden = torch.sigmoid(x_ftz.double())

    normal = finite_in & (golden >= _SMALLEST_NORMAL)
    err = _ulp_error(result[normal], golden[normal])
    worst = int(err.argmax())
    print(
        f"sigmoid sweep: {x.numel()} inputs, {int(normal.sum())} with a normal golden, max {err.max().item():.3f} ULP "
        f"at x={x[normal][worst].item()!r}, mean {err.mean().item():.4f} ULP, "
        f"{int((err > 0.5).sum())} not correctly rounded"
    )
    assert err.max().item() <= MAX_ULP

    tiny = finite_in & (golden < _SMALLEST_NORMAL)
    assert ((result[tiny] == 0) | (result[tiny] <= _SMALLEST_NORMAL * (1 + MAX_ULP * 2.0**-23))).all()
    assert (result[x == float("inf")] == 1).all()
    assert (result[x == float("-inf")] == 0).all()
    nan_out = result[nan_in]
    print(f"sigmoid sweep: NaN inputs -> {torch.unique(nan_out.view(torch.int32))[:8].tolist()} (bit patterns)")
    assert (torch.isnan(nan_out) | (nan_out == 0) | (nan_out == 1)).all()
