# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for tilize (Phase 0: bfloat16 -> bfloat16, DRAM interleaved).

tilize is a pure re-lay of bytes, so the expected result is bit-identical: PCC 1,
every error metric 0, 0 ULP. The metrics are still measured (not assumed) and
printed, so a future refinement that routes values through a different datapath
(fp32 DEST, a pack-time cast, a tiny-tile LLK) has a baseline to diff against.

Metrics per case: PCC (assert_with_pcc), comp_allclose deltas, max / mean abs
error, relative RMS error, max ULP distance (bf16 bit patterns), bit-mismatch
count, and the got/true ratio spread (median, p5, p95 over finite non-zero
reference elements) — the scale-bug detector: a tight cluster around a non-1.0
constant would be a structural scale bug, not rounding.
"""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.tilize import tilize


def _ordered_bf16(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns mapped to a monotonic integer line (ULP distance = difference)."""
    bits = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _input(shape, dist):
    g = torch.Generator().manual_seed(1234)
    if dist == "normal":
        return torch.randn(shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    if dist == "wide_exponent":
        # magnitudes spread over ~2^-120 .. 2^120, both signs: exercises exponent bits
        mant = torch.rand(shape, generator=g) + 1.0
        exp = torch.randint(-120, 120, shape, generator=g).float()
        sign = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return (sign * mant * torch.pow(2.0, exp)).to(torch.bfloat16)
    raise ValueError(dist)


CASES = [
    pytest.param((1, 1, 32, 32), "normal", id="1x1x32x32-normal"),
    pytest.param((2, 3, 64, 96), "normal", id="2x3x64x96-normal"),
    pytest.param((1, 1, 2048, 64), "normal", id="1x1x2048x64-normal"),
    pytest.param((1, 1, 16384, 64), "normal", id="1x1x16384x64-normal"),
    pytest.param((1, 1, 2048, 64), "wide_exponent", id="1x1x2048x64-wide_exponent"),
]


@pytest.mark.parametrize("shape, dist", CASES)
def test_tilize_precision_baseline(device, shape, dist):
    x = _input(shape, dist)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    assert out.layout == ttnn.TILE_LAYOUT
    y = ttnn.to_torch(out)
    assert y.dtype == torch.bfloat16 and list(y.shape) == list(shape)

    _, pcc_msg = assert_with_pcc(x.float(), y.float(), pcc=0.99999)
    _, allclose_msg = comp_allclose(x.float(), y.float(), rtol=0.0, atol=0.0)

    diff = (y.float() - x.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / x.float().pow(2).mean().sqrt()).item()
    ulp = (_ordered_bf16(y) - _ordered_bf16(x)).abs()
    max_ulp = int(ulp.max().item())
    bit_mismatches = int((y.view(torch.int16) != x.view(torch.int16)).sum().item())

    ref = x.float()
    mask = torch.isfinite(ref) & (ref != 0)
    ratio = (y.float()[mask] / ref[mask]).double()
    r_med = ratio.median().item()
    r_p5, r_p95 = (torch.quantile(ratio[:1_000_000], q).item() for q in (0.05, 0.95))

    print(
        f"\nPRECISION {shape} {dist}: {pcc_msg} | {allclose_msg} | max_abs={max_abs:.3e} "
        f"mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} max_ulp={max_ulp} bit_mismatches={bit_mismatches} "
        f"ratio median={r_med:.6f} p5={r_p5:.6f} p95={r_p95:.6f}"
    )
    # A re-lay is bit-identical: no tolerance is the right tolerance.
    assert bit_mismatches == 0, f"{bit_mismatches} of {x.numel()} elements changed bits (max {max_ulp} ULP)"
