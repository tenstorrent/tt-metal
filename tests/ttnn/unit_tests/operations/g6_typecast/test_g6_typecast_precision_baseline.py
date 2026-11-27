# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for g6_typecast (Phase 0 SUPPORTED: bf16 -> {float32, bfloat8_b, int32}, TILE).

Production shapes only (BB-22). Typecast has one exact result per value, so the gates are:
  float32 / int32 destination: bit-exact (0 ULP, 0 mismatches) -- helpers.TOLERANCES "exact";
  bfloat8_b destination: within one quantization step of the host rendering per group of 16
  (helpers.TOLERANCES "bfp8_group"), plus PCC / relative RMS (RMS err / std(ref)) as in eval/metrics.py.
The got/true ratio spread (median, p5, p95) is the scale-bug detector: noise must center on 1.0.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_typecast import g6_typecast

MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

CASES = [
    # case,   shape,               destination,     buffer
    ("Q2191", (16, 1, 12), ttnn.float32, "dram"),
    ("Q2185", (16, 4096, 12), ttnn.float32, "dram"),
    ("Q2229", (1, 1, 32, 128), ttnn.int32, "dram"),
    ("Q2197", (1, 1, 31, 256), ttnn.bfloat8_b, "dram"),
    ("Q2187", (1, 1, 3844, 256), ttnn.bfloat8_b, "dram"),
    ("P39", (1, 128, 256, 256), ttnn.bfloat8_b, "l1"),
]


def _input(shape, dst):
    torch.manual_seed(0)
    n = 1
    for d in shape:
        n *= d
    if dst == ttnn.int32:
        u = torch.rand(n) * 32 - 2
        sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        return (sign * torch.pow(2.0, u)).to(torch.bfloat16).reshape(shape)
    return torch.randn(shape).to(torch.bfloat16)


def _ulp_fp32(a, e):
    """ULP distance between two float32 tensors (bit-pattern ordering)."""
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    ei = e.contiguous().view(torch.int32).to(torch.int64)
    ai = torch.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
    ei = torch.where(ei < 0, -(ei & 0x7FFFFFFF), ei)
    return (ai - ei).abs()


def _bfp8_steps(actual, exact):
    """Max deviation from the host bfloat8_b rendering, in units of its group's quantization step."""
    host = ttnn.to_torch(ttnn.from_torch(exact, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).to(torch.float64)
    pad = (-exact.shape[-1]) % 16

    def g(t):
        t = t.to(torch.float64)
        if pad:
            t = torch.nn.functional.pad(t, (0, pad))
        return t.reshape(-1, 16)

    a, h, x = g(actual), g(host), g(exact)
    gmax = x.abs().amax(dim=1, keepdim=True)
    _, exp = torch.frexp(gmax)
    step = torch.where(gmax > 0, torch.pow(2.0, (exp - 7).to(torch.float64)), torch.ones_like(gmax))
    return float(((a - h).abs() / step).max())


@pytest.mark.parametrize("case, shape, dst, buffer", CASES, ids=[c[0] for c in CASES])
def test_precision_baseline(device, case, shape, dst, buffer):
    x = _input(shape, dst)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=MEM[buffer])
    out = ttnn.to_torch(g6_typecast(tx, dst))

    if dst == ttnn.int32:
        expected = x.to(torch.int32)
        got, ref = out.to(torch.float64), expected.to(torch.float64)
    else:
        expected = x.to(torch.float32)
        got, ref = out.to(torch.float32).to(torch.float64), expected.to(torch.float64)

    _, pcc = comp_pcc(ref, got)
    _, allclose_msg = comp_allclose(ref, got)
    err = got - ref
    max_abs = float(err.abs().max())
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / ref.std())

    nz = (ref != 0) & torch.isfinite(ref) & torch.isfinite(got)
    r = (got[nz] / ref[nz]).to(torch.float64)
    q = torch.quantile(r[: 1 << 24], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))

    extra = ""
    if dst == ttnn.float32:
        ulp = _ulp_fp32(out.to(torch.float32), expected)
        extra = f"max_ulp={int(ulp.max())}"
        assert int(ulp.max()) == 0, f"{case}: float32 result not bit-exact ({extra})"
    elif dst == ttnn.int32:
        mism = int((out.to(torch.int64) != expected.to(torch.int64)).sum())
        extra = f"int_mismatches={mism}"
        assert mism == 0, f"{case}: int32 result not exact ({extra})"
    else:
        steps = _bfp8_steps(out.to(torch.float32), expected)
        extra = f"max_bfp8_steps={steps:.3f}"
        assert steps <= 1.0, f"{case}: bfloat8_b beyond one step of host rendering ({extra})"
        assert pcc >= 0.99 and rel_rms <= 0.05, f"{case}: pcc={pcc} rel_rms={rel_rms}"

    print(
        f"\nPRECISION {case} {shape} bf16->{dst}: pcc={pcc:.6f} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"rel_rms={rel_rms:.6g} ratio[p5,med,p95]=[{q[0]:.5f},{q[1]:.5f},{q[2]:.5f}] {extra} | {allclose_msg}"
    )
