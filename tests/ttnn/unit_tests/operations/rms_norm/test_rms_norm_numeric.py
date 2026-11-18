# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1 — numerical configurability: fp32_dest_acc_en=False (Float16_b intermediates) and bfloat8_b
(input and γ). Covers both regimes (LOCAL / W_SPLIT), TILE + ROW_MAJOR, TILE + RM γ, non-aligned shapes, and the
perf-profile config (bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ) at its 0.9995 PCC gate.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def _ref(x, gamma, eps=1e-6):
    xf = x.float()
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        y = y * gamma.float().reshape(-1)
    return y


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _run(device, shape, dtype, layout, *, fp32_acc, gamma_dtype=None, gamma_layout=ttnn.TILE_LAYOUT,
         fidelity=ttnn.MathFidelity.HiFi4, poison=None):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g, tg = None, None
    if gamma_dtype is not None:
        g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = ttnn.from_torch(g, dtype=gamma_dtype, layout=gamma_layout, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if poison is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, poison)
        if tg is not None and gamma_layout == ttnn.TILE_LAYOUT:
            tg = ttnn.fill_implicit_tile_padding(tg, poison)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    assert out.dtype == dtype and out.layout == layout
    actual = ttnn.to_torch(out).float()
    # The block-float round-trip of x / γ is part of the input, so compare against the dequantized inputs.
    xq = ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)) if dtype == ttnn.bfloat8_b else x
    gq = g
    if gamma_dtype == ttnn.bfloat8_b:
        gq = ttnn.to_torch(ttnn.from_torch(g, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT))
    expected = _ref(xq, gq)
    assert torch.isfinite(actual).all()
    return _pcc(expected, actual)


SHAPES = [
    (1, 1, 32, 64),  # LOCAL single tile-row
    (2, 4, 128, 512),  # LOCAL many rows
    (1, 1, 32, 4096),  # W_SPLIT by occupancy
    (1, 1, 32, 16384),  # W_SPLIT by residency
    (1, 1, 32, 4095),  # W_SPLIT + partial last W tile
    (2, 1, 100, 47),  # both non-aligned
]


@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["gTILE", "gRM"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_bf16_acc_off(device, shape, layout, gamma_layout):
    pcc = _run(device, shape, ttnn.bfloat16, layout, fp32_acc=False, gamma_dtype=ttnn.bfloat16,
               gamma_layout=gamma_layout)
    assert pcc >= PCC[ttnn.bfloat16], pcc


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("gamma_dtype", [None, ttnn.bfloat16, ttnn.bfloat8_b], ids=["nog", "g_bf16", "g_bf8b"])
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (4, 8, 32, 256), (1, 1, 32, 4096), (1, 1, 128, 8192)],
                         ids=lambda s: "x".join(map(str, s)))
def test_bf8b_input(device, shape, gamma_dtype, fp32_acc):
    pcc = _run(device, shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, fp32_acc=fp32_acc, gamma_dtype=gamma_dtype)
    assert pcc >= PCC[ttnn.bfloat8_b], pcc


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 1, 32, 4064), (2, 1, 100, 47)],
                         ids=lambda s: "x".join(map(str, s)))
def test_bf8b_gamma(device, shape, dtype, layout, fp32_acc):
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: float32 input requires fp32 DEST accumulation")
    pcc = _run(device, shape, dtype, layout, fp32_acc=fp32_acc, gamma_dtype=ttnn.bfloat8_b)
    assert pcc >= PCC[dtype], pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 72), (1, 1, 40, 40), (1, 1, 32, 4095)], ids=lambda s: "x".join(map(str, s)))
def test_bf16_acc_off_poisoned_padding(device, shape):
    pcc = _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, fp32_acc=False, gamma_dtype=ttnn.bfloat16,
               poison=1000.0)
    assert pcc >= PCC[ttnn.bfloat16], pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 1024), (1, 1, 32, 7168), (1, 1, 640, 6144), (1, 1, 7168, 1024)],
                         ids=lambda s: "x".join(map(str, s)))
def test_perf_profile_config(device, shape):
    """The perf loose-case config: bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ, PCC gate 0.9995."""
    pcc = _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, fp32_acc=False, gamma_dtype=ttnn.bfloat16,
               fidelity=ttnn.MathFidelity.HiFi2)
    assert pcc >= 0.9995, pcc


def test_fp32_acc_off_rejected_for_fp32_input(device):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with pytest.raises(NotImplementedError):
        rms_norm(x, compute_kernel_config=cfg)
