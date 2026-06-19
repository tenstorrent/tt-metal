# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — layout + non-tile-aligned shape matrix for rms_norm.

Authoritative layout-correctness test (memory-layouts skill §7). Exercises the
two layout pairs the op supports natively — TILE->TILE and ROW_MAJOR->ROW_MAJOR
(output layout always matches input) — across the full aligned / non-aligned
shape sweep and bf16 + fp32, with optional gamma in either layout.

Everything is done in-kernel: no ttnn.to_layout / tilize / untilize / pad /
slice on the host. The RMS denominator must reflect only valid (non-padding)
elements along W; the reference is computed in fp32 over the true shape.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt((xf**2).mean(dim=-1, keepdim=True) + eps)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


_TOL = {ttnn.bfloat16: (0.995, 0.04), ttnn.float32: (0.999, 0.02)}


@pytest.mark.parametrize(
    "dtype",
    [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="32x32_aligned_small"),
        pytest.param((1, 1, 32, 64), id="32x64_aligned"),
        pytest.param((1, 1, 64, 128), id="64x128_aligned"),
        pytest.param((2, 1, 128, 256), id="128x256_aligned_large"),
        pytest.param((1, 1, 32, 50), id="32x50_W_non_aligned"),
        pytest.param((1, 1, 50, 32), id="50x32_H_non_aligned"),
        pytest.param((1, 1, 50, 50), id="50x50_both_non_aligned"),
        pytest.param((4, 8, 47, 100), id="47x100_both_non_aligned_large"),
        pytest.param((1, 32, 17), id="3d_H_non_aligned"),
        pytest.param((128, 100), id="2d_W_non_aligned"),
    ],
)
@pytest.mark.parametrize(
    "input_layout,output_layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, id="tile_to_tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, id="rm_to_rm"),
    ],
)
@pytest.mark.parametrize(
    "gamma_mode",
    [
        pytest.param("no_gamma", id="no_gamma"),
        pytest.param("gamma_tile", id="gamma_tile"),
        pytest.param("gamma_rm", id="gamma_rm"),
    ],
)
def test_rms_norm_layout_matrix(device, shape, input_layout, output_layout, dtype, gamma_mode):
    # Block formats have no ROW_MAJOR representation — not exercised here (bf16/fp32 only).
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(
        torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    )

    gamma = None
    torch_gamma = None
    if gamma_mode != "no_gamma":
        W = shape[-1]
        torch_gamma = torch.randn(W, dtype=torch.float32).to(
            torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
        )
        glayout = ttnn.TILE_LAYOUT if gamma_mode == "gamma_tile" else ttnn.ROW_MAJOR_LAYOUT
        gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=glayout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False

    ttnn_in = ttnn.from_torch(
        torch_input, dtype=dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_out = rms_norm(ttnn_in, gamma=gamma, compute_kernel_config=cfg)

    # Output layout must match input layout.
    assert ttnn_out.layout == output_layout, f"expected {output_layout}, got {ttnn_out.layout}"

    out = ttnn.to_torch(ttnn_out)
    assert tuple(out.shape) == tuple(shape), f"shape {tuple(out.shape)} != {tuple(shape)}"

    expected = _ref(torch_input, gamma=torch_gamma)
    pcc_min, rms_max = _TOL[dtype]
    pcc = _pcc(out, expected)
    denom = expected.to(torch.float32).std().item() or 1.0
    rms = ((out.to(torch.float32) - expected).pow(2).mean().sqrt().item()) / denom
    assert pcc >= pcc_min, f"PCC {pcc:.5f} < {pcc_min} (shape={shape}, {input_layout}, {dtype}, {gamma_mode})"
    assert rms <= rms_max, f"relRMS {rms:.4f} > {rms_max} (shape={shape}, {input_layout}, {dtype}, {gamma_mode})"
