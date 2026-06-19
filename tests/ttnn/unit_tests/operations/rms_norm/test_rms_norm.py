# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the rms_norm operation (Phase 0 contract).

IMMUTABLE SPEC — the kernel implementer must NOT modify this file.

Phase 0 SUPPORTED rectangle:
  - dtype:   bfloat16
  - layout:  TILE_LAYOUT
  - shapes:  tile-aligned (H, W both multiples of 32), interleaved DRAM
  - gamma:   optional; bfloat16, TILE_LAYOUT, shape (1, 1, 1, W)
  - regimes: Regime A (row-parallel) AND Regime B (wide-W cross-core W-split)

The wide / few-row shapes below FORCE the Regime B cross-core path; they must
pass at the same PCC as the small row-parallel shapes.
"""

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm


# PCC tolerance keyed by dtype — same thresholds as the golden suite.
PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

EPSILON = 1e-6


def torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor = None, eps: float = EPSILON) -> torch.Tensor:
    """PyTorch reference: RMSNorm over the last dimension."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if gamma is not None:
        out = out * gamma
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# (shape, description) — single-tile, multi-tile, non-square, multi-batch,
# plus wide / few-row shapes that exercise the Regime B cross-core W-split.
SHAPES = [
    ((1, 1, 32, 32), "single-tile"),
    ((1, 1, 64, 128), "multi-tile"),
    ((1, 1, 32, 256), "non-square-wide"),
    ((2, 4, 128, 512), "multi-batch"),
    ((1, 1, 2048, 256), "tall-row-parallel"),
    ((1, 32, 1024), "rank3"),
    ((128, 512), "rank2"),
    # --- wide / few-row: force Regime B (cross-core W-split) ---
    ((1, 1, 32, 4096), "wide-W"),
    ((1, 1, 32, 16384), "very-wide-W-1row"),
    ((1, 1, 64, 12288), "wide-W-2rows"),
]


@pytest.mark.parametrize("shape,desc", SHAPES, ids=[d for _, d in SHAPES])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_rms_norm(device, shape, desc, with_gamma):
    torch.manual_seed(42)
    dtype = ttnn.bfloat16

    torch_input = torch.randn(shape, dtype=torch.float32)

    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        W = shape[-1]
        torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    torch_output = torch_rms_norm(torch_input, torch_gamma, eps=EPSILON)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=EPSILON)

    actual = ttnn.to_torch(ttnn_output).to(torch.float32).reshape(torch_output.shape)

    pcc = _pcc(torch_output, actual)
    assert pcc >= PCC_BY_DTYPE[dtype], f"PCC {pcc:.6f} below threshold for {desc} (with_gamma={with_gamma})"
