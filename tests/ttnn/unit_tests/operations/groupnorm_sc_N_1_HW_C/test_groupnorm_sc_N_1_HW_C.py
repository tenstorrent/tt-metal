# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for groupnorm_sc_N_1_HW_C.

This file is the immutable acceptance spec for the op. The implementer
SHOULD NOT modify it. It is the contract the kernel must satisfy.

Reference: torch.nn.functional.group_norm. Channel-last (N, 1, HW, C)
layout — we reshape to (N, C, HW) before calling torch's reference
(which expects channels at dim=1), then permute the output back.

Tolerances (keyed by dtype) match the golden suite:
  float32   → PCC ≥ 0.999
  bfloat16  → PCC ≥ 0.995
  bfloat8_b → PCC ≥ 0.99
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


PCC_THRESHOLDS = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two flattened tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b).item() / denom


def _torch_groupnorm(
    x_nhwc: torch.Tensor,
    num_groups: int,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """
    Reference GroupNorm.
    x_nhwc has shape (N, 1, HW, C). torch.nn.functional.group_norm expects
    channels at dim=1, so we permute to (N, C, HW) for the reference call,
    then permute back.
    """
    N, one, HW, C = x_nhwc.shape
    assert one == 1, "dim[1] must be 1"
    x_ncl = x_nhwc.reshape(N, HW, C).permute(0, 2, 1).contiguous()  # (N, C, HW)
    gamma_1d = gamma.reshape(C) if gamma is not None else None
    beta_1d = beta.reshape(C) if beta is not None else None
    y_ncl = F.group_norm(x_ncl, num_groups=num_groups, weight=gamma_1d, bias=beta_1d, eps=eps)
    y_nhwc = y_ncl.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()
    return y_nhwc


# Parametrize set — both tile-aligned and non-aligned-C shapes, both layouts,
# both affine variants, with the SDXL-realistic (C/G) % 32 != 0 case included.
SHAPES = [
    # (N, 1, HW, C, num_groups, id)
    ((1, 1, 32, 32), 1, "single_tile"),
    ((1, 1, 64, 64), 2, "multi_tile_aligned"),
    ((1, 1, 64, 128), 4, "tile_aligned_groupx4"),
    ((1, 1, 128, 256), 8, "square_aligned"),
    ((2, 1, 64, 128), 4, "multi_batch"),
    # SDXL-style: (C/G) % 32 != 0 path — exercises intra-group masking
    ((1, 1, 32, 320), 32, "sdxl_C320_G32_Cg10"),
    ((1, 1, 64, 640), 32, "sdxl_C640_G32_Cg20"),
    # c_non_aligned (C % 32 != 0)
    ((1, 1, 64, 48), 2, "c_non_aligned_Cg24"),
    ((1, 1, 64, 80), 4, "c_non_aligned_Cg20"),
]


@pytest.mark.parametrize("shape, num_groups, shape_id", SHAPES, ids=[s[2] for s in SHAPES])
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "rm"],
)
@pytest.mark.parametrize(
    "affine_mode",
    ["gamma_beta", "gamma_only", "no_affine"],
    ids=["gamma_beta", "gamma_only", "no_affine"],
)
def test_groupnorm_sc_N_1_HW_C(device, shape, num_groups, shape_id, layout, affine_mode):
    """
    Acceptance test.

    Validates:
      - output shape == input shape
      - output dtype == input dtype (Phase 0: bfloat16)
      - output layout == TILE_LAYOUT (always — independent of input layout)
      - PCC vs torch.nn.functional.group_norm ≥ dtype threshold
    """
    torch.manual_seed(42)

    dtype = ttnn.bfloat16
    eps = 1e-5
    N, one, HW, C = shape
    assert C % num_groups == 0, f"shape spec error: C={C} not divisible by num_groups={num_groups}"

    x_torch = torch.randn(shape, dtype=torch.float32)
    gamma_torch = None
    beta_torch = None
    if affine_mode in ("gamma_beta", "gamma_only"):
        gamma_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    if affine_mode == "gamma_beta":
        beta_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)

    y_ref = _torch_groupnorm(x_torch, num_groups, gamma_torch, beta_torch, eps)

    x_tt = ttnn.from_torch(
        x_torch.to(torch.bfloat16),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma_tt = None
    beta_tt = None
    if gamma_torch is not None:
        gamma_tt = ttnn.from_torch(
            gamma_torch.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if beta_torch is not None:
        beta_tt = ttnn.from_torch(
            beta_torch.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=gamma_tt, beta=beta_tt, eps=eps)

    # --- Output contract ---
    assert tuple(y_tt.shape) == shape, f"shape: got {tuple(y_tt.shape)}, expected {shape}"
    assert y_tt.dtype == dtype, f"dtype contract: output {y_tt.dtype} != input {dtype}"
    assert y_tt.layout == ttnn.TILE_LAYOUT, f"output layout must always be TILE_LAYOUT; got {y_tt.layout}"

    y_out = ttnn.to_torch(y_tt).to(torch.float32)

    threshold = PCC_THRESHOLDS[dtype]
    pcc = _pcc(y_out, y_ref)
    assert pcc >= threshold, (
        f"PCC below threshold for shape={shape}, num_groups={num_groups}, "
        f"layout={layout}, affine={affine_mode}, dtype={dtype}: "
        f"pcc={pcc:.5f} < {threshold}"
    )


# -----------------------------------------------------------------------------
# Argument-validation tests — ValueError on structural input errors.
# Separate from the registry-model NotImplementedError gates: these check
# that the op rejects malformed input shapes / group counts / weight shapes
# regardless of what's in SUPPORTED.
# -----------------------------------------------------------------------------


def test_groupnorm_rejects_non_rank_4(device):
    """A 3D tensor must be rejected with ValueError."""
    x = ttnn.from_torch(
        torch.randn((1, 32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(x, num_groups=1)


def test_groupnorm_rejects_non_unit_dim1(device):
    """A 4D tensor with dim[1] != 1 must be rejected with ValueError."""
    x = ttnn.from_torch(
        torch.randn((1, 2, 32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(x, num_groups=1)


def test_groupnorm_rejects_indivisible_groups(device):
    """C not divisible by num_groups must be rejected with ValueError."""
    x = ttnn.from_torch(
        torch.randn((1, 1, 32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(x, num_groups=3)  # 64 % 3 != 0


def test_groupnorm_rejects_bad_gamma_shape(device):
    """gamma shape != (1, 1, 1, C) must be rejected with ValueError."""
    x = ttnn.from_torch(
        torch.randn((1, 1, 32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_gamma = ttnn.from_torch(
        torch.randn((1, 1, 1, 32), dtype=torch.bfloat16),  # C=32 ≠ input C=64
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(x, num_groups=2, gamma=bad_gamma)
