# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 2 test: gamma/beta in TILE_LAYOUT.

The reader path under R2 reads one tile per Ct directly from DRAM into
cb_gamma_tile / cb_beta_tile (no replicate-32 staging). Since the
logical weight shape is (1, 1, 1, C), only row 0 of the read tile is
valid (rows 1-31 are tile padding) — the compute kernel handles this
by switching the apply-phase mul/add to BroadcastDim::ROW.

This test exercises:
- Both layout pairs that R2 newly enables: input TILE × affine TILE, and
  input ROW_MAJOR × affine TILE.
- Aligned and non-aligned (C % 32 != 0) shapes.
- All three affine modes (gamma_beta, gamma_only, and no_affine — the
  no_affine cell is the canonical sanity check that R2's CT-arg threading
  doesn't break the no-affine path).
- bf16 and fp32 affine dtypes. (bf8b affine is exercised by a smoke check
  at the bottom; tolerance is wider per the dtype tolerance table.)

PCC tolerances mirror the golden suite (bf16: 0.995, fp32: 0.999,
bf8b: 0.99).
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
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b).item() / denom


def _torch_groupnorm(x_nhwc, num_groups, gamma, beta, eps):
    N, one, HW, C = x_nhwc.shape
    assert one == 1
    x_ncl = x_nhwc.reshape(N, HW, C).permute(0, 2, 1).contiguous()
    gamma_1d = gamma.reshape(C) if gamma is not None else None
    beta_1d = beta.reshape(C) if beta is not None else None
    y_ncl = F.group_norm(x_ncl, num_groups=num_groups, weight=gamma_1d, bias=beta_1d, eps=eps)
    return y_ncl.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()


# Shapes exercising both tile_aligned and c_non_aligned, plus an
# SDXL-style (C/G) % 32 != 0 case.
SHAPES = [
    # (shape, num_groups, id)
    ((1, 1, 32, 32), 1, "single_tile"),
    ((1, 1, 64, 64), 2, "multi_tile_aligned"),
    ((1, 1, 64, 128), 4, "tile_aligned_groupx4"),
    ((1, 1, 128, 256), 8, "square_aligned"),
    ((2, 1, 64, 128), 4, "multi_batch"),
    # SDXL-style: (C/G) % 32 != 0
    ((1, 1, 32, 320), 32, "sdxl_C320_G32_Cg10"),
    ((1, 1, 64, 640), 32, "sdxl_C640_G32_Cg20"),
    # c_non_aligned (C % 32 != 0)
    ((1, 1, 64, 48), 2, "c_non_aligned_Cg24"),
    ((1, 1, 64, 80), 4, "c_non_aligned_Cg20"),
]


@pytest.mark.parametrize("shape, num_groups, shape_id", SHAPES, ids=[s[2] for s in SHAPES])
@pytest.mark.parametrize(
    "input_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["input_tile", "input_rm"],
)
@pytest.mark.parametrize(
    "affine_mode",
    ["gamma_beta", "gamma_only"],
    ids=["gamma_beta", "gamma_only"],
)
@pytest.mark.parametrize(
    "affine_dtype",
    [ttnn.bfloat16, ttnn.float32],
    ids=["affine_bf16", "affine_fp32"],
)
def test_groupnorm_affine_tile_layout(device, shape, num_groups, shape_id, input_layout, affine_mode, affine_dtype):
    """Exercise affine_layout=TILE across input layouts, affine modes, dtypes, and shapes."""
    torch.manual_seed(42)

    dtype = ttnn.bfloat16
    eps = 1e-5
    N, one, HW, C = shape
    assert C % num_groups == 0

    torch_affine_dtype = torch.bfloat16 if affine_dtype == ttnn.bfloat16 else torch.float32

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
        layout=input_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma_tt = None
    beta_tt = None
    if gamma_torch is not None:
        gamma_tt = ttnn.from_torch(
            gamma_torch.to(torch_affine_dtype),
            dtype=affine_dtype,
            layout=ttnn.TILE_LAYOUT,  # the new R2 path
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if beta_torch is not None:
        beta_tt = ttnn.from_torch(
            beta_torch.to(torch_affine_dtype),
            dtype=affine_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=gamma_tt, beta=beta_tt, eps=eps)

    # --- Output contract ---
    assert tuple(y_tt.shape) == shape, f"shape: got {tuple(y_tt.shape)}, expected {shape}"
    assert y_tt.dtype == dtype, f"dtype contract: output {y_tt.dtype} != input {dtype}"
    assert y_tt.layout == ttnn.TILE_LAYOUT, f"output layout must be TILE_LAYOUT; got {y_tt.layout}"

    y_out = ttnn.to_torch(y_tt).to(torch.float32)
    threshold = PCC_THRESHOLDS[dtype]
    pcc = _pcc(y_out, y_ref)
    assert pcc >= threshold, (
        f"PCC below threshold for shape={shape}, num_groups={num_groups}, "
        f"input_layout={input_layout}, affine={affine_mode}, affine_dtype={affine_dtype}: "
        f"pcc={pcc:.5f} < {threshold}"
    )


# -----------------------------------------------------------------------------
# Smoke test: no_affine with input in TILE_LAYOUT should still work — this
# verifies that the R2 CT-arg threading doesn't break the no-affine path
# (where affine_layout is unused but must still travel cleanly through the
# descriptor).
# -----------------------------------------------------------------------------


def test_groupnorm_no_affine_with_tile_input_smoke(device):
    """no_affine path is unaffected by the affine_layout axis. Sanity check."""
    torch.manual_seed(7)
    shape = (1, 1, 64, 128)
    num_groups = 4
    eps = 1e-5

    x_torch = torch.randn(shape, dtype=torch.float32)
    y_ref = _torch_groupnorm(x_torch, num_groups, None, None, eps)

    x_tt = ttnn.from_torch(
        x_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=None, beta=None, eps=eps)
    y_out = ttnn.to_torch(y_tt).to(torch.float32)
    pcc = _pcc(y_out, y_ref)
    assert pcc >= 0.995, f"no_affine smoke: pcc={pcc:.5f}"


# -----------------------------------------------------------------------------
# bf8b affine smoke test — bf8b is a block-quantized format with no RM
# representation, so the only way to test it is via affine_layout=TILE.
# Input dtype is bf16 (bf8b INPUT is EXCLUSIONS-gated per R1). PCC threshold
# follows the bf8b tolerance band (0.99).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("shape, num_groups, shape_id", SHAPES, ids=[s[2] for s in SHAPES])
def test_groupnorm_affine_tile_bf8b(device, shape, num_groups, shape_id):
    """affine_dtype=bf8b is only reachable via affine_layout=TILE (R2)."""
    torch.manual_seed(11)
    dtype = ttnn.bfloat16
    affine_dtype = ttnn.bfloat8_b
    eps = 1e-5
    N, one, HW, C = shape

    x_torch = torch.randn(shape, dtype=torch.float32)
    gamma_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    beta_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)

    y_ref = _torch_groupnorm(x_torch, num_groups, gamma_torch, beta_torch, eps)

    x_tt = ttnn.from_torch(
        x_torch.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # bf8b has no native torch representation; the ttnn.from_torch call
    # quantizes the bf16 tensor into bf8b on device.
    gamma_tt = ttnn.from_torch(
        gamma_torch.to(torch.bfloat16),
        dtype=affine_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tt = ttnn.from_torch(
        beta_torch.to(torch.bfloat16),
        dtype=affine_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=gamma_tt, beta=beta_tt, eps=eps)
    y_out = ttnn.to_torch(y_tt).to(torch.float32)
    pcc = _pcc(y_out, y_ref)
    # bf8b tolerance band — the per-block shared-exponent quantization
    # introduces additional precision floor that bf16/fp32 do not have.
    threshold = PCC_THRESHOLDS[affine_dtype]
    assert pcc >= threshold, (
        f"bf8b affine PCC below threshold for shape={shape}, num_groups={num_groups}: " f"pcc={pcc:.5f} < {threshold}"
    )
