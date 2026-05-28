# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 3 test: hw_non_aligned shapes (HW % 32 != 0, C aligned).

The kernel now iterates over Ht = ceil(HW / 32) HW-tile-rows; the last
tile-row covers HW_LAST_ROWS valid rows (1..32). Two correctness things
must hold:

1. **Reductions** (mean, sumsq) use the *true* N_per_g = HW * Cg in the
   inv_N scaler, not the padded HW. For RM input, the reader zero-fills
   the padding rows in the last HW-tile-row so they contribute zero to
   the masked-tile reduction. For TILE input, ttnn's TILE_LAYOUT tile
   padding is zero-filled.

2. **Output shape** stays (N, 1, HW, C) — the user-visible logical shape
   is unchanged, even though the physical TILE_LAYOUT output has
   ceil(HW / 32) tile-rows.

This test exercises the four golden hw_non_aligned shapes (from
eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py::INPUTS), plus
a few additional edge cases targeting boundary values of HW % 32 (1, 31,
the partial-tile single-row case HW < 32, and multi-tile combined with
SDXL-style (C/G) % 32 != 0 mid-flight).

Tolerance per dtype matches the golden suite:
  bf16  → PCC ≥ 0.995
  fp32  → PCC ≥ 0.999
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


# hw_non_aligned shapes — every shape has HW % 32 != 0 and C % 32 == 0.
# Includes the four golden-test cases plus boundary-coverage cases.
HW_NON_ALIGNED_SHAPES = [
    # Golden-suite shapes (eval/golden_tests/.../feature_spec.py)
    ((1, 1, 17, 64), 1, "golden_17x64_G1"),  # HW=17 (Ht=1, last_rows=17, single partial)
    ((1, 1, 50, 128), 1, "golden_50x128_G1"),  # HW=50 (Ht=2, last_rows=18)
    ((1, 1, 47, 256), 1, "golden_47x256_G1"),  # HW=47 (Ht=2, last_rows=15)
    ((2, 1, 100, 128), 1, "golden_100x128_G1_N2"),  # HW=100 (Ht=4, last_rows=4), N=2
    # Boundary HW % 32 values not covered by golden
    ((1, 1, 1, 64), 1, "hw_1_single_row"),  # extreme: HW=1
    ((1, 1, 31, 64), 1, "hw_31_minus_one"),  # last_rows=31 (just under tile)
    ((1, 1, 33, 64), 1, "hw_33_plus_one"),  # last_rows=1
    ((1, 1, 63, 64), 1, "hw_63_two_tiles"),  # last_rows=31, Ht=2
    # hw_non_aligned combined with multi-tile C (more reduce iterations)
    ((1, 1, 47, 128), 2, "hw_47_C128_G2"),
    ((1, 1, 50, 64), 2, "hw_50_C64_G2"),
]


@pytest.mark.parametrize(
    "shape, num_groups, shape_id", HW_NON_ALIGNED_SHAPES, ids=[s[2] for s in HW_NON_ALIGNED_SHAPES]
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("affine_mode", ["gamma_beta", "gamma_only", "no_affine"])
def test_groupnorm_hw_non_aligned(device, shape, num_groups, shape_id, layout, affine_mode):
    """
    Per (shape, layout, affine_mode):
      - Output shape stays (N, 1, HW, C) — the logical (unpadded) shape.
      - PCC vs torch.nn.functional.group_norm meets the dtype threshold.
    """
    torch.manual_seed(42)

    dtype = ttnn.bfloat16
    eps = 1e-5
    N, one, HW, C = shape
    assert one == 1
    assert HW % 32 != 0, "test only covers hw_non_aligned"
    assert C % 32 == 0, "test only covers hw_non_aligned (C must be aligned)"
    assert C % num_groups == 0

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
    assert (
        tuple(y_tt.shape) == shape
    ), f"output shape must match input logical shape; got {tuple(y_tt.shape)}, expected {shape}"
    assert y_tt.dtype == dtype
    assert y_tt.layout == ttnn.TILE_LAYOUT  # output is always TILE_LAYOUT per design

    y_out = ttnn.to_torch(y_tt).to(torch.float32)
    threshold = PCC_THRESHOLDS[dtype]
    pcc = _pcc(y_out, y_ref)
    assert pcc >= threshold, (
        f"PCC below threshold for shape={shape}, num_groups={num_groups}, "
        f"layout={layout}, affine={affine_mode}: pcc={pcc:.5f} < {threshold}"
    )


# Targeted fp32 smoke — the bf16 sweep above is the bulk of the matrix;
# this confirms fp32 activations also work on the partial-HW path.
@pytest.mark.parametrize(
    "shape, num_groups",
    [
        ((1, 1, 50, 64), 2),
        ((1, 1, 47, 128), 1),
        ((1, 1, 17, 64), 1),
    ],
    ids=["50x64_G2", "47x128_G1", "17x64_G1"],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_groupnorm_hw_non_aligned_fp32(device, shape, num_groups, layout):
    torch.manual_seed(42)
    dtype = ttnn.float32
    eps = 1e-5
    N, one, HW, C = shape
    assert HW % 32 != 0

    x_torch = torch.randn(shape, dtype=torch.float32)
    gamma_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    beta_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    y_ref = _torch_groupnorm(x_torch, num_groups, gamma_torch, beta_torch, eps)

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gamma_tt = ttnn.from_torch(
        gamma_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    beta_tt = ttnn.from_torch(
        beta_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=gamma_tt, beta=beta_tt, eps=eps)

    assert tuple(y_tt.shape) == shape
    y_out = ttnn.to_torch(y_tt).to(torch.float32)
    pcc = _pcc(y_out, y_ref)
    assert pcc >= PCC_THRESHOLDS[dtype], (
        f"PCC below threshold for shape={shape}, num_groups={num_groups}, layout={layout}: " f"pcc={pcc:.5f}"
    )
