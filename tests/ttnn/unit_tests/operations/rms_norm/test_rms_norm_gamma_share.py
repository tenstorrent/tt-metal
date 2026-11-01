# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 — interleaved occupancy split + gamma multicast.

Pins:
  * the per-core-cost W split (mid tile-row counts now split k > 1 to fill the grid), and
  * the writer-side gamma load distributed by one multicast per W slice, in each regime of
    `_gamma_mcast_groups`: k == 1 single group (incl. a partial last grid row of row-groups), k > 1 one group
    per grid column, k > 1 with a partial grid row (per-core fallback), and a W slice held by one Tensix core
    (sender with no receivers) — across TILE / ROW_MAJOR input, TILE / ROW_MAJOR gamma, bf16 / fp32 /
    bfloat8_b gamma, and a non-aligned W / H tail.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y


def pcc(actual, expected):
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def perf_config():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def gamma_regime(device, shape, layout):
    """The `_gamma_mcast_groups` regime the op will pick for this shape (host-side, same solve as the op)."""
    geo = pd.derive_geometry(shape)
    sz = pd._Sizes(ttnn.bfloat16, ttnn.bfloat16, layout == ttnn.ROW_MAJOR_LAYOUT, True, False)
    grid = device.compute_with_storage_grid_size()
    blk = pd.select_blocking(geo, sz, grid.x, grid.y, pd.cb_l1_budget(device), 2)
    return blk, pd._gamma_mcast_groups(blk, grid.x)


def run(device, shape, *, layout, gamma_dtype, gamma_layout, cfg=None, pcc_min=0.9995):
    torch.manual_seed(3)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    W = shape[-1]
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    g = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
    tt_g = ttnn.from_torch(g, dtype=gamma_dtype, layout=gamma_layout, device=device)
    g_ref = ttnn.to_torch(tt_g).to(torch.float32).reshape(-1)[:W]
    out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=cfg or perf_config())
    actual = ttnn.to_torch(out).to(torch.float32)
    expected = torch_rms_norm(x, g_ref)
    p = pcc(actual, expected)
    assert p >= pcc_min, f"PCC {p:.6f} < {pcc_min} shape={shape} layout={layout} gamma={gamma_dtype}/{gamma_layout}"


# (shape, expected k > 1, expected gamma regime) on the 13x10 BH p150b grid; other grids only check numerics.
CASES = [
    pytest.param((1, 1, 640, 2048), "column", id="k6_column_groups"),
    pytest.param((1, 1, 1024, 512), "column", id="k3_column_groups"),
    pytest.param((1, 1, 352, 1024), "per_core", id="k6_partial_row_fallback"),
    pytest.param((1, 1, 3200, 256), "single", id="k1_single_group_partial_row"),
    pytest.param((1, 1, 96, 1024), "column", id="k8_one_group_per_column"),
    # Refinement 5: k = 16 > grid_x -> groups span grid rows, a slice's Tensix cores are not one column
    pytest.param((1, 1, 96, 4096), "per_core", id="k16_spanning_rows_per_core"),
    pytest.param((1, 1, 32, 2048), "column", id="k8_single_core_slices"),
    pytest.param((2, 1, 200, 1500), None, id="non_aligned_w_h"),
]


@pytest.mark.parametrize("shape, regime", CASES)
def test_rms_norm_gamma_share_tile(device, shape, regime):
    grid = device.compute_with_storage_grid_size()
    if regime is not None and (grid.x, grid.y) == (13, 10):
        blk, groups = gamma_regime(device, shape, ttnn.TILE_LAYOUT)
        got = "per_core" if groups is None else ("single" if blk.k == 1 else "column")
        assert got == regime, (shape, blk.k, groups)
    run(device, shape, layout=ttnn.TILE_LAYOUT, gamma_dtype=ttnn.bfloat16, gamma_layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(
    "gamma_dtype, gamma_layout",
    [
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.float32, ttnn.TILE_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ],
    ids=["g_bf16_rm", "g_fp32_tile", "g_bf8b_tile"],
)
@pytest.mark.parametrize("shape", [(1, 1, 640, 2048), (1, 1, 3200, 256)], ids=["k6_column", "k1_single"])
def test_rms_norm_gamma_share_gamma_formats(device, shape, gamma_dtype, gamma_layout):
    run(device, shape, layout=ttnn.TILE_LAYOUT, gamma_dtype=gamma_dtype, gamma_layout=gamma_layout, pcc_min=0.999)


@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 640, 2048), (1, 1, 1056, 384), (2, 1, 200, 1500)], ids=["k6", "k1_33groups", "non_aligned"]
)
def test_rms_norm_gamma_share_row_major(device, shape, gamma_layout):
    run(device, shape, layout=ttnn.ROW_MAJOR_LAYOUT, gamma_dtype=ttnn.bfloat16, gamma_layout=gamma_layout)
