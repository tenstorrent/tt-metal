# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused recurrence's scalar-page layout: [B,T,H] -> [B*H*T,1,1], one crossing, not two.

The kernel broadcasts decay and beta from element [0,0] of one page per (head, token), so the wrapper
has to move 48 per-head scalars out of a single tile row into 48 tile pages. It used to do that with
permute + (untilize, reshape, tilize), i.e. TWO crossings of the tile boundary, and the layer paid
~22 us of device time for the pair (profile p1_after -> p2_after, tp1_fused region).

headvec_scalars() now crosses once at T == 1 with a single ttnn.reshape. That is only valid if the
values land in the same place, which is what this test pins: element [h,0,0] of the result must be
input element h, and the whole [BH,1,1] tensor must equal torch's flatten. Values are distinct per
index so a misplacement cannot pass by luck.
"""
import pytest
import torch
from loguru import logger

import ttnn


def _mk(mesh_device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("n", [48, 32, 64, 96])
def test_single_crossing_matches_row_major(mesh_device, n):
    """A bare reshape and the untilize/reshape/tilize path must agree exactly, and both must match torch."""
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(1, 1, n)
    x = _mk(mesh_device, src)

    # The op's previous path: cross the tile boundary through ROW_MAJOR.
    t = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, (n, 1, 1))
    via_rm = ttnn.to_torch(ttnn.to_layout(t, ttnn.TILE_LAYOUT)).reshape(n)

    # The op's current path: one ttnn.reshape.
    via_one = ttnn.to_torch(ttnn.reshape(x, (n, 1, 1))).reshape(n)

    want = src.reshape(n)
    logger.info(f"n={n}: one-reshape matches torch {torch.equal(via_one, want)}, row-major path {torch.equal(via_rm, want)}")
    assert torch.equal(via_rm, want), f"the row-major reference path is wrong for n={n}: {via_rm[:4].tolist()}"
    assert torch.equal(via_one, want), f"single reshape misplaced values for n={n}: {via_one[:4].tolist()} vs {want[:4].tolist()}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pair_layout_as_the_op_builds_it(mesh_device):
    """The real call: fp32, exp applied to g on the compact form, one reshape each, values preserved."""
    torch.manual_seed(0)
    g = torch.randn(1, 1, 48, dtype=torch.float32)
    b = torch.sigmoid(torch.randn(1, 1, 48, dtype=torch.float32))
    g_tt = _mk(mesh_device, g, dtype=ttnn.bfloat16)  # the layer hands over bf16
    b_tt = _mk(mesh_device, b, dtype=ttnn.bfloat16)

    d = ttnn.reshape(ttnn.exp(ttnn.typecast(g_tt, ttnn.float32)), (48, 1, 1))
    bb = ttnn.reshape(ttnn.typecast(b_tt, ttnn.float32), (48, 1, 1))

    d_dev = ttnn.to_torch(d).reshape(48)
    b_dev = ttnn.to_torch(bb).reshape(48)
    d_ref = torch.exp(g.bfloat16().float()).reshape(48)
    b_ref = b.bfloat16().float().reshape(48)
    logger.info(f"decay max err {(d_dev - d_ref).abs().max():.3e}, beta max err {(b_dev - b_ref).abs().max():.3e}")
    assert torch.allclose(d_dev, d_ref, rtol=1e-2, atol=1e-6), "decay pages are wrong"
    assert torch.allclose(b_dev, b_ref, rtol=1e-2, atol=1e-6), "beta pages are wrong"
