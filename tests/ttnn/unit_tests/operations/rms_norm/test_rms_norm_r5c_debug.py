# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 5c debug/regression test — RM cross-core + TILE gamma.
# DO NOT DELETE — documents the 5c verification.
#
# Lifts the {ROW_MAJOR, WIDTH/BLOCK_SHARDED, gamma_layout=TILE} exclusion for
# fp32/bf16 gamma. Each RM cross-core core owns a sub-tile W-slice at a sub-tile
# global column offset (w_col_start = i*Ws, Ws in {4,8,16,...}), so a TILE-stored
# gamma can't be read as whole tiles aligned to local col 0. The reader extracts
# the containing global gamma tile(s)' ROW-0 sub-columns (face-aware L1 byte copy)
# into cb_gamma_rm and reuses the RM-gamma compute tilize leg unchanged.
#
# bf8b TILE gamma stays EXCLUDED (block-float sub-tile extraction needs an
# in-reader dequant) — the last assertion guards that carve-out.

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def _ref(x, gamma):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


# (shape, memory_layout): mix of sub-tile Ws (8/4), tile-aligned Ws (64), w_non
# (masked / ragged), narrow-H BLOCK (sub-tile Hs), multi-image (multi-round groups).
R5C_SHAPES = [
    ((1, 1, 32, 64), WIDTH),  # sub-tile Ws=8, 8-core group, 1 tile-row
    ((1, 1, 32, 4096), WIDTH),  # tile-aligned Ws=64 (spans 2 gamma tiles/core)
    ((2, 4, 128, 512), WIDTH),  # multi-image, multi-round (32 tile-rows)
    ((1, 1, 32, 50), WIDTH),  # w_non (bf16 rectangular 7c / fp32 ragged 13c)
    ((4, 8, 32, 47), WIDTH),  # w_non, multi-round (32 tile-rows)
    ((1, 1, 32, 64), BLOCK),  # sub-tile Hs=4 AND Ws=8
    ((1, 1, 256, 512), BLOCK),  # design loose case, tile-aligned per-core
    ((1, 1, 64, 17), BLOCK),  # w_non BLOCK (rectangular)
    ((1024, 1024), BLOCK),  # 2D, tile-aligned per-core
]


@pytest.mark.parametrize("shape,memory_layout", R5C_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32])
def test_r5c_rm_cross_core_tile_gamma(device, shape, memory_layout, dtype, gamma_dtype):
    """RM (bf16/fp32) WIDTH/BLOCK cross-core + TILE gamma (bf16/fp32, incl. mixed
    precision) matches the reference and keeps the input's shard spec."""
    torch.manual_seed(0)
    W = shape[-1]
    x = torch.randn(shape, dtype=_TORCH[dtype])
    g = torch.randn(W, dtype=_TORCH[gamma_dtype])
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    xin = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    # gamma stored TILE at a possibly-different dtype (mixed precision) — the 5c case.
    gin = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(xin, gamma=gin, compute_kernel_config=_cfg(), memory_config=xin.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape).to(torch.float32)
    expected = _ref(x, g).to(torch.float32)
    assert_with_pcc(expected, actual, PCC[dtype])


def test_r5c_bf8b_tile_gamma_now_supported(device):
    """R5d lifted the last 5c carve-out: bf8b TILE gamma + RM cross-core is now SUPPORTED.
    The reader dequants the block-float row-0 sub-columns into the float cb_gamma_rm and
    reuses the RM-gamma tilize leg (see test_rms_norm_r5d_debug.py for the full matrix)."""
    torch.manual_seed(0)
    shape = (1, 1, 64, 512)
    x = torch.randn(shape, dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), BLOCK, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xin = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    g = torch.randn(512, dtype=torch.bfloat16)
    gin = ttnn.from_torch(g.reshape(1, 1, 1, 512), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(xin, gamma=gin, compute_kernel_config=_cfg(), memory_config=xin.memory_config())
    assert out.memory_config().memory_layout == BLOCK
    actual = ttnn.to_torch(out).reshape(shape).to(torch.float32)
    expected = _ref(x, g).to(torch.float32)
    assert_with_pcc(expected, actual, PCC[ttnn.bfloat16])
