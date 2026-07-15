# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 5d debug/regression test — RM cross-core + bf8b TILE gamma.
# DO NOT DELETE — documents the 5d verification.
#
# Lifts the {ROW_MAJOR, WIDTH/BLOCK_SHARDED, gamma_dtype=bf8b, gamma_layout=TILE}
# EXCLUSIONS (the last 5c carve-out). A bf8b tile is block-float (16 elements share
# one 8-bit exponent), so the row-0 sub-column extraction R5c does for fp32/bf16 is
# not a byte copy — the reader DEQUANTS each row-0 datum (block-float decode) into a
# real float cb_gamma_rm stick, then reuses the SAME RM-gamma compute tilize leg.
#
# Correctness is inherited from the R2 INTERLEAVED bf8b-gamma path: the reader
# dequant reconstructs exactly what the hardware unpacker produces (bf8b->float is a
# lossless widening), so feeding it through the tilize leg is numerically identical
# to the direct-TILE-read FPU-unpack path R2 already validated.

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
# bf8b gamma quantization error is absorbed by the input-dtype tolerance (same as the
# R2 INTERLEAVED bf8b-gamma cells). Keep the golden PCC floors.
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


def _run(device, shape, memory_layout, dtype, W):
    """RM (bf16/fp32) WIDTH/BLOCK cross-core + bf8b TILE gamma vs the fp32 reference."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=_TORCH[dtype])
    # gamma quantized to bf8b TILE (bf8b + RM is INVALID). Reference uses the pre-quant
    # bf16 gamma; the tolerance absorbs the bf8b quantization (R2's bf8b-gamma contract).
    g = torch.randn(W, dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    xin = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    gin = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(xin, gamma=gin, compute_kernel_config=_cfg(), memory_config=xin.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape).to(torch.float32)
    expected = _ref(x, g).to(torch.float32)
    assert_with_pcc(expected, actual, PCC[dtype])


def test_r5d_minimal(device):
    """Cheapest possible case: sub-tile Ws=8, 8-core WIDTH group, 1 tile-row, bf16."""
    _run(device, (1, 1, 32, 64), WIDTH, ttnn.bfloat16, 64)


# (shape, memory_layout): mix of sub-tile Ws (8/4), tile-aligned Ws (64), w_non
# (masked / ragged), narrow-H BLOCK (sub-tile Hs), multi-image (multi-round groups).
R5D_SHAPES = [
    ((1, 1, 32, 64), WIDTH),  # sub-tile Ws=8, 8-core group, 1 tile-row
    ((1, 1, 32, 4096), WIDTH),  # tile-aligned Ws=64 (spans 2 gamma tiles/core)
    ((2, 4, 128, 512), WIDTH),  # multi-image, multi-round (32 tile-rows)
    ((1, 1, 32, 50), WIDTH),  # w_non (bf16 rectangular 7c / fp32 ragged 13c)
    ((4, 8, 32, 47), WIDTH),  # w_non, multi-round (32 tile-rows)
    ((1, 1, 32, 64), BLOCK),  # sub-tile Hs=4 AND Ws=8
    ((1, 1, 256, 512), BLOCK),  # design loose case, tile-aligned per-core
    ((1, 1, 64, 17), BLOCK),  # w_non BLOCK (rectangular)
    ((1024, 1024), BLOCK),  # 2D, tile-aligned per-core (W_BLOCK_TILES=4, multi-round)
]


@pytest.mark.parametrize("shape,memory_layout", R5D_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_r5d_rm_cross_core_bf8b_tile_gamma(device, shape, memory_layout, dtype):
    """RM (bf16/fp32) WIDTH/BLOCK cross-core + bf8b TILE gamma matches the reference
    (within the input-dtype tolerance, which absorbs the bf8b gamma quantization) and
    keeps the input's shard spec."""
    _run(device, shape, memory_layout, dtype, shape[-1])
