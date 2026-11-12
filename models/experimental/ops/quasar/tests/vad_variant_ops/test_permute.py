# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.permute`` (ROW_MAJOR and TILE).

Shape source (kernel cache): permute kernels take only tensor-accessor ct args; the shape
lives in named compile-time args recovered from DWARF (``num_rows, x_dim, num_blocks_total,
X_p, W_p, rows_per_x, ...``; ``permute_{rm,tiled}_program_factory.cpp``):
  * P1 RM [1, 8, 6, 10000] -> dims (0, 2, 3, 1): num_rows 48, x_dim 1, 1878 = 6 x 313 blocks
    [M-H; dims[-1] = 1 certain, rest most plausible]
  * P5 TILE [8, 6, 10000] -> (1, 2, 0): X_p 32, W_p 10016, rows_per_x 6, output L1  [M]
  * P2/P3 small RM swaps of the last two dims  [M];  P4/P6 row-invariant permutes  [L shape]
Permute is value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q = U.NUM_QUERY
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# fmt: off
# (shape, dims, layout, in_mem, out_mem)
CASES = [
    pytest.param((1, U.NUM_HEADS, U.NUM_CAMS, Q), (0, 2, 3, 1), RM, "dram", "dram", id="P1-rm-bev-attn-0231"),
    pytest.param((1, 1, 1, 32), (0, 1, 3, 2), RM, "dram", "dram", id="P2-rm-1x1x1x32"),
    pytest.param((1, 1, 2, 8), (0, 1, 3, 2), RM, "l1", "l1", id="P3-rm-1x1x2x8"),
    pytest.param((1, 4, 8, 32), (0, 2, 1, 3), RM, "dram", "dram", id="P4-rm-row-invariant"),
    pytest.param((U.NUM_HEADS, U.NUM_CAMS, Q), (1, 2, 0), TILE, "dram", "l1", id="P5-tile-8x6x10000-120"),
    pytest.param((1, 300, U.NUM_HEADS, U.HEAD_DIM), (0, 2, 1, 3), TILE, "dram", "dram", id="P6-tile-row-invariant"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dims, layout, in_mem, out_mem", CASES)
def test_permute(ttnn_mesh_device, reset_seeds, shape, dims, layout, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, in_mem, layout=layout)
    out = ttnn.permute(x, dims, memory_config=U.resolve_memcfg(out_mem, mesh))
    U.assert_lossless(x_torch.permute(dims).contiguous(), out, mesh_device=mesh)
