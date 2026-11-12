# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reshape`` (TILE and ROW_MAJOR).

Shape source (kernel cache):
  * Tiled reshapes (``reader/writer_reshape_tiled``): only the mapping-table size
    (``max_map_entries`` = 64 / 128 / 32) survives; several shapes produce each value, so
    these are the model-plausible head split/merge candidates  [L]
  * RM ``rm_reshape_interleaved`` ct [.., src_page 512 B, dst_page 64 B, ..]: [..., 256] ->
    [..., 8, 32] in L1 (split embed into heads)  [M-H];  16 B -> 32 B rows in DRAM  [M]
Reshape is value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# fmt: off
# (in_shape, out_shape, layout, mem)
CASES = [
    pytest.param((1, Q, 8, 32), (1, Q, E), TILE, "dram", id="R1-merge-heads-dram"),
    pytest.param((1, Q, 8, 32), (1, Q, E), TILE, "l1", id="R2-merge-heads-l1"),
    pytest.param((1, Q, 8, 8), (1, Q, 64), TILE, "dram", id="R3-merge-8x8"),
    pytest.param((1, Q, 8, 4, 2), (1, Q, 8, 8), TILE, "dram", id="R6-points-xy-dram"),
    # No L1 variant of R6: TILE pads the last two dims (4, 2) to 32x32, so the input is
    # 80000 tiles = 164 MB, far beyond total L1. The model can only have held it in DRAM.
    pytest.param((1, Q, E), (1, Q, 8, 32), RM, "l1", id="R9-rm-split-heads-l1"),
    pytest.param((1, 100, 2, 8), (1, 100, 16), RM, "dram", id="R10-rm-16B-to-32B-rows"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("in_shape, out_shape, layout, mem", CASES)
def test_reshape(ttnn_mesh_device, reset_seeds, in_shape, out_shape, layout, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(in_shape)
    x = U.to_tt_mem(x_torch, mesh, mem, layout=layout)
    out = ttnn.reshape(x, out_shape, memory_config=U.resolve_memcfg(mem, mesh))
    assert tuple(out.shape) == tuple(out_shape)
    U.assert_lossless(x_torch.reshape(out_shape), out, mesh_device=mesh)
