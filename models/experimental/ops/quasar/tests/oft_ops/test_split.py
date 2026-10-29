# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.split`` — the ResNet stem splits its conv1 output into two 32-channel halves
so each GroupNorm half fits in L1 (``TTResNetFeatures.forward``:
``ttnn.split(conv1, conv1.shape[-1] // 2, dim=3)``).

Shape source: second kernel cache (in-tree OFT, factory at ``628a4797bf7``):
``reader_tm_tile_layout_split_two_chunks/494498`` ct = [z=1, per_core_tiles_x=768,
per_core_tiles_y=1, z_stride=7680, y_stride=2, DRAM, 2048 B] and
``writer_tm_tile_layout_split_two_chunks/129852`` (both outputs DRAM, 2048-B tiles):
2 width tiles (C=64) split over 2 y-cores, 3840 height tiles over 5 x-cores
-> [1, 1, 192*640, 64] bf16 TILE DRAM interleaved, split 32 on dim 3.                   [H]

Value-preserving: checked bit-exact. The mainline split factory is Metal 2.0 (ProgramSpec), so
it is called directly.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


def _run(mesh, shape):
    x = U.torch_rand(shape)
    tt_x = U.to_tt(x, mesh)  # DRAM interleaved bf16 TILE
    half = shape[-1] // 2
    outs = ttnn.split(tt_x, half, dim=3)
    assert len(outs) == 2, f"expected 2 chunks, got {len(outs)}"
    for i, out in enumerate(outs):
        assert out.memory_config().buffer_type == ttnn.BufferType.DRAM, "split output left DRAM"
        U.assert_lossless(x[..., i * half : (i + 1) * half], out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, U.STEM_H * U.STEM_W, 64), id="stem-conv1-1x1x122880x64-split32-494498"),
        pytest.param((1, 1, 2048, 64), id="proxy-1x1x2048x64-split32"),
    ],
)
def test_split_last_dim_two_chunks(ttnn_mesh_device, reset_seeds, shape):
    _run(ttnn_mesh_device, shape)
