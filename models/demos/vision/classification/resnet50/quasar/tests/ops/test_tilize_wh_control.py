# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
CONTROL / A-B experiment #2: run the MAINLINE ttnn.tilize (data_movement) on WH at the same geometry the
quasar conv fork's Program A (conv_tilize_only_metal2) deadlocks on, to decide LLK-bug vs fork-plumbing.

Context: with the SyncHalf experiment (factory forces SyncFull only for block_sharded), the split path's
Program A still deadlocks on WH -- now INSIDE fast_tilize_block (MATH mid-MOVB2D at llk_math_fast_tilize.h:250,
PACK mid-PACR at llk_pack_fast_tilize.h:331), a MATH<->PACK DEST-recycle stall. Mainline ttnn.conv2d already
PASSED on the same conv shape (test_regular_conv2d_wh_control), i.e. its fast_tilize_block works -- but that
path tilizes into an in-kernel CB feeding its own matmul. The fork instead tilizes into a BORROWED OUTPUT DFB
(exact-fill) drained by a SEPARATE drain_out program with disable_dfb_implicit_sync. This standalone tilize
isolates the LLK fast_tilize_block from the fork's borrowed-output/cross-program plumbing.

READING IT (WH):
  * PASSES -> the LLK fast_tilize_block is fine at this width/height on WH; the fork's deadlock is its
    borrowed-output + drain_out + disable_dfb_implicit_sync plumbing (or init ordering) -> CALLER-FIXABLE
    (rework Program A to tilize into a plain intermediate DFB + real writer, like the non-split path).
  * HANGS   -> the LLK fast_tilize_block itself deadlocks at this geometry on WH -> LLK fix needed, not caller.

Dims mirror the fork's Program A tilize: K = 16 tiles wide (512 cols, = in_channels 32 * 4 * 4 / 32, block_w=16),
per-core M large enough (>= 8 tiles) to exercise multiple height blocks (the fork uses act_block_h=128 = 4-tile
blocks, >= 2 blocks/core). Run WITHOUT any TT_METAL_QSR_* env.

Run (WH):
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_tilize_wh_control.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999


def _run_tilize(mesh_device, *, m_rows, k_cols, target_rows_per_core=256):
    """Standalone mainline ttnn.tilize of a height-sharded bf16 [1,1,m_rows,k_cols] tensor -> TILE layout.
    Round-trips back to torch and PCC-checks (tilize is layout-only, so out == in logically)."""
    device = mesh_device
    torch.manual_seed(0)
    torch_in = torch.randn((1, 1, m_rows, k_cols), dtype=torch.bfloat16).float()

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    # pick a core count that divides m_rows AND gives a tile-aligned, large-ish per-core height (>= 8 tiles)
    want = max(1, m_rows // target_rows_per_core)
    num_cores = max(
        (c for c in range(1, min(max_cores, m_rows // 32) + 1) if (m_rows % c == 0) and ((m_rows // c) % 32 == 0)),
        key=lambda c: (abs(c - want) * -1, c),
    )
    shard_h = m_rows // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, k_cols),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_in = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)

    # mainline tilize -> TILE layout (same fast_tilize_block the fork's Program A uses on WH).
    tt_tiled = ttnn.tilize(tt_in, use_multicore=True)

    out = ttnn.to_torch(ttnn.from_device(tt_tiled)).reshape(1, 1, m_rows, k_cols)
    assert_with_pcc(torch_in, out.float(), pcc=PCC)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "m_rows, k_cols",
    [
        (512, 512),  # K=16 tiles wide; ~8 M-tiles/core on a 2-core shard (matches the fork's per-core M)
        (2048, 512),  # taller: more height blocks per core, same K=16 width
    ],
    ids=["M512_K512", "M2048_K512"],
)
def test_tilize_wh_control(mesh_device, m_rows, k_cols):
    """Standalone WH fast_tilize at the fork's Program A geometry. PASS => LLK ok, fork plumbing is the bug;
    HANG => LLK fast_tilize_block bug at this geometry."""
    _run_tilize(mesh_device, m_rows=m_rows, k_cols=k_cols)
