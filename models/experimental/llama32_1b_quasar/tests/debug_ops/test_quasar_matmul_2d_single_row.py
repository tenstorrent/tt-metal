# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guard for the 2D-mcast matmul on a SINGLE-ROW grid (the 2-node emulator, 2x1).

The llama prefill fused-QKV matmul runs a 2D-mcast config on the 2-node emulator, whose usable grid is a
single row (2x1: cores (0,0),(1,0)). The mainline MatmulMultiCoreReuseMcast2DProgramFactory computed the
multicast-rectangle "+1" corner unconditionally -- top_core_plus_one = (core.x, start_core_y + 1) -- so on a
single-row grid it addressed (0,1), which has no core, and worker_core_from_logical_core threw:
    RuntimeError: No core coordinate found at location: (0, 1, TENSIX, LOGICAL)

Fix (ported from the experimental/quasar 2D factory): clamp the "+1" corner to the sender's own coord when
the dimension has a single core (num_cores_with_work_* == 1); the degenerate mcast then covers 0 receivers so
the clamped rectangle is unused. Multi-row/col grids are unchanged. See
matmul_multicore_reuse_mcast_2d_program_factory.cpp.

This test builds a 2D-mcast matmul on a 2x1 grid with per_core_M = full M (num_blocks_y = 1 -> a single work
row, i.e. num_cores_with_work_r == 1), which is exactly the degenerate case: WITHOUT the clamp it FATALs at
program creation with "No core (0,1)"; WITH it, it builds and computes. N is split across the 2 columns and M
streams via out_block_h. Inputs are built via quasar.tilize (from_torch(TILE) faults on Quasar).

Run (Quasar, SLOW dispatch = the true single-row grid):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_matmul_2d_single_row.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# [M, K, N] in tiles * 32. A 2x1 grid: M (2 tiles) on the single row, N (2 tiles) split across the 2 columns.
M, K, N = 64, 64, 64  # 2 x 2 x 2 tiles
GRID = (2, 1)  # gx=2 cols, gy=1 row -> single-row 2D mcast (num_cores_with_work_r == 1)


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.timeout(3600)
def test_matmul_2d_mcast_single_row(mesh_device):
    """2D-mcast matmul on a 2x1 grid (single work row). Fails at program creation ('No core (0,1)') without
    the single-row mcast-corner clamp; builds + PCC-checks with it."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    at = _tile_bf16_dram(a, mesh_device)
    bt = _tile_bf16_dram(b, mesh_device)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GRID[0], GRID[1]),
        in0_block_w=K // 32,  # 2 tiles, one K block
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=M // 32,  # 2 tiles; streams M (per_core_M) in out_block_h chunks
        out_block_w=1,
        per_core_M=M // 32,  # full M on the single row -> num_blocks_y = 1 -> num_cores_with_work_r == 1
        per_core_N=(N // 32) // GRID[0],  # N split across the 2 columns
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    out = ttnn.matmul(at, bt, program_config=program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)

    ref = a.float().reshape(M, K) @ b.float().reshape(K, N)
    pcc = _pcc(o, ref)
    logger.info(f"[mm-2d-single-row] out {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}")
    assert tuple(o.shape) == (1, 1, M, N), f"unexpected shape {tuple(o.shape)}"
    assert pcc > 0.99, f"2D single-row matmul PCC {pcc}"
