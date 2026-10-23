# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_linear.py — do NOT regenerate this file
# from a capture.
#
# The captured ttnn.linear cases run the DRAM-sharded matmul
# (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig): weight DRAM-width-sharded
# across 12 banks. On a small device that shard grid does not exist, so the case is
# skipped. Two problems make a faithful small version impossible:
#   1. The DRAM-sharded path calls get_optimal_dram_bank_to_logical_worker_assignment
#      (core_assignment.cpp), which TT_ASSERTs arch in {WORMHOLE_B0, BLACKHOLE} — it
#      is not implemented for Quasar.
#   2. The mainline matmul (what ttnn.linear uses) has no Quasar/Metal-2.0 factory at
#      all, so ttnn.linear cannot run on Quasar under any config.
#
# So this is a DIRECT test (not graph_case-driven): it exercises the Quasar-ported
# matmul (ttnn.experimental.quasar.matmul) as a regular 1D mcast_in0 matmul with the
# weight in INTERLEAVED DRAM (not DRAM-sharded) — the arrangement that actually runs
# on Quasar. Small dims (M=32=1 tile, K=64, N=128) keep it sim-tractable.
# ---------------------------------------------------------------------------
"""Small-grid Quasar matmul (1D mcast_in0, interleaved-DRAM weight) — stands in for ttnn.linear."""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

qsr = ttnn._ttnn.operations.experimental.quasar

# 2 compute cores; activation + output L1-width-sharded across them, weight interleaved in DRAM.
_GRID_XY = (2, 1)
_M, _K, _N = 32, 64, 128  # M=1 tile (mcast_in0 needs M in one per_core_M block), K=2 tiles, N=4 tiles


@G.with_default_mesh()
def test_linear_small_grid(ttnn_mesh_device, reset_seeds):
    mesh_device = ttnn_mesh_device
    dev = mesh_device.compute_with_storage_grid_size()
    if dev.x < _GRID_XY[0] or dev.y < _GRID_XY[1]:
        pytest.skip(f"case needs a {_GRID_XY[0]}x{_GRID_XY[1]} compute grid; device grid is {dev.x}x{dev.y}")

    worker_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GRID_XY[0] - 1, _GRID_XY[1] - 1))}
    )
    # in0: [M, K] L1 width-sharded across the 2 cores (K split, one K-tile per core).
    in0_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, [_M, _K // _GRID_XY[0]], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # out: [M, N] L1 width-sharded across the same 2 cores (N split, per_core_N tiles per core).
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, [_M, _N // _GRID_XY[0]], ttnn.ShardOrientation.ROW_MAJOR),
    )

    program_config = qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=_GRID_XY,
        in0_block_w=1,  # divides K (2 tiles)
        out_subblock_h=1,
        out_subblock_w=2,  # divides per_core_N (2)
        per_core_M=1,  # M = 1 tile
        per_core_N=2,  # N = 4 tiles / 2 cores
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        untilize_out=False,
    )

    torch.manual_seed(0)
    a = torch.randn([1, 1, _M, _K], dtype=torch.bfloat16)
    b = torch.randn([1, 1, _K, _N], dtype=torch.bfloat16)

    a_t = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in0_mc,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    # The "DRAM input" (weight) is INTERLEAVED, not DRAM-sharded.
    b_t = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )

    out_t = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=program_config,
        memory_config=out_mc,
        dtype=ttnn.bfloat16,
    )

    ref = torch.matmul(a.float(), b.float())
    U.assert_pcc(ref, out_t, pcc=0.99, mesh_device=mesh_device)
