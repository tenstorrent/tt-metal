# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro of the Quasar 2D block-sharded multicast matmul HANG.

Reached in resnet50/quasar after the layer3 block-sharding change: a
`matmul_multicore_reuse_mcast_2d` (MatmulDeviceOperation) whose in0 is BLOCK_SHARDED on the 8x4
grid deadlocks in the mcast sender/receiver semaphore handshake. Watcher: compute
`bmm_large_block_zm_fused_bias_activation` at UABW, the mcast readers
`reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded` / `..._in1_receiver_writer` at
WFW; the op never reaches synchronize_device.

NOTE (root cause still open): the async_full_barrier is NOT the cause — craq-sim stubs
`scmdbuf_tr_ack` to 0 so that barrier passes. The stuck cores are upstream in the sender_sem /
receiver_sem handshake. The in0 sender/receiver kernel is instrumented (ring-buffer markers
0x5E4D0001 sender-wait, 0x5E4D0002 receiver-ack, 0x5E4D0003 receiver-wait) — grep the watcher log
for those after this hangs to see which wait is stuck and the sem values.

Config is copied verbatim from the resnet run (log 108704):
  in0  A = [1,1,784,512] bf16, BLOCK_SHARDED, grid (0,0)-(7,3), shard [224,64]  (GRID_2D)
  in1  B = [1,1,512,256] bf16, DRAM interleaved
  program_config = MatmulMultiCoreReuseMultiCastProgramConfig(grid=8x4, in0_block_w=2,
                   out_subblock_h=7/w=1, out_block_h=7/w=1, per_core_M=7, per_core_N=1,
                   transpose_mcast=0)
  output = BLOCK_SHARDED [224,32]

out_subblock_h=1 also hangs, and made this a parameter.

Run (craq-sim, slow dispatch, forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest tests/ttnn/unit_tests/operations/matmul/test_quasar_mcast2d_matmul_hang.py

A healthy matmul returns; the bug hangs in the 2D-mcast handshake (never reaches the assert below).
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("out_subblock_h", [1, 7])
def test_quasar_mcast2d_matmul_hang(mesh_device, out_subblock_h):
    device = mesh_device

    # Resized for the 2-compute-core emulator (was 8x4=32 cores; over-sharded -> bank_manager TT_FATAL
    # "num_shards 32 <= 2 L1 banks"). Same 2D block-sharded mcast shape on a 2x1 grid: M over grid_y=1,
    # K over grid_x=2, N tiled over grid_x=2. The matmul kernels are currently no-op'd (LLK dest-sync
    # deadlock, issue filed), so this only needs to allocate + launch, not reproduce the hang.
    M, K, N = 224, 128, 64
    grid_x, grid_y = 1, 2  # in0 block-sharded across a 1x2 grid (M over y, K over x)

    a_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    b_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    # BLOCK_SHARDED A: 224 rows over grid_y=1 -> 224 (7 tiles); 128 cols over grid_x=2 -> 64 (2 tiles).
    # With a CoreRangeSet, pass the per-core SHARD shape + use_height_and_width_as_shard_shape=True.
    a_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 224, 64),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    a = a.to(device, a_mem_config)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Config classes live under the raw binding path (as the resnet model uses for the 1D variant).
    # grid=2x1: num_blocks_x = N_tiles(2)/per_core_N(1) = 2 = grid_x; num_blocks_y = per_core_M/out_block_h = 1.
    program_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=2,
        out_subblock_h=out_subblock_h,
        out_subblock_w=1,
        out_block_h=7,
        out_block_w=1,
        per_core_M=7,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )

    # Output BLOCK_SHARDED: shard [224,32] per core on the 2x1 grid.
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 224, 32),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out = ttnn.experimental.quasar.matmul(
        a,
        b,
        program_config=program_config,
        memory_config=out_mem_config,
        dtype=ttnn.bfloat16,
    )

    # The 2D-mcast handshake deadlock hangs above; a healthy run reaches here.
    ttnn.synchronize_device(device)
    assert out is not None
