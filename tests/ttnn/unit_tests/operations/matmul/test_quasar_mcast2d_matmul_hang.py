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

    # Resized for the 2-compute-core emulator (was 8x4=32 cores). Block-sharding K (or N) across a
    # multi-COLUMN grid requires grid_x = number of width-shards; the emulator can't provide extra columns,
    # so a [224,128] tensor with a [224,64] shard needs 2 columns and trips tensor_spec.cpp:153 "shards
    # along width (2) must not exceed columns (1)". Fix: run on a SINGLE core (1x1 grid) with the WHOLE
    # tensor as one block shard (shard shape == full tensor, no K/N split). The matmul kernels are no-op'd
    # (LLK dest-sync deadlock, issue filed), so this only needs to allocate + launch, not hang.
    M, K, N = 224, 128, 32
    grid_x, grid_y = 1, 1  # single compute core; whole A/out is one block shard (no K/N split)

    a_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    b_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    # BLOCK_SHARDED A on 1 core: the whole tensor is one shard [224,128] (7 x 4 tiles). The shard width
    # equals the full tensor width, so it fits a single-column grid (no "shards along width" violation).
    a_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, M, K),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    a = a.to(device, a_mem_config)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Config classes live under the raw binding path (as the resnet model uses for the 1D variant).
    # grid=1x1: num_blocks_x = N_tiles(1)/per_core_N(1) = 1 = grid_x; num_blocks_y = per_core_M/out_block_h = 1.
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

    # Output BLOCK_SHARDED on 1 core: the whole output is one shard [224,32] (full N=32).
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, M, N),
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
