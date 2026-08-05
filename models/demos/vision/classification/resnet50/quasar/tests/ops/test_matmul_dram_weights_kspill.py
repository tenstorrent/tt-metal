# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option-A validation (#48552): does a Quasar matmul with WEIGHTS IN DRAM, K-SPILLED, run correctly on the 2-core
emulator without hanging?

This is the capability the layer4 L1-fit plan depends on. Layer4 conv2 (512->512 3x3) can't fit HEIGHT_SHARDED
(full-N x full-K weights ~4.6 MB > 3.7 MB L1) and block-sharded on 2 cores hits the fused conv_bmm 0x19. The
proposed fix (option A) is: on the SPLIT path, keep the weights DRAM-resident and stream them K-block-by-K-block
(in0_block_w < K, num K-blocks > 1) so only one K-block of weights is L1-resident at a time, accumulating partials.

This test exercises the underlying matmul DIRECTLY (bypassing the conv, whose split path does not K-spill yet):
it replicates the exact program config the height-sharded conv-as-matmul uses
(determine_matmul_op_config_from_conv_op_config_qsr, conv2d.cpp:275):
    MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=False, in0_block_w=<K-block>, per_core_M, per_core_N, ...)
with the activation HEIGHT_SHARDED over M and the weights in DRAM interleaved, at layer4 GEMM dims.

  * mcast_in0=False -> weights (in1) are broadcast to the M-sharded cores (the height-sharded conv pattern),
    read from DRAM; in0_block_w < K forces the K-spill (weight streaming) that makes it fit.

Outcomes:
  * PASS (PCC ok, no hang) -> option A is viable; wire the split Program B to keep weights in DRAM + K-spill.
  * HANG / FATAL -> the weights-in-DRAM mcast_in1 + K-spill matmul has its own issue (e.g. the in1-sender
    async_full_barrier / partials-accumulate path) and option A needs that fixed first.

The fc test (test_linear.py) already validates the mcast_in0=TRUE (width-sharded) weights-in-DRAM K-spill path;
this validates the mcast_in0=FALSE (height-sharded) path that layer4 conv2 would actually use.

Run (craq-sim / emulator, forced JIT):
  TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_matmul_dram_weights_kspill.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import nearest_32
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32


# (name, M_tiles, K_tiles, N_tiles)  -- layer4 GEMM equivalents (weights [K,N] in tiles)
# fmt: off
_GEMMS = [
    ("layer4_conv2_512to512_3x3",  2, 144, 16),   # M=out 7x7 padded 64=2t; K=512*9/32=144t; N=512/32=16t; weights 4.6MB
    ("layer4_downsample_1024to2048", 2, 32, 64),  # 1x1 s2: K=1024/32=32t; N=2048/32=64t; weights 4MB
]
# fmt: on


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("in0_block_w", [4, 8], ids=["kblk4", "kblk8"])
@pytest.mark.parametrize("name, m_tiles, k_tiles, n_tiles", _GEMMS, ids=[g[0] for g in _GEMMS])
def test_quasar_matmul_dram_weights_kspill(mesh_device, name, m_tiles, k_tiles, n_tiles, in0_block_w):
    device = mesh_device
    torch.manual_seed(0)

    assert k_tiles % in0_block_w == 0, f"in0_block_w={in0_block_w} must divide K_tiles={k_tiles}"
    num_k_blocks = k_tiles // in0_block_w
    if num_k_blocks < 2:
        pytest.skip(f"in0_block_w={in0_block_w} does not spill K_tiles={k_tiles} (need >=2 blocks)")

    M = m_tiles * TILE
    K = k_tiles * TILE
    N = n_tiles * TILE

    # Height-shard M across cores (mirrors the height-sharded conv: 1 M-tile/core on the 2-core part).
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, min(max_cores, m_tiles) + 1) if m_tiles % c == 0)
    per_core_M = m_tiles // num_cores
    per_core_N = n_tiles  # mcast_in0=False -> weights broadcast, each core computes full N for its M rows
    # out_subblock: per_core_M(=1 on 2-core) rows; widest w that divides N and keeps h*w <= 8 DEST tiles.
    out_subblock_h = 1
    out_subblock_w = max(w for w in range(1, 9) if per_core_N % w == 0 and out_subblock_h * w <= 8)
    mm_grid = (num_cores, 1)
    print(
        f"  DIAG {name}: M={M}(t{m_tiles}) K={K}(t{k_tiles}) N={N}(t{n_tiles}) | cores={num_cores} "
        f"per_core_M={per_core_M} per_core_N={per_core_N} in0_block_w={in0_block_w} "
        f"num_k_blocks={num_k_blocks} osub=({out_subblock_h},{out_subblock_w})"
    )

    matmul_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=mm_grid,
        in0_block_w=in0_block_w,  # < K_tiles => K-spill: only in0_block_w x N of weights resident at a time
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,  # weights (in1) broadcast to the M-sharded cores == the height-sharded conv pattern
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    act_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    weight_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)  # ttnn matmul: act @ weight, weight stored [K,N]
    golden = torch.matmul(act_torch.float(), weight_torch.float())  # [1,1,M,N]

    # Activation HEIGHT_SHARDED over M on the matmul grid (mcast_in0=False requires input sharding == matmul grid).
    act_core_grid = ttnn.CoreGrid(x=num_cores, y=1)
    act_mem_config = ttnn.create_sharded_memory_config_(
        [nearest_32(M) // num_cores, K],
        act_core_grid,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )
    act = ttnn.from_torch(act_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, act_mem_config)

    # Weights in DRAM interleaved -> streamed K-block-by-K-block (the whole point of option A).
    weight = ttnn.from_torch(
        weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.quasar.matmul(
        act,
        weight,
        program_config=matmul_config,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )

    got = ttnn.to_torch(ttnn.from_device(out)).float().reshape(1, 1, M, N)
    # bf16 + LoFi over K up to 4608 is noisy -> 0.98 (raise once the path is trusted).
    assert_with_pcc(golden, got, pcc=0.98)
    print(f"  {name} kblk{in0_block_w} PASSED (weights-in-DRAM K-spill matmul, {num_k_blocks} K-blocks)")
