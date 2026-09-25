# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar 1D-mcast matmul DFB tile-counter underflow (llama32_1b decode QKV matmul).

The llama decode QKV matmul (forced interleaved on Quasar) is a 1D mcast_in0 matmul:
    in0 [1,1,32,2048] bf16 DRAM-interleaved  (M=32, K=2048 -> 64 K-tiles)
    w   [1,1,2048,3072] bf16 DRAM-interleaved (K=2048, N=3072 = Q2048+K512+V512)
    program_config: MatmulMultiCoreReuseMultiCast1DProgramConfig(grid 8x4, in0_block_w=2,
        per_core_M=1, per_core_N=3, out_subblock_h=1, out_subblock_w=3, mcast_in0=1) -> WIDTH_SHARDED L1

It runs to completion (after the mcast-rectangle fix) but then aborts:
    ERROR: UndefinedBehavior: qsr_tile_counter_check_error:
        tile counter occupancy=65535 exceeds capacity=4 (posted=64 acked=65)
occupancy = posted - acked = 64 - 65 = -1: the in0 DFB (cap 4 = 2 blocks x in0_block_num_tiles=2) is popped
ONE more than pushed (64 K-tiles pushed by the mcast receiver, 65 acked). This isolates just that matmul so
it can be run under TTSIM_QSR_DFB_TRACE=1 to name the counter and the extra-ack event.

Inputs are built bf16 via row-major upload + quasar.tilize (not from_torch(TILE), which hangs on the sim).

Run (Quasar sim, DFB trace to a file):
    TTSIM_QSR_DFB_TRACE=1 MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest tests/ttnn/unit_tests/operations/test_quasar_qkv_matmul_dfb.py 2> qkv_matmul_dfb_trace.txt
"""

import pytest
import torch
from loguru import logger

import ttnn

M, K, N = 32, 2048, 3072
GRID_X, GRID_Y = 8, 4


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[qkv-mm-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_qkv_matmul_mcast_in0(mesh_device):
    """1D mcast_in0 matmul at the QKV decode shape. FAILS on Quasar with the in0 DFB tile-counter underflow."""
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < GRID_X or grid.y < GRID_Y:
        pytest.skip(f"needs an {GRID_X}x{GRID_Y} grid; device has {grid.x}x{grid.y}")

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    at = _tile_bf16_dram(a, mesh_device)
    wt = _tile_bf16_dram(w, mesh_device)

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(GRID_X, GRID_Y),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=1,
        per_core_N=3,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    out_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    logger.info(f"[qkv-mm-repro] linear a[1,1,{M},{K}] x w[1,1,{K},{N}] mcast_in0 grid {GRID_X}x{GRID_Y}")
    out = ttnn.linear(
        at,
        wt,
        program_config=prog_cfg,
        memory_config=out_memcfg,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_cfg,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG))
    logger.info(f"[qkv-mm-repro] out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "QKV matmul produced non-finite output"
