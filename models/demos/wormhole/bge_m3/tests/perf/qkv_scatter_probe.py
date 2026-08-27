# SPDX-License-Identifier: Apache-2.0
"""Probe: can minimal_matmul_split (Python/JIT) beat minimal_matmul + head-split?

At the exact DP QKV shape (B6, S8192, K=1024, N=3072). Compares device-kernel time:
  A = minimal_matmul (fused) + fused head-split K-BF4         [current #217]
  B = minimal_matmul_split(chunks=3) + per-tensor head reshape

The scatter-writer win (fusing head-transpose into the matmul writer) needs a
C++ rebuild. This probe checks whether the EXISTING Python-bound split op offers
any of that win without a rebuild.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*a, **k):
        return None


B, S, HIDDEN, NUM_HEADS, DH = 6, 8192, 1024, 16, 64
QKV_N = 3 * NUM_HEADS * DH  # 3072
N_ITERS = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_qkv_scatter_probe(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 required")

    from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=16,
        K_block_size=16,
        N_block_size=8,
        subblock_h=4,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False
    )

    torch.manual_seed(0)
    hs = ttnn.from_torch(
        torch.randn(B, 1, S, HIDDEN, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(HIDDEN, QKV_N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def approach_a():
        qkv = ttnn.experimental.minimal_matmul(
            input_tensor=hs,
            weight_tensor=w,
            bias_tensor=None,
            fused_activation=None,
            config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
        )
        q, k, v = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=4, k_out_dtype=ttnn.bfloat4_b)
        ttnn.deallocate(qkv)
        return q, k, v

    def approach_b():
        outs = ttnn.experimental.minimal_matmul_split(
            hs,
            w,
            chunks=3,
            dim=-1,
            bias_tensor=None,
            fused_activation=None,
            config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
        )
        return tuple(outs)  # [B,1,S,H*D] each — width-split only, no head transpose

    for name, fn in [("A_matmul_plus_headsplit", approach_a), ("B_matmul_split", approach_b)]:
        try:
            for _ in range(2):
                outs = fn()
                for t in outs:
                    ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            signpost(name)
            for _ in range(N_ITERS):
                outs = fn()
                for t in outs:
                    ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"OK {name}")
        except Exception as e:
            logger.error(f"FAIL {name}: {str(e)[:160]}")
    signpost("end")
