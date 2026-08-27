# SPDX-License-Identifier: Apache-2.0
"""Standalone QKV head-split head_groups sweep at the exact DP shape.
Measures DEVICE KERNEL DURATION (via tracy signposts), not wall clock.

Finding 3: head_groups=4 is inherited from B8/B32/S512, never tuned for
B6/S8192. At B6/S8192 there are 6*256=1536 batch*seq_tile units before head
grouping -> all 64 cores saturate at head_groups=1. Groups 2/4 only shrink
units and multiply reader/writer/CB-sync frequency. Class ceiling 35.4ms/pass.

Shape: qkv_fused [B6, 1, S8192, 3*16*64=3072] BFP8 (QKV matmul output).
Each head_groups value runs under its own signpost.
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


B, NUM_HEADS, S, DH = 6, 16, 8192, 64
QKV_W = 3 * NUM_HEADS * DH  # 3072
N_ITERS = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_headsplit_groups_probe(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 required")

    from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

    qkv = ttnn.from_torch(
        torch.randn(B, 1, S, QKV_W, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for hg in [1, 2, 4, 8, 16]:
        name = f"head_groups_{hg}"
        try:
            # warmup (compile) outside signpost
            for _ in range(2):
                q, k, v = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=hg)
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            ttnn.synchronize_device(mesh_device)
            signpost(name)
            for _ in range(N_ITERS):
                q, k, v = bge_qkv_heads_headsplit(qkv, num_heads=NUM_HEADS, head_groups=hg)
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"OK {name}")
        except Exception as e:
            logger.error(f"FAIL {name}: {str(e)[:160]}")
    signpost("end")
    ttnn.deallocate(qkv)
