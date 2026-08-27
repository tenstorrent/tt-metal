# SPDX-License-Identifier: Apache-2.0
"""Probe: custom head-split concat-heads vs stock nlp_concat_heads at S8192.

At S8192 the model currently uses STOCK ttnn.experimental.nlp_concat_heads
(~13.8ms/pass in the profile). The custom bge_concat_heads_headsplit kernel
exists but is only wired for S512. This probe checks whether enabling it at
S8192 (like the QKV head-split) is faster, and sweeps head_groups.

Context shape: [B6, 16, S8192, 64] BFP8 (SDPA output).
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
N_ITERS = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_concat_heads_probe(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 required")

    from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads.op import (
        bge_concat_heads_headsplit,
        bge_concat_heads_stock,
    )

    torch.manual_seed(0)
    ctx = ttnn.from_torch(
        torch.randn(B, NUM_HEADS, S, DH, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_mem = ttnn.DRAM_MEMORY_CONFIG

    # correctness reference (stock)
    ref = ttnn.to_torch(
        bge_concat_heads_stock(ctx, out_memcfg=out_mem), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:B]

    configs = [("stock_nlp_concat", None)]
    for hg in [1, 2, 4, 8, 16]:
        configs.append((f"headsplit_hg{hg}", hg))

    for name, hg in configs:

        def fn():
            if hg is None:
                return bge_concat_heads_stock(ctx, out_memcfg=out_mem)
            return bge_concat_heads_headsplit(ctx, head_groups=hg, out_memcfg=out_mem)

        try:
            # correctness
            got = ttnn.to_torch(fn(), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B]
            pcc = torch.corrcoef(torch.stack([got.float().flatten(), ref.float().flatten()]))[0, 1].item()
            for _ in range(2):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            signpost(name)
            for _ in range(N_ITERS):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            logger.info(f"OK {name} pcc={pcc:.5f}")
        except Exception as e:
            logger.error(f"FAIL {name}: {str(e)[:160]}")
    signpost("end")
