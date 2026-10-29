# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH op-level bench/dump for the fused GDN chunk op (phased prep + scan) at the Qwen3.6-27B TP=4 shape.

Per device: flat q/k [1, 2048, 4*128] bf16, v [1, 2048, 12*128] bf16, beta/g [1, 2048, 12] bf16, chunk 32,
12 value heads, K=V=128 -> scan on 48 cores (12 heads x 4 V-blocks). Runs GDN_REPEATS iterations between tracy
signposts start_gdn/stop_gdn and dumps (o, final_state) to profiles/gdn_scan_<GDN_DUMP>.pt so two processes
(e.g. TT_GDN_SCAN_MCAST unset vs =1) can be compared bit-for-bit with profiles/gdn_dump_compare.py.

  source profiles/env.sh; profiles/prof_gdn.sh <name> [ENV=VAL ...]
"""
import math
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
    build_fused_const_tiles,
    chunk_gated_delta_rule_fused_adapter,
)

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
T, NK, DK, NV, DV = 2048, 4, 128, 12, 128


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_scan_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    repeats = int(os.environ.get("GDN_REPEATS", "10"))
    dump = os.environ.get("GDN_DUMP", "")
    torch.manual_seed(0)
    q = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    k = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    v = torch.randn(1, T, NV * DV, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, NV)).to(torch.bfloat16)
    g = (-torch.nn.functional.softplus(torch.randn(1, T, NV)) * 0.1).to(torch.bfloat16)  # log decay <= 0
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def up(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    tq, tk, tv, tb, tg = up(q), up(k), up(v), up(beta), up(g)
    consts = build_fused_const_tiles(mesh)

    def run():
        return chunk_gated_delta_rule_fused_adapter(
            tq,
            tk,
            tv,
            tb,
            tg,
            scale=1.0 / math.sqrt(DK),
            initial_state=None,
            device=mesh,
            qkv_head_dims=(NK, DK, NV, DV),
            return_o_bh=True,
            const_tiles=consts,
        )

    o, fs = run()
    ttnn.synchronize_device(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    o_t = ttnn.to_torch(o, mesh_composer=comp).float()
    fs_t = ttnn.to_torch(fs, mesh_composer=comp).float()
    logger.info(f"[GDN_BENCH] o {tuple(o_t.shape)} fs {tuple(fs_t.shape)} finite={bool(torch.isfinite(o_t).all())}")
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    t0 = time.perf_counter()
    signpost("start_gdn")
    for _ in range(repeats):
        o, fs = run()
        ttnn.deallocate(o)
        ttnn.deallocate(fs)
    ttnn.synchronize_device(mesh)
    signpost("stop_gdn")
    wall = (time.perf_counter() - t0) / repeats * 1e6
    if dump:
        torch.save({"o": o_t, "fs": fs_t}, f"/home/ttuser/experiments/qwen36_27b/profiles/gdn_scan_{dump}.pt")
    print(f"GDN_BENCH_RESULT wall_us={wall:.1f} dump={dump}")
