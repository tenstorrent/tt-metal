# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH op-level bench/dump for the fused GDN chunk op (phased prep + scan) at the Qwen3.8-27B TP=2 shape (lane L).

Per device: flat q/k [1, 2048, NK*128] bf16, v [1, 2048, NV*128], beta/g [1, 2048, NV], chunk 32, K=V=128.
GDN_NV (default 24 = TP=2; 12 = TP=4) picks the value heads (NK = NV/3). GDN_REPEATS iterations between tracy
signposts start_gdn/stop_gdn; GDN_DUMP=<tag> saves (o, final_state) to $PROFILE_OUT_DIR/gdn_scan_<tag>.pt for
bit-exact A/B across processes (e.g. a factory change vs the shipped build).
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

DEVICE_PARAMS = [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
T, DK, DV = 2048, 128, 128


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_scan_bench_tp2(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    NV = int(os.environ.get("GDN_NV", "24"))
    NK = NV // 3
    repeats = int(os.environ.get("GDN_REPEATS", "10"))
    dump = os.environ.get("GDN_DUMP", "")
    torch.manual_seed(0)
    q = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    k = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    v = torch.randn(1, T, NV * DV, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, NV)).to(torch.bfloat16)
    g = (-torch.nn.functional.softplus(torch.randn(1, T, NV)) * 0.1).to(torch.bfloat16)  # log decay <= 0
    s0 = torch.randn(1, NV, DK, DV, dtype=torch.float32) * 0.1
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def up(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    tq, tk, tv, tb, tg = up(q), up(k), up(v), up(beta), up(g)
    ts0 = up(s0, ttnn.float32)
    consts = build_fused_const_tiles(mesh)

    def run():
        return chunk_gated_delta_rule_fused_adapter(
            tq,
            tk,
            tv,
            tb,
            tg,
            scale=1.0 / math.sqrt(DK),
            initial_state=ts0,
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
    logger.info(
        f"[GDN_BENCH] NV={NV} o {tuple(o_t.shape)} fs {tuple(fs_t.shape)} finite={bool(torch.isfinite(o_t).all())}"
    )
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
        out_dir = os.environ.get("PROFILE_OUT_DIR", "/home/ttuser/experiments/qwen36_27b/profiles/opt_round5/laneL")
        torch.save({"o": o_t, "fs": fs_t}, f"{out_dir}/gdn_scan_{dump}.pt")
    print(f"GDN_BENCH_RESULT NV={NV} wall_us={wall:.1f} dump={dump}")
