# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH op-level bench/dump for the fused GDN conv (ttnn.experimental.kda.qkv_causal_conv1d_silu, TILE input) at the
Qwen3.8-27B TP=2 prefill shape (lane L): x [1, 2048, 5120] bf16 TILE (q 1024 | k 1024 | v 3072 per device), history
[1, 3, 5120] TILE, four [1, 1, 5120] tap tiles, channel chunk KDA_CHUNK (default 512 = the model's), HiFi4 fp32 acc
(the model's config). KDA_REPEATS iterations between tracy signposts; KDA_DUMP=<tag> saves q/k/v to
$PROFILE_OUT_DIR/kda_conv_<tag>.pt for bit-exact A/B across processes.
"""
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

DEVICE_PARAMS = [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
T, QW, KW, VW = 2048, 1024, 1024, 3072


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_kda_conv_bench_tp2(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    C = QW + KW + VW
    repeats = int(os.environ.get("KDA_REPEATS", "10"))
    chunk = int(os.environ.get("KDA_CHUNK", "512"))
    dump = os.environ.get("KDA_DUMP", "")
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def up(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )

    x = up(torch.randn(1, T, C))
    hist = up(torch.randn(1, 3, C))
    taps = [up(torch.randn(1, 1, C) * 0.5) for _ in range(4)]
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def run():
        return ttnn.experimental.kda.qkv_causal_conv1d_silu(
            x,
            hist,
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            KW,
            KW,
            VW,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=cfg,
        )

    q, k, v = run()
    ttnn.synchronize_device(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    outs = {n: ttnn.to_torch(t, mesh_composer=comp) for n, t in (("q", q), ("k", k), ("v", v))}
    logger.info(f"[KDA_BENCH] chunk={chunk} q {tuple(outs['q'].shape)} v {tuple(outs['v'].shape)}")
    for t in (q, k, v):
        ttnn.deallocate(t)
    t0 = time.perf_counter()
    signpost("start_kda")
    for _ in range(repeats):
        for t in run():
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    signpost("stop_kda")
    wall = (time.perf_counter() - t0) / repeats * 1e6
    if dump:
        out_dir = os.environ.get("PROFILE_OUT_DIR", "/home/ttuser/experiments/qwen36_27b/profiles/opt_round5/laneL")
        torch.save(outs, f"{out_dir}/kda_conv_{dump}.pt")
    print(f"KDA_BENCH_RESULT chunk={chunk} wall_us={wall:.1f} dump={dump}")
