# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH microbench: all_gather_minimal_matmul_async (AGMM) core-layout sweep at the exact Qwen3.6-27B TP=4
in-projection configs (mirrors tp_common.all_gather_matmul_prefill / all_gather_swiglu_prefill).

Variants (env AGMM_VARIANTS, comma list of name:GXxGY:T|F:links:workers_per_link), default:
  base:8x9:T:2:4  u10x10:10x10:F:2:5  t10x9:10x9:T:2:5  t11x9:11x9:T:2:6  u9x10:9x10:F:2:5
Shapes (env AGMM_SHAPES, default gdn_in,attn_in,mlp): gdn_in N=4120 / attn_in N=3584 (HiFi2 fp32-acc, bf8 weight,
M4/K8/N<=8, 1x4) and mlp N=8704 (LoFi fp32-acc, bf4 [gate|up] weight, fuse_swiglu, M8/K8/N16, 1x4).
Each variant runs 1 warm + AGMM_REPEATS timed iterations between tracy signposts start_<shape>_<variant>/stop_...;
PCC vs torch is checked for gdn/attn (and for mlp assuming tile-column-pair [gate|up] interleave).

  source profiles/env.sh; profiles/prof_agmm.sh <name> [ENV=VAL ...]
"""
import math
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
S = 2048
TILE = 32
SHAPES = {
    "gdn_in": (5120, 4120),
    "attn_in": (5120, 3584),
    "mlp": (5120, 8704),
    "gdn_out": (4096, 1280),
}  # (K_total, N_local)
DEFAULT_VARIANTS = "base:8x9:T:2:4,u10x10:10x10:F:2:5,t10x9:10x9:T:2:5,t11x9:11x9:T:2:6,u9x10:9x10:F:2:5"


def _parse_variants():
    out = []
    for v in os.environ.get("AGMM_VARIANTS", DEFAULT_VARIANTS).split(","):
        name, grid, tr, links, nwpl = v.split(":")
        gx, gy = (int(t) for t in grid.split("x"))
        out.append((name, (gx, gy), tr == "T", int(links), int(nwpl)))
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_agmm_grid(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    repeats = int(os.environ.get("AGMM_REPEATS", "10"))
    shapes = os.environ.get("AGMM_SHAPES", "gdn_in,attn_in,mlp").split(",")
    do_pcc = os.environ.get("AGMM_PCC", "1") == "1"
    n_block_override = os.environ.get("AGMM_NBLOCK")  # optional int for gdn/attn/gdn_out
    n_block_mlp = int(os.environ.get("AGMM_NBLOCK_MLP", "16"))
    m_block_mlp = int(os.environ.get("AGMM_MBLOCK_MLP", "8"))
    nbuf = int(os.environ.get("AGMM_NBUF", "8"))  # num_buffers_per_channel
    kblock_override = os.environ.get("AGMM_KBLOCK")  # optional int
    mode = os.environ.get(
        "AGMM_MODE", "agmm"
    )  # agmm | mm (ttnn.experimental.minimal_matmul on the pre-gathered x: same chains, no fabric relay)
    cfg_hifi2 = tpc.COMPUTE_HIFI2
    cfg_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    torch.manual_seed(0)
    results = []
    for shape in shapes:
        K_TOTAL, n_local = SHAPES[shape]
        is_mlp = shape == "mlp"
        x = torch.randn(1, 1, S, K_TOTAL, dtype=torch.bfloat16)
        x_sh = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        x_full = None
        if mode == "mm":
            x_full = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        w = torch.randn(K_TOTAL, n_local * 4, dtype=torch.bfloat16) * 0.02
        w_dtype = ttnn.bfloat4_b if is_mlp else ttnn.bfloat8_b
        w_sh = ttnn.from_torch(
            w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)
        )
        # reference (device-0 shard): x_full @ w[:, :n_local] with the dtype-rounded weight
        ref = None
        if do_pcc:
            w0 = ttnn.to_torch(ttnn.from_torch(w[:, :n_local], dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
            y = x.float().reshape(S, K_TOTAL) @ w0
            if is_mlp:
                # assume tile-column-pair interleave: even tile columns = gate, odd = up
                yt = y.reshape(S, n_local // (2 * TILE), 2, TILE)
                y = (torch.nn.functional.silu(yt[:, :, 0, :]) * yt[:, :, 1, :]).reshape(S, n_local // 2)
            ref = y
        for name, grid, transpose, links, nwpl in _parse_variants():
            in1_axis = grid[1] if transpose else grid[0]
            if is_mlp:
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=m_block_mlp,
                    K_block_size=int(kblock_override) if kblock_override else tpc.agmm_k_block_size(K_TOTAL // 4),
                    N_block_size=n_block_mlp,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                )
                ckc = cfg_lofi
            else:
                n_tiles_per_core = max(1, math.ceil(n_local / TILE / in1_axis))
                n_block = int(n_block_override) if n_block_override else min(8, n_tiles_per_core)
                sub_w = max(d for d in (4, 2, 1) if n_block % d == 0)
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=int(kblock_override) if kblock_override else tpc.agmm_k_block_size(K_TOTAL // 4),
                    N_block_size=n_block,
                    subblock_h=1,
                    subblock_w=sub_w,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                )
                ckc = cfg_hifi2

            def run():
                if mode == "mm":
                    return ttnn.experimental.minimal_matmul(
                        x_full,
                        w_sh,
                        config=cfg,
                        compute_kernel_config=ckc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                        fuse_swiglu=is_mlp,
                    )
                return ttnn.experimental.all_gather_minimal_matmul_async(
                    input_tensor=x_sh,
                    weight_tensor=w_sh,
                    config=cfg,
                    compute_kernel_config=ckc,
                    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                    num_links=links,
                    topology=topo,
                    cluster_axis=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    force_transpose=transpose,
                    num_workers_per_link=nwpl,
                    num_buffers_per_channel=nbuf,
                    fuse_swiglu=is_mlp,
                )[0]

            tag = f"{shape}_{name}"
            try:
                out = run()
                ttnn.synchronize_device(mesh)
                pcc = None
                if do_pcc:
                    got = (
                        ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].reshape(S, -1).float()
                    )
                    _, pcc = comp_pcc(ref, got, 0.99)
                ttnn.deallocate(out)
                t0 = time.perf_counter()
                signpost(f"start_{tag}")
                for _ in range(repeats):
                    o = run()
                    ttnn.deallocate(o)
                ttnn.synchronize_device(mesh)
                signpost(f"stop_{tag}")
                wall_us = (time.perf_counter() - t0) / repeats * 1e6
                msg = f"AGMM_GRID {mode} {shape:8s} {name:8s} grid={grid} T={transpose} links={links} nwpl={nwpl} mblk={cfg.M_block_size} nblk={cfg.N_block_size} kblk={cfg.K_block_size} nbuf={nbuf} wall_us={wall_us:8.1f} pcc={pcc}"
            except Exception as e:  # noqa: BLE001
                msg = f"AGMM_GRID {mode} {shape:8s} {name:8s} grid={grid} T={transpose} links={links} nwpl={nwpl} ERR {str(e).splitlines()[0][:160]}"
            logger.info(msg)
            results.append(msg)
        ttnn.deallocate(w_sh)
        ttnn.deallocate(x_sh)
        if x_full is not None:
            ttnn.deallocate(x_full)
    for m in results:
        print(m)
    print("AGMM_GRID_DONE")
