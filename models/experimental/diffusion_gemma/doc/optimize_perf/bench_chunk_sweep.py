# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense prefill-MoE chunk and sparse-matmul geometry sweep.

The denoise MoE (gemma4 prefill_forward) splits the 256-token canvas into chunks of
PREFILL_CHUNK_SIZE=32 (group_size=1 each) and pays the expensive expert-major transpose
reorder once per chunk (8×). This first tests larger chunks on a grid that can fit their
output blocks, then keeps the correct 32-token chunk and sweeps K blocking for gate/up/down.
It reuses all existing sparse-matmul math and reports device wall-clock plus exactness/PCC
against the stock chunk=32 geometry.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from loguru import logger

import ttnn
import models.demos.gemma4.tt.experts.prefill as PF
from models.demos.gemma4.tt.experts.prefill import prefill_forward
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _pcc(a, b):
    a = a.flatten().to(torch.float32) - a.flatten().to(torch.float32).mean()
    b = b.flatten().to(torch.float32) - b.flatten().to(torch.float32).mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else (torch.dot(a, b) / d).item()


def _to_host(t):
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def _time(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def _full_grid_config_builder(
    mesh,
    *,
    gate_up_in0=1,
    down_in0=1,
    gate_up_grid=None,
    down_grid=None,
):
    """Build sparse-matmul geometry that packs every sequence group on the full device grid."""
    device_grid = mesh.compute_with_storage_grid_size()
    full_grid = (int(device_grid.x), int(device_grid.y))

    def build(m, n, in0_block_w=1):
        group_size = PF.PREFILL_CHUNK_SIZE // ttnn.TILE_SIZE
        n_tiles = math.ceil(n / ttnn.TILE_SIZE)
        grid_x, grid_y = (gate_up_grid or full_grid) if n_tiles <= 8 else (down_grid or full_grid)
        num_cores = grid_x * grid_y
        max_n_blocks = max(1, num_cores // group_size)
        per_core_n = math.ceil(n_tiles / max_n_blocks)
        block_w = gate_up_in0 if n_tiles <= 8 else down_in0
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=max(ttnn.TILE_SIZE, m) // ttnn.TILE_SIZE,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    return build


def run(num_layers, canvas_length, iters, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        H = tt_model.hf_config.hidden_size
        moe = None
        for layer in tt_model.layers:
            if getattr(layer, "enable_moe_block", False):
                moe = layer.moe
                break
        experts = moe.experts
        weights, cfg = experts.weights, experts.config
        mesh_config, ccl = experts.mesh_config, experts.ccl_manager

        def mk_hidden():
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * 0.1
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        ri = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)
        ri.deallocate(True)
        expert_input = mk_hidden()

        orig_chunk = PF.PREFILL_CHUNK_SIZE
        orig_builder = PF._build_sparse_matmul_config
        baseline_host = None
        sweeps = [
            ("baseline", orig_builder, [32]),
            ("full_grid", _full_grid_config_builder(mesh), [32, 64, 128]),
        ]
        sweeps.extend(
            (
                f"full_grid_g{gate_up_in0}_d1",
                _full_grid_config_builder(mesh, gate_up_in0=gate_up_in0),
                [32],
            )
            for gate_up_in0 in (2, 4, 8, 11, 22, 44, 88)
        )
        sweeps.extend(
            (
                f"full_grid_g{gate_up_in0}_d{down_in0}",
                _full_grid_config_builder(mesh, gate_up_in0=gate_up_in0, down_in0=down_in0),
                [32],
            )
            for gate_up_in0 in (22, 44)
            for down_in0 in (2, 3, 6)
        )
        sweeps.extend(
            (
                f"compact_g6x1_d{down_grid[0]}x{down_grid[1]}",
                _full_grid_config_builder(
                    mesh,
                    gate_up_in0=44,
                    down_in0=3,
                    gate_up_grid=(6, 1),
                    down_grid=down_grid,
                ),
                [32],
            )
            for down_grid in ((11, 8), (11, 4), (11, 2), (11, 1), (8, 1))
        )
        for geometry, builder, chunks in sweeps:
            PF._build_sparse_matmul_config = builder
            for chunk in chunks:
                PF.PREFILL_CHUNK_SIZE = chunk
                try:

                    def call():
                        out = prefill_forward(
                            expert_input,
                            dense_routing,
                            weights,
                            cfg,
                            experts.prefill_sparsity,
                            mesh_config=mesh_config,
                            mesh_device=mesh,
                            ccl_manager=ccl,
                        )
                        out.deallocate(True)

                    ms = _time(call, iters, mesh)
                    out = prefill_forward(
                        expert_input,
                        dense_routing,
                        weights,
                        cfg,
                        experts.prefill_sparsity,
                        mesh_config=mesh_config,
                        mesh_device=mesh,
                        ccl_manager=ccl,
                    )
                    host = _to_host(out)
                    out.deallocate(True)
                    if geometry == "baseline" and chunk == 32:
                        baseline_host = host
                        pcc = 1.0
                        exact = True
                        max_abs = 0.0
                    else:
                        pcc = _pcc(baseline_host, host)
                        exact = torch.equal(baseline_host, host)
                        max_abs = float((baseline_host - host).abs().max())
                    logger.info(
                        f"[geometry={geometry} chunk={chunk} group_size={chunk//32}] "
                        f"ms/call={ms:.2f} PCC_vs_baseline={pcc:.5f} exact={exact} max_abs={max_abs:.6g}"
                    )
                    print(
                        f"RESULT_CHUNK geometry={geometry} chunk={chunk} ms={ms:.2f} "
                        f"pcc={pcc:.5f} exact={int(exact)} max_abs={max_abs:.6g}",
                        flush=True,
                    )
                except Exception as e:
                    logger.warning(f"geometry={geometry} chunk={chunk} FAILED: {type(e).__name__}: {str(e)[:300]}")
                    print(
                        f"RESULT_CHUNK_BLOCKED geometry={geometry} chunk={chunk} "
                        f"{type(e).__name__}: {str(e)[:200]}",
                        flush=True,
                    )
        PF.PREFILL_CHUNK_SIZE = orig_chunk
        PF._build_sparse_matmul_config = orig_builder

        expert_input.deallocate(True)
        dense_routing.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.iters, args.max_seq_len)


if __name__ == "__main__":
    main()
