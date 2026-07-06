# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decompose the tuned sparse-MoE forward into its device-time components.

After OPT-004 the 5 matmuls are geometry-tuned (verify_opt004_fullmoe: full MoE 2.89 ms,
matmuls ~1 ms). This bench times each COMPONENT of ``sparse_experts_forward`` separately
(dispatch build, transpose, gather matmul, batched experts, combine matmul, all-reduce,
reshapes) so we can see what actually dominates the tuned MoE and where the remaining
in-repo headroom is. All timing is pipelined device compute (async dispatch + one final
sync), the same ``_time`` harness as the OPT-004 bench, so numbers are comparable.

Run on QB2 (device free + warm page cache):
    DG_SPARSE_MOE_TUNED=1 DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
      python models/experimental/diffusion_gemma/doc/optimize_perf/bench_moe_decomp.py

Markers:
  RESULT_DECOMP comp=.. ms=.. pct=..
  RESULT_DECOMP_FULL full_ms=..
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import sparse_moe as SM

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TILE = 32


def _time(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def run(iters, capacity):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=512, num_layers=2, create_kv_cache=True
        )
        tt_model = mi.tt_model
        H = tt_model.hf_config.hidden_size
        moe = None
        for layer in tt_model.layers:
            if getattr(layer, "enable_moe_block", False):
                moe = layer.moe
                break
        experts = moe.experts
        weights = experts.weights
        cfg = experts.config
        mesh_config = experts.mesh_config
        ccl = experts.ccl_manager
        E = cfg.num_experts
        I = weights.intermediate_size_per_device
        C = capacity
        S = 256
        EC = E * C
        ckcfg = SM.default_sparse_moe_compute_kernel_config()
        tuned = SM.build_tuned_configs(mesh, E, C, H, I, S)
        os.environ["DG_SPARSE_MOE_TUNED"] = "1"
        logger.info(f"E={E} H={H} I/dev={I} S={S} C={C} EC={EC} tuned={SM.tuned_configs_enabled()}")

        def mk_hidden(scale=0.1):
            host = torch.randn(1, 1, S, H, dtype=torch.float32) * scale
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
        hidden = mk_hidden()

        comps = {}

        # 1. dispatch build
        def build():
            disp, comb = SM.build_capacity_dispatch(dense_routing, E, C, cfg.top_k)
            disp.deallocate(True)
            comb.deallocate(True)

        comps["dispatch_build"] = _time(build, iters, mesh)

        # persistent disp/comb for the rest
        disp, comb = SM.build_capacity_dispatch(dense_routing, E, C, cfg.top_k)

        # 2. transpose disp
        def do_transpose():
            t = ttnn.transpose(disp, 2, 3)
            t.deallocate(True)

        comps["transpose_disp"] = _time(do_transpose, iters, mesh)

        disp_t = ttnn.transpose(disp, 2, 3)

        # 3. gather matmul
        def gather():
            out = ttnn.matmul(
                disp_t,
                hidden,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ckcfg,
                program_config=tuned["gather"],
            )
            out.deallocate(True)

        comps["gather_matmul"] = _time(gather, iters, mesh)

        dispatched = ttnn.matmul(
            disp_t,
            hidden,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ckcfg,
            program_config=tuned["gather"],
        )
        gathered = ttnn.reshape(dispatched, (1, E, C, H))

        # 4. batched experts (gate/up/geglu/down)
        def batched():
            out = SM._batched_experts(gathered, weights, ckcfg, program_configs=tuned)
            out.deallocate(True)

        comps["batched_experts"] = _time(batched, iters, mesh)

        down = SM._batched_experts(gathered, weights, ckcfg, program_configs=tuned)
        down_flat = ttnn.reshape(down, (1, 1, EC, H))

        # 5. combine matmul
        def combine():
            out = ttnn.matmul(
                comb,
                down_flat,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ckcfg,
                program_config=tuned["combine"],
            )
            out.deallocate(True)

        comps["combine_matmul"] = _time(combine, iters, mesh)

        out = ttnn.matmul(
            comb,
            down_flat,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ckcfg,
            program_config=tuned["combine"],
        )

        # 6. all-reduce
        if mesh_config is not None and mesh_config.tp > 1:

            def allreduce():
                o = ccl_allreduce(out, mesh_config, ccl)
                # ccl_allreduce may return a new tensor; don't deallocate `out` (reused)
                if o is not out:
                    o.deallocate(True)

            comps["all_reduce"] = _time(allreduce, iters, mesh)

        for t in (disp, comb, disp_t, dispatched, gathered, down, down_flat, out):
            try:
                t.deallocate(True)
            except Exception:
                pass

        # full forward
        expert_input = mk_hidden()

        def full():
            o = SM.sparse_experts_forward(experts, expert_input, dense_routing, capacity=C)
            o.deallocate(True)

        full_ms = _time(full, iters, mesh)

        total_comp = sum(comps.values())
        for name, ms in comps.items():
            print(f"RESULT_DECOMP comp={name} ms={ms:.3f} pct={100*ms/full_ms:.1f}", flush=True)
        print(f"RESULT_DECOMP comp=SUM_COMPONENTS ms={total_comp:.3f} pct={100*total_comp/full_ms:.1f}", flush=True)
        print(f"RESULT_DECOMP_FULL full_ms={full_ms:.3f}", flush=True)

        expert_input.deallocate(True)
        dense_routing.deallocate(True)
        hidden.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--capacity", type=int, default=32)
    args = ap.parse_args()
    run(args.iters, args.capacity)


if __name__ == "__main__":
    main()
