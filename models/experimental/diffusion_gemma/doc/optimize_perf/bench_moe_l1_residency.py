# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-08 L1-residency pass — measure the MoE token-gather L1 levers (HIGH-1 gather, HIGH-2 down).

Single mesh-open session (the QB2 box is SHARED/contended, so one window measures everything):
  1. Per-component device-time decomposition of the tuned sparse MoE (dispatch/gather/experts/combine)
     for the DRAM baseline vs each L1 mode — shows WHICH matmul's time actually moves.
  2. Full ``sparse_experts_forward`` device time per DG_MOE_L1 mode (off/down/gather/both).
  3. Correctness gates per mode:
       - PCC vs the OFF (DRAM) output  -> must be ~1.0 (proves placement-only, same math).
       - PCC vs the dense moe.experts  -> must match OFF-vs-dense (the true-sparse MoE PCC ~0.9997).
  All timing is pipelined device compute (async dispatch + one final ``synchronize_device``), the same
  ``_time`` harness as bench_moe_decomp.py, so numbers are comparable across modes.

Run (device free window):
  DG_SPARSE_MOE_TUNED=1 DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/bench_moe_l1_residency.py --iters 30

Markers: RESULT_L1_FULL mode=.. ms=.. pcc_vs_off=.. pcc_vs_dense=..
         RESULT_L1_DECOMP mode=.. comp=.. ms=..
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import sparse_moe as SM

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TILE = 32
MODES = ["off", "down", "gather", "both"]


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (torch.dot(a, b) / denom).item()


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
        os.environ["DG_SPARSE_MOE_TUNED"] = "1"
        tuned = SM.build_tuned_configs(mesh, E, C, H, I, S)
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
        expert_input = mk_hidden()

        # ---- dense reference (the true-sparse MoE PCC target) ----
        dense_out = moe.experts(expert_input, dense_routing)
        dense_host = _to_host(dense_out)
        dense_out.deallocate(True)

        # ---- full forward + correctness per mode ----
        off_host = None
        for mode in MODES:
            os.environ["DG_MOE_L1"] = mode

            def full():
                o = SM.sparse_experts_forward(experts, expert_input, dense_routing, capacity=C)
                o.deallocate(True)

            try:
                ms = _time(full, iters, mesh)
            except Exception as e:
                print(f"RESULT_L1_FULL mode={mode} ERROR {type(e).__name__}: {str(e)[:220]}", flush=True)
                logger.warning(f"mode={mode} full failed: {type(e).__name__}: {str(e)[:400]}")
                continue
            out = SM.sparse_experts_forward(experts, expert_input, dense_routing, capacity=C)
            host = _to_host(out)
            out.deallocate(True)
            if mode == "off":
                off_host = host
            pcc_off = _pcc(off_host, host) if off_host is not None else 1.0
            pcc_dense = _pcc(dense_host, host)
            print(
                f"RESULT_L1_FULL mode={mode} ms={ms:.3f} pcc_vs_off={pcc_off:.6f} pcc_vs_dense={pcc_dense:.5f}",
                flush=True,
            )

        # ---- per-component decomposition: DRAM baseline vs each L1 mode ----
        disp, comb = SM.build_capacity_dispatch(dense_routing, E, C, cfg.top_k)
        disp_t = ttnn.transpose(disp, 2, 3)

        for mode in MODES:
            l1_gather = mode in ("gather", "both", "all")
            l1_down = mode in ("down", "both", "all")

            def gather():
                out = ttnn.matmul(
                    disp_t,
                    expert_input,
                    memory_config=SM._l1_or_dram(l1_gather),
                    compute_kernel_config=ckcfg,
                    program_config=tuned["gather"],
                )
                out.deallocate(True)

            g_ms = _time(gather, iters, mesh)

            dispatched = ttnn.matmul(
                disp_t,
                expert_input,
                memory_config=SM._l1_or_dram(l1_gather),
                compute_kernel_config=ckcfg,
                program_config=tuned["gather"],
            )
            gathered = ttnn.reshape(dispatched, (1, E, C, H))

            def batched():
                out = SM._batched_experts(gathered, weights, ckcfg, program_configs=tuned, l1_down=l1_down)
                out.deallocate(True)

            b_ms = _time(batched, iters, mesh)

            down = SM._batched_experts(gathered, weights, ckcfg, program_configs=tuned, l1_down=l1_down)
            down_flat = ttnn.reshape(down, (1, 1, EC, H))

            def combine():
                out = ttnn.matmul(
                    comb,
                    down_flat,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckcfg,
                    program_config=tuned["combine"],
                )
                out.deallocate(True)

            c_ms = _time(combine, iters, mesh)

            print(f"RESULT_L1_DECOMP mode={mode} comp=gather_matmul ms={g_ms:.3f}", flush=True)
            print(f"RESULT_L1_DECOMP mode={mode} comp=batched_experts ms={b_ms:.3f}", flush=True)
            print(f"RESULT_L1_DECOMP mode={mode} comp=combine_matmul ms={c_ms:.3f}", flush=True)
            for t in (dispatched, gathered, down, down_flat):
                try:
                    t.deallocate(True)
                except Exception:
                    pass

        for t in (disp, comb, disp_t):
            try:
                t.deallocate(True)
            except Exception:
                pass
        expert_input.deallocate(True)
        dense_routing.deallocate(True)
        os.environ["DG_MOE_L1"] = "off"
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--capacity", type=int, default=32)
    args = ap.parse_args()
    run(args.iters, args.capacity)


if __name__ == "__main__":
    main()
