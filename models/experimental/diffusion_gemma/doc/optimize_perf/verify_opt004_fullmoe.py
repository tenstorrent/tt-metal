# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OPT-004 focused full-MoE verify: sparse_experts_forward tuned vs untuned + PCC vs dense.

The comprehensive bench_opt004_matmul_geometry.py also runs a per-candidate geometry SWEEP
(many program-config compiles) which is slow; this focused verify skips the sweep and reports
only the land-decision numbers on the real 26B layer-0 MoE (mesh (1,4), TP=4, capacity 32):
  RESULT_OPT004_MOE untuned_ms=.. tuned_ms=.. speedup=.. pcc_untuned_vs_dense=.. pcc_tuned_vs_dense=.. pcc_tuned_vs_untuned=..

Pure geometry change (same dtype/fidelity) -> pcc_tuned_vs_dense must equal pcc_untuned_vs_dense.
*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
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
from models.experimental.diffusion_gemma.tt.sparse_moe import sparse_experts_forward

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


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
    if hasattr(t.device(), "get_num_devices") and t.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return ttnn.to_torch(t)


def _time_sparse(moe, expert_input, dense_routing, capacity, iters, mesh):
    # warm
    warm = sparse_experts_forward(moe.experts, expert_input, dense_routing, capacity=capacity)
    host = _to_host(warm)
    warm.deallocate(True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = sparse_experts_forward(moe.experts, expert_input, dense_routing, capacity=capacity)
        out.deallocate(True)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3 / iters
    return ms, host


def run(num_layers, canvas_length, max_seq_len, capacity, iters, skip_dense):
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

        def mk_hidden(scale=0.1):
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * scale
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

        # dense reference (moe.experts dense-128 path) — OPTIONAL: its first-call compile is very slow,
        # and the UNTUNED sparse path is already the verified-correct baseline (PCC 0.9997 vs dense,
        # verify_sparse_moe.py). So pcc_tuned_vs_untuned on all 5 configs is the real land gate.
        dense_host = None
        if not skip_dense:
            dense_out = moe.experts(expert_input, dense_routing)
            dense_host = _to_host(dense_out)
            dense_out.deallocate(True)

        os.environ["DG_SPARSE_MOE_TUNED"] = "0"
        un_ms, un_host = _time_sparse(moe, expert_input, dense_routing, capacity, iters, mesh)
        os.environ["DG_SPARSE_MOE_TUNED"] = "1"
        tu_ms, tu_host = _time_sparse(moe, expert_input, dense_routing, capacity, iters, mesh)
        os.environ["DG_SPARSE_MOE_TUNED"] = "0"

        un_pcc = _pcc(dense_host, un_host) if dense_host is not None else float("nan")
        tu_pcc = _pcc(dense_host, tu_host) if dense_host is not None else float("nan")
        tu_un_pcc = _pcc(un_host, tu_host)
        logger.info(f"untuned {un_ms:.3f} ms  tuned {tu_ms:.3f} ms  speedup {un_ms/max(tu_ms,1e-6):.2f}x")
        print(
            f"RESULT_OPT004_MOE capacity={capacity} untuned_ms={un_ms:.3f} tuned_ms={tu_ms:.3f} "
            f"speedup={un_ms/max(tu_ms,1e-6):.2f} pcc_untuned_vs_dense={un_pcc:.5f} "
            f"pcc_tuned_vs_dense={tu_pcc:.5f} pcc_tuned_vs_untuned={tu_un_pcc:.5f}",
            flush=True,
        )
        dense_routing.deallocate(True)
        expert_input.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--capacity", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--with-dense", action="store_true", help="also compute the slow dense-128 PCC reference")
    args = ap.parse_args()
    run(
        args.num_layers, args.canvas_length, args.max_seq_len, args.capacity, args.iters, skip_dense=not args.with_dense
    )


if __name__ == "__main__":
    main()
