# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever-3 prototype: true per-token top-8 sparse MoE vs dense-128 prefill, on the 256 canvas.

The denoise MoE uses the gemma4 PREFILL expert path: all-ones sparsity, nnz=num_experts (compute
ALL 128 experts for every 32-token tile-group, then zero 120/128 via routing weights). The DECODE
expert path instead passes the real dense routing as the sparsity with nnz=top_k=8, computing only
the 8 active experts per token. If the decode path generalizes to a 256-token canvas (each token has
exactly 8 non-zero → the nnz=8 invariant holds per-token), it is TRUE per-token sparse (16x fewer
expert products) and must be numerically equal to the dense path (dense = sparse + explicit zeros).

This prototype:
  A) times the current dense-128 prefill_forward (baseline);
  B) tries decode_forward (per-token nnz=8) on the SAME 256-token canvas + routing;
  C) checks PCC(A_output, B_output).

Measures device wall-clock (no profiler). If B errors or OOMs, the error is the finding.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.experts.decode import decode_forward
from models.demos.gemma4.tt.experts.prefill import prefill_forward
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF


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


def _time(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def _to_host(t):
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


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
        weights = experts.weights
        cfg = experts.config
        mesh_config = experts.mesh_config
        ccl = experts.ccl_manager
        logger.info(f"num_experts={cfg.num_experts} top_k={cfg.top_k} inter/dev={weights.intermediate_size_per_device}")

        def mk_hidden():
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * 0.1
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # real dense routing (8 non-zero per token) from the router
        ri = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)  # [1,1,S,E], 8 non-zero/token
        ri.deallocate(True)
        # sanity: count non-zero per token on host
        r_host = _to_host(dense_routing)  # [1,1,S,E]
        nz = (r_host.abs() > 0).sum(-1)  # [1,1,S]
        logger.info(f"routing non-zero/token: min={int(nz.min())} max={int(nz.max())} (expect top_k={cfg.top_k})")

        expert_input = mk_hidden()

        # ---------- A) dense-128 prefill baseline ----------
        def dense_call():
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

        dense_ms = _time(dense_call, iters, mesh)
        dense_out = prefill_forward(
            expert_input,
            dense_routing,
            weights,
            cfg,
            experts.prefill_sparsity,
            mesh_config=mesh_config,
            mesh_device=mesh,
            ccl_manager=ccl,
        )
        dense_host = _to_host(dense_out)
        dense_out.deallocate(True)
        logger.info(f"[A dense-128 prefill] ms/call = {dense_ms:.2f}")
        print(f"RESULT_DENSE ms={dense_ms:.2f}", flush=True)

        # ---------- B) per-token nnz=8 decode path on 256 tokens ----------
        try:

            def sparse_call():
                out = decode_forward(
                    expert_input,
                    dense_routing,
                    weights,
                    cfg,
                    mesh_config=mesh_config,
                    mesh_device=mesh,
                    ccl_manager=ccl,
                )
                out.deallocate(True)

            sparse_ms = _time(sparse_call, iters, mesh)
            sparse_out = decode_forward(
                expert_input,
                dense_routing,
                weights,
                cfg,
                mesh_config=mesh_config,
                mesh_device=mesh,
                ccl_manager=ccl,
            )
            sparse_host = _to_host(sparse_out)
            sparse_out.deallocate(True)
            pcc = _pcc(dense_host, sparse_host)
            logger.info(f"[B sparse nnz=8 decode-path on 256] ms/call = {sparse_ms:.2f}  PCC_vs_dense={pcc:.5f}")
            print(
                f"RESULT_SPARSE ms={sparse_ms:.2f} pcc={pcc:.5f} speedup={dense_ms/max(sparse_ms,1e-6):.2f}x",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"sparse decode-path on 256 FAILED: {type(e).__name__}: {str(e)[:400]}")
            print(f"RESULT_SPARSE_BLOCKED {type(e).__name__}: {str(e)[:250]}", flush=True)

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
