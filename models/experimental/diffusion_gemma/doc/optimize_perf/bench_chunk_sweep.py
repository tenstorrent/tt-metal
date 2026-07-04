# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever-4 quick sweep: denoise MoE prefill chunk size (transpose-reorder overhead).

The denoise MoE (gemma4 prefill_forward) splits the 256-token canvas into chunks of
PREFILL_CHUNK_SIZE=32 (group_size=1 each) and pays the expensive expert-major transpose
reorder once per chunk (8×). This sweeps PREFILL_CHUNK_SIZE ∈ {32,64,128,256} (group_size
1/2/4/8) by monkeypatching the module constant — reusing ALL existing sparse_matmul logic,
so any faster chunk is a bit-equivalent, no-semantics-change win. Measures clean device
wall-clock + PCC vs the chunk=32 baseline.
"""
from __future__ import annotations

import argparse
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
        baseline_host = None
        for chunk in [32, 64, 128, 256]:
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
                if chunk == 32:
                    baseline_host = host
                    pcc = 1.0
                else:
                    pcc = _pcc(baseline_host, host)
                logger.info(f"[chunk={chunk} group_size={chunk//32}] ms/call={ms:.2f} PCC_vs_c32={pcc:.5f}")
                print(f"RESULT_CHUNK chunk={chunk} ms={ms:.2f} pcc={pcc:.5f}", flush=True)
            except Exception as e:
                logger.warning(f"chunk={chunk} FAILED: {type(e).__name__}: {str(e)[:300]}")
                print(f"RESULT_CHUNK_BLOCKED chunk={chunk} {type(e).__name__}: {str(e)[:200]}", flush=True)
        PF.PREFILL_CHUNK_SIZE = orig_chunk

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
