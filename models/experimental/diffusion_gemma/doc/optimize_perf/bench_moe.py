# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Clean (non-profiled) microbench of the denoise MoE forward on the real 256-token canvas.

Answers the flagship lever-3/4 questions decisively:
  1. What dtype are the expert weights ACTUALLY loaded at (bf16 vs bfp8)?
  2. What is the clean wall-clock MoE cost per layer for the 256-token canvas?
  3. What fraction of the full denoise layer forward is the MoE (vs attention + norms + shared_mlp)?

Builds a reduced-layer real-checkpoint model, grabs the first MoE-enabled layer, and times
`_denoise_moe_forward` and the full `_denoise_layer_forward` warmed over N iters.
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

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


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

        # find first MoE-enabled layer
        moe_layer_idx = None
        for i, layer in enumerate(tt_model.layers):
            if getattr(layer, "enable_moe_block", False):
                moe_layer_idx = i
                break
        logger.info(f"first MoE layer idx={moe_layer_idx} (num_layers={num_layers})")
        if moe_layer_idx is None:
            print("NO_MOE_LAYER", flush=True)
            return
        layer = tt_model.layers[moe_layer_idx]
        moe = layer.moe

        # report actual expert weight dtypes
        gw = moe.experts.weights
        logger.info(
            f"EXPERT DTYPES gate={gw.gate_proj.dtype} up={gw.up_proj.dtype} down={gw.down_proj.dtype} "
            f"inter_per_dev={gw.intermediate_size_per_device}"
        )
        print(
            f"RESULT_MOE_DTYPE gate={gw.gate_proj.dtype} up={gw.up_proj.dtype} down={gw.down_proj.dtype}",
            flush=True,
        )

        def mk_hidden():
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * 0.1
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # ---- MoE-only forward timing ----
        def moe_only():
            router_input = mk_hidden()
            expert_input = mk_hidden()
            out = DF._denoise_moe_forward(moe, router_input, expert_input)
            router_input.deallocate(True)
            expert_input.deallocate(True)
            out.deallocate(True)

        moe_ms = _time(moe_only, iters, mesh)
        logger.info(f"[MoE-only] ms/call (256 tokens) = {moe_ms:.2f}")

        # ---- router-only forward timing (isolate router from experts) ----
        def router_only():
            ri = mk_hidden()
            dr = DF._denoise_router_forward(moe.router, ri)
            ri.deallocate(True)
            dr.deallocate(True)

        router_ms = _time(router_only, iters, mesh)
        logger.info(f"[router-only] ms/call = {router_ms:.2f}")

        # ---- experts-only (given routing) ----
        ri0 = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri0)
        ri0.deallocate(True)

        def experts_only():
            ei = mk_hidden()
            out = moe.experts(ei, dense_routing)
            ei.deallocate(True)
            out.deallocate(True)

        experts_ms = _time(experts_only, iters, mesh)
        logger.info(f"[experts-only] ms/call (256 tokens, dense-128) = {experts_ms:.2f}")

        # ---- shared_mlp-only ----
        def shared_only():
            ni = mk_hidden()
            out = layer.shared_mlp(ni)
            ni.deallocate(True)
            out.deallocate(True)

        shared_ms = _time(shared_only, iters, mesh)
        logger.info(f"[shared_mlp-only] ms/call = {shared_ms:.2f}")

        print(
            f"RESULT_MOE_BENCH num_layers={num_layers} canvas={canvas_length} "
            f"moe_ms={moe_ms:.2f} router_ms={router_ms:.2f} experts_ms={experts_ms:.2f} shared_mlp_ms={shared_ms:.2f}",
            flush=True,
        )
        dense_routing.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.iters, args.max_seq_len)


if __name__ == "__main__":
    main()
