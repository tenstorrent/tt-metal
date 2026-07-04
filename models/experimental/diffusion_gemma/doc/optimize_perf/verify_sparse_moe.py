# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify the in-repo true-sparse token-gather MoE (tt/sparse_moe.py) vs the dense-128 path.

Checks, on a real 26B MoE layer:
  A) sparse_experts_forward(experts, x, routing) PCC vs moe.experts(x, routing) — the MoE output.
  B) full _denoise_layer_forward PCC with DG_SPARSE_MOE off vs on — the whole layer output.
  C) real max expert load per capacity, and the PCC as capacity varies (32/48/64) — to pick a
     drop-safe capacity.

Uses real router routing (from the real router on a real normed hidden). Random-ish input, but the
routing is the model's own, so the load distribution is representative of the model's routing.
"""
from __future__ import annotations

import argparse
import os

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
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def run(num_layers, canvas_length, max_seq_len):
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
        E = moe.experts.config.num_experts

        def mk_hidden(scale=0.1):
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * scale
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # ---- real routing from the real router ----
        ri = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)  # [1,1,S,E]
        ri.deallocate(True)
        routing_host = _to_host(dense_routing).reshape(canvas_length, E)
        load = (routing_host.abs() > 0).sum(0)  # tokens per expert
        logger.info(f"expert load: min={int(load.min())} max={int(load.max())} mean={load.float().mean():.1f}")
        print(
            f"RESULT_LOAD max={int(load.max())} mean={load.float().mean():.2f} over_32={int((load>32).sum())}",
            flush=True,
        )

        expert_input = mk_hidden()

        # ---- A) MoE output PCC: dense vs sparse (module path) ----
        dense_out = moe.experts(expert_input, dense_routing)
        dense_host = _to_host(dense_out)
        dense_out.deallocate(True)
        for C in (32, 64):  # capacity must be tile-aligned (multiple of 32)
            sp = sparse_experts_forward(moe.experts, expert_input, dense_routing, capacity=C)
            sp_host = _to_host(sp)
            sp.deallocate(True)
            pcc = _pcc(dense_host, sp_host)
            dropped = int((load - C).clamp(min=0).sum())
            print(f"RESULT_MOE_PCC capacity={C} pcc={pcc:.5f} dropped_pairs={dropped}", flush=True)

        # ---- B) full denoise LAYER PCC: dense vs sparse ----
        moe_layer_idx = None
        for i, layer in enumerate(tt_model.layers):
            if getattr(layer, "enable_moe_block", False):
                moe_layer_idx = i
                break
        S = canvas_length
        prompt = mk_hidden()  # prompt-prefix source (concat path)
        attn_mask = None

        def run_layer(scale):
            hs_host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * scale

            def _mk():
                return ttnn.from_torch(
                    hs_host,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )

            def _layer(inp):
                out = DF._denoise_layer_forward(tt_model, moe_layer_idx, inp, prompt, attn_mask, q_rope_offset=S)
                h = _to_host(out)
                out.deallocate(True)
                return h

            os.environ["DG_SPARSE_MOE"] = "0"
            dh1 = _layer(_mk())
            dh2 = _layer(_mk())  # dense-vs-dense noise floor
            os.environ["DG_SPARSE_MOE"] = "1"
            os.environ["DG_SPARSE_MOE_CAPACITY"] = "32"
            sh = _layer(_mk())
            os.environ["DG_SPARSE_MOE"] = "0"
            return _pcc(dh1, dh2), _pcc(dh1, sh)

        for scale in (0.1, 1.0):
            try:
                noise_pcc, layer_pcc = run_layer(scale)
                print(
                    f"RESULT_LAYER_PCC scale={scale} dense_vs_dense={noise_pcc:.5f} dense_vs_sparse={layer_pcc:.5f}",
                    flush=True,
                )
            except Exception as e:
                logger.warning(f"layer PCC failed: {type(e).__name__}: {str(e)[:400]}")
                print(f"RESULT_LAYER_PCC_ERR scale={scale} {type(e).__name__}: {str(e)[:200]}", flush=True)

        expert_input.deallocate(True)
        dense_routing.deallocate(True)
        prompt.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.max_seq_len)


if __name__ == "__main__":
    main()
