# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever A step 2: measure the ON-DEVICE dispatch-index build for the token-gather MoE.

bench_gather_moe.py proved the token-gather compute is 12.5x cheaper than dense-128 but used
HOST-built dispatch indices. The traced denoise path can't read routing back to host, so the
dispatch must be built ON-DEVICE from the router topk. This bench builds the whole dispatch
on-device and measures whether the ~11ms/layer win survives.

On-device GShard-style capacity dispatch from dense routing [S, E] (exactly 8 non-zero/token):
  1. vals, idx = topk(routing, k=8)                     # per-token expert ids + weights
  2. mask[S,E]  = scatter(0, idx, 1)                     # 1 where token->expert
  3. excl[S,E]  = cumsum(mask, tokens) - mask            # # earlier tokens on same expert
  4. pos[S,8]   = gather(excl, idx)                      # slot within expert bucket
  5. col[S,8]   = idx*C + pos ; overflow(pos>=C)->dead   # column in E*C dispatch buffer
  6. disp[S,EC] = scatter(0, col, 1)  ; comb[S,EC] = scatter(0, col, vals)
  7. gather  : dispatched[EC,H] = disp^T @ x            # matmul
     experts : batched gate/up/geglu/down on [E,C,H]
     combine : out[S,H] = comb @ down_flat ; all-reduce

Times each phase + PCC vs the dense-128 path.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.experts.prefill import prefill_forward
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.doc.optimize_perf.bench_gather_moe import batched_expert_compute
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


def build_dispatch_ondevice(dense_routing, mesh, S, E, C, top_k):
    """Build disp[1,1,S,E*C] and comb[1,1,S,E*C] masks on-device from dense routing.

    Returns (disp, comb) both TILE bf16.
    """
    EC = E * C
    # 1. topk -> vals[1,1,S,k], idx[1,1,S,k]
    vals, idx = ttnn.topk(dense_routing, k=top_k, dim=-1)  # vals bf16, idx uint16/uint32
    idx_i = ttnn.typecast(idx, ttnn.uint32)  # scatter + gather both require UINT32/UINT16 index
    idx_f = ttnn.typecast(idx, ttnn.float32)

    ones_sk = ttnn.ones([1, 1, S, top_k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

    # 2. mask[1,1,S,E]
    zeros_se = ttnn.zeros([1, 1, S, E], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    mask = ttnn.scatter(zeros_se, dim=-1, index=idx_i, src=ones_sk)

    # 3. exclusive position along token dim (dim=2)
    cum = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)
    mask_f = ttnn.typecast(mask, ttnn.float32)
    excl = ttnn.sub(cum, mask_f)  # [1,1,S,E] f32

    # 4. per-slot position
    pos = ttnn.gather(excl, dim=-1, index=idx_i)  # [1,1,S,k] f32

    # 5. col = idx*C + pos ; overflow -> dead column EC
    col = ttnn.add(ttnn.mul(idx_f, float(C)), pos)  # [1,1,S,k] f32
    overflow = ttnn.ge(pos, float(C))  # 1.0 where pos>=C
    dead = ttnn.full([1, 1, S, top_k], float(EC), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    col = ttnn.where(overflow, dead, col)
    # combine weights: zero overflow slots
    valid = ttnn.sub(ttnn.ones_like(overflow), overflow)
    vals_masked = ttnn.mul(vals, ttnn.typecast(valid, ttnn.bfloat16))

    col_i = ttnn.typecast(col, ttnn.uint32)

    # 6. dispatch & combine masks over E*C (+1 dead col), then slice off dead
    zeros_ecd = ttnn.zeros([1, 1, S, EC + 1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    disp = ttnn.scatter(zeros_ecd, dim=-1, index=col_i, src=ones_sk)
    zeros_ecd2 = ttnn.zeros([1, 1, S, EC + 1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    comb = ttnn.scatter(zeros_ecd2, dim=-1, index=col_i, src=vals_masked)
    disp = ttnn.slice(disp, [0, 0, 0, 0], [1, 1, S, EC])
    comb = ttnn.slice(comb, [0, 0, 0, 0], [1, 1, S, EC])

    for t in (
        vals,
        idx,
        idx_i,
        idx_f,
        ones_sk,
        mask,
        cum,
        mask_f,
        excl,
        pos,
        col,
        overflow,
        dead,
        valid,
        vals_masked,
        col_i,
    ):
        t.deallocate(True)
    return disp, comb


def run(num_layers, canvas_length, iters, max_seq_len, capacity):
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
        E = cfg.num_experts
        C = capacity
        S = canvas_length
        EC = E * C
        top_k = cfg.top_k
        logger.info(f"E={E} top_k={top_k} C={C} S={S} H={H} EC={EC}")

        ckcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )

        def mk_hidden():
            host = torch.randn(1, 1, S, H, dtype=torch.float32) * 0.1
            return (
                ttnn.from_torch(
                    host,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
                host,
            )

        ri, _ = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)
        ri.deallocate(True)
        expert_input, _ = mk_hidden()

        # ---- dense baseline ----
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
        print(f"RESULT_DENSE ms={dense_ms:.2f}", flush=True)

        # ---- dispatch-build only ----
        def dispatch_only():
            disp, comb = build_dispatch_ondevice(dense_routing, mesh, S, E, C, top_k)
            disp.deallocate(True)
            comb.deallocate(True)

        disp_ms = _time(dispatch_only, iters, mesh)
        print(f"RESULT_DISPATCH_BUILD ms={disp_ms:.2f}", flush=True)

        # ---- full on-device path ----
        def full_call():
            disp, comb = build_dispatch_ondevice(dense_routing, mesh, S, E, C, top_k)
            disp_T = ttnn.transpose(disp, 2, 3)  # [1,1,EC,S]
            dispatched = ttnn.matmul(
                disp_T, expert_input, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
            )  # [1,1,EC,H]
            disp.deallocate(True)
            disp_T.deallocate(True)
            gathered = ttnn.reshape(dispatched, (1, E, C, H))
            dispatched.deallocate(True)
            down = batched_expert_compute(gathered, weights, E, C, H, ckcfg)  # [1,E,C,H] partial
            gathered.deallocate(True)
            down_flat = ttnn.reshape(down, (1, 1, EC, H))
            down.deallocate(True)
            out = ttnn.matmul(comb, down_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
            comb.deallocate(True)
            down_flat.deallocate(True)
            if mesh_config is not None and mesh_config.tp > 1:
                out = ccl_allreduce(out, mesh_config, ccl)
            out.deallocate(True)

        full_ms = _time(full_call, iters, mesh)
        print(f"RESULT_FULL_ONDEVICE ms={full_ms:.2f} speedup_vs_dense={dense_ms/max(full_ms,1e-6):.2f}x", flush=True)

        # ---- PCC ----
        disp, comb = build_dispatch_ondevice(dense_routing, mesh, S, E, C, top_k)
        disp_T = ttnn.transpose(disp, 2, 3)
        dispatched = ttnn.matmul(
            disp_T, expert_input, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
        )
        disp.deallocate(True)
        disp_T.deallocate(True)
        gathered = ttnn.reshape(dispatched, (1, E, C, H))
        dispatched.deallocate(True)
        down = batched_expert_compute(gathered, weights, E, C, H, ckcfg)
        gathered.deallocate(True)
        down_flat = ttnn.reshape(down, (1, 1, EC, H))
        down.deallocate(True)
        out = ttnn.matmul(comb, down_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
        comb.deallocate(True)
        down_flat.deallocate(True)
        if mesh_config is not None and mesh_config.tp > 1:
            out = ccl_allreduce(out, mesh_config, ccl)
        full_host = _to_host(out)
        out.deallocate(True)
        pcc = _pcc(dense_host, full_host)
        print(
            f"RESULT_PCC pcc={pcc:.5f} capacity={C} dense_ms={dense_ms:.2f} "
            f"dispatch_ms={disp_ms:.2f} full_ondevice_ms={full_ms:.2f}",
            flush=True,
        )

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
    ap.add_argument("--capacity", type=int, default=32)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.iters, args.max_seq_len, args.capacity)


if __name__ == "__main__":
    main()
