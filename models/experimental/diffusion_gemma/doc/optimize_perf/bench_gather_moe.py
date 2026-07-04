# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever A prototype: TRUE-SPARSE token-gather MoE vs the dense-128 prefill path.

The denoise MoE runs the gemma4 PREFILL expert path: for each 32-token tile-group it computes
ALL 128 experts (all-ones sparsity, nnz=128), then zeros 120/128 via routing weights. Only top-8
experts/token are active, so ~16x of that compute is wasted. This prototype builds the true-sparse
alternative: GATHER each active expert's assigned tokens into a fixed-capacity buffer, run only the
active experts' gate/up/down as a BATCHED matmul (one 32-token tile per expert), then SCATTER back
weighted by the routing weights and all-reduce.

TP note: the canvas input is REPLICATED across the (1,4) TP mesh (experts are TP-sharded on the
intermediate dim, not expert-parallel). So gather/scatter over the TOKEN dim is LOCAL per device;
only the down-projection needs the existing all-reduce. No cross-device token dispatch.

Measures device wall-clock (no profiler) in stages so GO/NO-GO is unambiguous:
  0) dense-128 prefill baseline (the wall: ~137 ms/layer).
  1) expert compute ONLY — batched matmul gate/up/geglu/down on [E, C, H] (no gather/scatter).
     This is the theoretical floor of the approach. If already >= dense, the batched matmul overhead
     kills it -> NO-GO before we even build gather/scatter.
  2) FULL path — embedding-gather + batched experts + combine-matmul + all-reduce. If ~= dense ->
     gather/scatter washout -> NO-GO. If << dense -> GO.
  3) PCC(full, dense) at the chosen capacity + dropped-pair diagnostics.

Host-built dispatch indices are fine for a GO/NO-GO timing prototype (the real impl builds them
on-device from the router topk). We are measuring whether the DEVICE compute (gather + batched
matmul + combine) beats the dense path.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.prefill import prefill_forward
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TILE = 32


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


def build_dispatch(routing_host, num_experts, capacity):
    """Host-build the token-gather assignment from dense routing [S, E] (8 non-zero/token).

    Returns:
        gather_idx  [E*C] int64  token id for each expert-capacity slot (pad -> token 0)
        slot_weight [E, C] f32    routing weight for each slot (pad -> 0)
        n_dropped   int           number of (token, expert) pairs dropped by capacity
        n_pairs     int           total active (token, expert) pairs
    """
    S, E = routing_host.shape
    gather_idx = torch.zeros(E * capacity, dtype=torch.int64)
    slot_weight = torch.zeros(E, capacity, dtype=torch.float32)
    n_dropped = 0
    n_pairs = 0
    for e in range(E):
        toks = torch.nonzero(routing_host[:, e] > 0, as_tuple=False).flatten()
        n_pairs += len(toks)
        if len(toks) > capacity:
            n_dropped += len(toks) - capacity
            toks = toks[:capacity]
        for c, t in enumerate(toks.tolist()):
            gather_idx[e * capacity + c] = t
            slot_weight[e, c] = routing_host[t, e].item()
    return gather_idx, slot_weight, n_dropped, n_pairs


def batched_expert_compute(gathered, weights, num_experts, capacity, hidden_size, ckcfg):
    """Batched matmul over active experts on [E, C, H] gathered tokens.

    gathered: [1, E, C, H] tiled bf16 (each expert's capacity tokens)
    Returns: [E, C, H] partial (TP-sharded down, pre all-reduce)
    """
    # weights are [1, E, H, I] (gate/up, col-parallel) and [1, E, I, H] (down, row-parallel)
    gate = ttnn.matmul(
        gathered, weights.gate_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
    )  # [1, E, C, I]
    up = ttnn.matmul(
        gathered, weights.up_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
    )  # [1, E, C, I]
    down_input = apply_geglu(gate, up)  # [1, E, C, I]
    gate.deallocate(True)
    up.deallocate(True)
    down = ttnn.matmul(
        down_input, weights.down_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
    )  # [1, E, C, H] partial
    down_input.deallocate(True)
    return down


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
        I = weights.intermediate_size_per_device
        C = capacity
        logger.info(f"num_experts={E} top_k={cfg.top_k} inter/dev={I} capacity={C} H={H} canvas={canvas_length}")

        ckcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        def mk_hidden():
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * 0.1
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

        # real dense routing (8 non-zero per token)
        ri, _ = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)  # [1,1,S,E]
        ri.deallocate(True)
        routing_host = _to_host(dense_routing).reshape(canvas_length, E)  # [S, E]
        nz = (routing_host.abs() > 0).sum(-1)
        logger.info(f"routing non-zero/token: min={int(nz.min())} max={int(nz.max())}")

        expert_input, x_host = mk_hidden()  # [1,1,S,H] replicated

        # =========================================================
        # Stage 0: dense-128 prefill baseline
        # =========================================================
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
        logger.info(f"[Stage0 dense-128] ms/call = {dense_ms:.2f}")
        print(f"RESULT_DENSE ms={dense_ms:.2f}", flush=True)

        # Build host dispatch
        gather_idx, slot_weight, n_dropped, n_pairs = build_dispatch(routing_host, E, C)
        logger.info(f"dispatch: pairs={n_pairs} dropped={n_dropped} ({100.0*n_dropped/max(n_pairs,1):.2f}%)")

        # gather indices on device: [1, E*C] uint32 ROW_MAJOR
        gather_idx_dev = ttnn.from_torch(
            gather_idx.to(torch.int32).reshape(1, E * C),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # embedding table: [S, H] ROW_MAJOR from the canvas hidden
        x_2d = ttnn.from_torch(
            x_host.reshape(canvas_length, H),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # slot weights: [1, E, C, 1] tiled bf16
        slot_w_dev = ttnn.from_torch(
            slot_weight.reshape(1, E, C, 1),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # combine matrix: [S, E*C], C_mat[t, s] = slot_weight[s] if gather_idx[s]==t else 0
        combine_mat = torch.zeros(canvas_length, E * C, dtype=torch.float32)
        sw_flat = slot_weight.reshape(E * C)
        for s in range(E * C):
            if sw_flat[s] != 0:
                combine_mat[gather_idx[s], s] = sw_flat[s]
        combine_mat_dev = ttnn.from_torch(
            combine_mat.reshape(1, 1, canvas_length, E * C),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def do_gather():
            emb = ttnn.embedding(gather_idx_dev, x_2d, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, E*C, H]
            emb = ttnn.reshape(emb, (1, E, C, H))
            emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
            return emb

        # =========================================================
        # Stage 1: expert compute ONLY (batched matmul), no gather/scatter
        # =========================================================
        gathered_fixed = do_gather()  # reuse a fixed gathered buffer to isolate compute

        def expert_only():
            down = batched_expert_compute(gathered_fixed, weights, E, C, H, ckcfg)
            down.deallocate(True)

        expert_ms = _time(expert_only, iters, mesh)
        logger.info(f"[Stage1 batched-experts only] ms/call = {expert_ms:.2f}")
        print(
            f"RESULT_EXPERTS_ONLY ms={expert_ms:.2f} speedup_vs_dense={dense_ms/max(expert_ms,1e-6):.2f}x", flush=True
        )
        gathered_fixed.deallocate(True)

        # =========================================================
        # Stage 2: FULL path — gather + experts + combine + all-reduce
        # =========================================================
        def full_call():
            gathered = do_gather()  # [1, E, C, H]
            down = batched_expert_compute(gathered, weights, E, C, H, ckcfg)  # [1, E, C, H] partial
            gathered.deallocate(True)
            # combine: reshape down -> [1,1,E*C,H], matmul combine_mat [1,1,S,E*C] @ down -> [1,1,S,H]
            down_flat = ttnn.reshape(down, (1, 1, E * C, H))
            out = ttnn.matmul(
                combine_mat_dev, down_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
            )
            down.deallocate(True)
            down_flat.deallocate(True)
            if mesh_config is not None and mesh_config.tp > 1:
                out = ccl_allreduce(out, mesh_config, ccl)
            out.deallocate(True)

        full_ms = _time(full_call, iters, mesh)
        logger.info(f"[Stage2 FULL gather+experts+combine] ms/call = {full_ms:.2f}")
        print(f"RESULT_FULL ms={full_ms:.2f} speedup_vs_dense={dense_ms/max(full_ms,1e-6):.2f}x", flush=True)

        # =========================================================
        # Stage 3: PCC vs dense
        # =========================================================
        gathered = do_gather()
        down = batched_expert_compute(gathered, weights, E, C, H, ckcfg)
        gathered.deallocate(True)
        down_flat = ttnn.reshape(down, (1, 1, E * C, H))
        out = ttnn.matmul(
            combine_mat_dev, down_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
        )
        down.deallocate(True)
        down_flat.deallocate(True)
        if mesh_config is not None and mesh_config.tp > 1:
            out = ccl_allreduce(out, mesh_config, ccl)
        full_host = _to_host(out)
        out.deallocate(True)
        pcc = _pcc(dense_host, full_host)
        logger.info(f"[Stage3 PCC] full_vs_dense = {pcc:.5f}")
        print(
            f"RESULT_PCC pcc={pcc:.5f} capacity={C} dropped_pct={100.0*n_dropped/max(n_pairs,1):.2f} "
            f"dense_ms={dense_ms:.2f} experts_only_ms={expert_ms:.2f} full_ms={full_ms:.2f}",
            flush=True,
        )

        for t in (gather_idx_dev, x_2d, slot_w_dev, combine_mat_dev, expert_input, dense_routing):
            t.deallocate(True)
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
