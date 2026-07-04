# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever-4: denoise MoE sparse_matmul intermediates DRAM vs L1.

The gemma4 PREFILL expert path (used by denoise for the 256 canvas) places the gate/up/down
sparse_matmul outputs in DRAM (`ttnn.DRAM_MEMORY_CONFIG`). The optimize skill flags explicit
DRAM on decode intermediates as a perf smell; the expensive expert-major transpose then operates
on DRAM tensors. Per 32-token chunk each intermediate is ~1.5 MB, which may fit L1. This benches an
in-repo copy of the prefill chunk with L1 memory configs vs the DRAM baseline (bit-equivalent),
reporting clean device wall-clock + PCC.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.prefill import prefill_forward
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TILE_SIZE = 32


def _process_chunk_l1(hidden_states, routing_weights, weights, config, prefill_sparsity, memcfg):
    """Copy of gemma4 _process_prefill_chunk with a configurable memory_config for the
    sparse_matmul intermediates (DRAM baseline vs L1)."""
    chunk_len = hidden_states.shape[2]
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    group_size = chunk_len // TILE_SIZE
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
    sparsity = ttnn.repeat(prefill_sparsity, (1, 1, group_size, 1))
    nnz = num_experts * group_size
    output_tile = ttnn.Tile([32, 32])
    intermediate_size = weights.intermediate_size_per_device
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, intermediate_size)
    down_config = _build_sparse_matmul_config(TILE_SIZE, hidden_size)

    gate = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=memcfg,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    sm_intermediate = gate.shape[-1]
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (1, num_experts, chunk_len, sm_intermediate))
    up = ttnn.sparse_matmul(
        hidden_grouped,
        weights.up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=memcfg,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    hidden_grouped.deallocate(True)
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (1, num_experts, chunk_len, sm_intermediate))
    down_input = apply_geglu(gate, up)
    down_input = ttnn.reshape(down_input, (1, num_experts, chunk_len, sm_intermediate))
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=prefill_sparsity,
        nnz=num_experts,
        memory_config=memcfg,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    next_states = ttnn.reshape(down, (1, num_experts, chunk_len, hidden_size))
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
    next_states = ttnn.mul(next_states, routing_permuted)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(next_states, (1, 1, chunk_len, hidden_size))
    return next_states


def prefill_forward_l1(
    hidden_states,
    routing_weights,
    weights,
    config,
    prefill_sparsity,
    mesh_config,
    mesh_device,
    ccl_manager,
    memcfg,
    chunk=32,
):
    seq_len = hidden_states.shape[2]
    h_chunks = ttnn.split(hidden_states, chunk, dim=2) if seq_len > chunk else [hidden_states]
    r_chunks = ttnn.split(routing_weights, chunk, dim=2) if seq_len > chunk else [routing_weights]
    acc = None
    for hc, rc in zip(h_chunks, r_chunks):
        res = _process_chunk_l1(hc, rc, weights, config, prefill_sparsity, memcfg)
        if acc is None:
            acc = res
        else:
            cat = ttnn.concat([acc, res], dim=2)
            acc.deallocate(True)
            res.deallocate(True)
            acc = cat
    if mesh_config is not None and mesh_config.tp > 1:
        acc = ccl_allreduce(acc, mesh_config, ccl_manager)
    return acc


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
        moe = next(layer.moe for layer in tt_model.layers if getattr(layer, "enable_moe_block", False))
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

        # baseline (shared gemma4 prefill, DRAM)
        def base_call():
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

        base_ms = _time(base_call, iters, mesh)
        base_out = prefill_forward(
            expert_input,
            dense_routing,
            weights,
            cfg,
            experts.prefill_sparsity,
            mesh_config=mesh_config,
            mesh_device=mesh,
            ccl_manager=ccl,
        )
        base_host = _to_host(base_out)
        base_out.deallocate(True)
        logger.info(f"[baseline DRAM] ms={base_ms:.2f}")
        print(f"RESULT_MOEMEM variant=dram_baseline ms={base_ms:.2f} pcc=1.0", flush=True)

        for name, memcfg in [("dram_copy", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)]:
            try:

                def call():
                    out = prefill_forward_l1(
                        expert_input,
                        dense_routing,
                        weights,
                        cfg,
                        experts.prefill_sparsity,
                        mesh_config,
                        mesh,
                        ccl,
                        memcfg,
                    )
                    out.deallocate(True)

                ms = _time(call, iters, mesh)
                out = prefill_forward_l1(
                    expert_input, dense_routing, weights, cfg, experts.prefill_sparsity, mesh_config, mesh, ccl, memcfg
                )
                host = _to_host(out)
                out.deallocate(True)
                pcc = _pcc(base_host, host)
                logger.info(f"[{name}] ms={ms:.2f} PCC={pcc:.5f}")
                print(
                    f"RESULT_MOEMEM variant={name} ms={ms:.2f} pcc={pcc:.5f} speedup={base_ms/max(ms,1e-6):.2f}x",
                    flush=True,
                )
            except Exception as e:
                logger.warning(f"{name} FAILED: {type(e).__name__}: {str(e)[:300]}")
                print(f"RESULT_MOEMEM_BLOCKED variant={name} {type(e).__name__}: {str(e)[:200]}", flush=True)

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
