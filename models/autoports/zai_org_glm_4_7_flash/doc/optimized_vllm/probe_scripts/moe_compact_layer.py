# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prototype + validate a compact (indexed) batch>1 decode MoE against the
shipped union path, on one real MoE layer at B=32.

``moe_union_vs_compact.json`` showed the indexed/gather form of the expert
chain is 2-3x cheaper than the union form at every active-expert count, because
the union form's output group axis is the full ``E=64`` (skipped groups are
zero-filled) and every post-matmul op then runs over all 64 groups.

This probe builds the real thing: routing -> inactive-row mask -> union ->
top-KC union ids -> indexed sparse_matmul -> compact chain, and checks it
against the shipped union path bit-for-bit-ish (PCC) plus wall time.

    python models/autoports/zai_org_glm_4_7_flash/doc/optimized_vllm/probe_scripts/moe_compact_layer.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tests import utils  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "moe_compact_layer.json"

B = 32
MAX_CONTEXT = 1024
CHUNK = 1024
ITERS = 30
WARM = 5


def routing_of(dec, x, active_mask=None):
    """Shipped union-path routing, optionally zeroing inactive rows."""
    scores, centered_bf16 = dec._router_scores_decode(x)
    _, idx = ttnn.topk(centered_bf16, k=dec.top_k, dim=-1, sorted=True)
    T = idx.shape[2]
    src = dec.scatter_ones
    if src.shape[2] != T:
        src = ttnn.slice(dec.scatter_ones, [0, 0, 0, 0], [1, 1, T, dec.top_k])
    mask_bf16 = ttnn.scatter(ttnn.zeros_like(centered_bf16), dim=-1, index=idx, src=src)
    ttnn.deallocate(centered_bf16)
    ttnn.deallocate(idx)
    mask = ttnn.typecast(mask_bf16, ttnn.float32)
    ttnn.deallocate(mask_bf16)
    picked = ttnn.multiply(scores, mask)
    ttnn.deallocate(scores)
    ttnn.deallocate(mask)
    denom = ttnn.add(ttnn.sum(picked, dim=-1, keepdim=True), 1e-20)
    inv = ttnn.reciprocal(denom)
    ttnn.deallocate(denom)
    weights = ttnn.multiply(picked, inv)
    ttnn.deallocate(picked)
    ttnn.deallocate(inv)
    routing = ttnn.typecast(weights, ttnn.bfloat16)  # [1,1,B,E]
    ttnn.deallocate(weights)
    if active_mask is not None:
        masked = ttnn.multiply(routing, active_mask)
        ttnn.deallocate(routing)
        routing = masked
    return routing


def union_routed(dec, x, routing):
    E, inter = dec.n_experts, dec.moe_inter
    Bx = x.shape[2]
    union = ttnn.max(routing, dim=2, keepdim=True)
    sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(union)
    gu = ttnn.sparse_matmul(
        x,
        dec.experts_gate_up,
        sparsity=sparsity,
        nnz=None,
        program_config=dec.sparse_gu_pc_union,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=dec.ck_expert,
        dtype=ttnn.bfloat16,
    )
    gu = ttnn.reshape(gu, (1, E, Bx, 2 * inter))
    gate = ttnn.slice(gu, [0, 0, 0, 0], [1, E, Bx, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
    up = ttnn.slice(gu, [0, 0, 0, inter], [1, E, Bx, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(gu)
    h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1,E,B,1]
    h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(rw)
    down = ttnn.sparse_matmul(
        h,
        dec.experts_down,
        sparsity=sparsity,
        nnz=None,
        is_input_a_sparse=True,
        program_config=dec.sparse_dn_pc_union,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=dec.ck_expert,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(h)
    ttnn.deallocate(sparsity)
    routed = ttnn.sum(down, dim=1, keepdim=True)
    ttnn.deallocate(down)
    return routed


def compact_routed(dec, x, routing, kc):
    E, inter = dec.n_experts, dec.moe_inter
    Bx = x.shape[2]
    union = ttnn.max(routing, dim=2, keepdim=True)  # [1,1,1,E] bf16 TILE
    _, uidx = ttnn.topk(union, k=kc, dim=-1, sorted=True)  # [1,1,1,kc] uint16 TILE
    ttnn.deallocate(union)
    uidx_rm = ttnn.to_layout(uidx, ttnn.ROW_MAJOR_LAYOUT)
    uidx_rm = ttnn.reshape(uidx_rm, (1, kc))
    uidx_i = ttnn.typecast(uidx, ttnn.uint32)
    ttnn.deallocate(uidx)
    uidx_b = ttnn.repeat(uidx_i, ttnn.Shape([1, 1, Bx, 1]))  # [1,1,B,kc]
    ttnn.deallocate(uidx_i)
    rw = ttnn.gather(routing, dim=-1, index=uidx_b)  # [1,1,B,kc]
    ttnn.deallocate(uidx_b)
    rw = ttnn.permute(rw, (0, 3, 2, 1))  # [1,kc,B,1]

    gu = ttnn.sparse_matmul(
        x,
        dec.experts_gate_up,
        sparsity=dec.ones_e,
        indices=uidx_rm,
        is_input_b_sparse=True,
        program_config=dec.sparse_gu_pc,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=dec.ck_expert,
        dtype=ttnn.bfloat16,
    )
    gu = ttnn.reshape(gu, (1, kc, Bx, 2 * inter))
    gate = ttnn.slice(gu, [0, 0, 0, 0], [1, kc, Bx, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
    up = ttnn.slice(gu, [0, 0, 0, inter], [1, kc, Bx, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(gu)
    h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(rw)
    down = ttnn.sparse_matmul(
        h,
        dec.experts_down,
        sparsity=dec.ones_e,
        indices=uidx_rm,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=dec.sparse_dn_pc,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=dec.ck_expert,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(h)
    ttnn.deallocate(uidx_rm)
    routed = ttnn.sum(down, dim=1, keepdim=True)
    ttnn.deallocate(down)
    return routed


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def time_traced(dev, fn, label):
    out = fn()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(WARM):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    s = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        s.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(out)
    m = round(statistics.median(s), 4)
    print(f"  {label}: {m} ms", flush=True)
    return m


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=200_000_000)
    payload = {}
    try:
        cfg = utils.hf_config()
        layer_idx = utils.LAYER_KINDS["moe"]
        sd = utils.load_real_layer_state_dict(cfg, layer_idx)
        dec = OptimizedDecoder.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=dev,
            max_batch_size=B,
            max_context=MAX_CONTEXT,
            prefill_chunk_size=CHUNK,
        )
        E = dec.n_experts
        print(f"E={E} inter={dec.moe_inter} hidden={dec.hidden} top_k={dec.top_k}", flush=True)

        torch.manual_seed(0)
        x_t = torch.randn(1, 1, B, dec.hidden) * 0.02
        x = ttnn.from_torch(
            x_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # ---------- correctness: all 32 rows active ----------
        r_union = ttnn.to_torch(union_routed(dec, x, routing_of(dec, x)))
        results = {"pcc_vs_union": {}, "ms": {}}
        for kc in (16, 32, 64):
            r_c = ttnn.to_torch(compact_routed(dec, x, routing_of(dec, x), kc))
            results["pcc_vs_union"][f"all32rows_kc{kc}"] = round(pcc(r_union, r_c), 6)
            print(f"  PCC all-32-rows kc={kc}: {results['pcc_vs_union'][f'all32rows_kc{kc}']}", flush=True)

        # ---------- correctness: 1 active row (mask), kc=4 ----------
        am_t = torch.zeros(1, 1, B, 1)
        am_t[0, 0, 0, 0] = 1.0
        am = ttnn.from_torch(
            am_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        r_union_1 = ttnn.to_torch(union_routed(dec, x, routing_of(dec, x, am)))
        for kc in (4, 8):
            r_c1 = ttnn.to_torch(compact_routed(dec, x, routing_of(dec, x, am), kc))
            # only row 0 is meaningful; the masked rows are zero in both
            p = pcc(r_union_1[..., :1, :], r_c1[..., :1, :])
            results["pcc_vs_union"][f"row0_masked_kc{kc}"] = round(p, 6)
            print(f"  PCC 1-active-row (masked) kc={kc}, row 0: {p}", flush=True)
            results["pcc_vs_union"][f"masked_rows_zero_kc{kc}"] = float(r_c1[..., 1:, :].abs().max())

        # ---------- timing ----------
        results["ms"]["union_all32rows"] = time_traced(
            dev, lambda: union_routed(dec, x, routing_of(dec, x)), "union (routing+chain), 32 rows active"
        )
        results["ms"]["union_1row_masked"] = time_traced(
            dev, lambda: union_routed(dec, x, routing_of(dec, x, am)), "union masked to 1 active row"
        )
        for kc in (4, 8, 16, 32, 64):
            results["ms"][f"compact_kc{kc}_1row_masked"] = time_traced(
                dev, lambda kc=kc: compact_routed(dec, x, routing_of(dec, x, am), kc), f"compact kc={kc}, 1 active row"
            )
        for kc in (32, 64):
            results["ms"][f"compact_kc{kc}_all32rows"] = time_traced(
                dev, lambda kc=kc: compact_routed(dec, x, routing_of(dec, x), kc), f"compact kc={kc}, 32 rows active"
            )

        payload = {
            "purpose": (
                "Prototype of a compact (sparse_matmul INDEXED/GATHER) batch>1 decode MoE plus an "
                "inactive-row routing mask, validated for PCC against the shipped union path and timed "
                "against it, on one real MoE layer at B=32. Includes the full routing prologue, so these "
                "are whole-routed-expert-block numbers, not just the post-matmul chain."
            ),
            "config": {
                "batch_rows": B,
                "n_experts": E,
                "top_k": dec.top_k,
                "expert_dtype": "bfloat4_b (deployment policy)",
                "iters": ITERS,
            },
            "results": results,
            "moe_layers_in_model": 46,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"WROTE {OUT}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
