# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the batch-32 decode MoE cost the expert matmuls, or the dense-over-64
post-matmul chain?

``OptimizedDecoder._moe_decode_union`` (the path every ``max_batch>1`` decode
takes, and therefore the *only* path vLLM serving takes, because the adapter
always builds 32 physical rows) runs ``slice``/``silu-mul``/``mul``/``sum``
over the full ``E=64`` expert axis regardless of how few experts the union
actually selected. ``_moe_decode_indexed`` (``max_batch==1`` only) instead uses
``sparse_matmul``'s INDEXED/GATHER mode, whose output group axis is compact
(``num_active``), so the same chain runs over ``k=4``.

This probe times both chains on one real MoE layer at B=32, sweeping the number
of active experts, so the union-vs-compact difference is measured rather than
assumed. Traced replay, warmed, no profiler.

    python models/autoports/zai_org_glm_4_7_flash/doc/optimized_vllm/probe_scripts/moe_union_vs_compact.py
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
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "moe_union_vs_compact.json"

B = 32
MAX_CONTEXT = 1024
CHUNK = 1024
ITERS = 30
WARM = 5


def _time_traced(dev, build_fn, label):
    """Capture `build_fn` once and time warmed replays."""
    out = build_fn()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = build_fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(WARM):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(out)
    res = round(statistics.median(samples), 4)
    print(f"  {label}: {res} ms/layer", flush=True)
    return res


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=200_000_000)
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
        inter = dec.moe_inter
        hidden = dec.hidden
        print(f"E={E} inter={inter} hidden={hidden} top_k={dec.top_k}", flush=True)

        x = ttnn.from_torch(
            torch.randn(1, 1, B, hidden) * 0.02,
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        def union_chain(active: int):
            sp_t = torch.zeros(1, 1, 1, E)
            sp_t[0, 0, 0, :active] = 1.0
            sp = ttnn.from_torch(sp_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            rw = ttnn.from_torch(
                torch.rand(1, E, B, 1),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            def build():
                gu = ttnn.sparse_matmul(
                    x,
                    dec.experts_gate_up,
                    sparsity=sp,
                    nnz=None,
                    program_config=dec.sparse_gu_pc_union,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=dec.ck_expert,
                    dtype=ttnn.bfloat16,
                )
                gu = ttnn.reshape(gu, (1, E, B, 2 * inter))
                gate = ttnn.slice(gu, [0, 0, 0, 0], [1, E, B, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
                up = ttnn.slice(gu, [0, 0, 0, inter], [1, E, B, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(gu)
                h = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
                down = ttnn.sparse_matmul(
                    h,
                    dec.experts_down,
                    sparsity=sp,
                    nnz=None,
                    is_input_a_sparse=True,
                    program_config=dec.sparse_dn_pc_union,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=dec.ck_expert,
                    dtype=ttnn.bfloat16,
                )
                ttnn.deallocate(h)
                routed = ttnn.sum(down, dim=1, keepdim=True)
                ttnn.deallocate(down)
                return routed

            return build, (sp, rw)

        def compact_chain(kc: int):
            idx = ttnn.from_torch(
                torch.arange(kc, dtype=torch.int32).reshape(1, kc).to(torch.int32),
                device=dev,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ones = ttnn.from_torch(
                torch.ones(1, 1, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            rw = ttnn.from_torch(
                torch.rand(1, kc, B, 1),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            def build():
                gu = ttnn.sparse_matmul(
                    x,
                    dec.experts_gate_up,
                    sparsity=ones,
                    indices=idx,
                    is_input_b_sparse=True,
                    program_config=dec.sparse_gu_pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=dec.ck_expert,
                    dtype=ttnn.bfloat16,
                )
                gu = ttnn.reshape(gu, (1, kc, B, 2 * inter))
                gate = ttnn.slice(gu, [0, 0, 0, 0], [1, kc, B, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
                up = ttnn.slice(gu, [0, 0, 0, inter], [1, kc, B, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(gu)
                h = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
                down = ttnn.sparse_matmul(
                    h,
                    dec.experts_down,
                    sparsity=ones,
                    indices=idx,
                    is_input_a_sparse=True,
                    is_input_b_sparse=True,
                    program_config=dec.sparse_dn_pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=dec.ck_expert,
                    dtype=ttnn.bfloat16,
                )
                ttnn.deallocate(h)
                routed = ttnn.sum(down, dim=1, keepdim=True)
                ttnn.deallocate(down)
                return routed

            return build, (idx, ones, rw)

        results = {"union": {}, "compact": {}}
        keep = []
        for active in (4, 8, 16, 32, 64):
            build, held = union_chain(active)
            keep.extend(held)
            results["union"][str(active)] = _time_traced(dev, build, f"union active={active}")
        for kc in (4, 8, 16, 32, 64):
            build, held = compact_chain(kc)
            keep.extend(held)
            results["compact"][str(kc)] = _time_traced(dev, build, f"compact kc={kc}")

        payload = {
            "purpose": (
                "Per-MoE-layer traced decode cost at B=32: union (dense E=64 post-matmul chain, the path "
                "every max_batch>1 decode takes) vs indexed/compact (post-matmul chain over num_active only). "
                "Isolates how much of the served 45 ms/token is the dense-over-64 chain rather than expert "
                "matmul work."
            ),
            "config": {
                "batch_rows": B,
                "n_experts": E,
                "moe_inter": inter,
                "hidden": hidden,
                "top_k": dec.top_k,
                "expert_dtype": "bfloat4_b (deployment policy)",
                "iters": ITERS,
                "measure": "median ms per traced replay of one MoE layer's expert chain",
            },
            "ms_per_layer": results,
            "moe_layers_in_model": 46,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"WROTE {OUT}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
