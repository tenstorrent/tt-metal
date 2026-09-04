# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does the compact decode MoE's routing prologue actually cost, and can
the compact-specific part of it be made cheaper?

Two questions the first version of this stage's report answered badly:

1. It quoted "the prologue costs ~0.17 ms/layer, so there is ~7.8 ms/token of
   headroom". Most of that prologue is the router + top-k + normalize the
   *union* path pays too; only the union-max/top-k/index/gather tail is
   compact-specific. This splits them.
2. It named the compact-specific tail as a candidate and did not try it.
   `_moe_decode_indexed` already records that `ttnn.gather` is a ~37 us
   single-core kernel at [1,1,32,64] while an `ttnn.embedding` chain measures
   ~18 us, so there is a concrete adapted alternative to measure, plus a
   one-hot matmul variant that avoids the index broadcast entirely.

One real MoE layer, B=32, deployment dtypes, traced replay, warmed.

    python .../probe_scripts/moe_prologue_ablation.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tests import utils  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "moe_prologue_ablation.json"

B = 32
KC = 4
ITERS = 40
WARM = 6


def time_traced(dev, fn, label, out=None):
    try:
        o = fn()
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        o = fn()
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
        ttnn.deallocate(o)
        m = round(statistics.median(s), 4)
        print(f"  {label}: {m} ms", flush=True)
        return m
    except Exception:  # noqa: BLE001
        err = traceback.format_exc().strip().splitlines()[-1][:300]
        print(f"  {label}: BLOCKED {err}", flush=True)
        return {"blocked": err}


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=200_000_000)
    try:
        cfg = utils.hf_config()
        idx_layer = utils.LAYER_KINDS["moe"]
        dec = OptimizedDecoder.from_state_dict(
            utils.load_real_layer_state_dict(cfg, idx_layer),
            hf_config=cfg,
            layer_idx=idx_layer,
            mesh_device=dev,
            max_batch_size=B,
            max_context=1024,
            prefill_chunk_size=1024,
        )
        E = dec.n_experts
        torch.manual_seed(0)
        x = ttnn.from_torch(
            torch.randn(1, 1, B, dec.hidden) * 0.02,
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        am_t = torch.zeros(1, 1, B, 1)
        am_t[0, 0, 0, 0] = 1.0
        am = ttnn.from_torch(
            am_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        expert_ids = ttnn.from_torch(
            torch.arange(E, dtype=torch.float32).reshape(1, 1, E, 1),
            device=dev,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        res = {}

        # -- shared: the router/top-k/normalize both paths pay --------------
        res["routing_weights_only"] = time_traced(
            dev, lambda: dec._routing_weights_decode(x, am), "routing weights only (both paths pay this)"
        )

        # -- union path's own tail: max + to_layout -------------------------
        def union_tail():
            r = dec._routing_weights_decode(x, am)
            u = ttnn.max(r, dim=2, keepdim=True)
            sp = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(u)
            ttnn.deallocate(r)
            return sp

        res["union_prologue"] = time_traced(dev, union_tail, "union prologue (routing + max + to_layout)")

        # -- shipped compact tail: max + topk + idx prep + repeat + gather ---
        def compact_gather():
            r = dec._routing_weights_decode(x, am)
            u = ttnn.max(r, dim=2, keepdim=True)
            _, uidx = ttnn.topk(u, k=KC, dim=-1, sorted=True)
            ttnn.deallocate(u)
            idx_rm = ttnn.reshape(ttnn.to_layout(uidx, ttnn.ROW_MAJOR_LAYOUT), (1, KC))
            u32 = ttnn.typecast(uidx, ttnn.uint32)
            ttnn.deallocate(uidx)
            rows = ttnn.repeat(u32, ttnn.Shape([1, 1, B, 1]))
            ttnn.deallocate(u32)
            rw = ttnn.gather(r, dim=-1, index=rows)
            ttnn.deallocate(rows)
            ttnn.deallocate(r)
            rw = ttnn.permute(rw, (0, 3, 2, 1))
            ttnn.deallocate(idx_rm)
            return rw

        res["compact_prologue_first_implementation_repeat_gather"] = time_traced(
            dev, compact_gather, "compact prologue, first implementation (repeat + gather)"
        )

        # -- candidate A: one-hot built by broadcast-eq, then a small matmul --
        def compact_onehot():
            r = dec._routing_weights_decode(x, am)
            u = ttnn.max(r, dim=2, keepdim=True)
            _, uidx = ttnn.topk(u, k=KC, dim=-1, sorted=True)
            ttnn.deallocate(u)
            idx_rm = ttnn.reshape(ttnn.to_layout(uidx, ttnn.ROW_MAJOR_LAYOUT), (1, KC))
            uf = ttnn.typecast(uidx, ttnn.float32)  # [1,1,1,kc]
            ttnn.deallocate(uidx)
            onehot = ttnn.eq(expert_ids, uf)  # [1,1,E,1] x [1,1,1,kc] -> [1,1,E,kc]
            ttnn.deallocate(uf)
            onehot = ttnn.typecast(onehot, ttnn.bfloat16)
            rw = ttnn.matmul(r, onehot, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1,1,B,kc]
            ttnn.deallocate(onehot)
            ttnn.deallocate(r)
            rw = ttnn.permute(rw, (0, 3, 2, 1))
            ttnn.deallocate(idx_rm)
            return rw

        res["compact_prologue_candidate_onehot_matmul"] = time_traced(
            dev, compact_onehot, "compact prologue, candidate A (one-hot + matmul)"
        )

        # -- candidate B: ttnn.embedding over a [E, B] routing table ---------
        def compact_embedding():
            r = dec._routing_weights_decode(x, am)
            u = ttnn.max(r, dim=2, keepdim=True)
            _, uidx = ttnn.topk(u, k=KC, dim=-1, sorted=True)
            ttnn.deallocate(u)
            u32 = ttnn.typecast(uidx, ttnn.uint32)
            ttnn.deallocate(uidx)
            idx_rm = ttnn.reshape(ttnn.to_layout(u32, ttnn.ROW_MAJOR_LAYOUT), (1, KC))
            ttnn.deallocate(u32)
            table = ttnn.transpose(r, -2, -1)  # [1,1,E,B]
            ttnn.deallocate(r)
            table = ttnn.reshape(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), (E, B))
            rw = ttnn.embedding(idx_rm, table)  # [1, kc, B]
            ttnn.deallocate(table)
            ttnn.deallocate(idx_rm)
            rw = ttnn.to_layout(ttnn.reshape(rw, (1, KC, B, 1)), ttnn.TILE_LAYOUT)
            return rw

        res["compact_prologue_candidate_embedding"] = time_traced(
            dev, compact_embedding, "compact prologue, candidate B (embedding over [E,B] table)"
        )

        # -- candidate B', the exact shape a real implementation needs: the
        # uint16 ROW_MAJOR stick sparse_matmul's indexed mode requires AND the
        # uint32 index ttnn.embedding requires, derived from it.
        def compact_embedding_real():
            r = dec._routing_weights_decode(x, am)
            u = ttnn.max(r, dim=2, keepdim=True)
            _, uidx = ttnn.topk(u, k=KC, dim=-1, sorted=True)
            ttnn.deallocate(u)
            idx_rm = ttnn.reshape(ttnn.to_layout(uidx, ttnn.ROW_MAJOR_LAYOUT), (1, KC))  # uint16, sparse_matmul
            ttnn.deallocate(uidx)
            idx_u32 = ttnn.typecast(idx_rm, ttnn.uint32)  # embedding index
            table = ttnn.transpose(r, -2, -1)  # [1,1,E,B]
            ttnn.deallocate(r)
            table = ttnn.reshape(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), (E, B))
            rw = ttnn.embedding(idx_u32, table)  # [1, kc, B]
            ttnn.deallocate(table)
            ttnn.deallocate(idx_u32)
            ttnn.deallocate(idx_rm)
            rw = ttnn.to_layout(ttnn.reshape(rw, (1, KC, B, 1)), ttnn.TILE_LAYOUT)
            return rw

        res["compact_prologue_SHIPPED_embedding_both_indices"] = time_traced(
            dev, compact_embedding_real, "compact prologue, candidate B' (embedding + uint16 stick for sparse_matmul)"
        )

        payload = {
            "purpose": (
                "Split the compact decode MoE's routing prologue into the part the union path pays too and the "
                "compact-specific tail, and measure two adapted alternatives for that tail. One real MoE layer, "
                "B=32, kc=4, one live row, traced warmed replay."
            ),
            "shipped_variant": "compact_prologue_SHIPPED_embedding_both_indices",
            "note": (
                "The repeat+gather form was this stage's first implementation and is NOT what ships; the embedding "
                "form that produces both the uint16 stick sparse_matmul's indexed mode needs and the uint32 index "
                "ttnn.embedding needs is."
            ),
            "config": {"batch_rows": B, "kc": KC, "n_experts": E, "iters": ITERS},
            "ms_per_layer": res,
            "moe_layers_in_model": 46,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print("WROTE", OUT, flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
