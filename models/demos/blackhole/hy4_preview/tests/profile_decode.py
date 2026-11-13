# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage traced latency of ONE decode step of a Hy4 attention layer (4x8 mesh, one user per row, context P built by the prefill path). Env: LAYER (1), KVFMT, POS (2560), V1, GOLD.
Stages (each its own trace, inputs prepared outside): front (q_a gather + norm, kv proj, lat norm, k_pe rope), indexer (score + topk), qpath (q_b, heads, absorb, rope, concat), sdpa (pad heads, op, slice),
finish (w_uv, gate, o_proj, all-reduce). Prints RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "1"))
KVFMT = os.environ.get("KVFMT", "bf16")
P = int(os.environ.get("POS", "2560"))
V1 = os.environ.get("V1", "0") == "1"


def tr(md, name, fn, reps=10):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps)
    ttnn.release_trace(md, tid)
    print(
        "RES "
        + json.dumps(
            {"layer": L, "kv": KVFMT, "v1": V1, "pos": P, "n": 512, "stage": name, "traced_us": round(min(ts) * 1e6, 1)}
        ),
        flush=True,
    )


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=800_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        a = Hy4Attention(
            md,
            mesh_4x8(),
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt=KVFMT,
            rope_interleave=g["conv"]["rope_interleave"],
            idx_rope_interleave=g["conv"]["idx_rope_interleave"],
        )
        for p0 in range(0, P, 512):
            h = upload(md, g["h"][p0 : p0 + 512].reshape(1, 1, 512, -1).to(torch.bfloat16))
            ids_in = (
                upload(
                    md,
                    g["idx"][p0 : p0 + 512].to(torch.int32).reshape(1, 1, 512, -1),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                if not a.has_indexer
                else None
            )
            a.prefill_chunk(h, p0, ids_in)
        h = upload(md, g["h"][P].reshape(1, 1, 1, -1).to(torch.bfloat16))
        ids_in = (
            upload(md, g["idx"][P].to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if not a.has_indexer
            else None
        )
        tabs = a.rope_tabs(P, 1)
        cos, sin, icos, isin = tabs
        # one real step to leave row P in the caches (append), then every stage runs append-free on static state
        a.decode_step(h, P, ids_in, v1=V1)
        ttnn.synchronize_device(md)
        # precomputed stage inputs
        qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
        qa = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
        kvc = a.cache.rows

        def front():
            qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
            qa_g = a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3)
            qn = a._rms(qa_g, a.q_a_norm)
            kv = ttnn.linear(h, a.w_kv_a, compute_kernel_config=a.ckc)
            lat = a._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, 1, 512]), a.kv_a_norm)
            kpe = a.rope.apply(ttnn.slice(kv, [0, 0, 0, 512], [1, 1, 1, 576]), cos, sin, a.ckc)
            return qn, lat, kpe

        tr(md, "front(qa gather+norm, kv, lat norm, k_pe rope)", front)
        if a.has_indexer:
            tr(
                md,
                "indexer(q/k/w proj, rope, score, topk)",
                lambda: a.indexer.select_decode(h, qa, P, icos, isin, append=False),
            )
            ids = a.indexer.select_decode(h, qa, P, icos, isin, append=False)
        else:
            ids = ids_in

        def qpath():
            q = ttnn.linear(qa, a.w_q_b, compute_kernel_config=a.ckc)
            qn, qp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=a.hc, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            qabs = ttnn.matmul(qn, a.w_uk, compute_kernel_config=a.ckc)
            qp = a.rope.apply(qp, cos, sin, a.ckc)
            return ttnn.concat([qabs, qp], dim=3)

        tr(md, "qpath(q_b, heads, absorb, rope, concat)", qpath)
        qfull = qpath()
        tr(
            md,
            f"sdpa({'split v1' if V1 else 'stock v0'}: pad heads, op, slice)",
            lambda: a._sparse_decode(qfull, kvc, ids, V1),
        )
        o_lat = a._sparse_decode(qfull, kvc, ids, V1)
        tr(md, "finish(w_uv, gate, o_proj, all-reduce)", lambda: a._finish(o_lat, h))
        tr(md, "ALL decode_step (append-free)", lambda: a.decode_step(h, P, ids_in, v1=V1, append=False, tabs=tabs))
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
