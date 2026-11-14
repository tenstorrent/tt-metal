# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""prefill-roof phase 2: ONE 512-token prefill chunk of a Hy4 attention layer on a 1x8 slot (per-chip work identical to one row of the 4x8 mesh), traced, per stage + whole chunk,
for config variants, with PCC of the chunk output vs the oracle golden and (layer 1) the bf8-query indexer-score experiment on real data.
Env: LAYER (1 full | 2 shared), P0 (2560), VARIANTS (comma: base,hifi2,hifi2na,l1,hifi2l1), EXP_BF8Q (1: real-data bf8 q score PCC/overlap + timing, layer 1 only), GOLD, HY4_CKPT. Prints PF json lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import idx_setmatch, load_sd, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden")
CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
L = int(os.environ.get("LAYER", "1"))
P0 = int(os.environ.get("P0", "2560"))
N = 512
VARS = os.environ.get("VARIANTS", "base").split(",")


def PF(**kw):
    print("PF " + json.dumps(kw), flush=True)


def tr(md, fn, reps=5, n=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(min(ts), 1)


def set_variant(a, md, v):
    orig = getattr(set_variant, "orig", None)
    if orig is None:
        set_variant.orig = orig = (ttnn.linear, ttnn.matmul, a.ckc, a.indexer.ckc if a.indexer else None)
    ttnn.linear, ttnn.matmul = orig[0], orig[1]
    if not hasattr(set_variant, "o2"):
        set_variant.o2 = (ttnn.experimental.all_to_all_async_generic, ttnn.transformer.sparse_sdpa)
    ttnn.experimental.all_to_all_async_generic, ttnn.transformer.sparse_sdpa = set_variant.o2
    if "nl2" in v:
        f0 = set_variant.o2[0]
        ttnn.experimental.all_to_all_async_generic = lambda *x, **k: f0(*x, **{**k, "num_links": 2})
    if "kc64" in v:
        f1 = set_variant.o2[1]
        ttnn.transformer.sparse_sdpa = lambda *x, **k: f1(*x, **{**k, "k_chunk_size": 64})
    a.ckc = orig[2]
    if a.indexer:
        a.indexer.ckc = orig[3]
    if "hifi2" in v:
        c = compute_cfg(md, "hifi2", "na" not in v)
        a.ckc = c
        if a.indexer:
            a.indexer.ckc = c
    if "tuned" in v:
        if "hifi2" in v or "tuned2" in v:
            c = compute_cfg(md, "hifi2", True)
            a.ckc = c
            if a.indexer:
                a.indexer.ckc = c
        lin0 = orig[0]

        def tlin(x, w, **k):
            M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
            if (
                "program_config" not in k
                and M % 256 == 0
                and K % 32 == 0
                and N % 32 == 0
                and len(x.shape) == 4
                and x.shape[1] == 1
                and len(w.shape) == 4
            ):
                mt, kt, nt = M // 32, K // 32, N // 32
                gy = 8 if mt % 8 == 0 else 4
                gx = max(d for d in range(1, 13) if nt % d == 0)
                pn = nt // gx
                bw = max(d for d in (16, 8, 4, 2, 1) if kt % d == 0)
                sw = max(d for d in (4, 2, 1) if pn % d == 0)
                act = k.pop("activation", None)
                fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if act == "sigmoid" else None
                k["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=bw,
                    out_subblock_h=1,
                    out_subblock_w=sw,
                    per_core_M=mt // gy,
                    per_core_N=pn,
                    transpose_mcast=False,
                    fused_activation=fa,
                    fuse_batch=True,
                )
            return lin0(x, w, **k)

        ttnn.linear = tlin
    if v.endswith("l1"):

        def wrap(f):
            def g(*x, **k):
                k.setdefault("memory_config", ttnn.L1_MEMORY_CONFIG)
                return f(*x, **k)

            return g

        ttnn.linear, ttnn.matmul = wrap(orig[0]), wrap(orig[1])


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=800_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        a = Hy4Attention(
            md,
            mesh_1x8(),
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt="bf16",
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=ttnn.bfloat8_b,
        )
        shared = not a.has_indexer
        for p0 in range(0, P0 + N, N):
            h = upload(md, g["h"][p0 : p0 + N].reshape(1, 1, N, -1).to(torch.bfloat16))
            ids_in = (
                upload(
                    md, g["idx"][p0 : p0 + N].to(torch.int32).reshape(1, 1, N, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                )
                if shared
                else None
            )
            out, ids = a.prefill_chunk(h, p0, ids_in)
        ttnn.synchronize_device(md)
        ref = g["out"][P0 : P0 + N].float()
        tabs = a.rope_tabs(P0, N)
        cos, sin, icos, isin = tabs
        kvc = a.cache.rows
        for v in VARS:
            set_variant(a, md, v)
            o, ids2 = a.prefill_chunk(h, P0, ids_in, append=False, tabs=tabs)
            ttnn.synchronize_device(md)
            got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(N, -1).float()
            r = {"layer": L, "variant": v, "P0": P0, "pcc_out": round(pcc(got, ref), 6)}
            if not shared:
                ov, same = idx_setmatch(
                    ttnn.to_torch(ttnn.get_device_tensors(ids2)[0]).reshape(N, -1).long(), g["idx"][P0 : P0 + N]
                )
                r["idx_overlap"] = round(ov, 4)
            r["all_us"] = tr(md, lambda: a.prefill_chunk(h, P0, ids_in, append=False, tabs=tabs), reps=3, n=5)

            # stages
            def front():
                qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
                qn = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
                kv = ttnn.linear(h, a.w_kv_a, compute_kernel_config=a.ckc)
                lat = a._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, N, 512]), a.kv_a_norm)
                kpe = a.rope.apply(ttnn.slice(kv, [0, 0, 0, 512], [1, 1, N, 576]), cos, sin, a.ckc)
                return qn, lat, kpe

            r["front_us"] = tr(md, front)
            qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
            qa = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
            if not shared:
                r["indexer_us"] = tr(md, lambda: a.indexer.select(h, qa, P0, icos, isin, None, append=False), reps=3)

            def qpath():
                q = ttnn.linear(qa, a.w_q_b, compute_kernel_config=a.ckc)
                qn_, qp = ttnn.experimental.nlp_create_q_heads_split(
                    q, num_heads=a.hc, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                qabs = ttnn.matmul(qn_, a.w_uk, compute_kernel_config=a.ckc)
                qp = a.rope.apply(qp, cos, sin, a.ckc)
                return ttnn.concat([qabs, qp], dim=3)

            r["qpath_us"] = tr(md, qpath)
            qfull = qpath()
            r["sdpa_us"] = tr(md, lambda: a._sparse_prefill(qfull, kvc, ids2), reps=3)
            o_lat = a._sparse_prefill(qfull, kvc, ids2)
            r["finish_us"] = tr(md, lambda: a._finish(o_lat, h))
            PF(**r)
        set_variant(a, md, "base")
        if os.environ.get("EXP_SDPA") == "1":
            qfull = ttnn.concat(
                [
                    ttnn.zeros([1, 8, N, 512], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md),
                    ttnn.zeros([1, 8, N, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md),
                ],
                dim=3,
            )
            qr = ttnn.to_torch(ttnn.get_device_tensors(a.cache.rows)[0])
            # real q: recompute from the model path
            qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
            qa = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
            q = ttnn.linear(qa, a.w_q_b, compute_kernel_config=a.ckc)
            qn_, qp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=a.hc, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            qfull = ttnn.concat(
                [ttnn.matmul(qn_, a.w_uk, compute_kernel_config=a.ckc), a.rope.apply(qp, cos, sin, a.ckc)], dim=3
            )
            R = {}
            q_t = ttnn.experimental.all_to_all_async_generic(
                qfull, in_dim=1, out_dim=2, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
            )
            R["a2a_q_us"] = tr(
                md,
                lambda: ttnn.experimental.all_to_all_async_generic(
                    qfull, in_dim=1, out_dim=2, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
                ),
                reps=3,
            )
            for nl in (2, 3, 4):
                try:
                    f = lambda: ttnn.experimental.all_to_all_async_generic(
                        qfull, in_dim=1, out_dim=2, num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
                    )
                    o2 = f()
                    ok = bool(
                        (
                            ttnn.to_torch(ttnn.get_device_tensors(o2)[3])
                            == ttnn.to_torch(ttnn.get_device_tensors(q_t)[3])
                        ).all()
                    )
                    R[f"a2a_q_nl{nl}_us"] = tr(md, f, reps=3)
                    R[f"a2a_q_nl{nl}_same"] = ok
                except Exception as e:
                    R[f"a2a_q_nl{nl}_us"] = "ERR " + str(e).split(chr(10))[0][:100]
            R["to_rm_us"] = tr(md, lambda: ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT), reps=3)
            q_seq = ttnn.to_layout(q_t, ttnn.ROW_MAJOR_LAYOUT)
            R["mesh_partition_us"] = tr(md, lambda: ttnn.mesh_partition(ids2, dim=2, cluster_axis=1), reps=3)
            idx_c = ttnn.mesh_partition(ids2, dim=2, cluster_axis=1)
            for kc in (64, 128, 256, 512):
                try:
                    R[f"sdpa_op_kc{kc}_us"] = tr(
                        md,
                        lambda: ttnn.transformer.sparse_sdpa(
                            q_seq,
                            kvc,
                            idx_c,
                            cfg.kv_lora_rank,
                            kv_format=a._kv_format(),
                            scale=a.scale,
                            k_chunk_size=kc,
                            compute_kernel_config=a.ckc_sdpa,
                            attention_sink=a.sink_full,
                        ),
                        reps=3,
                    )
                except Exception as e:
                    R[f"sdpa_op_kc{kc}_us"] = "ERR " + str(e).split(chr(10))[0][:100]
            out = ttnn.transformer.sparse_sdpa(
                q_seq,
                kvc,
                idx_c,
                cfg.kv_lora_rank,
                kv_format=a._kv_format(),
                scale=a.scale,
                k_chunk_size=128,
                compute_kernel_config=a.ckc_sdpa,
                attention_sink=a.sink_full,
            )
            R["to_tile_us"] = tr(md, lambda: ttnn.to_layout(out, ttnn.TILE_LAYOUT), reps=3)
            out_t = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            R["a2a_back_us"] = tr(
                md,
                lambda: ttnn.experimental.all_to_all_async_generic(
                    out_t, in_dim=2, out_dim=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
                ),
                reps=3,
            )
            PF(exp="sdpa_pieces", **R)
        if os.environ.get("EXP_BF8Q") == "1" and not shared:
            ix = a.indexer
            qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
            qa = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
            T = ix.length
            q = ttnn.linear(qa, ix.w_q, compute_kernel_config=ix.ckc)
            q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q, num_heads=ix.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            q = ix._rot_first(q, icos, isin)
            w = ttnn.linear(h, ix.w_p, compute_kernel_config=ix.ckc)
            pc = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(T, 256), head_group_size=1)

            def score(qq):
                return ttnn.experimental.indexer_score_dsa(
                    qq, ix.keys, w, chunk_start_idx=P0, program_config=pc, compute_kernel_config=ix.ckc_score, kv_len=T
                )

            q8 = ttnn.typecast(q, ttnn.bfloat8_b)
            res = {}
            for nm, qq in (("bf16q", q), ("bf8q", q8)):
                s = ix._causal_fix(score(qq), P0, N, T)
                ids_ = ttnn.experimental.topk_large_indices(s, k=min(2048, T), valid_length=T)
                sh = ttnn.to_torch(ttnn.get_device_tensors(s)[0]).reshape(N, -1).float()
                ii = ttnn.to_torch(ttnn.get_device_tensors(ids_)[0]).reshape(N, -1).long()
                res[nm] = (sh, ii)
                ov, same = idx_setmatch(ii, g["idx"][P0 : P0 + N])
                PF(exp="bf8q", q=nm, overlap_vs_oracle=round(ov, 4), T=T, score_us=tr(md, lambda: score(qq), reps=3))
            fin = torch.isfinite(res["bf16q"][0]) & torch.isfinite(res["bf8q"][0])
            PF(
                exp="bf8q",
                pcc_score_bf8_vs_bf16=round(pcc(res["bf8q"][0][fin], res["bf16q"][0][fin]), 6),
                ids_overlap_bf8_vs_bf16=round(idx_setmatch(res["bf8q"][1], res["bf16q"][1])[0], 4),
            )
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
