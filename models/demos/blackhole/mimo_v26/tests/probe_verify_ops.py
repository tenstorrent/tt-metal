# SPDX-License-Identifier: Apache-2.0
"""Op probe for the speculative VERIFY step: 8 query tokens per user (positions pos..pos+7, causal inside the block, KV of the block already in the paged cache),
single Blackhole chip, per-chip TP8 shapes (8 Q heads, 1 KV head, QK 192, V padded to 192), SWA (window 128 incl. self, sink) and GA.
Candidates: V1 'virtual users' = the 1-token paged decode op with 8 virtual users per real user (page-table row repeated, cur_pos = pos+t);
            V2 chunked prefill SDPA (paged) with one 32-row q tile per user (8 real rows + pad), chunk_start = pos (arbitrary alignment).
    PROBE=v1,v2,perf PROBE_OUT=<jsonl> python probe_verify_ops.py"""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_sdpa_ops as P  # noqa: E402
from probe_sdpa_ops import (  # noqa: E402
    DQK,
    DV,
    HIFI2,
    HIFI4,
    SCALE,
    WINDOW,
    bench,
    dec_cfg,
    make_paged,
    pad_last,
    pcc,
    pre_cfg,
    rec,
    ref_decode,
    ref_prefill,
    run_case,
    sink_tt,
    tt,
)

BLK = 8
NH = 8


def make_case(B, Lc, swa, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(B, 1, Lc, DQK, generator=g).bfloat16()
    V = (torch.randn(B, 1, Lc, DV, generator=g) * 0.7).bfloat16()
    q = torch.randn(B, NH, BLK, DQK, generator=g).bfloat16()
    sink = ((torch.rand(NH, generator=g) * 0.58 + 0.56)).bfloat16() if swa else None
    return K, V, q, sink


def v1_case(dev, B, Lc, swa, pos, ring_mod=None):
    """virtual-user decode: B real users x 8 block tokens."""
    K, V, q, sink = make_case(B, Lc, swa)
    pk, pt = make_paged(K)
    pv, _ = make_paged(pad_last(V, DQK))
    pos = torch.as_tensor(pos, dtype=torch.int32)
    # virtual users: user b token t -> row b*8+t
    vpt = pt.repeat_interleave(BLK, 0)
    vpos = (pos[:, None] + torch.arange(BLK)[None]).reshape(-1).to(torch.int32)
    vq = q.permute(0, 2, 1, 3).reshape(1, B * BLK, NH, DQK)  # [1, B*8, nh, d]
    kw = dict(
        cur_pos_tensor=tt(dev, vpos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        scale=SCALE,
        program_config=dec_cfg(dev, 128),
        compute_kernel_config=HIFI4(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if swa:
        nhp = 32
        s = torch.zeros(nhp, 32)
        s[:NH, 0] = sink.float() / SCALE
        kw.update(sliding_window_size=WINDOW, attention_sink=tt(dev, s.bfloat16()))
    extra = {"cache_position_modulo": ring_mod} if ring_mod else {}
    tq, tk, tv, tpt = tt(dev, vq), tt(dev, pk), tt(dev, pv), tt(dev, vpt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def op():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tq, tk, tv, page_table_tensor=tpt, **kw, **extra
        )

    return op, (K, V, q, sink, pos)


def v1_check(dev, B, Lc, swa, pos):
    op, (K, V, q, sink, pos_t) = v1_case(dev, B, Lc, swa, pos)
    o = ttnn.to_torch(op())[0, :, :NH, :DV].float().reshape(B, BLK, NH, DV)
    pcs = []
    for b in range(B):
        for t in range(BLK):
            ref = ref_decode(
                q[b : b + 1, :, t],
                K[b : b + 1],
                V[b : b + 1],
                torch.tensor([int(pos_t[b]) + t]),
                sink=sink,
                window=WINDOW if swa else None,
            )
            pcs.append(pcc(ref[0], o[b, t]))
    return {
        "pcc_min": round(min(pcs), 6),
        "pcc_mean": round(sum(pcs) / len(pcs), 6),
        "ms": round(bench(dev, op, 10), 3),
    }


def v2_case(dev, B, Lc, swa, pos):
    """chunked prefill SDPA per user, q tile of 32 rows (8 real + 24 pad), chunk_start = pos[b]."""
    (
        K,
        V,
        q,
        sink,
    ) = make_case(B, Lc, swa)
    pk, pt = make_paged(K)
    pv, _ = make_paged(pad_last(V, DQK))
    tk, tv = tt(dev, pk), tt(dev, pv)
    qpad = torch.nn.functional.pad(q, (0, 0, 0, 32 - BLK))  # [B,nh,32,d]
    args = dict(program_config=pre_cfg(dev, 32, 32), compute_kernel_config=HIFI2())
    if swa:
        args.update(sliding_window_size=WINDOW, attention_sink=sink_tt(dev, sink, "scale"))
    tqs = [tt(dev, qpad[b : b + 1]) for b in range(B)]
    tpts = [tt(dev, pt[b : b + 1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for b in range(B)]

    def op():
        outs = [
            ttnn.transformer.chunked_scaled_dot_product_attention(tqs[b], tk, tv, tpts[b], int(pos[b]), **args)
            for b in range(B)
        ]
        return outs

    return op, (K, V, q, sink, pos)


def v2_check(dev, B, Lc, swa, pos):
    op, (K, V, q, sink, pos_t) = v2_case(dev, B, Lc, swa, pos)
    outs = op()
    pcs = []
    for b in range(B):
        h = ttnn.to_torch(outs[b])[0, :, :BLK, :DV].float()
        ref = ref_prefill(
            q[b : b + 1],
            K[b : b + 1, :, : int(pos_t[b]) + BLK],
            V[b : b + 1, :, : int(pos_t[b]) + BLK],
            q_start=int(pos_t[b]),
            sink=sink,
            window=WINDOW if swa else None,
        )
        pcs.append(pcc(ref[0], h))
    for o in outs:
        ttnn.deallocate(o)

    def timed():
        o = op()
        for t in o[1:]:
            ttnn.deallocate(t)
        return o[0]

    return {"pcc_min": round(min(pcs), 6), "ms": round(bench(dev, timed, 10), 3)}


def g_v1(dev):
    for swa in (True, False):
        for B, pos in ((1, [1000]), (4, [1000, 1001, 1500, 129]), (4, [127, 128, 1999, 2030])):
            run_case("v1_virtual_users", lambda: v1_check(dev, B, 2048, swa, pos), swa=swa, B=B, pos=pos, Lc=2048)


def g_v2(dev):
    for swa in (True, False):
        for B, pos in ((1, [1000]), (1, [1003]), (1, [1024]), (4, [1000, 1001, 1500, 129]), (4, [96, 128, 1984, 2016])):
            run_case("v2_chunked_q32", lambda: v2_check(dev, B, 2048, swa, pos), swa=swa, B=B, pos=pos, Lc=2048)


def g_perf(dev):
    # GA at 64k and SWA, B users: virtual-user decode vs chunked-q32 per user vs the plain 1-token decode (same B)
    for swa, Lc in ((False, 65536), (True, 2048)):
        for B in (1, 4, 8):
            pos = [Lc - 64 - 7 * i for i in range(B)]

            def f():
                res = {}
                op, _ = v1_case(dev, B, Lc, swa, pos)
                res["v1_ms"] = round(bench(dev, op, 10), 3)
                op1, _ = v1_case(dev, B, Lc, swa, pos)  # same data, but 1 token per user: B virtual users only
                try:
                    op2, _ = v2_case(dev, B, Lc, swa, pos)

                    def timed():
                        o = op2()
                        for t in o[1:]:
                            ttnn.deallocate(t)
                        return o[0]

                    res["v2_ms"] = round(bench(dev, timed, 10), 3)
                except Exception as e:  # noqa
                    res["v2_err"] = P.err(e)
                # 1-token decode at B users (baseline)
                c = P.DecodeCase(dev, B, NH, 1, Lc, pos, swa)
                res["decode_1tok_ms"] = round(bench(dev, c.op_pad, 10), 3)
                return res

            run_case("verify_perf", f, swa=swa, Lc=Lc, B=B)


P.GROUPS.update(v1=g_v1, v2=g_v2, perf=g_perf)

if __name__ == "__main__":
    groups = os.environ.get("PROBE", "v1,v2").split(",")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        rec("env", groups=groups)
        for g in groups:
            t0 = time.time()
            try:
                P.GROUPS[g](dev)
            except Exception as e:  # noqa
                rec("group_fail", group=g, error=P.err(e))
            rec("group_done", group=g, secs=round(time.time() - t0, 1))
    finally:
        ttnn.close_mesh_device(dev)
