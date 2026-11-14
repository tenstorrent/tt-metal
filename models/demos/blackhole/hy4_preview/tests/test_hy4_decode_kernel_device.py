"""Hy4 decode split-kernel device tests (one chip). Run inside the DSV4 image via program/decode-kernel/run_dk.sh:  run_dk.sh <tag> test_hy4_decode_kernel_device.py TEST=plain|merge|lat|scaled|all
Every case prints one `RES {json}` line. PCC gate 0.999 (bf16 KV vs fp32 torch on the same bf16 inputs; SCALED_FP8 vs the dequantised reference)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dk_dev import *

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt"))
import seqshard as SQ

H, HR, TOPK, D, V = 32, 8, 2048, 576, 512
GATE = 0.999


def gen_data(kind, T, S, seed=0):
    """kv [T,576] bf16, q [HR,S,576] bf16 (real heads), idx later.  kinds: random (N(0,1)); real (heavy-tailed channel scales + a few massive channels + shared mean, q aligned with a few keys so attention is peaked: top weight ~0.3-0.9)"""
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        kv = torch.randn(T, D, generator=g)
        q = torch.randn(HR, S, D, generator=g)
    else:
        ch = torch.exp(torch.randn(D, generator=g) * 0.8)  # per-channel scale spread (lognormal)
        ch[torch.randint(0, 512, (4,), generator=g)] *= 12.0  # massive-activation channels
        kv = (torch.randn(T, D, generator=g) * ch + 0.5 * torch.randn(D, generator=g) * ch) * 0.35
        kv[:, 512:] *= 2.0  # rope part carries larger magnitude
        q = torch.randn(HR, S, D, generator=g) * ch * 0.35
    return kv.to(torch.bfloat16), q.to(torch.bfloat16)


def align_q_to_keys(kind, q, kv, idx, strength=14.0):
    """real-like: make each head's query point at a random selected key (peaked softmax, the regime where bf16 max/sum errors would show)."""
    if kind == "random":
        return q
    q = q.clone().float()
    S = q.shape[1]
    for s in range(S):
        ids = idx[s][idx[s] >= 0]
        for h in range(HR):
            k = kv[ids[torch.randint(0, ids.numel(), (1,))]].float()[0]
            q[h, s] = 0.5 * q[h, s] + strength * k / (k.norm() + 1e-6) * (D**0.5) / 16.0 * 0.25
    return q.to(torch.bfloat16)


def dev_inputs(md, q, kv, idx, sink_logit=None):
    qp = torch.zeros(H, q.shape[1], D, dtype=torch.bfloat16)
    qp[:HR] = q
    S = q.shape[1]
    q_dev = to_dev(md, qp.reshape(1, H, S, D), ttnn.bfloat16)
    kv_dev = to_dev(md, kv.reshape(1, 1, kv.shape[0], D), ttnn.bfloat16)
    sink_dev = None
    if sink_logit is not None:
        sp = torch.zeros(H)
        sp[:HR] = sink_logit
        sink_dev = to_dev(md, (sp / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16)
    return q_dev, kv_dev, idx_dev(md, idx), sink_dev


def test_plain(md, mod):
    """full softmax incl. sink, single column, B=4 (S=1/4 rows), B=8 (S=2/8), verify blocks, and the formerly broken S>=12"""
    for kind in ("random", "real"):
        for S, G, kc in [
            (1, 8, 128),
            (2, 8, 128),
            (4, 8, 128),
            (4, 4, 256),
            (8, 4, 256),
            (12, 4, 256),
            (16, 4, 256),
            (15, 8, 128),
        ]:
            T = 65536
            kv, q = gen_data(kind, T, S)
            per = T // S
            idx = make_idx(S, TOPK, [(s * per, (s + 1) * per) for s in range(S)], TOPK)
            q = align_q_to_keys(kind, q, kv, idx)
            sl = torch.linspace(-2, 2, HR)
            q_dev, kv_dev, i_dev, s_dev = dev_inputs(md, q, kv, idx, sl)
            out = mod.sparse_sdpa_split(q_dev, kv_dev, i_dev, V, SCALE, kc, G, ckc_of(md), s_dev)
            o = ttnn.to_torch(out)[0, :HR].float()
            ref = ref_sparse(q, kv, idx, sl)
            c = pcc(o, ref)
            emit(
                {
                    "test": "plain",
                    "data": kind,
                    "S": S,
                    "G": G,
                    "kc": kc,
                    "pcc": round(c, 5),
                    "pcc_min_row": round(min(pcc(o[:, s], ref[:, s]) for s in range(S)), 5),
                    "pass": c >= GATE,
                }
            )
            for t in (q_dev, kv_dev, i_dev, s_dev):
                ttnn.deallocate(t)


def column_partials(md, mod, q, kv_nat, ids_global, n_pages, G=8, kc=128, S=None, scaled=False):
    """ids_global [S,K] natural positions (SENT tail). Runs the 8 column shards one after the other on the chip. -> list over columns of (o [HR,S,V] f32, m [HR,S] scaled max, l [HR,S]), plus valid count per column [8,S]"""
    S = q.shape[1]
    kv_cols = SQ.shard_kv(kv_nat.float().numpy(), n_pages)  # [8, n_pages*64, D]
    loc, cnt = SQ.split_ids(ids_global.numpy().astype(np.int64))  # [8,S,K] uint32 , [8,S]
    parts = []
    kv_deqs = []
    for c in range(8):
        kvc = torch.from_numpy(kv_cols[c]).to(torch.bfloat16)
        ix = torch.from_numpy(loc[c].astype(np.int64))
        ix = torch.where(ix == SQ.SENTINEL, torch.full_like(ix, -1), ix)
        q_dev, kv_dev, i_dev, _ = dev_inputs(md, q, kvc, ix)
        if scaled:
            ttnn.deallocate(kv_dev)
            kv_dev, kv_deq = pack_scaled(md, kvc)
            kv_deqs.append(kv_deq)
        out, stats = mod.sparse_sdpa_split(q_dev, kv_dev, i_dev, V, SCALE, kc, G, ckc_of(md), None, emit_stats=True)
        o = ttnn.to_torch(out)[0, :HR].float()
        st = ttnn.to_torch(stats).reshape(S, 64).float()
        m = st[:, :HR].T * SCALE  # [HR,S] raw max -> scaled logits
        l = st[:, 32 : 32 + HR].T
        parts.append((o, m, l))
        for t in (q_dev, kv_dev, i_dev, out, stats):
            ttnn.deallocate(t)
    return (parts, cnt, kv_deqs) if scaled else (parts, cnt)


def merge_cpu(parts, sink_logit, s):
    ps = [(p[0][:, s].double().numpy(), p[1][:, s].double().numpy(), p[2][:, s].double().numpy()) for p in parts]
    return SQ.merge(ps, sink=sink_logit.double().numpy())


def test_merge(md, mod):
    """8 column shards, per-column (o, m, l) from the kernel, CPU merge == full softmax over the global top-2048 incl. sink; includes empty columns"""
    for kind in ("random", "real"):
        for name, ctx, S in [
            ("ctx64k_S1", 65536, 1),
            ("ctx64k_S2", 65536, 2),
            ("ctx64k_S4", 65536, 4),
            ("ctx100_S2_empty_cols", 100, 2),
            ("ctx3000_S1", 3000, 1),
            ("ctx1M_slice_S1", 262144, 1),
        ]:
            n_pages = max(1, -(-ctx // 512))
            T = n_pages * 512
            kv, q = gen_data(kind, T, S, seed=1)
            ids = torch.full((S, TOPK), -1, dtype=torch.int64)
            for s in range(S):
                n = min(TOPK, ctx)
                ids[s, :n] = torch.randperm(ctx)[:n]
            q = align_q_to_keys(kind, q, kv, ids)
            sl = torch.linspace(-2, 2, HR)
            ids_u = torch.where(ids < 0, torch.full_like(ids, SQ.SENTINEL), ids)
            parts, cnt = column_partials(md, mod, q, kv, ids_u, n_pages)
            ref = ref_sparse(q, kv, ids, sl)
            merged = torch.stack(
                [torch.from_numpy(merge_cpu(parts, sl, s)).float() for s in range(S)], dim=1
            )  # [HR,S,V]
            c = pcc(merged, ref)
            empties = int((torch.from_numpy(cnt) == 0).sum())
            # raw stats check vs CPU partial_attention (exact-math column partials)
            worst_m = worst_l = 0.0
            for col in range(8):
                for s in range(S):
                    if cnt[col, s] == 0:
                        assert (
                            torch.isinf(parts[col][1][:, s]).all() and (parts[col][2][:, s] == 0).all()
                        ), "empty column must emit m=-inf, l=0"
                        continue
                    kvc = SQ.shard_kv(kv.float().numpy(), n_pages)[col]
                    loc, _ = SQ.split_ids(ids_u.numpy().astype(np.int64))
                    o_r, m_r, l_r = SQ.partial_attention(
                        q[:, s].double().numpy(), kvc.astype(np.float64), loc[col, s].astype(np.int64), V, SCALE
                    )
                    worst_m = max(worst_m, float(np.abs(parts[col][1][:, s].numpy() - m_r).max()))
                    worst_l = max(worst_l, float(np.abs(parts[col][2][:, s].numpy() / l_r - 1).max()))
            emit(
                {
                    "test": "merge",
                    "data": kind,
                    "case": name,
                    "S": S,
                    "empty_col_rows": empties,
                    "pcc_merged_vs_full": round(c, 5),
                    "max_abs_dm": round(worst_m, 4),
                    "max_rel_dl": round(worst_l, 4),
                    "pass": c >= GATE,
                }
            )


def pack_scaled(md, kv):
    dp = ttnn.experimental.deepseek_prefill
    T = kv.shape[0]
    t_lat = to_dev(md, kv[:, :V].reshape(1, 1, T, V), ttnn.bfloat16)
    t_rope = to_dev(md, kv[:, V:].reshape(1, 1, T, D - V), ttnn.bfloat16)
    t_fp8, t_sc = dp.per_token_cast_to_fp8(t_lat, round_scale_to_power_of_two=True)
    kv_dev = dp.pack_scaled_fp8_kv_cache(t_fp8, t_sc, t_rope)
    rec = ttnn.to_torch(dp.per_token_cast_back(t_fp8, t_sc, output_dtype=ttnn.bfloat16)).reshape(T, V)
    for t in (t_lat, t_rope, t_fp8, t_sc):
        ttnn.deallocate(t)
    return kv_dev, torch.cat([rec.to(torch.bfloat16), kv[:, V:]], dim=-1)


def test_merge_scaled(md, mod):
    """SCALED_FP8 column shards + stats: merged result vs the dequantised full-softmax reference"""
    for kind in ("random", "real"):
        ctx, S = 65536, 2
        n_pages = ctx // 512
        T = n_pages * 512
        kv, q = gen_data(kind, T, S, seed=2)
        ids = torch.stack([torch.randperm(ctx)[:TOPK] for _ in range(S)])
        q = align_q_to_keys(kind, q, kv, ids)
        sl = torch.linspace(-2, 2, HR)
        parts, cnt, deqs = column_partials(md, mod, q, kv, ids, n_pages, scaled=True)
        # dequantised natural pool for the reference: undo the column sharding
        kv_cols = torch.stack(deqs).float().numpy()
        t = np.arange(T)
        _, col, loc = SQ.locate(t)
        kv_deq_nat = torch.from_numpy(kv_cols[col, loc]).to(torch.bfloat16)
        ref = ref_sparse(q, kv_deq_nat, ids, sl)
        merged = torch.stack([torch.from_numpy(merge_cpu(parts, sl, s)).float() for s in range(S)], dim=1)
        c = pcc(merged, ref)
        emit(
            {
                "test": "merge_scaled_fp8",
                "data": kind,
                "S": S,
                "pcc_merged_vs_dequant_full": round(c, 5),
                "pcc_vs_bf16_full": round(pcc(merged, ref_sparse(q, kv, ids, sl)), 5),
                "pass": c >= GATE,
            }
        )


def test_scaled(md, mod):
    for kind in ("random", "real"):
        for S, G, kc in [(1, 8, 128), (4, 8, 128), (4, 4, 256), (8, 4, 256)]:
            T = 65536
            kv, q = gen_data(kind, T, S)
            per = T // S
            idx = make_idx(S, TOPK, [(s * per, (s + 1) * per) for s in range(S)], TOPK)
            q = align_q_to_keys(kind, q, kv, idx)
            sl = torch.linspace(-2, 2, HR)
            kv_dev, kv_ref = pack_scaled(md, kv)
            q_dev, _, i_dev, s_dev = dev_inputs(md, q, kv[:16], idx, sl)
            try:
                out = mod.sparse_sdpa_split(q_dev, kv_dev, i_dev, V, SCALE, kc, G, ckc_of(md), s_dev)
                o = ttnn.to_torch(out)[0, :HR].float()
                ref_deq = ref_sparse(q, kv_ref, idx, sl)
                ref_bf = ref_sparse(q, kv, idx, sl)
                emit(
                    {
                        "test": "scaled_fp8",
                        "data": kind,
                        "S": S,
                        "G": G,
                        "kc": kc,
                        "pcc_vs_dequant_ref": round(pcc(o, ref_deq), 5),
                        "pcc_vs_bf16_ref": round(pcc(o, ref_bf), 5),
                        "pass": pcc(o, ref_deq) >= GATE,
                    }
                )
            except Exception as e:
                emit(
                    {
                        "test": "scaled_fp8",
                        "data": kind,
                        "S": S,
                        "G": G,
                        "kc": kc,
                        "error": str(e).strip().splitlines()[0][:300],
                    }
                )
            ttnn.deallocate(kv_dev)


def test_lat(md, mod):
    """per-layer latency (traced replay, 10 calls): valid keys 2048/1024/512/128, S=1 (B=4), 2 (B=8), 4/8 (verify), emit vs plain, bf16 vs scaled"""
    for fmt in ("bf16", "scaled"):
        for S in (1, 2, 4, 8):
            T = 65536
            kv, q = gen_data("random", T, S)
            if fmt == "scaled":
                kv_dev, _ = pack_scaled(md, kv)
            else:
                kv_dev = to_dev(md, kv.reshape(1, 1, T, D), ttnn.bfloat16)
            for valid in (2048, 1024, 512, 128):
                per = T // S
                idx = make_idx(S, TOPK, [(s * per, (s + 1) * per) for s in range(S)], valid)
                sl = torch.linspace(-2, 2, HR)
                q_dev, _kv, i_dev, s_dev = dev_inputs(md, q, kv[:16], idx, sl)
                ttnn.deallocate(_kv)
                for G, kc in [(8, 128), (4, 256)]:
                    for emit_stats in (False, True):
                        try:
                            if emit_stats:
                                call = lambda: mod.sparse_sdpa_split(
                                    q_dev, kv_dev, i_dev, V, SCALE, kc, G, ckc_of(md), None, emit_stats=True
                                )
                            else:
                                call = lambda: mod.sparse_sdpa_split(
                                    q_dev, kv_dev, i_dev, V, SCALE, kc, G, ckc_of(md), s_dev
                                )
                            us = timed_traced(md, call)
                            emit(
                                {
                                    "test": "lat",
                                    "fmt": fmt,
                                    "S": S,
                                    "valid": valid,
                                    "G": G,
                                    "kc": kc,
                                    "emit_stats": emit_stats,
                                    "us_per_layer": us,
                                }
                            )
                        except Exception as e:
                            emit(
                                {
                                    "test": "lat",
                                    "fmt": fmt,
                                    "S": S,
                                    "valid": valid,
                                    "G": G,
                                    "kc": kc,
                                    "emit_stats": emit_stats,
                                    "error": str(e).strip().splitlines()[0][:200],
                                }
                            )
                            raise
                for t in (q_dev, i_dev, s_dev):
                    ttnn.deallocate(t)
            ttnn.deallocate(kv_dev)


if __name__ == "__main__":
    mod = load_split()
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    which = os.environ.get("TEST", "all")
    try:
        for name, fn in [
            ("plain", test_plain),
            ("merge", test_merge),
            ("scaled", test_scaled),
            ("merge_scaled", test_merge_scaled),
            ("lat", test_lat),
        ]:
            if which in (name, "all"):
                try:
                    fn(md, mod)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    emit({"test": name, "error": str(e).strip().splitlines()[0][:300]})
    finally:
        ttnn.close_mesh_device(md)
