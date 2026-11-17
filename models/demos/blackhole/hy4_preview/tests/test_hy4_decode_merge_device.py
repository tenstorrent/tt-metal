"""Hy4 decode attention block (q all-gather + split kernel + all_to_all + merge) on a 1x8 mesh row: PCC vs the torch full softmax over the global top-2048 ids, and per-layer latency.
Run inside the DSV4 image: program/decode-merge/run_dm.sh <tag> test_hy4_decode_merge_device.py TEST=pcc|lat|smoke [CASES=..]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hy4_decode_kernel_device as DK
from dk_dev import SCALE, pcc, ref_sparse, timed_traced
from dm_block import NC, Block, close_row, from_mesh, open_row, to_mesh

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt"))
import seqshard as SQ

DK.HR = 64  # gen_data / align_q_to_keys generate all 64 heads
TOPK = 2048
GATE = 0.999


def emit(r):
    print("RES " + json.dumps(r), flush=True)


def pack_scaled_mesh(md, kv_cols):
    """kv_cols [8,Tloc,576] bf16 -> mesh SCALED_FP8 packed kv + dequantised reference [8,Tloc,576]"""
    dp = ttnn.experimental.deepseek_prefill
    Tl = kv_cols.shape[1]
    lat = to_mesh(md, kv_cols[:, :, :512].reshape(NC, 1, Tl, 512), ttnn.bfloat16)
    rope = to_mesh(md, kv_cols[:, :, 512:].reshape(NC, 1, Tl, 64), ttnn.bfloat16)
    f8, sc = dp.per_token_cast_to_fp8(lat, round_scale_to_power_of_two=True)
    kv_dev = dp.pack_scaled_fp8_kv_cache(f8, sc, rope)
    rec = from_mesh(md, dp.per_token_cast_back(f8, sc, output_dtype=ttnn.bfloat16)).reshape(NC, Tl, 512)
    for t in (lat, rope, f8, sc):
        ttnn.deallocate(t)
    return kv_dev, torch.cat([rec.to(torch.bfloat16), kv_cols[:, :, 512:]], dim=-1)


class Case:
    def __init__(self, md, kind, ctx, S, fmt, seed=0, topk_valid=None, skew=False):
        n_pages = max(1, -(-ctx // 512))
        T = n_pages * 512
        gkind, strength = (
            ("real", 14.0) if kind == "real" else (("real", 60.0) if kind == "hot" else (kind, 14.0))
        )  # hot: logits far above the sink (overflow margin test)
        kv, q = DK.gen_data(gkind, T, S, seed=seed)  # q [64,S,576]
        ids = torch.full((S, TOPK), -1, dtype=torch.int64)
        for s in range(S):
            n = min(topk_valid or TOPK, ctx)
            if (
                skew
            ):  # every selected key lives in column 0 (worst case for the per-column kernel: 2048 keys on one column)
                pool = torch.arange(ctx)[(torch.arange(ctx) % 512) // 64 == 0]
                ids[s, :n] = pool[torch.randperm(pool.numel())[:n]]
            else:
                ids[s, :n] = torch.randperm(ctx)[:n]
        q = DK.align_q_to_keys(gkind, q, kv, ids, strength)
        sc0 = (q[:, 0].float() @ kv[ids[0][ids[0] >= 0]].float().T) * SCALE
        self.max_logit = round(float(sc0.max()), 1)  # largest scaled logit of user 0 (the sink is in [-2, 2])
        self.sl = torch.linspace(-2, 2, 64)
        ids_u = torch.where(ids < 0, torch.full_like(ids, SQ.SENTINEL), ids)
        loc, self.cnt = SQ.split_ids(ids_u.numpy().astype(np.int64))  # [8,S,K]
        kv_cols = torch.from_numpy(SQ.shard_kv(kv.float().numpy(), n_pages)).to(torch.bfloat16)  # [8,Tl,576]
        Tl = kv_cols.shape[1]
        qh = 8 if os.environ.get("Q8", "1") == "1" else 32  # q rows per column: 8 real heads (default) or padded to 32
        qp = torch.zeros(NC, qh, S, 576, dtype=torch.bfloat16)
        for c in range(NC):
            qp[c, :8] = q[8 * c : 8 * c + 8]
        self.q_dev = to_mesh(md, qp, ttnn.bfloat16)
        ix = torch.from_numpy(loc.astype(np.int64))
        ix = torch.where(ix == SQ.SENTINEL, torch.full_like(ix, -1), ix).to(torch.int32).reshape(NC, 1, S, TOPK)
        self.idx_dev = to_mesh(md, ix, ttnn.uint32)
        sk = torch.zeros(NC, 1, 8 * S, 1)
        for c in range(NC):
            for h in range(8):
                sk[c, 0, h * S : (h + 1) * S, 0] = self.sl[8 * c + h]
        self.sink_dev = to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sr = torch.zeros(NC, 1, 8 * S, 544)
        sr[:, :, :, 512] = sk[:, :, :, 0]
        sr[:, :, :, 513] = 1.0
        sr[:, :, :, 514] = 1.0
        self.sinkrow_dev = to_mesh(md, sr, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sr3 = torch.zeros(
            NC, S, 32, 1
        )  # tile-output owner layout: sink logit per head row (8 real rows), same for every s
        for c in range(NC):
            sr3[c, :, :8, 0] = self.sl[8 * c : 8 * c + 8][None, :]
        self.sinkrow3_dev = to_mesh(md, sr3, ttnn.bfloat16, ttnn.TILE_LAYOUT)  # [1,S,32,1]
        if fmt == "scaled":
            self.kv_dev, deq = pack_scaled_mesh(md, kv_cols)
            kv_ref_cols = deq
        else:
            self.kv_dev = to_mesh(md, kv_cols.reshape(NC, 1, Tl, 576), ttnn.bfloat16)
            kv_ref_cols = kv_cols
        # natural reference pool
        t = np.arange(T)
        _, col, lr = SQ.locate(t)
        kv_nat = kv_ref_cols.float().numpy()[col, lr]
        self.ref = ref_sparse(q, torch.from_numpy(kv_nat).to(torch.bfloat16), ids, self.sl)  # [64,S,512]
        self.ref_bf16 = ref_sparse(q, kv, ids, self.sl) if fmt == "scaled" else None
        self.S, self.ctx, self.kind, self.fmt = S, ctx, kind, fmt

    def free(self):
        for t in (self.q_dev, self.idx_dev, self.sink_dev, self.sinkrow_dev, self.sinkrow3_dev, self.kv_dev):
            ttnn.deallocate(t)


def result_to_heads(md, out, S, tile=False):
    r = from_mesh(md, out).float()
    if (
        tile and S == 1 and os.environ.get("HY4_ATTN_HM", "0") == "1" and r.numel() == NC * 8 * 512
    ):  # B26: head-major [8c,8,1,512] (HM writer)
        return r.reshape(NC * 8, 1, 512)
    if tile:  # [8c,S,32,512] (8 real head rows)
        return r.reshape(NC, S, 32, 512)[:, :, :8].permute(0, 2, 1, 3).reshape(64, S, 512)
    return r.reshape(NC, 8, S, 512).reshape(64, S, 512)  # [8c,1,8S,512] rows (h,s)


def mk_block(md, S, **over):
    kw = {}
    if os.environ.get("BLK_G"):
        kw["G"], kw["kc"] = int(os.environ["BLK_G"]), int(os.environ["BLK_KC"])
    kw.setdefault("v2", os.environ.get("V2", "1") == "1")
    kw.setdefault("tile", os.environ.get("TILE", "1") == "1")
    kw.setdefault("softmax", os.environ.get("SOFTMAX", "0") == "1")
    kw.setdefault("safe", os.environ.get("SAFE", "0") == "1")
    kw.setdefault("num_links", int(os.environ.get("NL", "2")))
    kw.setdefault("fid", os.environ.get("FID", "hifi4"))
    kw.update(over)
    return Block(md, S, **kw)


def run_sweep(md):
    """knob sweep of the whole block (traced us): kernel split G/kc, fidelity, links; balanced random keys, ctx 64k, bf16 and scaled; plus the one-column worst case"""
    for fmt in ("bf16", "scaled"):
        for S in (1, 2):
            for skew in (False, True):
                c = Case(md, "random", 65536, S, fmt, seed=5, skew=skew)
                for G, kc, fid, nl in [
                    (8, 128, "hifi4", 2),
                    (4, 256, "hifi4", 2),
                    (8, 256, "hifi4", 2),
                    (8, 128, "hifi2", 2),
                    (4, 256, "hifi2", 2),
                    (8, 128, "hifi4", 1),
                    (8, 128, "hifi4", 4),
                ]:
                    if S * 8 * G > 120:
                        continue
                    try:
                        blk = mk_block(md, S, G=G, kc=kc, fid=fid, num_links=nl)
                        idx8 = blk.idx_rows(c.idx_dev)
                        us = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, sink_of(blk, c)))
                        emit(
                            {
                                "test": "sweep",
                                "fmt": fmt,
                                "S": S,
                                "skew": skew,
                                "G": G,
                                "kc": kc,
                                "fid": fid,
                                "nl": nl,
                                "us": us,
                            }
                        )
                    except Exception as e:
                        emit(
                            {
                                "test": "sweep",
                                "fmt": fmt,
                                "S": S,
                                "skew": skew,
                                "G": G,
                                "kc": kc,
                                "fid": fid,
                                "nl": nl,
                                "error": str(e).strip().splitlines()[0][:160],
                            }
                        )
                        try:
                            ttnn.end_trace_capture(md, 0, cq_id=0)
                        except Exception:
                            pass
                c.free()


def sink_of(blk, c):
    return c.sinkrow3_dev if blk.tile else (c.sinkrow_dev if blk.v2 else c.sink_dev)


def run_pcc(md):
    cases = [
        ("random", 65536, 1, "bf16"),
        ("real", 65536, 1, "bf16"),
        ("random", 65536, 2, "bf16"),
        ("real", 65536, 2, "bf16"),
        ("random", 3000, 1, "bf16"),
        ("random", 100, 1, "bf16"),
        ("real", 100, 2, "bf16"),
        ("random", 65536, 1, "scaled"),
        ("real", 65536, 2, "scaled"),
        ("real", 100, 1, "scaled"),
        ("random", 1048576, 1, "scaled"),
        ("hot", 65536, 1, "bf16"),
        ("hot", 65536, 2, "scaled"),
    ]
    sel = os.environ.get("CASES")
    for kind, ctx, S, fmt in cases:
        tag = f"{kind}_{ctx}_S{S}_{fmt}"
        if sel and tag not in sel.split(","):
            continue
        try:
            c = Case(md, kind, ctx, S, fmt, seed=3)
            for safe in (True, False):
                blk = mk_block(md, S, safe=safe)
                idx8 = blk.idx_rows(c.idx_dev)
                out = blk.run(c.q_dev, c.kv_dev, idx8, sink_of(blk, c))
                got = result_to_heads(md, out, S, blk.tile)
                if safe:
                    emit(
                        {
                            "test": "pcc_safe_max",
                            "case": tag,
                            "pcc": round(pcc(got, c.ref), 5),
                            "max_logit": c.max_logit,
                        }
                    )
                    continue
            p = pcc(got, c.ref)
            per_row = min(pcc(got[:, s], c.ref[:, s]) for s in range(S))
            per_head = min(pcc(got[h], c.ref[h]) for h in range(64))
            r = {
                "test": "pcc",
                "case": tag,
                "pcc": round(p, 5),
                "pcc_min_row": round(per_row, 5),
                "pcc_min_head": round(per_head, 5),
                "pass": p >= GATE and per_row >= GATE,
            }
            if c.ref_bf16 is not None:
                r["pcc_vs_bf16_full"] = round(pcc(got, c.ref_bf16), 5)
            r["max_logit"] = c.max_logit
            r["empty_col_rows"] = int((torch.from_numpy(c.cnt) == 0).sum())
            emit(r)
            c.free()
        except Exception as e:
            import traceback

            traceback.print_exc()
            emit({"test": "pcc", "case": tag, "error": str(e).strip().splitlines()[0][:300]})


def run_replay(md):
    """trace replay correctness: two blocks (different inputs) captured back to back in ONE trace, replayed 3x, outputs compared with the torch reference (the CCL ping-pong semaphores flip inside the trace)"""
    for fmt, S in (("bf16", 1), ("scaled", 2)):
        c1 = Case(md, "real", 65536, S, fmt, seed=11)
        c2 = Case(md, "random", 65536, S, fmt, seed=12)
        blk = mk_block(md, S)
        i1, i2 = blk.idx_rows(c1.idx_dev), blk.idx_rows(c2.idx_dev)
        for _ in range(2):
            blk.run(c1.q_dev, c1.kv_dev, i1, sink_of(blk, c1))
            blk.run(c2.q_dev, c2.kv_dev, i2, sink_of(blk, c2))
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        o1 = blk.run(c1.q_dev, c1.kv_dev, i1, sink_of(blk, c1))
        o2 = blk.run(c2.q_dev, c2.kv_dev, i2, sink_of(blk, c2))
        ttnn.end_trace_capture(md, tid, cq_id=0)
        res = []
        for rep in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            res.append(
                (
                    round(pcc(result_to_heads(md, o1, S, True), c1.ref), 5),
                    round(pcc(result_to_heads(md, o2, S, True), c2.ref), 5),
                )
            )
        ttnn.release_trace(md, tid)
        emit(
            {
                "test": "replay",
                "fmt": fmt,
                "S": S,
                "pcc_per_replay_(case1,case2)": res,
                "pass": all(min(r) >= GATE for r in res),
            }
        )
        c1.free()
        c2.free()


def run_lat(md):
    for fmt in ("bf16", "scaled"):
        for S in (1, 2):
            c = Case(md, "random", 65536, S, fmt, seed=5)
            blk = mk_block(md, S)
            idx8 = blk.idx_rows(c.idx_dev)
            prev = 0.0
            for upto, name in [
                (1, "q_allgather"),
                (2, "+kernel"),
                (3, "+to_owner"),
                (4, "+all_to_all"),
                (99, "+merge"),
            ]:
                us = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, sink_of(blk, c), upto=upto))
                emit(
                    {
                        "test": "lat",
                        "fmt": fmt,
                        "S": S,
                        "stage": name,
                        "cum_us": us,
                        "delta_us": round(us - prev, 1),
                        "G": blk.G,
                        "kc": blk.kc,
                    }
                )
                prev = us
            us = timed_traced(md, lambda: blk.idx_rows(c.idx_dev))
            emit({"test": "lat", "fmt": fmt, "S": S, "stage": "idx_rows(once per index layer)", "us": us})
            c.free()


if __name__ == "__main__":
    full, md = open_row()
    which = os.environ.get("TEST", "pcc")
    try:
        for name, fn in [("pcc", run_pcc), ("replay", run_replay), ("lat", run_lat), ("sweep", run_sweep)]:
            if which in (name, "all"):
                try:
                    fn(md)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    emit({"test": name, "error": str(e).strip().splitlines()[0][:300]})
    finally:
        close_row(full, md)
