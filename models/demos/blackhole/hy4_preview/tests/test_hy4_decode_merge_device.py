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
from dk_dev import pcc, ref_sparse, timed_traced
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
    def __init__(self, md, kind, ctx, S, fmt, seed=0, topk_valid=None):
        n_pages = max(1, -(-ctx // 512))
        T = n_pages * 512
        kv, q = DK.gen_data(kind, T, S, seed=seed)  # q [64,S,576]
        ids = torch.full((S, TOPK), -1, dtype=torch.int64)
        for s in range(S):
            n = min(topk_valid or TOPK, ctx)
            ids[s, :n] = torch.randperm(ctx)[:n]
        q = DK.align_q_to_keys(kind, q, kv, ids)
        self.sl = torch.linspace(-2, 2, 64)
        ids_u = torch.where(ids < 0, torch.full_like(ids, SQ.SENTINEL), ids)
        loc, self.cnt = SQ.split_ids(ids_u.numpy().astype(np.int64))  # [8,S,K]
        kv_cols = torch.from_numpy(SQ.shard_kv(kv.float().numpy(), n_pages)).to(torch.bfloat16)  # [8,Tl,576]
        Tl = kv_cols.shape[1]
        qp = torch.zeros(NC, 32, S, 576, dtype=torch.bfloat16)
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
        for t in (self.q_dev, self.idx_dev, self.sink_dev, self.kv_dev):
            ttnn.deallocate(t)


def result_to_heads(md, out, S):
    r = from_mesh(md, out).float()  # [8c,1,8S,512] rows (h,s)
    return r.reshape(NC, 8, S, 512).reshape(64, S, 512)


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
    ]
    sel = os.environ.get("CASES")
    for kind, ctx, S, fmt in cases:
        tag = f"{kind}_{ctx}_S{S}_{fmt}"
        if sel and tag not in sel.split(","):
            continue
        try:
            c = Case(md, kind, ctx, S, fmt, seed=3)
            blk = Block(md, S)
            idx8 = blk.idx_rows(c.idx_dev)
            out = blk.run(c.q_dev, c.kv_dev, idx8, c.sink_dev)
            got = result_to_heads(md, out, S)
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
            r["empty_col_rows"] = int((torch.from_numpy(c.cnt) == 0).sum())
            emit(r)
            c.free()
        except Exception as e:
            import traceback

            traceback.print_exc()
            emit({"test": "pcc", "case": tag, "error": str(e).strip().splitlines()[0][:300]})


def run_lat(md):
    for fmt in ("bf16", "scaled"):
        for S in (1, 2):
            c = Case(md, "random", 65536, S, fmt, seed=5)
            blk = Block(md, S)
            idx8 = blk.idx_rows(c.idx_dev)
            prev = 0.0
            for upto, name in [
                (1, "q_allgather"),
                (2, "+kernel"),
                (3, "+to_owner"),
                (4, "+all_to_all"),
                (99, "+merge"),
            ]:
                us = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sink_dev, upto=upto))
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
        for name, fn in [("pcc", run_pcc), ("lat", run_lat)]:
            if which in (name, "all"):
                try:
                    fn(md)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    emit({"test": name, "error": str(e).strip().splitlines()[0][:300]})
    finally:
        close_row(full, md)
