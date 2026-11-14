"""fuse-sparse-merge B1: decode indexer score kernel (tt/hy4_idx_score.py) vs torch (dequantised bf8 keys) and vs the stock indexer_score_dsa, PCC + top-2048 overlap + latency. 1x8 slot, all chips run the same data (replicated).
Run: run_fsm.sh <tag> test_hy4_idx_score_device TEST=pcc|lat [NS=131072,32768] [SS=1,2]"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
from dk_dev import pcc, timed_traced
from dm_block import close_row, open_row

import ttnn
import hy4_idx_score as IS

NS = [int(x) for x in os.environ.get("NS", "131072,32768,8192,2048").split(",")]
SS = [int(x) for x in os.environ.get("SS", "1,2").split(",")]
HEADS, DIM, TOPK = 32, 128, 2048


def emit(d):
    print("RES " + json.dumps(d), flush=True)


def up(md, t, dt, lay):
    return ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))


def dev0(md, t):
    r = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
    return r[: r.shape[0] // 8].float()


def data(n, S, mode, seed=1):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n, DIM, generator=g)
    q = torch.randn(HEADS, S, DIM, generator=g)
    w = torch.rand(S, HEADS, generator=g) * 0.2
    if mode == "heavy":
        k = k * torch.exp(torch.randn(n, 1, generator=g))
        for s in range(S):
            hot = torch.randint(0, n, (400,), generator=g)
            k[hot] += 6 * q[:, s, :].mean(0, keepdim=True)
    if mode == "recent":
        k[n - 2048 :] += 8 * q.mean((0, 1))[None]
    return k, q, w


def run(md, n, S, mode, kv_len=None, lat=False):
    kv_len = kv_len or n
    k, q, w = data(n, S, mode)
    kd = up(md, k.reshape(1, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    kdeq = dev0(md, kd).reshape(n, DIM)
    qh = q.permute(1, 0, 2).to(torch.bfloat16)  # [S,32,128]
    q_my = up(md, qh.reshape(S, 1, HEADS, DIM), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    wp = torch.zeros(1, 1, 32, 32)
    wp[0, 0, :S, :] = w
    w_d = up(md, wp.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = IS.idx_score(q_my, kd, w_d, kv_len, S)
    got = dev0(md, out).reshape(S, n)
    qf = qh.float()
    ref = torch.zeros(S, n)
    for s in range(S):
        sc = torch.relu(qf[s] @ kdeq.T)  # [32,n]
        ref[s] = (sc * w[s].to(torch.bfloat16).float()[:, None]).sum(0)
    r = {"test": "idx_score_pcc", "n": n, "S": S, "mode": mode, "kv_len": kv_len}
    valid = torch.arange(n) < kv_len
    r["finite_valid"] = bool(torch.isfinite(got[:, valid]).all())
    r["masked_ok"] = bool((got[:, ~valid] == float("-inf")).all()) if (~valid).any() else True
    r["pcc"] = round(pcc(got[:, valid], ref[:, valid]), 6)
    ov = []
    for s in range(S):
        a = torch.topk(got[s, valid], min(TOPK, int(valid.sum()))).indices
        b = torch.topk(ref[s, valid], min(TOPK, int(valid.sum()))).indices
        ov.append(len(set(a.tolist()) & set(b.tolist())) / a.numel())
    r["top2048_overlap_vs_fp32ref"] = [round(x, 4) for x in ov]
    emit(r)
    if lat:
        us = timed_traced(md, lambda: IS.idx_score(q_my, kd, w_d, kv_len, S))
        # stock op on the same data (q [1,32,32,128] rows padded, causal geometry: chunk_start n-32)
        qs = torch.zeros(1, HEADS, 32, DIM)
        qs[0, :, :S] = q
        q_st = up(md, qs.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
        f = lambda: ttnn.experimental.indexer_score_dsa(q_st, kd, w_d, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0)
        try:
            us_st = timed_traced(md, f)
            st = dev0(md, f()).reshape(32, n)[:S]
            m = slice(0, n - 64)
            pst = round(pcc(got[:, m], st[:, m]), 6)
        except Exception as e:
            us_st, pst = None, str(e).splitlines()[0][:120]
        emit({"test": "idx_score_lat", "n": n, "S": S, "mine_us": us, "stock_us": us_st, "pcc_vs_stock": pst})
        ttnn.deallocate(q_st)
    for t in (kd, q_my, w_d, out):
        ttnn.deallocate(t)


if __name__ == "__main__":
    full, md = open_row()
    try:
        which = os.environ.get("TEST", "pcc")
        for n in NS:
            for S in SS:
                try:
                    if which in ("pcc", "all"):
                        modes = os.environ.get("MODES", "random,heavy,recent").split(",")
                        for mode in modes:
                            run(md, n, S, mode, lat=(which == "all" and mode == modes[0]))
                        if n >= 4096:
                            run(md, n, S, "random", kv_len=n - 100)
                    elif which == "lat":
                        run(md, n, S, "random", lat=True)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    emit({"test": "idx_score", "n": n, "S": S, "error": str(e).splitlines()[0][:300]})
    finally:
        close_row(full, md)
