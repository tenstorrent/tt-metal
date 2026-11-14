"""fuse-sparse-merge B: whole decode indexer chain (score -> select_v2) with the fused decode score kernel vs the stock score; exactness (tie-only, overlap vs fp32 torch top-2048) + traced latency per layer.
Run: run_fsm.sh <tag> test_hy4_idx_chain_device TOTALS=16384,65536,262144,1048576 SS=1,2 MODES=random,heavy,recent"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probes", "idx_1m"))
from dk_dev import pcc, timed_traced
from dm_block import close_row, open_row
import hy4_idx_score as IS
from idx_pipe import Ccl, colid_tensor, select_v2
from idx_common import gid_of

import ttnn

COLS, HEADS, DIM, TOPK = 8, 32, 128, 2048
TOTALS = [int(x) for x in os.environ.get("TOTALS", "16384,65536,262144,1048576").split(",")]
SS = [int(x) for x in os.environ.get("SS", "1,2").split(",")]
MODES = os.environ.get("MODES", "random,heavy,recent").split(",")
CFG = tuple(int(v) for v in os.environ.get("CFG", "16:512:512:4").split(":"))  # B:kb:kloc:split


def emit(d):
    print("RES " + json.dumps(d), flush=True)


def mk(md, t, dt, lay, shard=False):
    mm = ttnn.ShardTensorToMesh(md, dim=0) if shard else ttnn.ReplicateTensorToMesh(md)
    return ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mm)


def data(total, S, mode, seed):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(total, DIM, generator=g)
    q = torch.randn(HEADS, S, DIM, generator=g)
    w = torch.rand(S, HEADS, generator=g) * 0.2
    if mode == "heavy":
        G = G * torch.exp(torch.randn(total, 1, generator=g))
        for s in range(S):
            hot = torch.randint(0, total, (400,), generator=g)
            G[hot] += 6 * q[:, s, :].mean(0, keepdim=True)
        G[total - 700 :] += 3.2 * q[:, 0, :].mean(0, keepdim=True)
    if mode == "recent":
        G[total - 2048 :] += 8 * q.mean((0, 1))[None]
    return G, q, w


def main():
    full, md = open_row()
    ccl, colid = Ccl(md), colid_tensor(md)
    B0, kb, kl, sp = CFG
    try:
        for total in TOTALS:
            n = total // COLS
            B = max(1, min(B0, n // kb))  # blocks of >= kb keys (16k total: 4 blocks)
            for S in SS:
                for mode in MODES:
                    G, q, w = data(total, S, mode, 3 + S)
                    gid = gid_of(torch.arange(COLS)[:, None], torch.arange(n)[None, :], "slice64")  # [8,n] global token id of local slot
                    kd = mk(md, G[gid].reshape(COLS, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard=True)
                    kdeq = ttnn.to_torch(kd, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(COLS, n, DIM)
                    qh = q.permute(1, 0, 2).to(torch.bfloat16)
                    q_my = mk(md, qh.reshape(S, 1, HEADS, DIM), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    wp = torch.zeros(1, 1, 32, 32)
                    wp[0, 0, :S] = w
                    w_d = mk(md, wp.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    # fp32 torch oracle over the global pool (dequantised keys, bf16 q, bf16 weights)
                    ref = torch.zeros(S, total)
                    for s in range(S):
                        for c in range(COLS):
                            sc = torch.relu(qh[s].float() @ kdeq[c].T)
                            ref[s, gid[c]] = (sc * w[s].to(torch.bfloat16).float()[:, None]).sum(0)
                    score = IS.idx_score(q_my, kd, w_d, n, S)  # [1,1,S,n] per column
                    sc_all = ttnn.to_torch(score, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(COLS, S, n)
                    dev_scores = torch.zeros(S, total)
                    for c in range(COLS):
                        dev_scores[:, gid[c]] = sc_all[c]
                    ids = select_v2(ccl, colid, score, n, B, kb, kl, TOPK, sp, False, total)
                    idh = ttnn.to_torch(ids, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(COLS, S, TOPK).long()
                    r = {"test": "idx_chain_exact", "total": total, "S": S, "mode": mode, "cols_identical": bool((idh == idh[0:1]).all())}
                    idh = idh[0]
                    tie_rows, ov32, valid_rows = 0, [], 0
                    for s in range(S):
                        dv = idh[s]
                        dv = dv[dv < total]
                        orc = torch.topk(dev_scores[s], TOPK).indices
                        a, b = set(dv.tolist()), set(orc.tolist())
                        thr = dev_scores[s][orc].min()
                        mm = a - b
                        tie_rows += int(all(float(dev_scores[s][m]) == float(thr) for m in mm) and len(dv) == TOPK and len(set(dv.tolist())) == TOPK)
                        o32 = torch.topk(ref[s], TOPK).indices
                        ov32.append(round(len(a & set(o32.tolist())) / TOPK, 4))
                    r.update({"tie_only_rows": tie_rows, "of": S, "overlap_vs_fp32_torch": ov32, "score_pcc": round(pcc(dev_scores, ref), 6)})
                    emit(r)
                    if mode == MODES[0]:
                        us_sc = timed_traced(md, lambda: IS.idx_score(q_my, kd, w_d, n, S))
                        us_sel = timed_traced(md, lambda: select_v2(ccl, colid, score, n, B, kb, kl, TOPK, sp, False, total))
                        us_all = timed_traced(md, lambda: select_v2(ccl, colid, IS.idx_score(q_my, kd, w_d, n, S), n, B, kb, kl, TOPK, sp, False, total))
                        qs = torch.zeros(1, HEADS, 32, DIM)
                        qs[0, :, :S] = q
                        q_st = mk(md, qs.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)

                        def stock():
                            sc = ttnn.experimental.indexer_score_dsa(q_st, kd, w_d, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0)
                            rows = ttnn.slice(ttnn.reshape(sc, [1, 1, 32, n]), [0, 0, 0, 0], [1, 1, S, n])
                            return select_v2(ccl, colid, rows, n, B, kb, kl, TOPK, sp, False, total)

                        us_stock = timed_traced(md, stock)
                        emit({"test": "idx_chain_lat", "total": total, "S": S, "cfg": CFG, "score_us": us_sc, "select_us": us_sel, "chain_fused_score_us": us_all, "chain_stock_score_us": us_stock})
                        ttnn.deallocate(q_st)
                    for t in (kd, q_my, w_d, score, ids):
                        ttnn.deallocate(t)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
