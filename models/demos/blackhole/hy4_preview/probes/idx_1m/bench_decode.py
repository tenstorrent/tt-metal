# Decode-step indexer on the 1x8 slot: score + local top-k + merge, latency per stage and exactness. Env: TOTALS (total keys list), SMOD (data mode: random|heavy), SS (S rows list for timing)
import json

import torch
from idx_common import *
from idx_pipe import *

import ttnn

LAYOUT = 1


def make_data(total, mode, S, gen):
    N = total
    G = torch.randn(N, DIM, generator=gen)
    if mode == "heavy":
        G = G * torch.exp(1.0 * torch.randn(N, 1, generator=gen))  # lognormal row norms
    q = torch.randn(HEADS, 32, DIM, generator=gen)
    w = torch.rand(32, HEADS, generator=gen) * 0.2
    if mode == "heavy":
        # plant 4000 'hot' keys aligned with a mix of the query rows (needle-like), in blocks (recent-window like) and scattered
        for s in range(32):
            hot = torch.randint(0, N, (400,), generator=gen)
            G[hot] += 1.5 * q[:, s, :].mean(0, keepdim=True) * 4
        G[N - 700 :] += 0.8 * q[:, 0, :].mean(0, keepdim=True) * 4  # recent window block
    return G, q, w


def shard_keys(G, n):
    c = torch.arange(COLS)[:, None]
    l = torch.arange(n)[None, :]
    gid = gid_of(c, l, "slice64")  # [8,n]
    return G[gid].reshape(COLS, 1, n, DIM), gid


def main():
    totals = [int(x) for x in os.environ.get("TOTALS", "16384,65536,262144,524288,1048576").split(",")]
    modes = os.environ.get("SMOD", "random,heavy").split(",")
    SS = [int(x) for x in os.environ.get("SS", "1,2").split(",")]
    md = open_md()
    ccl = Ccl(md)
    colid = colid_tensor(md)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    gen = torch.Generator().manual_seed(1)
    for total in totals:
        n = total // COLS
        for mode in modes:
            res = {"total": total, "n_local": n, "mode": mode}
            G, q, w = make_data(total, mode, 32, gen)
            kloc, gid = shard_keys(G, n)
            k = up(md, kloc.to(torch.bfloat16), ttnn.bfloat8_b, shard=True)
            kq = (
                ttnn.to_torch(k, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(COLS, n, DIM)
            )  # dequantised
            Gq = torch.zeros(total, DIM)
            Gq[gid.reshape(-1)] = kq.reshape(-1, DIM)
            qd = up(md, q[None].to(torch.bfloat16))  # [1,32,32,128]
            wd = up(md, w[None, None].to(torch.bfloat16))  # [1,1,32,32]
            qf = ttnn.to_torch(qd, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:1].reshape(HEADS, 32, DIM)
            wf = ttnn.to_torch(wd, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:1].reshape(32, HEADS)
            ref = torch.zeros(32, total)
            for s in range(32):
                ref[s] = (torch.relu(qf[:, s, :] @ Gq.T) * wf[s][:, None]).sum(0)
            vis = (
                torch.arange(n)[None, :] <= (n - 32 + torch.arange(32))[:, None]
            )  # [32,n] local visibility (same for every column)
            for c in range(COLS):
                for s_ in range(32):
                    ref[s_, gid[c][~vis[s_]]] = float("-inf")
            # ---- exactness on all 32 query rows
            sc = ttnn.experimental.indexer_score_dsa(
                qd, k, wd, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
            )
            scf = (
                ttnn.to_torch(sc, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(COLS, 32, n)
            )  # device scores per column
            dev_scores = torch.zeros(32, total)
            dev_scores[:, gid.reshape(-1)] = scf.permute(1, 0, 2).reshape(32, -1)
            fin = torch.isfinite(ref)
            res["score_pcc"] = float(torch.corrcoef(torch.stack([dev_scores[fin], ref[fin]]))[0, 1])
            res["dev_finite_match"] = bool((torch.isfinite(dev_scores) == fin).all())
            rows = ttnn.reshape(sc, [1, 1, 32, n])
            ids = select_unblocked(ccl, colid, rows, n, TOPK, None, total, LAYOUT)
            idh = ttnn.to_torch(ids, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(COLS, 32, TOPK).long()
            res["cols_identical"] = bool((idh == idh[0:1]).all())
            idh = idh[0]
            ok_sets = ok_ties = 0
            mism_tot = 0
            ov_ref = 0.0
            for s in range(32):
                dv = idh[s]
                dv = dv[dv < total]
                orc = torch.topk(dev_scores[s], TOPK).indices
                thr = dev_scores[s][orc].min()
                a, b = set(dv.tolist()), set(orc.tolist())
                mism = a - b
                tie = all(float(dev_scores[s][m]) == float(thr) for m in mism)
                ok_sets += len(mism) == 0
                ok_ties += tie
                mism_tot += len(mism)
                ov_ref += len(a & set(torch.topk(ref[s], TOPK).indices.tolist())) / TOPK
                if s == 0:
                    res["sorted_ascending"] = bool((dv[1:] > dv[:-1]).all())
                    res["n_ids"] = int(dv.numel())
            res.update(
                rows=32,
                set_exact_rows=ok_sets,
                tie_only_rows=ok_ties,
                mismatch_ids=mism_tot,
                overlap_vs_fp32_ref=ov_ref / 32,
            )
            print("R", json.dumps(res), flush=True)
            ttnn.deallocate(sc)
            # ---- latency (traced), per stage and chain
            for S in SS:
                r2 = {"total": total, "n_local": n, "mode": mode, "S": S}
                sc = ttnn.experimental.indexer_score_dsa(
                    qd, k, wd, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
                )
                f_score = lambda: ttnn.experimental.indexer_score_dsa(
                    qd, k, wd, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
                )
                r2["score_us"] = traced_us(md, f_score)
                rs = ttnn.slice(sc, [0, 0, 0, 0], [1, 1, S, n])
                r2["slice_us"] = traced_us(md, lambda: ttnn.slice(sc, [0, 0, 0, 0], [1, 1, S, n]))
                r2["topk_local_us"] = traced_us(md, lambda: topk_ids(rs, TOPK, n))
                lids = ttnn.reshape(topk_ids(rs, TOPK, n), [S, 1, 1, TOPK])
                sr = ttnn.reshape(rs, [S, 1, 1, n])
                r2["cand_us"] = traced_us(md, lambda: cs_cand(sr, lids, colid, LAYOUT))
                vals, gids = cs_cand(sr, lids, colid, LAYOUT)
                r2["ag_vals_us"] = traced_us(md, lambda: ccl.ag(vals))
                r2["ag_gids_us"] = traced_us(md, lambda: ccl.ag(gids))
                cv, ci = ccl.ag(vals), ccl.ag(gids)
                W = int(cv.shape[-1])
                cvr = ttnn.reshape(cv, [1, 1, S, W])
                r2["topk_merge_us"] = traced_us(md, lambda: topk_ids(cvr, TOPK))
                pos = ttnn.reshape(topk_ids(cvr, TOPK), [S, 1, 1, TOPK])
                r2["final_us"] = traced_us(md, lambda: cs_final(pos, ci, total))
                r2["chain_select_us"] = traced_us(
                    md, lambda: select_unblocked(ccl, colid, rs, n, TOPK, None, total, LAYOUT)
                )
                r2["chain_total_us"] = traced_us(
                    md,
                    lambda: select_unblocked(
                        ccl, colid, ttnn.slice(f_score(), [0, 0, 0, 0], [1, 1, S, n]), n, TOPK, None, total, LAYOUT
                    ),
                )
                print("R", json.dumps(r2), flush=True)
            ttnn.deallocate(k)
    close_md(md)


main()
