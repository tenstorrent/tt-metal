# Decode select v2 (blockwise local top-k, multi-core cand/gather kernels): latency per stage + exactness vs oracle. Env: TOTALS, SMOD, CFGS ("B:kb:kloc:split;...")
import json

import torch
from idx_common import *
from idx_pipe import *

import ttnn


def make_data(total, mode, gen):
    N = total
    G = torch.randn(N, DIM, generator=gen)
    if mode == "heavy":
        G = G * torch.exp(1.0 * torch.randn(N, 1, generator=gen))
    q = torch.randn(HEADS, 32, DIM, generator=gen)
    w = torch.rand(32, HEADS, generator=gen) * 0.2
    if mode == "heavy":
        for s in range(32):
            hot = torch.randint(0, N, (400,), generator=gen)
            G[hot] += 6 * q[:, s, :].mean(0, keepdim=True)
        G[N - 700 :] += 3.2 * q[:, 0, :].mean(0, keepdim=True)
    if mode == "recent":
        G[N - 2048 :] += 8 * q[:, :, :].mean((0, 1))[None]
    return G, q, w


def main():
    totals = [int(x) for x in os.environ.get("TOTALS", "65536,262144,1048576").split(",")]
    modes = os.environ.get("SMOD", "random,heavy,recent").split(",")
    cfgs = [
        tuple(int(v) for v in c.split(":"))
        for c in os.environ.get("CFGS", "16:512:1024:4;16:512:512:4;32:512:512:4;32:256:512:4").split(";")
    ]
    md = open_md()
    ccl = Ccl(md)
    colid = colid_tensor(md)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    gen = torch.Generator().manual_seed(1)
    for total in totals:
        n = total // COLS
        for mode in modes:
            G, q, w = make_data(total, mode, gen)
            c_ = torch.arange(COLS)[:, None]
            l_ = torch.arange(n)[None, :]
            gid = gid_of(c_, l_, "slice64")
            k = up(md, G[gid].reshape(COLS, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, shard=True)
            qd = up(md, q[None].to(torch.bfloat16))
            wd = up(md, w[None, None].to(torch.bfloat16))
            f_score = lambda: ttnn.experimental.indexer_score_dsa(
                qd, k, wd, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
            )
            sc = f_score()
            scf = ttnn.to_torch(sc, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float().reshape(COLS, 32, n)
            dev_scores = torch.zeros(32, total)
            dev_scores[:, gid.reshape(-1)] = scf.permute(1, 0, 2).reshape(32, -1)
            orc = [torch.topk(dev_scores[s], TOPK).indices for s in range(32)]
            rows = ttnn.reshape(sc, [1, 1, 32, n])
            for B, kb, kl, sp in cfgs:
                r = {"total": total, "mode": mode, "B": B, "kb": kb, "kloc": kl, "split": sp}
                try:
                    for sort_final in (False, True):
                        ids = select_v2(ccl, colid, rows, n, B, kb, kl, TOPK, sp, sort_final, total)
                        idh = (
                            ttnn.to_torch(ids, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                            .reshape(COLS, 32, TOPK)
                            .long()
                        )
                        r["cols_identical"] = bool((idh == idh[0:1]).all())
                        idh = idh[0]
                        exact = tie = 0
                        mism = 0
                        for s in range(32):
                            dv = idh[s]
                            dv = dv[dv < total]
                            a, b = set(dv.tolist()), set(orc[s].tolist())
                            thr = dev_scores[s][orc[s]].min()
                            mm = a - b
                            exact += len(mm) == 0
                            mism += len(mm)
                            tie += all(float(dev_scores[s][m]) == float(thr) for m in mm) and len(dv) == TOPK
                        r[f"sorted{int(sort_final)}"] = {
                            "set_exact_rows": exact,
                            "tie_only_rows": tie,
                            "mismatch": mism,
                            "asc": bool((idh[:, 1:] > idh[:, :-1]).all()) if sort_final else None,
                        }
                except Exception as e:
                    r["err"] = str(e).split("\n")[0][:200]
                print("R", json.dumps(r), flush=True)
            # latency (S = 1, 2)
            for S in (1, 2):
                rs = ttnn.slice(sc, [0, 0, 0, 0], [1, 1, S, n])
                base = {"total": total, "mode": mode, "S": S, "score_us": traced_us(md, f_score)}
                if mode != "random":
                    continue
                for B, kb, kl, sp in cfgs:
                    r = dict(base, B=B, kb=kb, kloc=kl, split=sp)
                    try:
                        m = n // B
                        blk = ttnn.reshape(rs, [1, 1, S * B, m])
                        r["blk_topk_us"] = traced_us(md, lambda: topk_ids(blk, kb, m))
                        bids = ttnn.reshape(topk_ids(blk, kb, m), [S * B, 1, 1, kb])
                        scr = ttnn.reshape(rs, [S * B, 1, 1, m])
                        r["cand2_us"] = traced_us(md, lambda: cs_cand2(scr, bids, colid, 1, m, B, sp))
                        bv, bg = cs_cand2(scr, bids, colid, 1, m, B, sp)
                        cvals = ttnn.reshape(bv, [1, 1, S, B * kb])
                        r["local_merge_topk_us"] = traced_us(md, lambda: topk_ids(cvals, kl))
                        pos = ttnn.reshape(topk_ids(cvals, kl), [S, 1, 1, kl])
                        cg, cv = ttnn.reshape(bg, [S, 1, 1, B * kb]), ttnn.reshape(bv, [S, 1, 1, B * kb])
                        r["local_gather_us"] = traced_us(md, lambda: cs_gather(pos, cg, cv, sp))
                        lids, lvals = cs_gather(pos, cg, cv, sp)
                        r["ag_vals_us"] = traced_us(md, lambda: ccl.ag(lvals))
                        r["ag_ids_us"] = traced_us(md, lambda: ccl.ag(lids))
                        av, ai = ccl.ag(lvals), ccl.ag(lids)
                        W = int(av.shape[-1])
                        avr = ttnn.reshape(av, [1, 1, S, W])
                        r["merge_topk_us"] = traced_us(md, lambda: topk_ids(avr, TOPK))
                        pos2 = ttnn.reshape(topk_ids(avr, TOPK), [S, 1, 1, TOPK])
                        r["final_gather_us"] = traced_us(md, lambda: cs_gather(pos2, ai, None, sp))
                        r["final_sorted_us"] = traced_us(md, lambda: cs_final(pos2, ai, total))
                        r["chain_select_unsorted_us"] = traced_us(
                            md, lambda: select_v2(ccl, colid, rs, n, B, kb, kl, TOPK, sp, False, total)
                        )
                        r["chain_select_sorted_us"] = traced_us(
                            md, lambda: select_v2(ccl, colid, rs, n, B, kb, kl, TOPK, sp, True, total)
                        )
                        r["chain_total_unsorted_us"] = traced_us(
                            md,
                            lambda: select_v2(
                                ccl,
                                colid,
                                ttnn.slice(f_score(), [0, 0, 0, 0], [1, 1, S, n]),
                                n,
                                B,
                                kb,
                                kl,
                                TOPK,
                                sp,
                                False,
                                total,
                            ),
                        )
                    except Exception as e:
                        r["err"] = str(e).split("\n")[0][:200]
                    print("R", json.dumps(r), flush=True)
            ttnn.deallocate(k)
    close_md(md)


main()
