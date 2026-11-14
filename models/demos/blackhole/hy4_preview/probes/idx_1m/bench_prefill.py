# Prefill chunk (512 query rows) indexer stages per column: n_local history keys. Env: NS, KLOCS
import json

import torch
from idx_common import *
from idx_pipe import *

import ttnn


def main():
    ns = [int(x) for x in os.environ.get("NS", "2048,8192,32768,65536,131072").split(",")]
    klocs = [int(x) for x in os.environ.get("KLOCS", "2048,512").split(",")]
    R = 512
    md = open_md()
    ccl = Ccl(md)
    colid = colid_tensor(md)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    for n in ns:
        total = n * COLS
        k = up(md, torch.randn(8, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, shard=True)
        q = up(md, torch.randn(1, HEADS, R, DIM).to(torch.bfloat16))
        w = up(md, torch.rand(1, 1, R, HEADS).to(torch.bfloat16))
        r = {"total": total, "n_local": n, "R": R}
        f_score = lambda: ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
        )
        r["score_us"] = traced_us(md, f_score, n=3, reps=2)
        sc = f_score()
        print("R", json.dumps(r), flush=True)
        for kl in klocs:
            if kl > n:
                continue
            r2 = {"total": total, "n_local": n, "R": R, "k_loc": kl}
            try:
                r2["topk_local_us"] = traced_us(md, lambda: topk_ids(sc, kl, n), n=3, reps=2)
                lids = ttnn.reshape(topk_ids(sc, kl, n), [R, 1, 1, kl])
                sr = ttnn.reshape(sc, [R, 1, 1, n])
                r2["cand_us"] = traced_us(md, lambda: cs_cand(sr, lids, colid, 1), n=3, reps=2)
                vals, gids = cs_cand(sr, lids, colid, 1)
                r2["ag_vals_us"] = traced_us(md, lambda: ccl.ag(vals), n=3, reps=2)
                r2["ag_gids_us"] = traced_us(md, lambda: ccl.ag(gids), n=3, reps=2)
                cv, ci = ccl.ag(vals), ccl.ag(gids)
                W = int(cv.shape[-1])
                cvr = ttnn.reshape(cv, [1, 1, R, W])
                kk = TOPK
                r2["topk_merge_us"] = traced_us(md, lambda: topk_ids(cvr, kk), n=3, reps=2)
                pos = ttnn.reshape(topk_ids(cvr, kk), [R, 1, 1, kk])
                r2["final_us"] = traced_us(md, lambda: cs_final(pos, ci, total), n=3, reps=2)
                ttnn.deallocate(cv)
                ttnn.deallocate(ci)
            except Exception as e:
                r2["err"] = str(e)[:300]
            print("R", json.dumps(r2), flush=True)
        ttnn.deallocate(sc)
        ttnn.deallocate(k)
    close_md(md)


main()
