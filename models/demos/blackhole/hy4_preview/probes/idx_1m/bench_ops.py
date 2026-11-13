# Op-level latency: indexer_score_dsa and topk_large_indices at Hy4 shapes (per-column local keys n). Prints "R <json>" lines.
import json

import torch
from idx_common import *

import ttnn


def main():
    ns = [int(x) for x in os.environ.get("NS", "2048,8192,32768,65536,131072").split(",")]
    md = open_md()
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    torch.manual_seed(0)
    for n in ns:
        res = {"n_local": n, "total": n * 8}
        k = up(md, torch.randn(8, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, shard=True)
        for Sq in (32, 512):
            if Sq == 512 and n > 131072:
                continue
            q = up(md, torch.randn(1, HEADS, Sq, DIM).to(torch.bfloat16))
            w = up(md, torch.rand(1, 1, Sq, HEADS).to(torch.bfloat16))
            for hg in (0,):
                cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=hg)
                try:
                    f = lambda: ttnn.experimental.indexer_score_dsa(
                        q, k, w, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
                    )
                    res[f"score_Sq{Sq}_hg{hg}_us"] = traced_us(md, f, n=3 if Sq == 512 else 5)
                except Exception as e:
                    res[f"score_Sq{Sq}_hg{hg}_err"] = str(e)[:200]
            sc = ttnn.experimental.indexer_score_dsa(
                q, k, w, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
            )
            print("R", json.dumps(res), flush=True)
            ttnn.deallocate(sc)
        # top-k on row-major bf16 rows
        for R in (1, 2, 512):
            if R == 512 and n > 131072:
                continue
            x = up(md, torch.randn(1, 1, R, n).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            for kk in (512, 1024, 2048):
                if kk > n:
                    continue
                try:
                    f = lambda: ttnn.experimental.topk_large_indices(x, k=kk, valid_length=n)
                    res[f"topk_R{R}_k{kk}_us"] = traced_us(md, f, n=3 if R == 512 else 5)
                except Exception as e:
                    res[f"topk_R{R}_k{kk}_err"] = str(e)[:200]
            print("R", json.dumps(res), flush=True)
            ttnn.deallocate(x)
        # blockwise: one row of n viewed as B rows of n/B
        for B in (4, 8, 16, 32, 64):
            if n // B < 2048:
                continue
            x = up(md, torch.randn(1, 1, B, n // B).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            for kk in (512, 2048):
                f = lambda: ttnn.experimental.topk_large_indices(x, k=kk, valid_length=n // B)
                try:
                    res[f"topk_blk{B}_k{kk}_us"] = traced_us(md, f)
                except Exception as e:
                    res[f"topk_blk{B}_k{kk}_err"] = str(e)[:200]
            ttnn.deallocate(x)
        print("R", json.dumps(res), flush=True)
        ttnn.deallocate(k)
    close_md(md)


main()
