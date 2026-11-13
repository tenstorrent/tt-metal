# prefill-roof X10/X12: indexer prefill score (bf16 q / bf8 q) + local top-k (k_loc) per chunk of R query rows at n local keys per column. Env: NS, R (512|1024), KLOC (512).
import json

import torch
from idx_common import *
from idx_pipe import *

import ttnn


def main():
    ns = [int(x) for x in os.environ.get("NS", "8192,131072").split(",")]
    R = int(os.environ.get("R", "512"))
    kl = int(os.environ.get("KLOC", "512"))
    md = open_md()
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
    for n in ns:
        k = up(md, torch.randn(8, 1, n, DIM).to(torch.bfloat16), ttnn.bfloat8_b, shard=True)
        qt = torch.randn(1, HEADS, R, DIM).to(torch.bfloat16)
        w = up(md, torch.rand(1, 1, R, HEADS).to(torch.bfloat16))
        r = {"R": R, "n_local": n, "total": n * COLS}
        for nm, dt in (("bf16q", ttnn.bfloat16), ("bf8q", ttnn.bfloat8_b)):
            q = up(md, qt, dt)
            f = lambda: ttnn.experimental.indexer_score_dsa(
                q, k, w, chunk_start_idx=n - 32, kv_len=n, program_config=cfg, cache_batch_idx=0
            )
            r["score_" + nm + "_us"] = round(traced_us(md, f, n=3, reps=2), 1)
            if nm == "bf16q":
                sc = f()
                r["topk_local_us"] = round(traced_us(md, lambda: topk_ids(sc, kl, n), n=3, reps=2), 1)
                ttnn.deallocate(sc)
            ttnn.deallocate(q)
        r["per_row_score_bf16_ns"] = round(r["score_bf16q_us"] * 1e3 / R, 1)
        r["per_row_topk_ns"] = round(r["topk_local_us"] * 1e3 / R, 1)
        print("R", json.dumps(r), flush=True)
        ttnn.deallocate(k)
    close_md(md)


main()
