# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MIRACL-en through the traced runner directly, no server: packed rows (TracedReranker.score_ids) vs one pair per row.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/miracl_runner.py miracl_eval.pt [key=value] [buckets]

Times 40 requests of 25 documents (one score_ids call per query), then all 1000 pairs in one call, and
reports Pearson / mean error / nDCG@10 against the eval set's HF fp32 scores. Optional: RerankConfig overrides
as for demo/bench.py --cfg, e.g. ``mm_fidelity=LoFi;lofi_kinds=ff2,ff13s`` (several separated by ``;``,
``none`` = no override), and the batch buckets, e.g. ``1,8,32``.
"""
import sys
import time
from collections import defaultdict

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.demo.eval_miracl import ndcg_mrr
from models.demos.blackhole.llama_nemotron_rerank_1b.demo.generator_vllm import NemotronRerankForScoring
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import config_from_overrides

d = torch.load(sys.argv[1])
OVERRIDES = sys.argv[2].split(";") if len(sys.argv) > 2 and "=" in sys.argv[2] else []
print(f"config overrides {OVERRIDES}", flush=True)
dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=400_000_000)
m = NemotronRerankForScoring(
    device=dev, max_batch_size=32, max_seq_len=512, warmup=False, config=config_from_overrides(OVERRIDES)
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.runner import TracedReranker

BUCKETS = tuple(int(b) for b in (sys.argv[3] if len(sys.argv) > 3 else "1,8,32").split(","))
m.runner = r = TracedReranker(m.runner.model, max_seq_len=512, batch_buckets=BUCKETS)
print(f"batch buckets {BUCKETS}", flush=True)
r.warmup()


def run_rows(rows, S):
    """One batch of rows through the runner's trace; blocks until its scores are back."""
    out_host, lens, _inputs = r._launch(rows, S)
    ttnn.synchronize_device(dev)
    return r.model.scores_from_host(out_host, lens)


def one_per_row_512(ids_list):
    """One pair per row, every row 512 tokens: the packed shape without packing."""
    scores, pos = [0.0] * len(ids_list), 0
    for B in r._batch_plan(len(ids_list)):
        chunk = list(range(pos, min(pos + B, len(ids_list))))
        pos += len(chunk)
        rows = [[ids_list[i]] for i in chunk]
        rows += [rows[0]] * (B - len(rows))
        for i, s in zip(chunk, run_rows(rows, 512)):
            scores[i] = s
    return scores


def unpacked(ids_list):
    """The previous runner: group pairs by length bucket, one pair per row, batches 32/8/1 completed with repeats."""
    scores, groups = [0.0] * len(ids_list), {}
    for i, ids in enumerate(ids_list):
        groups.setdefault(next(b for b in r.seq_buckets if b >= len(ids)), []).append(i)
    for S, idx in groups.items():
        pos = 0
        for B in r._batch_plan(len(idx)):
            chunk = idx[pos : pos + B]
            pos += len(chunk)
            rows = [[ids_list[i]] for i in chunk]
            rows += [rows[0]] * (B - len(rows))
            for i, s in zip(chunk, run_rows(rows, S)):
                scores[i] = s
    return scores


by_q = defaultdict(list)
for i, q in enumerate(d["qid"]):
    by_q[q].append(i)
ref = torch.tensor(d["hf_fp32"])
for name, fn in (("unpacked", unpacked), ("packed", r.score_ids), ("unpacked", unpacked), ("packed", r.score_ids)):
    scores = [0.0] * len(d["ids"])
    t0 = time.perf_counter()
    for q, idx in by_q.items():
        for i, s in zip(idx, fn([d["ids"][i] for i in idx])):
            scores[i] = s
    t_req = time.perf_counter() - t0
    t0 = time.perf_counter()
    bulk = fn(d["ids"])
    t_bulk = time.perf_counter() - t0
    tt = torch.tensor(scores)
    pear = torch.corrcoef(torch.stack([tt, ref]))[0, 1].item()
    nd, _ = ndcg_mrr(d["qid"], d["rel"], scores)
    print(
        f"{name:9s} per-query: {len(by_q)} requests in {t_req:.2f} s ({t_req / len(by_q) * 1e3:.1f} ms each, "
        f"{len(scores) / t_req:.0f} pairs/s) | bulk 1000 pairs {t_bulk:.2f} s ({1000 / t_bulk:.0f} pairs/s) | "
        f"pearson {pear:.5f} |err| mean {(tt - ref).abs().mean():.3f} nDCG@10 {nd:.4f} | bulk==per-query max diff "
        f"{max(abs(a - b) for a, b in zip(bulk, scores)):.3f}",
        flush=True,
    )
m.release()
ttnn.close_device(dev)
