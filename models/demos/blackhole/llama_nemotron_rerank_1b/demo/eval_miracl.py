# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device scores vs HF fp32 on a MIRACL-en reranking subset.

    python models/demos/blackhole/llama_nemotron_rerank_1b/demo/eval_miracl.py --eval-set miracl_eval.pt

``--eval-set`` is a torch file with ``ids`` (token id lists), ``qid``, ``rel`` and
``hf_fp32`` (built by ``build_eval_set``). Pairs are scored in batches of
``--batch`` sequences that share a length bucket.
"""

import argparse
import math
import time
from collections import defaultdict

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, config_from_overrides
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.weights import default_cache_dir, load_state_dict


def ndcg_mrr(qid, rel, score, k=10):
    by = defaultdict(list)
    for q, r, s in zip(qid, rel, score):
        by[q].append((s, r))
    nd, mrr = [], []
    for v in by.values():
        ranked = [r for _, r in sorted(v, key=lambda x: -x[0])]
        ideal = sorted((r for _, r in v), reverse=True)
        dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ranked[:k]))
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
        nd.append(dcg / idcg if idcg else 0.0)
        mrr.append(next((1 / (i + 1) for i, r in enumerate(ranked[:k]) if r > 0), 0.0))
    return sum(nd) / len(nd), sum(mrr) / len(mrr)


def run_eval(model, d, batch):
    order = sorted(range(len(d["ids"])), key=lambda i: len(d["ids"][i]))
    scores = [0.0] * len(order)
    groups = defaultdict(list)
    for i in order:
        groups[model.bucket(len(d["ids"][i]))].append(i)
    for S, idx in groups.items():
        for b in range(0, len(idx), batch):
            chunk = idx[b : b + batch]
            ids = [d["ids"][i] for i in chunk]
            ids += [ids[0]] * (batch - len(ids))  # pad the batch with a repeat; its score is dropped
            out = model.score(ids, seq_len=S)
            for i, s in zip(chunk, out):
                scores[i] = s
    return scores


def report(d, scores, label):
    ref, tt = torch.tensor(d["hf_fp32"]), torch.tensor(scores)
    e = (tt - ref).abs()
    pear = torch.corrcoef(torch.stack([tt, ref]))[0, 1].item()
    nd, mrr = ndcg_mrr(d["qid"], d["rel"], scores)
    nd_ref, mrr_ref = ndcg_mrr(d["qid"], d["rel"], d["hf_fp32"])
    print(
        f"{label}: |err| mean {e.mean():.3f} p99 {e.quantile(0.99):.3f} max {e.max():.3f} pearson {pear:.5f} "
        f"| nDCG@10 {nd:.4f} (hf {nd_ref:.4f}) MRR@10 {mrr:.4f} (hf {mrr_ref:.4f})"
    )
    return {"err_mean": e.mean().item(), "err_max": e.max().item(), "pearson": pear, "ndcg": nd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cfg", nargs="*", default=[], help="RerankConfig overrides, key=value")
    args = ap.parse_args()
    d = torch.load(args.eval_set)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    try:
        t0 = time.time()
        model = NemotronReranker(
            dev, config_from_overrides(args.cfg), load_state_dict(MODEL_ID), default_cache_dir(MODEL_ID)
        )
        print(f"build {time.time() - t0:.1f}s")
        t0 = time.time()
        scores = run_eval(model, d, args.batch)
        print(f"{len(scores)} pairs in {time.time() - t0:.1f}s (batch {args.batch}, untraced)")
        report(d, scores, "tt")
        if "hf_bf16" in d:
            report(d, d["hf_bf16"], "hf-bf16")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
