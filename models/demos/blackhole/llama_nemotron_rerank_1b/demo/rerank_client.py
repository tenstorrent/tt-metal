# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check and load-test the rerank server.

    python models/demos/blackhole/llama_nemotron_rerank_1b/demo/rerank_client.py --url http://localhost:8000 \\
        --eval-set miracl_eval.pt --concurrency 4

1. The model-card example: the protein passage must rank first.
2. MIRACL-en: the same queries x candidates as ``eval_miracl.py`` (rebuilt from the
   dataset, in the same order), one /v1/rerank request per query, ``--concurrency``
   in flight. Reports server logits vs the HF fp32 scores in ``--eval-set``, nDCG@10,
   request latency and throughput.

Also works against tt-media-server (``MODEL=llama-nemotron-rerank-1b-v2``): pass
``--api-key`` (its bearer token) and ``--model`` (the request's ``model`` field).
"""

import argparse
import asyncio
import statistics
import time

import aiohttp
import torch

from models.demos.blackhole.llama_nemotron_rerank_1b.demo.eval_miracl import ndcg_mrr

CARD_QUERY = "how much protein should a female eat?"
CARD_DOCS = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional.",
]


def miracl_requests(n_queries, k):
    """(qid, query, [(cid, rel, text)]) in the order build_eval used."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    get = lambda f: pd.read_parquet(
        hf_hub_download("mteb/MIRACLReranking", f"en-{f}/dev-00000-of-00001.parquet", repo_type="dataset")
    )
    queries, corpus, top, qrels = get("queries"), get("corpus"), get("top_ranked"), get("qrels")
    qtext = dict(zip(queries["_id"], queries["text"]))
    ctext = {i: (t + " " + x).strip() if t else x for i, t, x in zip(corpus["_id"], corpus["title"], corpus["text"])}
    rel = {}
    for q, c, s in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
        rel.setdefault(q, {})[c] = int(s)
    out = []
    for q, cands in zip(top["query-id"], top["corpus-ids"]):
        if len(out) >= n_queries:
            break
        if q not in rel or not any(v > 0 for v in rel[q].values()):
            continue
        out.append((q, qtext[q], [(c, rel[q].get(c, 0), ctext[c]) for c in list(cands)[:k]]))
    return out


HEADERS = {}
MODEL = None


async def post(session, url, body):
    if MODEL:
        body = {"model": MODEL, **body}
    t = time.perf_counter()
    async with session.post(url, json=body, headers=HEADERS) as r:
        r.raise_for_status()
        js = await r.json()
    return js, (time.perf_counter() - t) * 1e3


async def main_async(args):
    base = args.url.rstrip("/")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        async with s.get(base + "/health") as r:
            print("health", await r.json())
        js, ms = await post(s, base + "/v1/rerank", {"query": CARD_QUERY, "documents": CARD_DOCS, "top_n": 3})
        print(f"model-card example ({ms:.1f} ms):")
        for r in js["results"]:
            print(f"   index {r['index']}  logit {r['logit']:8.3f}  relevance {r['relevance_score']:.4f}")
        assert js["results"][0]["index"] == 0, "the protein passage must rank first"

        reqs = miracl_requests(args.queries, args.k)
        sem = asyncio.Semaphore(args.concurrency)
        lat, got = [], {}

        async def one(q, qtext, cands):
            async with sem:
                js, ms = await post(
                    s,
                    base + "/v1/rerank",
                    {"query": qtext, "documents": [c[2] for c in cands], "return_documents": False},
                )
            lat.append(ms)
            got[q] = {r["index"]: r["logit"] for r in js["results"]}

        await asyncio.gather(*(one(*r) for r in reqs[:2]))  # warm the HTTP path
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*(one(*r) for r in reqs))
        wall = time.perf_counter() - t0
        qid, rel, sc = [], [], []
        for q, _, cands in reqs:
            for i, (_, r, _) in enumerate(cands):
                qid.append(q), rel.append(r), sc.append(got[q][i])
        pairs = len(sc)
        lat.sort()
        print(
            f"MIRACL-en {len(reqs)} requests x {args.k} docs, concurrency {args.concurrency}: "
            f"{len(reqs) / wall:.1f} req/s, {pairs / wall:.0f} pairs/s, latency p50 {statistics.median(lat):.0f} ms "
            f"p90 {lat[int(0.9 * (len(lat) - 1))]:.0f} ms"
        )
        nd, mrr = ndcg_mrr(qid, rel, sc)
        line = f"server nDCG@10 {nd:.4f} MRR@10 {mrr:.4f}"
        if args.eval_set:
            d = torch.load(args.eval_set)
            ref, tt = torch.tensor(d["hf_fp32"]), torch.tensor(sc)
            if len(ref) == len(tt):
                e = (tt - ref).abs()
                line += f" | vs HF fp32: |err| mean {e.mean():.3f} max {e.max():.3f} pearson {torch.corrcoef(torch.stack([tt, ref]))[0, 1]:.5f}"
        print(line)
        async with s.get(base + "/metrics") as r:
            if r.content_type == "application/json":  # demo/serve.py; tt-media-server serves Prometheus text
                print("server metrics", await r.json())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--api-key", default=None, help="bearer token (tt-media-server's API_KEY)")
    ap.add_argument("--model", default=None, help="the request's model field (tt-media-server requires it)")
    args = ap.parse_args()
    global MODEL
    MODEL = args.model
    if args.api_key:
        HEADERS["Authorization"] = f"Bearer {args.api_key}"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
