# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal async client + load test for the BGE-M3 embedding server (``serve.py``).
The server speaks the OpenAI ``/v1/embeddings`` schema, so the ``openai`` Python
SDK works too — this is a dependency-light ``aiohttp`` example that also doubles
as a quick throughput probe.

Adapted from
``models/demos/blackhole/pplx_embed_0_6b/demo/embed_client.py`` (the client is
model-agnostic; only the default model id differs).

Examples
--------
    # Embed a few inline texts (float vectors):
    python models/demos/wormhole/bge_m3/demo/embed_client.py \\
        "hello world" "what is a tensor?"

    # Concurrent load test: 2000 requests, 64 in flight, base64 payloads:
    python models/demos/wormhole/bge_m3/demo/embed_client.py \\
        --load 2000 --concurrency 64 --encoding base64
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from typing import List

import aiohttp
import numpy as np


def decode_embedding(item: dict) -> np.ndarray:
    """Decode one OpenAI ``data[i].embedding`` (float list or base64) -> ndarray."""
    emb = item["embedding"]
    if isinstance(emb, str):
        return np.frombuffer(base64.b64decode(emb), dtype="<f4")
    return np.asarray(emb, dtype=np.float32)


async def embed(session: aiohttp.ClientSession, url: str, inputs, model: str, encoding: str):
    payload = {"model": model, "input": inputs, "encoding_format": encoding}
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    return [decode_embedding(d) for d in body["data"]], body.get("usage", {})


async def _run_inline(args, url: str) -> None:
    async with aiohttp.ClientSession() as session:
        vecs, usage = await embed(session, url, args.texts, args.model, args.encoding)
    for text, v in zip(args.texts, vecs):
        preview = ", ".join(f"{x:+.4f}" for x in v[:8])
        print(f"  dim={v.shape[0]}  |emb|={np.linalg.norm(v):.4f}  [{preview}, ...]  <- {text[:48]!r}")
    print(f"  usage: {usage}")


def _base_url(embeddings_url: str) -> str:
    return embeddings_url.split("/v1/embeddings")[0].rstrip("/")


async def _fetch_metrics(session: aiohttp.ClientSession, base: str, reset: bool = False) -> dict:
    try:
        async with session.get(f"{base}/metrics", params={"reset": "true"} if reset else {}) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    except Exception:
        return {}


def _make_corpus(n: int, approx_tokens: int = 460) -> List[str]:
    """Pool of ``n`` DISTINCT ~512-token texts (defeats the tokenizer's per-input
    cache, so server-side tokenization cost is measured realistically under load).
    """
    import random

    rng = random.Random(0)
    words = (
        "model inference latency throughput tokenization embedding transformer attention "
        "kernel matmul pipeline scheduler request response server client vector retrieval "
        "semantic search index document corpus benchmark profile memory bandwidth cache miss "
        "thread process core socket numa affinity blackhole galaxy tensix accelerator device "
        "host dispatch trace replay prefill decode batch sequence length padding mask normalize"
    ).split()
    # ~1.3 tokens/word for this vocab -> scale word count to hit approx_tokens.
    n_words = int(approx_tokens / 1.3)
    return [" ".join(rng.choice(words) for _ in range(n_words)) + f" #{i}" for i in range(n)]


async def _run_load(args, url: str) -> None:
    # A ~512-token input (forces the full ISL-512 bucket) when --long is set;
    # otherwise a short input.
    if args.long:
        text = "the quick brown fox jumps over the lazy dog while the curious cat watches nearby " * 40
    else:
        text = "the quick brown fox jumps over the lazy dog " * 4
    corpus = _make_corpus(max(256, args.concurrency * 8)) if args.vary else None
    base = _base_url(url)
    sem = asyncio.Semaphore(args.concurrency)
    latencies: List[float] = []
    total_tokens = 0

    async with aiohttp.ClientSession() as session:
        # warm the connection pool / server, then clear server-side metrics
        await embed(session, url, text, args.model, args.encoding)
        await _fetch_metrics(session, base, reset=True)

        async def one(i: int):
            nonlocal total_tokens
            inp = corpus[i % len(corpus)] if corpus is not None else text
            async with sem:
                t0 = time.perf_counter()
                _vecs, usage = await embed(session, url, inp, args.model, args.encoding)
                latencies.append((time.perf_counter() - t0) * 1000)
                total_tokens += int(usage.get("total_tokens", 0))

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.load)))
        wall = time.perf_counter() - t0

        server = await _fetch_metrics(session, base)

    lat = np.array(latencies)
    print(
        f"\n  requests:    {args.load}  (concurrency {args.concurrency}, encoding {args.encoding}, "
        f"{'long~512tok' if args.long else 'short'})"
    )
    print(f"  wall:        {wall:.2f}s")
    print(f"  throughput:  {args.load / wall:,.0f} req/s   {total_tokens / wall:,.0f} tok/s")
    print(
        f"  client e2e ms (incl. HTTP):  p50={np.percentile(lat, 50):.1f}  "
        f"p90={np.percentile(lat, 90):.1f}  p99={np.percentile(lat, 99):.1f}  max={lat.max():.1f}"
    )

    if server and server.get("count"):
        print(f"\n  server-side per-request breakdown ({server['count']} samples):")
        for key, label in (
            ("device", "on-device forward (encoder+pool)"),
            ("worker_total", "worker total (H2D+device+D2H+host)"),
            ("server_wall", "server wall (dispatch->result)"),
            ("overhead", "scheduling/queue overhead"),
            ("server_tok", "  hop: server_tok (server-side tokenize)"),
            ("chip_wait", "  hop: chip_wait (await idle chip)"),
            ("task_q", "  hop: task_q (dispatch->worker)"),
            ("prepare", "  hop: prepare (tokenize+host tensors)"),
            ("replay", "  hop: replay (== worker_total)"),
            ("result", "  hop: result (worker->future resolved)"),
        ):
            s = server.get(key, {})
            if s:
                print(
                    f"    {label:<40} p50={s['p50']:.2f}  p90={s['p90']:.2f}  "
                    f"p99={s['p99']:.2f}  max={s['max']:.2f}  mean={s['mean']:.2f}  ms"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Async client / load test for the BGE-M3 embedding server.")
    parser.add_argument("texts", nargs="*", help="Texts to embed (inline mode).")
    parser.add_argument("--url", default="http://localhost:8000/v1/embeddings", help="Embeddings endpoint URL.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Model id to send.")
    parser.add_argument("--encoding", default="float", choices=["float", "base64"], help="encoding_format.")
    parser.add_argument("--load", type=int, default=0, help="Run a load test with this many requests.")
    parser.add_argument("--concurrency", type=int, default=64, help="Max in-flight requests in load mode.")
    parser.add_argument("--long", action="store_true", help="Use a ~512-token input (full ISL bucket) in load mode.")
    parser.add_argument(
        "--vary",
        action="store_true",
        help="Send DISTINCT ~512-token texts per request (defeats tokenizer input "
        "caching) so server-side tokenization cost is measured realistically.",
    )
    args = parser.parse_args()

    if args.load > 0:
        asyncio.run(_run_load(args, args.url))
    else:
        if not args.texts:
            args.texts = ["hello world", "what is a tensor?"]
        asyncio.run(_run_inline(args, args.url))


if __name__ == "__main__":
    main()
