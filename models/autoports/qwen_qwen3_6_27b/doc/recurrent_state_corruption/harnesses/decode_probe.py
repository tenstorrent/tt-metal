"""Measure per-user decode rate at N concurrent chat requests.

Mirrors the eval's serving shape (chat endpoint, thinking on) but caps the
output so a datapoint costs minutes, not hours.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

PROMPT = (
    "What is the correct answer to this question: A 25-year-old woman presents "
    "with episodic hypertension and headaches. Plasma metanephrines are elevated.\n"
    "Choices:\n(A) Pheochromocytoma\n(B) Essential hypertension\n"
    "(C) Renal artery stenosis\n(D) Hyperthyroidism\n"
    "Please reason step by step, and your final answer must be only (A,B,C or D) within \\boxed\nAnswer:"
)


async def one(session, idx, url, model, max_tokens, out):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    stamps = []
    usage_tokens = None
    async with session.post(url, json=body) as r:
        async for raw in r.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens")
            for _ch in chunk.get("choices", []):
                if "delta" in _ch:
                    stamps.append(time.perf_counter())
    if len(stamps) < 2:
        out[idx] = None
        return
    ttft = stamps[0] - t0
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    out[idx] = {
        "tokens": usage_tokens or len(stamps),
        "chunks": len(stamps),
        "ttft_s": ttft,
        "itl_median_ms": statistics.median(itls) * 1000,
        "tok_per_s_user": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
        "decode_s": stamps[-1] - stamps[0],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    a = ap.parse_args()
    out = {}
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=a.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        await asyncio.gather(*[one(s, i, a.url, a.model, a.max_tokens, out) for i in range(a.concurrency)])
    wall = time.perf_counter() - t0
    ok = [v for v in out.values() if v]
    print(f"concurrency={a.concurrency} max_tokens={a.max_tokens} wall={wall:.1f}s ok={len(ok)}/{a.concurrency}")
    if not ok:
        return
    tps = [v["tok_per_s_user"] for v in ok]
    itl = [v["itl_median_ms"] for v in ok]
    ttft = [v["ttft_s"] for v in ok]
    toks = sum(v["tokens"] for v in ok)
    print("  per-req:", [(v["tokens"], round(v["ttft_s"], 1), round(v["decode_s"], 1)) for v in ok])
    print(f"  tok/s/user   median {statistics.median(tps):.2f}  min {min(tps):.2f}  max {max(tps):.2f}")
    print(f"  ITL median   {statistics.median(itl):.1f} ms")
    print(f"  TTFT         median {statistics.median(ttft):.1f} s  max {max(ttft):.1f} s")
    print(f"  aggregate    {toks/wall:.1f} tok/s over {toks} tokens")
    print(f"  => tokens reachable inside lm-eval's 1800 s timeout: " f"{int(statistics.median(tps)*1800)}")


asyncio.run(main())
