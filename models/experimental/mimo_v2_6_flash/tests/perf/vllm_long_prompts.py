# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Long prompts against a running MiMo vLLM server (doc/vllm_integration/long_context_16x1m.md).

``--n`` completions, each of exactly ``--input-len`` random token ids (sent as a token list, so the server sees that
many tokens; distinct per request, so no prefix-cache hits), ``--output-len`` greedy tokens with ``ignore_eos``, at most
``--concurrency`` in flight. Each request streams; the script logs its TTFT (from its own send and from the run start),
its decode inter-chunk latency (the server may stream more than one token per chunk) and the prompt tokens the server
counted, then a summary, and writes everything to ``--out`` (JSON). Every line to grep starts with ``LONG``.

    python models/experimental/mimo_v2_6_flash/tests/perf/vllm_long_prompts.py --n 16 --concurrency 16 \\
        --input-len 1048000 --out c16.json

Needs aiohttp and numpy (vLLM's environment has both); not a pytest test.
"""

import argparse
import asyncio
import json
import time

import aiohttp
import numpy as np


async def one(session, args, idx, prompt, t_run, sem, results):
    async with sem:
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t_send = time.perf_counter()
        stamps, usage, text = [], None, []
        async with session.post(args.url, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                print(f"LONG req {idx}: HTTP {resp.status} {err[:300]}", flush=True)
                results[idx] = {"error": err[:1000]}
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("text"):
                        stamps.append(time.perf_counter())
                        text.append(choice["text"])
        if not stamps:
            print(f"LONG req {idx}: no tokens", flush=True)
            results[idx] = {"error": "no tokens", "usage": usage}
            return
        itl = np.diff(stamps) * 1e3
        r = {
            "ttft_s": stamps[0] - t_send,
            "first_token_at_s": stamps[0] - t_run,
            "sent_at_s": t_send - t_run,
            "decode_s": stamps[-1] - stamps[0],
            "chunks": len(stamps),
            "itl_mean_ms": float(itl.mean()) if itl.size else None,
            "itl_median_ms": float(np.median(itl)) if itl.size else None,
            "usage": usage,
            "text": "".join(text)[:200],
        }
        results[idx] = r
        print(
            f"LONG req {idx}: TTFT {r['ttft_s']:.1f} s (first token {r['first_token_at_s']:.1f} s into the run), "
            f"{len(stamps)} chunks in {r['decode_s']:.2f} s, inter-chunk mean {r['itl_mean_ms'] or 0:.1f} / median "
            f"{r['itl_median_ms'] or 0:.1f} ms, prompt tokens {usage and usage.get('prompt_tokens')}",
            flush=True,
        )


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="requests")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--input-len", type=int, default=1_048_000, help="prompt tokens (prompt + output <= max-model-len)")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0, help="the prompts' token ids")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Flash-MOPD")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    prompts = [rng.integers(100, 150_000, args.input_len).tolist() for _ in range(args.n)]
    results = {}
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=4 * 3600, sock_read=3 * 3600)
    t_run = time.perf_counter()
    print(f"LONG start: {args.n} requests x {args.input_len} tokens, concurrency {args.concurrency}", flush=True)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(one(session, args, i, prompts[i], t_run, sem, results) for i in range(args.n)))
    wall = time.perf_counter() - t_run
    ok = [r for r in results.values() if "ttft_s" in r]
    ttfts = sorted(r["ttft_s"] for r in ok)
    json.dump(
        {
            "n": args.n,
            "concurrency": args.concurrency,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "wall_s": wall,
            "ok": len(ok),
            "ttft_s": ttfts,
            "requests": results,
        },
        open(args.out, "w"),
        indent=1,
    )
    if ok:
        print(
            f"LONG done: {len(ok)}/{args.n} ok in {wall:.0f} s; TTFT min {ttfts[0]:.0f} median "
            f"{ttfts[len(ttfts) // 2]:.0f} max {ttfts[-1]:.0f} s; prefill {len(ok) * args.input_len / wall:.0f} tok/s "
            f"over the run",
            flush=True,
        )
    else:
        print(f"LONG done: 0/{args.n} ok in {wall:.0f} s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
