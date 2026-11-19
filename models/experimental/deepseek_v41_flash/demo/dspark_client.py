# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DSpark decode rate against a running ``demo/server.py``, for one user or several at once: the GSM8K prompt (by
default) as a streamed, greedy chat request from ``--users`` users in parallel, ``--runs`` times, each run from fresh
sessions. Prints the first user's first reply, then per run each user's time to the first token and decode tok/s/u
(streamed tokens after the first over the time from the first to the last) and the run's mean, slowest user and
total tok/s.

The numbers line up with ``tests/decode/test_dspark_spec.py`` / ``test_instances_dspark.py`` (GSM8K, greedy): the
server decodes in 4-row verify rounds (3 drafts + 1) below position ~508, then a step at a time. The server's stream
sends no token counts mid-reply, so tokens are counted by the stream's usage chunk, else by content chunks. The
default server (``--instances 5``) decodes up to five users in parallel, one per instance; more get a 429. Needs only
the standard library; it never opens the device. Start the server first, then::

    python models/experimental/deepseek_v41_flash/demo/dspark_client.py --url http://127.0.0.1:8000 --users 5
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
import uuid

PROMPTS = {
    "gsm8k": (
        "Solve these three problems step by step, and give each final answer as a number.\n"
        "1. Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
        "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?\n"
        "2. A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\n"
        "3. Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. How much profit did he make?"
    ),
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests.",
    "chat": "Give me a list of the top 10 best and worst movies of all time, with a one-line joke about each.",
}


def stream(url: str, model: str, user: str, prompt: str, max_tokens: int) -> dict:
    """One streamed greedy chat request: ``{text, tokens, ttft, decode_s, t_first, t_last}``."""
    payload = json.dumps(
        {
            "model": model,
            "user": user,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            # No temperature / top_p: the persistent-MoE DSpark server samples greedily on the device and turns away a
            # request that sets either (400).
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first = last = None
    chunks, usage, text = 0, None, []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                # A streamed request gets its error (e.g. the 429 when every instance is busy) as an SSE event
                # after a 200 header.
                err = chunk["error"]
                raise RuntimeError(f"{err.get('code')} {err.get('type')}: {err.get('message')}")
            if chunk.get("usage"):
                usage = chunk["usage"].get("completion_tokens")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    end = time.perf_counter()
    first = first or end
    last = last or end
    return {
        "text": "".join(text),
        "tokens": usage if usage is not None else chunks,
        "ttft": first - t0,
        "decode_s": max(last - first, 1e-9),
        "t_first": first,
        "t_last": last,
    }


def rate(r: dict) -> float:
    return (r["tokens"] - 1) / r["decode_s"] if r["tokens"] > 1 else 0.0


def run_users(args, prompt: str, run_id: str, run: int) -> list:
    """``args.users`` streams at once (each a fresh user); each user's result, or the exception it raised."""
    results: list = [None] * args.users
    start = threading.Barrier(args.users)

    def one(u: int) -> None:
        try:
            start.wait()
            results[u] = stream(args.url, args.model, f"dspark-{run_id}-{run}-{u}", prompt, args.max_tokens)
        except Exception as e:  # noqa: BLE001 - reported per user below
            results[u] = e

    threads = [threading.Thread(target=one, args=(u,), daemon=True) for u in range(args.users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="DSpark decode rate against server.py, one user or several at once")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--model", default="deepseek-v4.1-flash", help="model name")
    p.add_argument("--prompt", default="gsm8k", help=f"one of {sorted(PROMPTS)} or the prompt text itself")
    p.add_argument("--max-tokens", type=int, default=300, help="reply tokens per request (default 300)")
    p.add_argument("--users", type=int, default=1, help="users streaming at once, each its own session (default 1)")
    p.add_argument("--runs", type=int, default=3, help="runs, each from fresh sessions (default 3)")
    p.add_argument("--quiet", action="store_true", help="do not print the first reply")
    args = p.parse_args()
    if args.users < 1:
        p.error("--users takes 1 or more")

    prompt = PROMPTS.get(args.prompt, args.prompt)
    run_id = uuid.uuid4().hex[:8]
    means, totals = [], []
    for i in range(args.runs):
        results = run_users(args, prompt, run_id, i)
        ok = [r for r in results if isinstance(r, dict)]
        if i == 0 and ok and not args.quiet:
            print(f"\n===== reply =====\n{ok[0]['text']}\n", flush=True)
        for u, r in enumerate(results):
            if isinstance(r, dict):
                print(
                    f"run {i + 1} user {u + 1}: {r['tokens']} tokens, first token {1000 * r['ttft']:.0f} ms, "
                    f"{rate(r):.1f} tok/s/u",
                    flush=True,
                )
            else:
                detail = r
                if isinstance(r, urllib.error.HTTPError):
                    detail = f"HTTP {r.code}: {r.read().decode('utf-8', errors='replace')[:300]}"
                print(f"run {i + 1} user {u + 1}: failed: {detail}", flush=True)
        rates = [rate(r) for r in ok if r["tokens"] > 1]
        if rates:
            # Total: every user's decoded tokens over the span from the first user's first token to the last one's
            # last token.
            span = max(r["t_last"] for r in ok) - min(r["t_first"] for r in ok)
            total = sum(r["tokens"] - 1 for r in ok) / max(span, 1e-9)
            means.append(sum(rates) / len(rates))
            totals.append(total)
            print(
                f"run {i + 1}: {len(ok)}/{args.users} users, {means[-1]:.1f} tok/s/u mean, {min(rates):.1f} slowest, "
                f"{total:.1f} tok/s total",
                flush=True,
            )
    if len(means) > 1:
        print(
            f"mean over {len(means)} runs: {sum(means) / len(means):.1f} tok/s/u, "
            f"{sum(totals) / len(totals):.1f} tok/s total"
        )


if __name__ == "__main__":
    main()
