# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One user's DSpark decode rate against a running ``demo/server.py``: the GSM8K prompt (by default) as a streamed,
greedy chat request, ``--runs`` times, each from a fresh session, printing the first reply and per run the time to
the first token and the decode tok/s/u (streamed tokens after the first over the time from the first to the last).

The numbers line up with ``tests/decode/test_dspark_spec.py`` / ``test_instances_dspark.py`` (one user, GSM8K, greedy):
the server decodes in 4-row verify rounds (3 drafts + 1) below position ~508, then a step at a time. The server's
stream sends no token counts mid-reply, so tokens are counted by the stream's usage chunk, else by content chunks.
Needs only the standard library; it never opens the device. Start the server first, then::

    python models/experimental/deepseek_v41_flash/demo/dspark_client.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import time
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
    """One streamed greedy chat request: ``{text, tokens, ttft, decode_s}``."""
    payload = json.dumps(
        {
            "model": model,
            "user": user,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
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
    return {
        "text": "".join(text),
        "tokens": usage if usage is not None else chunks,
        "ttft": first - t0,
        "decode_s": max((last or end) - first, 1e-9),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="One user's DSpark decode rate against server.py")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--model", default="deepseek-v4.1-flash", help="model name")
    p.add_argument("--prompt", default="gsm8k", help=f"one of {sorted(PROMPTS)} or the prompt text itself")
    p.add_argument("--max-tokens", type=int, default=300, help="reply tokens per request (default 300)")
    p.add_argument("--runs", type=int, default=3, help="requests, each from a fresh session (default 3)")
    p.add_argument("--quiet", action="store_true", help="do not print the first reply")
    args = p.parse_args()

    prompt = PROMPTS.get(args.prompt, args.prompt)
    run_id = uuid.uuid4().hex[:8]
    rows = []
    for i in range(args.runs):
        r = stream(args.url, args.model, f"dspark-{run_id}-{i}", prompt, args.max_tokens)
        if i == 0 and not args.quiet:
            print(f"\n===== reply =====\n{r['text']}\n", flush=True)
        rows.append(r)
        tps = (r["tokens"] - 1) / r["decode_s"] if r["tokens"] > 1 else 0.0
        print(
            f"run {i + 1}: {r['tokens']} tokens, first token {1000 * r['ttft']:.0f} ms, {tps:.1f} tok/s/u", flush=True
        )
    if len(rows) > 1:
        rates = [(r["tokens"] - 1) / r["decode_s"] for r in rows if r["tokens"] > 1]
        if rates:
            print(f"mean over {len(rates)} runs: {sum(rates) / len(rates):.1f} tok/s/u")


if __name__ == "__main__":
    main()
