# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Benchmark client to measure toks/s/u across sequence lengths against the v4.1 Flash server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


def _parse_seq_len(spec: str) -> int:
    s = spec.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _make_prompt(target_tokens: int) -> str:
    # Random prefix defeats prefix/slot-reuse caching so the full prompt is prefilled.
    prefix = f"{uuid.uuid4().hex} "
    # ~1 token per word; repeat a short phrase to reach the requested prompt length.
    unit = "hello world "
    repeats = max(1, (target_tokens - 8) // 2)
    return prefix + (unit * repeats).strip()


def _query_stream(
    url: str, model: str, user: str, prompt: str, max_tokens: int, ignore_eos: bool = True
) -> tuple[int, int | None, float, float]:
    """Send one streaming chat completion request and return (completion_tokens, prompt_tokens, ttft_s, total_s)."""
    payload = json.dumps(
        {
            "model": model,
            "user": user,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
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
    first_token_t: float | None = None
    chunk_tokens = 0
    usage_completion_tokens: int | None = None
    usage_prompt_tokens: int | None = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage_completion_tokens = chunk["usage"].get("completion_tokens")
                usage_prompt_tokens = chunk["usage"].get("prompt_tokens")
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    chunk_tokens += 1
    t_end = time.perf_counter()
    ttft = (first_token_t - t0) if first_token_t is not None else (t_end - t0)
    tokens = usage_completion_tokens if usage_completion_tokens is not None else chunk_tokens
    return tokens, usage_prompt_tokens, ttft, t_end - t0


def main() -> None:
    p = argparse.ArgumentParser(description="Measure toks/s/u across sequence lengths against server.py")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--model", default="deepseek-v4.1-flash", help="Model name")
    p.add_argument("--seq-lens", default="128,1024,4096,16384", help="Comma-separated prompt lengths (tokens)")
    p.add_argument("--max-tokens", type=int, default=64, help="Output tokens per request")
    p.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep generating past EOS up to --max-tokens (default: on)",
    )
    args = p.parse_args()

    run_id = uuid.uuid4().hex[:8]
    seq_lens = [_parse_seq_len(s) for s in args.seq_lens.split(",") if s.strip()]
    print(f"{'seq_len':>10} {'prompt_toks':>12} {'tokens':>8} {'ttft_ms':>10} {'decode_s':>10} {'toks/s/u':>10}")
    print("-" * 65)
    for seq_len in seq_lens:
        tokens, prompt_toks, ttft, total = _query_stream(
            args.url,
            args.model,
            user=f"bench-{run_id}-{seq_len}",
            prompt=_make_prompt(seq_len),
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
        )
        decode_s = max(total - ttft, 1e-9)
        toks_per_s = max(tokens - 1, 1) / decode_s if tokens > 1 else tokens / max(total, 1e-9)
        p_toks_str = str(prompt_toks) if prompt_toks is not None else "-"
        print(f"{seq_len:>10} {p_toks_str:>12} {tokens:>8} {ttft * 1000:>10.2f} {decode_s:>10.3f} {toks_per_s:>10.2f}")


if __name__ == "__main__":
    main()
