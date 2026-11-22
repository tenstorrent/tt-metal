# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill / decode performance sweep against the V4.1 Flash ``server.py``.

For each prompt length in powers of two up to ``--max-len`` (1M by default), run two phases over ``--users`` users:

* prefill: each user in turn prefills a fresh prompt of that many tokens (a one-token reply), so TTFT and prefill
  tok/s/user are one user's alone;
* decode: all users then decode ``--decode-tokens`` (1K) at once. Each request continues its user's conversation
  (the prompt, the one-token reply and a short follow-up turn), so the slot's cached prompt is reused and only the
  follow-up is prefilled before decoding.

It reports TTFT, prefill tok/s/user, total decode tok/s (all users together) and decode tok/s/user, so sweeping
``--users`` shows how concurrent users trade per-user decode speed for total throughput. A length whose prompt plus
decode would not fit the server's context window is capped at ``context_window - decode_tokens -``
:data:`_CONTINUE_MARGIN` (so 1M prefills a little under 1M - 1K); a length above the context window is skipped.

The server takes text, not token ids, so the prompt is built from `` hello`` / `` world`` words and sized from two
short calibration requests, which measure the chat template's overhead and the tokens per word. Each prompt starts
with a random run of those words, so no slot's cache holds a prefix of it, and its session is deleted afterwards so
its KV blocks go back to the pools.

    python models/experimental/deepseek_v41_flash/demo/perf_client.py --url http://127.0.0.1:8000
    python models/experimental/deepseek_v41_flash/demo/perf_client.py --min-len 1k --max-len 128k --users 4
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass

_WORDS = ("hello", "world")
_RANDOM_HEAD_WORDS = 64
_CALIBRATION_WORDS = (256, 1280)
_CONTINUE_PROMPT = "Continue."
# Room in the context for the prefill's one-token reply and the decode phase's follow-up turn.
_CONTINUE_MARGIN = 64


@dataclass
class Result:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    content: str
    t_start: float
    t_first: float
    t_end: float

    @property
    def ttft(self) -> float:
        return self.t_first - self.t_start

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / max(self.ttft, 1e-9)

    @property
    def decode_tps(self) -> float:
        return max(self.completion_tokens - 1, 0) / max(self.t_end - self.t_first, 1e-9)


def _parse_len(spec: str) -> int:
    s = spec.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _fmt_len(n: int) -> str:
    if n % (1 << 20) == 0:
        return f"{n >> 20}M"
    if n % (1 << 10) == 0:
        return f"{n >> 10}K"
    return str(n)


def _make_prompt(num_words: int, rng: random.Random) -> str:
    head = min(_RANDOM_HEAD_WORDS, max(num_words - 1, 0))
    words = [_WORDS[0]]
    words += [rng.choice(_WORDS) for _ in range(head)]
    words += [_WORDS[i % 2] for i in range(max(num_words - 1 - head, 0))]
    return " ".join(words)


def _context_window(url: str, model: str) -> int | None:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as resp:
        records = json.load(resp).get("data") or []
    for record in records:
        if record.get("id") == model or len(records) == 1:
            return record.get("context_window")
    return None


def _delete_session(url: str, user: str) -> None:
    req = urllib.request.Request(f"{url}/v1/sessions/{urllib.parse.quote(user, safe='')}", method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=60).close()
    except urllib.error.URLError as e:
        print(f"  warning: could not delete session {user}: {e}")


def _query(url: str, model: str, user: str, messages: list[dict], max_tokens: int, timeout: float) -> Result:
    """One streaming chat completion; TTFT is to the first non-empty delta, token counts are the server's usage."""
    payload = json.dumps(
        {
            "model": model,
            "user": user,
            "messages": messages,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    t_start = time.perf_counter()
    t_first: float | None = None
    usage: dict | None = None
    content: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"].get("message", chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if t_first is None and (delta.get("content") or delta.get("reasoning_content")):
                    t_first = time.perf_counter()
                content.append(delta.get("content") or "")
    t_end = time.perf_counter()
    if usage is None:
        raise RuntimeError("stream ended without usage")
    return Result(
        prompt_tokens=usage["prompt_tokens"],
        cached_tokens=usage.get("prompt_cache_hit_tokens", 0),
        completion_tokens=usage["completion_tokens"],
        content="".join(content),
        t_start=t_start,
        t_first=t_first if t_first is not None else t_end,
        t_end=t_end,
    )


def _calibrate(url: str, model: str, run_id: str, rng: random.Random, timeout: float) -> tuple[float, float]:
    """Returns (tokens per word, constant token overhead) of a prompt built by :func:`_make_prompt`."""
    counts = []
    for i, n in enumerate(_CALIBRATION_WORDS):
        user = f"perf-{run_id}-calib{i}"
        try:
            messages = [{"role": "user", "content": _make_prompt(n, rng)}]
            counts.append(_query(url, model, user, messages, 1, timeout).prompt_tokens)
        finally:
            _delete_session(url, user)
    (n1, n2), (p1, p2) = _CALIBRATION_WORDS, counts
    slope = (p2 - p1) / (n2 - n1)
    return slope, p1 - slope * n1


def _error_text(i: int, e: BaseException) -> str:
    detail = e.read().decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else ""
    return f"user {i}: {e} {detail}".strip()


def _run_length(
    args, run_id: str, rng: random.Random, target: int, slope: float, overhead: float
) -> tuple[list[Result], list[Result]]:
    """Prefills every user one after another, then decodes them all at once. Returns (prefills, decodes)."""
    num_words = max(1, round((target - overhead) / slope))
    users = [f"perf-{run_id}-{target}-u{i}" for i in range(args.users)]
    conversations = [[{"role": "user", "content": _make_prompt(num_words, rng)}] for _ in range(args.users)]
    try:
        prefills = []
        for i, user in enumerate(users):
            try:
                prefills.append(_query(args.url, args.model, user, conversations[i], 1, args.timeout))
            except Exception as e:  # noqa: BLE001 - reraised with the server's error body
                raise RuntimeError(f"prefill {_error_text(i, e)}") from e

        for conversation, prefill in zip(conversations, prefills):
            conversation.append({"role": "assistant", "content": prefill.content})
            conversation.append({"role": "user", "content": _CONTINUE_PROMPT})
        decodes: list[Result | None] = [None] * args.users
        errors: list[BaseException | None] = [None] * args.users
        start = threading.Barrier(args.users)

        def worker(i: int) -> None:
            try:
                start.wait()
                decodes[i] = _query(args.url, args.model, users[i], conversations[i], args.decode_tokens, args.timeout)
            except BaseException as e:  # noqa: BLE001 - reported per user below
                errors[i] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.users)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f"decode {_error_text(i, e)}")
    finally:
        for user in users:
            _delete_session(args.url, user)
    return prefills, decodes


def main() -> None:
    p = argparse.ArgumentParser(description="Prefill / decode performance sweep against the V4.1 Flash server.py")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--model", default="deepseek-v4.1-flash", help="Model name sent with each request")
    p.add_argument("--min-len", default="128", help="Smallest prompt length (power of two; k/M suffixes ok)")
    p.add_argument("--max-len", default="1M", help="Largest prompt length (power of two; k/M suffixes ok)")
    p.add_argument("--decode-tokens", type=int, default=1024, help="Tokens decoded after each prefill")
    p.add_argument(
        "--users",
        type=int,
        default=1,
        help="Users per length, each with its own prompt: prefilled one at a time, then decoded together",
    )
    p.add_argument("--timeout", type=float, default=7200.0, help="Socket timeout per request (s)")
    p.add_argument("--seed", type=int, default=None, help="Seed for the prompts' random heads")
    args = p.parse_args()
    args.url = args.url.rstrip("/")

    rng = random.Random(args.seed)
    run_id = uuid.uuid4().hex[:8]
    min_len, max_len = _parse_len(args.min_len), _parse_len(args.max_len)
    lengths = []
    n = 1 << max(min_len - 1, 0).bit_length()
    while n <= max_len:
        lengths.append(n)
        n <<= 1

    context = _context_window(args.url, args.model)
    print(f"server {args.url}, context window {context}, {args.users} user(s), {args.decode_tokens} decode tokens")
    slope, overhead = _calibrate(args.url, args.model, run_id, rng, args.timeout)
    print(f"calibration: {slope:.3f} tokens/word, {overhead:.1f} tokens of template overhead\n")

    header = (
        f"{'length':>7} {'prompt':>8} {'TTFT_s':>9} {'prefill_t/s/u':>14} "
        f"{'decoded':>8} {'total_decode_t/s':>17} {'decode_t/s/u':>13}"
    )
    print(header)
    print("-" * len(header))
    for length in lengths:
        target = length
        if context is not None:
            if length > context:
                print(f"{_fmt_len(length):>7}  skipped: above the {context}-token context window")
                continue
            target = min(length, context - args.decode_tokens - _CONTINUE_MARGIN)
        try:
            prefills, decodes = _run_length(args, run_id, rng, target, slope, overhead)
        except Exception as e:  # noqa: BLE001 - one failing length should not end the sweep
            print(f"{_fmt_len(length):>7}  failed: {e}")
            continue

        n = len(prefills)
        prompt_tokens = sum(r.prompt_tokens for r in prefills) / n
        ttft = sum(r.ttft for r in prefills) / n
        prefill_tps_u = sum(r.prefill_tps for r in prefills) / n
        decoded = sum(r.completion_tokens for r in decodes) / n
        decode_tps_u = sum(r.decode_tps for r in decodes) / n
        decode_window = max(r.t_end for r in decodes) - min(r.t_first for r in decodes)
        total_decode_tps = sum(max(r.completion_tokens - 1, 0) for r in decodes) / max(decode_window, 1e-9)
        print(
            f"{_fmt_len(length):>7} {prompt_tokens:>8.0f} {ttft:>9.3f} {prefill_tps_u:>14.1f} "
            f"{decoded:>8.0f} {total_decode_tps:>17.2f} {decode_tps_u:>13.2f}"
        )
        cached = sum(r.cached_tokens for r in prefills)
        if cached:
            print(f"         warning: {cached} prompt tokens were cache hits; prefill numbers are optimistic")
        # The follow-up turn and the re-rendered tail of the first turn are always prefilled again: a few tokens.
        reprefilled = [d.prompt_tokens - d.cached_tokens for d in decodes]
        if max(reprefilled) > _CONTINUE_MARGIN:
            print(
                f"         warning: the decode phase missed the cache (up to {max(reprefilled)} prompt tokens "
                "re-prefilled per user); users did not start decoding together"
            )
        if any(r.prompt_tokens != target for r in prefills):
            print(f"         note: target was {target} prompt tokens")


if __name__ == "__main__":
    main()
