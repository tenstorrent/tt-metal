# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTFT and decode-rate sweep against a running ``demo/server.py``.

For every context length (32K to 1M tokens in powers of two by default) this sends one streamed chat request
whose prompt fills the context minus the reply, decodes ``--max-tokens`` (128) tokens and reports:

* TTFT: request sent to the first streamed token, i.e. the prefill plus the first decode step;
* tok/s/u: ``(tokens - 1) / (last token - first token)``, the decode rate of the one user.

The prompt is random numbered records under a fresh nonce, so no request reuses another's cached KV, followed by
an instruction that asks for a long reply so the turn is unlikely to stop on EOS before ``--max-tokens``. The
server sends no token counts in a stream, so the reply is counted by re-tokenizing the streamed text. Each run's
session is deleted afterwards to return its KV pages to the pool. Contexts above the server's context window are
skipped, so the 1M point needs a server started with ``--max-context 1048576``.

Needs only ``requests`` and ``transformers`` (the tokenizer from the checkpoint snapshot); it never opens the device.
Start the server first (see ``demo/server.py``), then::

    python models/experimental/deepseek_v41_flash/demo/test_server_perf.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

from transformers import AutoTokenizer

from models.experimental.deepseek_v41_flash import encoding_dsv41
from models.experimental.deepseek_v4_flash.demo.client import Client
from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir

_DEFAULT_MODEL_DIR = "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
_DEFAULT_CONTEXTS = [1 << p for p in range(15, 21)]  # 32768 .. 1048576
_INSTRUCTION = (
    "\n\nIgnore the records above. Write a long, detailed short story about a lighthouse keeper "
    "who discovers a hidden map. Use at least 1500 words."
)
# The server stops a turn once the session reaches its context limit minus one token.
_CONTEXT_SLACK = 1


def _messages(nonce: str, filler: str) -> list[dict]:
    return [{"role": "user", "content": f"Session {nonce}.\n{filler}{_INSTRUCTION}"}]


def _prompt_tokens(tokenizer, messages: list[dict]) -> int:
    text = encoding_dsv41.encode_messages(messages, thinking_mode="chat")
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _filler_corpus(tokenizer, n_tokens: int, seed: int) -> tuple[str, list[int]]:
    """Random numbered records of at least ``n_tokens`` tokens, and each token's end offset in the text."""
    rng = random.Random(seed)
    lines, chars = [], 0
    # Each record is ~10 tokens of ~4 chars; overshoot so one tokenization covers every context.
    while chars < n_tokens * 5:
        line = f"Record {len(lines)}: code {rng.randrange(10**9):09d}, value {rng.random():.6f}.\n"
        lines.append(line)
        chars += len(line)
    text = "".join(lines)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = [end for _, end in enc["offset_mapping"]]
    if len(ends) < n_tokens:
        raise RuntimeError(f"filler corpus has {len(ends)} tokens, needed {n_tokens}")
    return text, ends


def _build_prompt(tokenizer, corpus: str, ends: list[int], target: int) -> tuple[list[dict], int]:
    """Messages whose rendered prompt is as close to ``target`` tokens as possible without exceeding it."""
    nonce = uuid.uuid4().hex
    overhead = _prompt_tokens(tokenizer, _messages(nonce, ""))
    n_filler = target - overhead
    while n_filler > 0:
        messages = _messages(nonce, corpus[: ends[n_filler - 1]])
        n = _prompt_tokens(tokenizer, messages)
        if n <= target:
            return messages, n
        n_filler -= n - target
    raise ValueError(f"target {target} tokens is below the prompt template's {overhead}")


def _run_one(client: Client, tokenizer, messages: list[dict], args, user: str) -> dict:
    params = {"user": user, "max_tokens": args.max_tokens, "temperature": args.temperature}
    text = ""
    t_first = t_last = None
    t0 = time.perf_counter()
    try:
        for chunk in client.chat(messages, stream=True, **params):
            delta = chunk["choices"][0]["delta"]
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if not piece:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            text += piece
    finally:
        client.reset_session(user)
    if t_first is None:
        raise RuntimeError("the server streamed no tokens")
    n_out = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    decode_s = t_last - t_first
    return {
        "ttft_s": t_first - t0,
        "completion_tokens": n_out,
        "decode_s": decode_s,
        "tok_s_u": (n_out - 1) / decode_s if n_out > 1 and decode_s > 0 else 0.0,
        "e2e_s": t_last - t0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--model-dir", default=_DEFAULT_MODEL_DIR, help="checkpoint snapshot (for the tokenizer)")
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=_DEFAULT_CONTEXTS,
        help="context lengths (prompt + reply) to sweep; default 32768 .. 1048576 in powers of two",
    )
    p.add_argument("--max-tokens", type=int, default=128, help="tokens to decode per request")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default: greedy)")
    p.add_argument("--warmup-tokens", type=int, default=1024, help="prompt length of an untimed first request; 0 skips")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=7200, help="per-request timeout in seconds")
    p.add_argument("--json", help="also write the results to this file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(resolve_snapshot_dir(Path(args.model_dir).expanduser()))
    client = Client(args.base_url, model=args.model, timeout=args.timeout)

    record = client.models()["data"][0]
    window = record.get("context_window")
    print(f"server {client.base_url}: model {record['id']!r}, context window {window}")

    contexts = sorted(args.contexts)
    if window is not None:
        skipped = [c for c in contexts if c > window]
        if skipped:
            print(f"skipping contexts {skipped}: above the server's context window {window}")
        contexts = [c for c in contexts if c <= window]
    if not contexts:
        print("no context length fits the server", file=sys.stderr)
        return 1

    targets = {c: c - args.max_tokens - _CONTEXT_SLACK for c in contexts}
    t = time.perf_counter()
    corpus, ends = _filler_corpus(tokenizer, max(max(targets.values()), args.warmup_tokens), args.seed)
    print(f"built a {len(ends)}-token filler corpus in {time.perf_counter() - t:.1f}s")

    if args.warmup_tokens:
        messages, n = _build_prompt(tokenizer, corpus, ends, args.warmup_tokens)
        r = _run_one(client, tokenizer, messages, args, user=f"perf-warmup-{uuid.uuid4().hex[:8]}")
        print(f"warmup: {n} prompt tokens, TTFT {r['ttft_s']:.2f}s, {r['tok_s_u']:.2f} tok/s/u")

    results = []
    for ctx in contexts:
        messages, n_prompt = _build_prompt(tokenizer, corpus, ends, targets[ctx])
        print(f"context {ctx}: sending {n_prompt} prompt tokens, decoding {args.max_tokens} ...", flush=True)
        r = _run_one(client, tokenizer, messages, args, user=f"perf-{ctx}-{uuid.uuid4().hex[:8]}")
        r.update({"context": ctx, "prompt_tokens": n_prompt})
        results.append(r)
        stopped = "" if r["completion_tokens"] >= args.max_tokens else " (stopped early on EOS)"
        print(
            f"  TTFT {r['ttft_s']:.2f}s ({n_prompt / r['ttft_s']:.0f} prompt tok/s), "
            f"{r['completion_tokens']} tokens at {r['tok_s_u']:.2f} tok/s/u{stopped}"
        )

    print()
    print(f"{'context':>8} {'prompt':>8} {'TTFT (s)':>9} {'prefill tok/s':>14} {'out':>5} {'tok/s/u':>8}")
    for r in results:
        print(
            f"{r['context']:>8} {r['prompt_tokens']:>8} {r['ttft_s']:>9.2f} "
            f"{r['prompt_tokens'] / r['ttft_s']:>14.0f} {r['completion_tokens']:>5} {r['tok_s_u']:>8.2f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
