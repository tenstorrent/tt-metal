#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prompt lengths TTI can legally send that are not divisible by anything internal.

The release contract says a valid prompt length must work whether or not it
divides the autoport's prefill chunk (8192), page block (64), tile (32) or any
trace bucket.  The optimized-vLLM stage proved that through the runner; this
re-proves it through the same OpenAI-compatible endpoint the release workflow
uses, at lengths chosen to miss every one of those boundaries, including two
that straddle the 8192-token prefill chunk.

Lengths are measured on the *rendered* prompt, i.e. what the model actually
prefills, not on the user string: the chat template adds a system message, so
padding the user text to N tokens would not give a prefill of N.

Usage::

    python non_aligned_probe.py --out doc/tti_release/non_aligned_probe.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

MODEL = "meta-models/Muse-Glimmer-30B"

#: Chosen so that each rendered prefill length is odd relative to 32 (tile),
#: 64 (page block) and 8192 (prefill chunk); 8193 and 12345 also cross the
#: chunk boundary with a non-aligned remainder.
TARGET_PREFILL_LENS = [1, 37, 127, 129, 1023, 2049, 4097, 8193, 12345]


def render_len(tokenizer, text: str) -> int:
    msgs = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def build_prompt(tokenizer, target: int) -> tuple[str, int]:
    """A user string whose *rendered* prompt is as close to ``target`` as the
    template's fixed overhead allows, then exactly ``target`` by bisection on a
    filler word count."""
    base = render_len(tokenizer, "")
    if target <= base:
        return "", base
    lo, hi = 0, max(8, (target - base) * 2)
    best_text, best_len = "", base
    while lo <= hi:
        mid = (lo + hi) // 2
        text = " ".join(["word"] * mid)
        n = render_len(tokenizer, text)
        if n == target:
            return text, n
        if abs(n - target) < abs(best_len - target):
            best_text, best_len = text, n
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text, best_len


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    results = []
    for target in TARGET_PREFILL_LENS:
        text, rendered = build_prompt(tokenizer, target)
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{args.server_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1200))
            usage = r["usage"]
            row = {
                "target_prefill_len": target,
                "rendered_prefill_len": rendered,
                "server_prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "finish_reason": r["choices"][0]["finish_reason"],
                "mod_32": rendered % 32,
                "mod_64": rendered % 64,
                "mod_8192": rendered % 8192,
                "ok": usage["completion_tokens"] > 0,
                "seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:  # noqa: BLE001 - the failure IS the result
            row = {
                "target_prefill_len": target,
                "rendered_prefill_len": rendered,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 2),
            }
        results.append(row)
        print(
            f"{row['rendered_prefill_len']:>6} tokens "
            f"(mod32={row.get('mod_32')}, mod64={row.get('mod_64')}, "
            f"mod8192={row.get('mod_8192')}) -> ok={row['ok']} "
            f"{row.get('error', '')}",
            flush=True,
        )

    passed = sum(1 for r in results if r["ok"])
    doc = {
        "_what": (
            "valid prompt lengths that divide none of the autoport's internal "
            "sizes (tile 32, page block 64, prefill chunk 8192), sent through the "
            "same /v1/chat/completions endpoint the release workflow uses"
        ),
        "server_url": args.server_url,
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"NON_ALIGNED {'OK' if passed == len(results) else 'FAIL'} {passed}/{len(results)} -> {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
