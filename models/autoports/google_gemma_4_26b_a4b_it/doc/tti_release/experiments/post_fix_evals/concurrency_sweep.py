# SPDX-License-Identifier: Apache-2.0
"""Sweep serving concurrency with one fixed prompt.

The GPQA rerun showed some requests corrupted from their very first generated
token when ten ran together, while the identical prompts were clean standalone.
This sends N identical copies of one prompt at a time and reports, per copy,
whether the output matches the known-good standalone answer. Identical inputs
mean any divergence is a serving-path defect, not model behaviour.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"


def one_request(messages, max_tokens: int) -> dict:
    body = {
        "model": "google/gemma-4-26B-A4B-it",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": False,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        out = json.load(response)
    choice = out["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "completion_tokens": out["usage"]["completion_tokens"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=Path, required=True, help="failing_doc_N.json from the samples audit")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 10])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    doc = json.loads(args.doc.read_text(encoding="utf-8"))
    messages = json.loads(doc["arguments"]["gen_args_0"]["arg_0"][0])

    results = {}
    reference = None
    for n in args.concurrency:
        with futures.ThreadPoolExecutor(max_workers=n) as pool:
            batch = list(pool.map(lambda _: one_request(messages, args.max_tokens), range(n)))
        if reference is None:
            reference = batch[0]["text"]
        rows = []
        for index, item in enumerate(batch):
            rows.append(
                {
                    "copy": index,
                    "completion_tokens": item["completion_tokens"],
                    "finish_reason": item["finish_reason"],
                    "matches_reference": item["text"] == reference,
                    "head": item["text"][:70],
                }
            )
        identical = sum(1 for r in rows if r["matches_reference"])
        capped = sum(1 for r in rows if r["finish_reason"] == "length")
        print(f"concurrency {n:>3}: {identical}/{n} identical to the standalone answer, {capped} hit the token cap")
        for r in rows:
            flag = "ok " if r["matches_reference"] else "DIFF"
            print(f"    {flag} copy {r['copy']}: {r['completion_tokens']:>6} tok {r['finish_reason']:<7} {r['head']!r}")
        results[str(n)] = rows

    args.out.write_text(json.dumps({"reference_head": reference[:200], "results": results}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
