# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Re-verify non-aligned prompt-length serving **after** the stage-09 change.

The stage contract forbids trading non-aligned prompt support for an
aligned-only fast path, and requires it re-verified rather than assumed. This
sends token-id prompts whose length is divisible by none of 8, 32, 64, 128 or
1024 -- the page block, the tile, the sampling slot count and the benchmark's
own length -- straight at a live server, and requires

* a 200 with non-empty text,
* ``usage.prompt_tokens`` **equal to the length requested**, so nothing was
  capped, padded or truncated on the way in.

One natural-language prompt is included so the answer can also be read.

Usage:
    python non_aligned_prompt_lengths.py --server-url http://localhost:8100 --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

LENGTHS = (37, 131, 333, 1025, 4097)
TEXT_PROMPT = "Explain in two sentences why paged attention needs a block table."


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    url = args.server_url.rstrip("/") + "/v1/completions"
    rows = []
    for length in LENGTHS:
        prompt_ids = [1000 + (i % 5000) for i in range(length)]
        body = post(
            url,
            {
                "model": args.model,
                "prompt": prompt_ids,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
        )
        reported = int(body["usage"]["prompt_tokens"])
        rows.append(
            {
                "requested_prompt_tokens": length,
                "reported_prompt_tokens": reported,
                "not_divisible_by": {str(d): length % d != 0 for d in (8, 32, 64, 128, 1024)},
                "completion_tokens": int(body["usage"]["completion_tokens"]),
                "text": body["choices"][0]["text"],
                "passed": reported == length and int(body["usage"]["completion_tokens"]) > 0,
            }
        )

    body = post(
        url,
        {"model": args.model, "prompt": TEXT_PROMPT, "max_tokens": 48, "temperature": 0.0},
    )
    rows.append(
        {
            "requested_prompt_tokens": None,
            "natural_language_prompt": TEXT_PROMPT,
            "reported_prompt_tokens": int(body["usage"]["prompt_tokens"]),
            "not_divisible_by": {str(d): int(body["usage"]["prompt_tokens"]) % d != 0 for d in (8, 32, 64, 128, 1024)},
            "completion_tokens": int(body["usage"]["completion_tokens"]),
            "text": body["choices"][0]["text"],
            "passed": int(body["usage"]["completion_tokens"]) > 0,
        }
    )

    failed = [row for row in rows if not row["passed"]]
    out = {
        "what": "non-aligned prompt lengths through the optimized serving path",
        "server_url": args.server_url,
        "rows": rows,
        "all_passed": not failed,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    for row in rows:
        print(
            f"{'PASS' if row['passed'] else 'FAIL'}  requested={row['requested_prompt_tokens']} "
            f"reported={row['reported_prompt_tokens']} out={row['completion_tokens']}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} non-aligned prompt(s) failed")


if __name__ == "__main__":
    main()
