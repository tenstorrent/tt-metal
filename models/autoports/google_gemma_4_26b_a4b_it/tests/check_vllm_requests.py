# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Request boundary and concurrent allocator checks against a running server."""

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]

    records = []
    lock = threading.Lock()

    def request(case):
        length, token = case
        response = requests.post(
            args.url + "/v1/completions",
            json={
                "model": model,
                "prompt": [2] + [token] * (length - 1),
                "max_tokens": args.tokens,
                "temperature": 0,
                "ignore_eos": True,
                "return_token_ids": True,
            },
            timeout=600,
        )
        result = response.json()
        with lock:
            records.append(
                {
                    "prompt_tokens": length,
                    "prompt_token": token,
                    "status_code": response.status_code,
                    "response": result,
                }
            )
            args.output.with_suffix(".partial.json").write_text(json.dumps(records, indent=2) + "\n")
        response.raise_for_status()
        assert result["usage"]["prompt_tokens"] == length, result
        assert result["usage"]["completion_tokens"] == args.tokens, result
        assert len(result["choices"][0]["token_ids"]) == args.tokens, result
        return {"prompt_tokens": length, "prompt_token": token, "response": result}

    cases = [(31, 100), (63, 101), (95, 102)]
    control = [request(case) for case in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        concurrent_rows = list(pool.map(request, cases))
    for one, many in zip(control, concurrent_rows):
        assert one["response"]["choices"][0]["token_ids"] == many["response"]["choices"][0]["token_ids"], (one, many)
    tails = [request(case) for case in [(33, 103), (1057, 104), (33, 103)]]
    assert tails[0]["response"]["choices"][0]["token_ids"] == tails[-1]["response"]["choices"][0]["token_ids"]
    report = {
        "model": model,
        "generated_tokens_per_request": args.tokens,
        "control": control,
        "concurrent": concurrent_rows,
        "lifecycle": tails,
        "concurrency": 3,
        "all_requests_complete": True,
        "concurrent_matches_isolated": True,
        "scope": "Serving contract; reduced runs are not full-model quality or performance evidence.",
    }
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for section in ("control", "concurrent", "lifecycle"):
            for actual, expected in zip(report[section], reference[section], strict=True):
                assert (
                    actual["response"]["choices"][0]["token_ids"] == expected["response"]["choices"][0]["token_ids"]
                ), (section, actual, expected)
        report["matches_reference"] = str(args.reference)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
