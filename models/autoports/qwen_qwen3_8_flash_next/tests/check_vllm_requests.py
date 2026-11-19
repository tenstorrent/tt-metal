# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small live-server lifecycle probe. Reduced targets establish mechanics only."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reduced", action="store_true")
    args = parser.parse_args()
    response = requests.get(args.server_url + "/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    results = []

    def request(length, index):
        prompt = ([785, 13458, 13] * ((length + 2) // 3))[:length]
        start = time.monotonic()
        response = requests.post(
            args.server_url + "/v1/completions",
            json=dict(model=model, prompt=prompt, temperature=0, max_tokens=8, ignore_eos=True),
            timeout=600,
        )
        result = dict(length=length, index=index, status=response.status_code, seconds=time.monotonic() - start)
        try:
            result["response"] = response.json()
        except ValueError:
            result["response"] = response.text
        return result

    results.append(request(129, 0))
    for batch in ((130, 257), (131, 129), (257, 130)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, length, len(results) + i) for i, length in enumerate(batch)]
            results.extend(f.result() for f in futures)
    passed = all(
        row["status"] == 200
        and row["response"]["usage"]["prompt_tokens"] == row["length"]
        and row["response"]["usage"]["completion_tokens"] == 8
        for row in results
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                passed=passed,
                reduced=args.reduced,
                quality_verdict="not_assessed",
                prompt_mode="raw_token_continuation_stress",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{'PASS' if passed else 'FAIL'} {len(results)} requests: {args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
