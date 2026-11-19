# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HTTP-only mixed-shape/allocator-growth probe; use against one live server."""

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--long-prompt-length", type=int, default=517)
    parser.add_argument("--sampled-top-k", type=int, default=20)
    parser.add_argument("--four-row-bucket", action="store_true", help="Exercise extents 3, 4 and 5 for B2/B4/B32")
    args = parser.parse_args()
    if args.sampled_top_k < 1:
        parser.error("--sampled-top-k must be positive")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(dict(results=results), indent=2) + "\n")

    def request(length, variant, *, sampled=False, first_token=None):
        started = time.monotonic()
        # Distinct valid token histories; reduced runs check mechanics only.
        prompt = [9707 + variant] * length
        payload = dict(
            model="Qwen/Qwen3.6-35B-A3B",
            prompt=prompt,
            max_tokens=args.output_tokens,
            temperature=0.7 if sampled else 0.0,
            top_k=args.sampled_top_k if sampled else 1,
            seed=42,
            ignore_eos=True,
            return_token_ids=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=600, stream=True)
        arrivals, token_ids, fragments, usage = [], [], [], None
        if response.status_code == 200:
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    ids = choice.get("token_ids") or []
                    fragment = choice.get("text", "")
                    if ids or fragment:
                        arrivals.append(time.monotonic())
                        if first_token is not None:
                            first_token.set()
                        token_ids.extend(ids)
                        fragments.append(fragment)
            body = dict(token_ids=token_ids, text="".join(fragments), usage=usage)
        else:
            body = response.json()
        return dict(
            length=length,
            variant=variant,
            sampled=sampled,
            top_k=payload["top_k"],
            requested_output_tokens=args.output_tokens,
            started=started,
            duration=time.monotonic() - started,
            status=response.status_code,
            ttft_ms=None if not arrivals else 1000 * (arrivals[0] - started),
            itl_ms=[1000 * (right - left) for left, right in zip(arrivals, arrivals[1:])],
            arrivals=arrivals,
            response=body,
        )

    results.append(request(129, 0))
    save()
    if results[-1]["status"] != 200:
        raise RuntimeError("Initial request failed; inspect saved response/server log")
    # Admit the long prompt only after another request has emitted a token.
    # This distinguishes concurrent decode from merely owning prefill state.
    first_token = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        decoding = pool.submit(request, 130, 1, first_token=first_token)
        if not first_token.wait(600):
            raise RuntimeError("Background request did not begin decoding")
        arrivals = [pool.submit(request, args.long_prompt_length, 2), pool.submit(request, 131, 3)]
        results.extend([decoding.result(), *(future.result() for future in arrivals)])
    save()
    batches = [((131, 3), (129, 0), (130, 1))]
    if args.four_row_bucket:
        batches += [
            ((129, 0), (130, 1), (131, 3), (132, 4), (133, 5)),
            ((132, 4), (131, 3), (130, 1), (129, 0)),
        ]
    for cases in batches:
        with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
            results.extend(pool.map(lambda case: request(*case), cases))
        save()
        if any(result["status"] != 200 for result in results):
            raise RuntimeError("Concurrent request failed; inspect saved response/server log")
    results.append(request(130, 1, sampled=True))
    results.append(request(129, 0))
    save()
    greedy = [row["response"]["token_ids"] for row in results if row["length"] == 129 and not row["sampled"]]
    if any(tokens != greedy[0] for tokens in greedy[1:]):
        raise AssertionError("Repeated greedy prompt changed under concurrent load")
    if any(len(row["response"].get("token_ids", [])) != args.output_tokens for row in results):
        raise AssertionError("A required request did not emit the requested token count")
    print(json.dumps({"requests": len(results), "statuses": [row["status"] for row in results]}))


if __name__ == "__main__":
    main()
