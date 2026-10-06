# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare a seeded request alone and beside requests that finish earlier."""

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--peer-tokens", type=int, nargs="+", default=[3, 7])
    parser.add_argument(
        "--no-penalties", action="store_true", help="Verify seeded counts without token-history transport"
    )
    args = parser.parse_args()
    if any(not 1 <= count < args.tokens for count in args.peer_tokens):
        parser.error("peer token counts must be positive and smaller than --tokens")
    response = requests.get(args.url + "/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    target = dict(
        model=model,
        prompt="All: ",
        max_tokens=args.tokens,
        temperature=0.5,
        seed=6,
        repetition_penalty=1.0 if args.no_penalties else 1.5,
        presence_penalty=0.0 if args.no_penalties else 1.0,
        frequency_penalty=0.0 if args.no_penalties else 1.0,
        ignore_eos=True,
        return_token_ids=True,
    )
    records = []
    lock = threading.Lock()

    def request(label, payload, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=30)
        print(f"START {label}", flush=True)
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=900)
        body = response.json()
        record = dict(label=label, request=payload, status_code=response.status_code, response=body)
        with lock:
            records.append(record)
            args.output.with_suffix(".partial.json").write_text(json.dumps(records, indent=2) + "\n")
        response.raise_for_status()
        ids = body["choices"][0]["token_ids"]
        assert len(ids) == payload["max_tokens"], record
        assert body["usage"]["completion_tokens"] == len(ids), record
        print(f"DONE {label} tokens={len(ids)}", flush=True)
        return ids

    expected = request("isolated_before", target)
    comparisons = []
    for count in args.peer_tokens:
        peer = dict(
            model=model,
            prompt="Count: ",
            max_tokens=count,
            temperature=0,
            ignore_eos=True,
            return_token_ids=True,
        )
        for peer_first in (False, True):
            name = f"peer_{count}_{'first' if peer_first else 'second'}"
            barrier = threading.Barrier(2)
            ordered = [(name + "_target", target), (name + "_peer", peer)]
            if peer_first:
                ordered.reverse()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = {label: pool.submit(request, label, payload, barrier) for label, payload in ordered}
                observed = pending[name + "_target"].result()
                pending[name + "_peer"].result()
            mismatch = next((i for i, (left, right) in enumerate(zip(expected, observed)) if left != right), None)
            comparisons.append(dict(case=name, exact_match=mismatch is None, first_mismatch_index=mismatch))
    after = request("isolated_after", target)
    report = dict(
        model=model,
        target=target,
        controls_match=after == expected,
        comparisons=comparisons,
        records=records,
        scope="HTTP concurrency exercises completion boundaries; scheduler/device instrumentation is needed to prove co-batching.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["controls_match"], "Isolated seeded controls differ"
    assert all(item["exact_match"] for item in comparisons), comparisons


if __name__ == "__main__":
    main()
