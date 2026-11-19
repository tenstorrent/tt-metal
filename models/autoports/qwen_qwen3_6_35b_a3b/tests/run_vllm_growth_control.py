# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Short concurrent allocator-growth workload, identical with async on and off."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--async-scheduling", choices=("on", "off"), required=True)
    parser.add_argument("--decode-buckets", type=int, nargs="+", default=[2, 32])
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = parser.parse_args()
    fixtures = [
        dict(id=f"r{i}", prompt=[9707 + i] * length, max_tokens=count)
        for i, (length, count) in enumerate(((129, 160), (130, 112), (131, 80)))
    ]
    result = dict(
        status="RUNNING",
        async_scheduling=args.async_scheduling,
        fixtures=fixtures,
        runs=[],
        scope="raw-token mechanics stress; not qualitative evidence",
        sampling=dict(temperature=0.0, top_k=1, ignore_eos=True),
        server_config_required=dict(
            capacity=32,
            context=262144,
            decode_buckets=args.decode_buckets,
            max_num_batched_tokens=args.max_num_batched_tokens,
            precision="selected",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def request(fixture):
        payload = dict(
            model="Qwen/Qwen3.6-35B-A3B",
            prompt=fixture["prompt"],
            max_tokens=fixture["max_tokens"],
            return_token_ids=True,
            **result["sampling"],
        )
        started = time.monotonic()
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=1800)
        response.raise_for_status()
        body = response.json()
        tokens = body["choices"][0]["token_ids"]
        assert len(tokens) == fixture["max_tokens"], body
        return dict(id=fixture["id"], started=started, finished=time.monotonic(), token_ids=tokens, response=body)

    save()
    try:
        for order in ((0, 1, 2), (2, 1, 0)):
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                rows = list(pool.map(request, [fixtures[i] for i in order]))
            result["runs"].append(dict(admission_order=list(order), results=rows))
            save()
        reference = {row["id"]: row["token_ids"] for row in result["runs"][0]["results"]}
        assert all(row["token_ids"] == reference[row["id"]] for row in result["runs"][1]["results"])
        result["reordered_greedy_identical"] = True
        if args.compare:
            control = json.loads(args.compare.read_text())
            assert control["fixtures"] == fixtures and control["sampling"] == result["sampling"]
            assert control["async_scheduling"] != args.async_scheduling
            for run in control["runs"]:
                assert all(row["token_ids"] == reference[row["id"]] for row in run["results"])
            result.update(async_nonasync_tokens_identical=True, compared_artifact=str(args.compare))
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
    print(
        json.dumps(
            dict(
                status=result["status"],
                requests=6,
                async_scheduling=args.async_scheduling,
                matched_control=result.get("async_nonasync_tokens_identical"),
            )
        )
    )


if __name__ == "__main__":
    main()
