# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnose sampler changes when a host-only companion joins a live request."""

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--companion-feature",
        choices=("penalty", "top_k", "min_p", "logprobs", "logit_bias", "min_tokens"),
        default="penalty",
    )
    parser.add_argument(
        "--arrival-order", choices=("active-first", "companion-first", "together"), default="active-first"
    )
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()
    results = []
    for greedy in (False, True):
        for mixed in (False, True):
            first_token = threading.Event()
            companion_first_token = threading.Event()
            start = args.server_log.stat().st_size

            def request(companion=False):
                body = dict(
                    model="Qwen/Qwen3.6-27B",
                    prompt=[1000 + i % 200 for i in range(137 if companion else 129)],
                    max_tokens=(32 if args.arrival_order != "active-first" else 8) if companion else 64,
                    temperature=0.7 if companion or not greedy else 0.0,
                    top_k=20,
                    top_p=0.9,
                    seed=7 if companion else 42,
                    ignore_eos=True,
                    return_token_ids=True,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                if companion:
                    if args.companion_feature == "penalty":
                        body["presence_penalty"] = 1.0
                    elif args.companion_feature == "top_k":
                        body["top_k"] = 128
                    elif args.companion_feature == "min_p":
                        body["min_p"] = 0.1
                    elif args.companion_feature == "logprobs":
                        body["logprobs"] = 3
                    elif args.companion_feature == "logit_bias":
                        body["logit_bias"] = {"1234": 2.0}
                    else:
                        body["min_tokens"] = 4
                begun = time.time()
                response = requests.post("http://localhost:8000/v1/completions", json=body, stream=True, timeout=300)
                response.raise_for_status()
                ids, times, usage = [], [], None
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        new = choice.get("token_ids") or []
                        ids.extend(new)
                        if new:
                            times.append(time.time())
                            (companion_first_token if companion else first_token).set()
                assert len(ids) == body["max_tokens"] and usage["completion_tokens"] == len(ids)
                return dict(body=body, submitted_unix=begun, token_ids=ids, token_times_unix=times, usage=usage)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                companion_control = request(True) if mixed and args.require_stable else None
                # A solo control must not satisfy the mixed-arrival event.
                companion_first_token.clear()
                companion = None
                if mixed and args.arrival_order == "companion-first":
                    companion = pool.submit(request, True)
                    if not companion_first_token.wait(60):
                        raise TimeoutError("No companion first token before active admission")
                    active = pool.submit(request)
                else:
                    active = pool.submit(request)
                if mixed and args.arrival_order == "active-first":
                    if not first_token.wait(60):
                        raise TimeoutError("No first token before companion admission")
                    companion = pool.submit(request, True)
                elif mixed and args.arrival_order == "together":
                    companion = pool.submit(request, True)
                result = dict(greedy=greedy, mixed=mixed, active=active.result())
                if companion:
                    result["companion"] = companion.result()
                if companion_control is not None:
                    result["companion_control"] = companion_control
            end = args.server_log.stat().st_size
            with args.server_log.open("rb") as source:
                source.seek(start)
                lines = source.read(end - start).decode(errors="replace").splitlines()
            result.update(
                log_byte_start=start,
                log_byte_end=end,
                execution_events=[line for line in lines if "QWEN_VLLM_" in line or "TT_SAMPLER_OWNER" in line],
            )
            results.append(result)
            args.output.write_text(json.dumps(dict(results=results), indent=2) + "\n")
    comparisons = []
    for solo, mixed in zip(results[::2], results[1::2]):
        left, right = solo["active"]["token_ids"], mixed["active"]["token_ids"]
        comparisons.append(
            dict(
                greedy=solo["greedy"],
                equal=left == right,
                first_different_index=next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None),
                companion_equal=(
                    mixed["companion"]["token_ids"] == mixed["companion_control"]["token_ids"]
                    if "companion_control" in mixed
                    else None
                ),
            )
        )
    passed = all(item["equal"] and item["companion_equal"] is not False for item in comparisons)
    args.output.write_text(
        json.dumps(
            dict(results=results, comparisons=comparisons, arrival_order=args.arrival_order, passed=passed), indent=2
        )
        + "\n"
    )
    print(json.dumps(comparisons))
    if args.require_stable and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
