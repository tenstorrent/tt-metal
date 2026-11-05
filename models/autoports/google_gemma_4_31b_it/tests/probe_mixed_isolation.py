# SPDX-License-Identifier: Apache-2.0
"""Capture original heterogeneous requests under native and explicit host routes."""

import argparse
import json
import random
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def original_configs():
    return [
        dict(prompt="Count: ", max_tokens=5, temperature=0),
        dict(prompt="Random: ", max_tokens=5, temperature=2.0, top_k=32, seed=42),
        dict(prompt="test test. Say: ", max_tokens=5, temperature=0.5, repetition_penalty=3.0, seed=42),
        dict(prompt="List: ", max_tokens=10, temperature=0.5, presence_penalty=2.0, seed=42),
        dict(prompt="Word: ", max_tokens=3, temperature=2.0, top_k=1, seed=99),
        dict(prompt="Letter: ", max_tokens=3, temperature=0.01, seed=42),
        dict(prompt="Number: ", max_tokens=3, temperature=2.0, top_k=5, seed=42),
        dict(prompt="a a a. Next: ", max_tokens=5, temperature=0.5, frequency_penalty=2.0, seed=42),
        dict(
            prompt="All: ",
            max_tokens=8,
            temperature=0.5,
            repetition_penalty=1.5,
            presence_penalty=1.0,
            frequency_penalty=1.0,
            seed=6,
        ),
        dict(prompt="TopP: ", max_tokens=8, temperature=1.0, top_p=0.5, seed=7),
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--server-log", type=Path, default=root / "readiness_vllm/final_full/server.log")
    parser.add_argument("--output", type=Path, default=root / "readiness_vllm/mixed_isolation_probe.json")
    parser.add_argument("--native-pairs", type=int, default=3)
    parser.add_argument("--host-pairs", type=int, default=2)
    parser.add_argument("--neutral-bias-pairs", type=int, default=2)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def http(path, payload=None):
        req = urllib.request.Request(
            base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    model = args.model or http("/v1/models")["data"][0]["id"]
    configs = [dict(top_p=1.0, presence_penalty=0.0, frequency_penalty=0.0) | item for item in original_configs()]
    report = dict(
        status="running",
        model=model,
        original_configs=configs,
        waves=[],
        comparisons=[],
        scope="request isolation correctness; not performance",
        limitations=[
            "Native requests retain original sampling, seed, cap and prompt.",
            "return_token_ids is diagnostic metadata. Host controls add logprobs=0 or neutral logit_bias={0:0.0}.",
            "Log event attribution assumes this probe is the sole client; no cache reset is requested.",
            "HTTP concurrency does not guarantee simultaneous scheduler admission.",
        ],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def boundary():
        raw = args.server_log.read_bytes()
        return dict(byte=len(raw), line=raw.count(b"\n"))

    def request(item):
        identity, payload = item
        result = http("/v1/completions", dict(model=model, return_token_ids=True, **payload))
        ids = result["choices"][0].get("token_ids")
        assert ids and len(ids) == result["usage"]["completion_tokens"]
        return dict(config_id=identity, request=payload, response=result)

    def wave(name, order, mode):
        start = boundary()
        print("MIXED_ISOLATION", name, order, flush=True)
        began = time.time()
        extra = {"native": {}, "host": {"logprobs": 0}, "neutral_bias": {"logit_bias": {0: 0.0}}}[mode]
        requests = [(i, configs[i] | extra) for i in order]
        with ThreadPoolExecutor(max_workers=10) as pool:
            rows = list(pool.map(request, requests))
        end = boundary()
        raw = args.server_log.read_bytes()[start["byte"] : end["byte"]]
        events = []
        for line in raw.decode(errors="replace").splitlines():
            pos = line.find('{"event": "sampling_mode"')
            if pos >= 0:
                events.append(json.loads(line[pos:]))
        entry = dict(
            name=name,
            order=order,
            started_unix=began,
            finished_unix=time.time(),
            log_start=start,
            log_end=end,
            server_log=str(args.server_log),
            rows=rows,
            route_counts=dict(Counter(f"{e['phase']}:{e['mode']}" for e in events)),
            sampling_events=events,
            prefill_signatures=[
                dict(mode=e["mode"], starts=e["starts"], ends=e["ends"]) for e in events if e["phase"] == "prefill"
            ],
        )
        report["waves"].append(entry)
        save()
        return {r["config_id"]: r["response"]["choices"][0] for r in rows}

    controls = {}
    for mode, pairs in [
        ("native", args.native_pairs),
        ("host", args.host_pairs),
        ("neutral_bias", args.neutral_bias_pairs),
    ]:
        for repeat in range(pairs):
            order = list(range(10))
            first = wave(f"{mode}_{repeat}_original", order, mode)
            shuffled = order.copy()
            random.Random(20261105 + repeat).shuffle(shuffled)
            second = wave(f"{mode}_{repeat}_shuffled", shuffled, mode)
            controls[(mode, repeat, "original")] = first
            controls[(mode, repeat, "shuffled")] = second
            report["comparisons"].append(
                dict(
                    mode=mode,
                    repeat=repeat,
                    rows=[
                        dict(
                            config_id=i,
                            token_ids_equal=first[i]["token_ids"] == second[i]["token_ids"],
                            text_equal=first[i]["text"] == second[i]["text"],
                            first_difference=next(
                                (
                                    j
                                    for j, (a, b) in enumerate(zip(first[i]["token_ids"], second[i]["token_ids"]))
                                    if a != b
                                ),
                                None,
                            ),
                        )
                        for i in order
                    ],
                )
            )
            save()
    report["cross_host_controls"] = [
        dict(
            repeat=repeat,
            order=order,
            rows=[
                dict(
                    config_id=i,
                    token_ids_equal=controls[("host", repeat, order)][i]["token_ids"]
                    == controls[("neutral_bias", repeat, order)][i]["token_ids"],
                    text_equal=controls[("host", repeat, order)][i]["text"]
                    == controls[("neutral_bias", repeat, order)][i]["text"],
                )
                for i in range(10)
            ],
        )
        for repeat in range(min(args.host_pairs, args.neutral_bias_pairs))
        for order in ("original", "shuffled")
    ]
    report["status"] = "captured_for_review"
    save()
    print(json.dumps(report["comparisons"]), flush=True)


if __name__ == "__main__":
    main()
