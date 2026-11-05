# SPDX-License-Identifier: Apache-2.0
"""Capture token-level evidence for temperature variety and frequency batch failures."""

import argparse
import json
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/sampling_failure_probe.json"),
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def http(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    model = args.model or http("/v1/models")["data"][0]["id"]
    report = {
        "source": "vllm",
        "model": model,
        "scope": "targeted correctness diagnosis, not performance",
        "waves": [],
        "limitations": [
            "Unseeded temperature variation is recorded, not assumed for a finite sample.",
            "Character repetition is not a token-level frequency-penalty correctness criterion.",
            "logprobs=0 explicitly holds TP4 controls on the stock host sampling route; native waves retain original routing.",
        ],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def request(item):
        payload = dict(model=model, return_token_ids=True, **item)
        response = http("/v1/completions", payload)
        choice = response["choices"][0]
        ids = choice.get("token_ids")
        assert isinstance(ids, list) and ids, response
        assert response["usage"]["completion_tokens"] == len(ids), response
        return {
            "request": item,
            "text": choice["text"],
            "token_ids": ids,
            "first_token_id": ids[0],
            "first_character": choice["text"][:1],
            "finish_reason": choice.get("finish_reason"),
            "usage": response["usage"],
            "token_counts": dict(Counter(ids)),
            "character_a_count": choice["text"].count("a"),
        }

    def wave(name, requests):
        print("SAMPLING_FAILURE_PROBE", name, len(requests), flush=True)
        began = time.time()
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            rows = list(pool.map(request, requests))
        entry = {
            "name": name,
            "started_unix": began,
            "finished_unix": time.time(),
            "concurrency": len(requests),
            "rows": rows,
        }
        report["waves"].append(entry)
        save()
        return rows

    for repeat in range(args.temperature_repeats):
        rows = wave(
            f"temperature_native_B7_{repeat}",
            [dict(prompt="Random letter: ", max_tokens=10, temperature=2.0, top_p=1.0) for _ in range(7)],
        )
        report["waves"][-1]["variety"] = {
            "unique_outputs": len({tuple(r["token_ids"]) for r in rows}),
            "unique_first_token_ids": len({r["first_token_id"] for r in rows}),
            "unique_first_characters": len({r["first_character"] for r in rows}),
        }
        save()

    prompt = "a a a a a a a a a"

    def frequency(penalty, host=False):
        item = dict(prompt=prompt, max_tokens=15, temperature=0.0, top_p=1.0, frequency_penalty=penalty)
        if host:
            item["logprobs"] = 0
        return item

    native = wave("frequency_native_mixed", [frequency(0.0 if i % 2 == 0 else 2.0) for i in range(args.batch_size)])
    solo = {penalty: wave(f"frequency_solo_host_{penalty}", [frequency(penalty, True)])[0] for penalty in (0.0, 2.0)}
    controls = {}
    for name, penalties in (
        ("clean", [0.0] * args.batch_size),
        ("penalized", [2.0] * args.batch_size),
        ("mixed", [0.0 if i % 2 == 0 else 2.0 for i in range(args.batch_size)]),
    ):
        controls[name] = wave(f"frequency_host_{name}", [frequency(value, True) for value in penalties])
    comparisons = []
    for i, mixed in enumerate(controls["mixed"]):
        penalty = 0.0 if i % 2 == 0 else 2.0
        homogeneous = controls["clean" if penalty == 0 else "penalized"][i]
        comparisons.append(
            {
                "slot": i,
                "penalty": penalty,
                "same_slot_homogeneous_ids_equal": mixed["token_ids"] == homogeneous["token_ids"],
                "solo_ids_equal": mixed["token_ids"] == solo[penalty]["token_ids"],
                "native_mixed_ids_equal": mixed["token_ids"] == native[i]["token_ids"],
            }
        )
    report["frequency_controls"] = {
        "comparisons": comparisons,
        "native_first_pair_a_counts": [row["character_a_count"] for row in native[:2]],
        "host_first_pair_token_ids_equal": controls["mixed"][0]["token_ids"] == controls["mixed"][1]["token_ids"],
        "host_first_token_ids": [row["first_token_id"] for row in controls["mixed"]],
        "interpretation": "Frequency penalty applies generated-token counts, so equal first-token decisions are expected before any output history exists. Inspect later token-level changes and same-policy controls.",
    }
    report["status"] = "captured_for_review"
    save()
    print(json.dumps({"output": str(args.output), "waves": len(report["waves"])}), flush=True)


if __name__ == "__main__":
    main()
