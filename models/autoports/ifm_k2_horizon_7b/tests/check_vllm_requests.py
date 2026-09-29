"""Repeatable allocator-growth and request-lifecycle requests to a live server."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--output", required=True)
    p.add_argument("--compare")
    p.add_argument("--boundary", action="store_true")
    p.add_argument("--sampled", action="store_true")
    a = p.parse_args()
    cases = [
        {"id": f"growth{i}", "prompt": [500 + i * 100 + j for j in range(length)], "max_tokens": steps}
        for i, (length, steps) in enumerate([(31, 129), (33, 101), (65, 77), (127, 35)])
    ]
    if a.boundary:
        cases += [
            {
                "id": f"boundary{length}",
                "prompt": ([500, 600, 700, 800] * ((length + 3) // 4))[:length],
                "max_tokens": 5,
            }
            for length in [4095, 4096, 4097, 4353]
        ]
    result = {"server_url": a.url, "sampled": a.sampled, "cases": cases, "serial": [], "concurrent": [], "repeat": []}

    def save():
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")

    def request(case):
        payload = {
            "model": "IFM/K2-Horizon-7B",
            "prompt": case["prompt"],
            "max_tokens": case["max_tokens"],
            "temperature": 0.7 if a.sampled else 0,
            "top_k": 32,
            "top_p": 0.9 if a.sampled else 1.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "seed": 17,
        }
        started = time.time()
        response = requests.post(a.url + "/v1/completions", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        assert "error" not in data, data
        tokens = data["choices"][0]["token_ids"]
        assert len(tokens) == case["max_tokens"], data
        assert data["usage"]["prompt_tokens"] == len(case["prompt"]), data
        return {"id": case["id"], "elapsed_seconds": time.time() - started, "response": data}

    result["metrics_before"] = requests.get(a.url + "/metrics", timeout=10).text
    for case in cases:
        result["serial"].append(request(case))
        save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result["concurrent"] = list(pool.map(request, cases[:4]))
    save()
    for case in reversed(cases[:4]):
        result["repeat"].append(request(case))
        save()
    control = {row["id"]: row["response"]["choices"][0]["token_ids"] for row in result["serial"]}
    for phase in ("concurrent", "repeat"):
        for row in result[phase]:
            assert row["response"]["choices"][0]["token_ids"] == control[row["id"]], (phase, row["id"])
    if a.compare:
        previous = json.loads(Path(a.compare).read_text())
        previous_control = {row["id"]: row["response"]["choices"][0]["token_ids"] for row in previous["serial"]}
        for key, value in previous_control.items():
            assert control[key] == value, ("comparison_control", key)
        result["comparison_control"] = a.compare
    result["metrics_after"] = requests.get(a.url + "/metrics", timeout=10).text
    result["passed"] = True
    save()
    print("REQUEST_LIFECYCLE_PASS", a.output)


if __name__ == "__main__":
    main()
