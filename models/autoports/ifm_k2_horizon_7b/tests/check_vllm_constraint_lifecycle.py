"""Repeat real allowlist-to-bias request transitions against a live server."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    controls = json.loads(Path(args.control).read_text())
    by_id = {row["id"]: row for row in controls["cases"]}
    candidate = controls["bias_control"]["candidate"]
    alternative = controls["bias_control"]["alternative"]
    records, failures = [], []

    def request(case):
        endpoint = "/v1/chat/completions" if case["chat"] else "/v1/completions"
        response = requests.post(args.url + endpoint, json=case["request"], timeout=300)
        response.raise_for_status()
        data = response.json()
        assert "error" not in data, data
        return dict(id=case["id"], request=case["request"], response=data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for cycle in range(args.cycles):
            warm = list(pool.map(request, [row for row in controls["cases"] if row["id"].startswith("allowed_")]))
            inputs = [by_id["bias_unbanned"], by_id["bias_banned"]]
            if cycle % 2:
                inputs.reverse()
            results = list(pool.map(request, inputs))
            for row in results:
                expected = candidate if row["id"] == "bias_unbanned" else alternative
                if row["response"]["choices"][0]["token_ids"] != [expected]:
                    failures.append(dict(cycle=cycle, **row))
            records.append(dict(cycle=cycle, warm=warm, control=results))
            Path(args.output).write_text(
                json.dumps(
                    dict(control=args.control, records=records, failures=failures, passed=not failures), indent=2
                )
                + "\n"
            )
            print(cycle, "failures", len(failures), flush=True)
    assert not failures, f"{len(failures)} biased requests lost their constraint"
    print("CONSTRAINT_LIFECYCLE_PASS", args.output)


if __name__ == "__main__":
    main()
