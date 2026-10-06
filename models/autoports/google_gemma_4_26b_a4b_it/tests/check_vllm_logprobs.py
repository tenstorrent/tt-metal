# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numerical logprob reproducibility through the live serving scheduler."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    prompts = {"A": [2] + [100] * 30, "B": [2] + [101] * 62}
    records = []

    def request(name):
        response = requests.post(
            args.url + "/v1/completions",
            json=dict(
                model=model,
                prompt=prompts[name],
                max_tokens=1,
                temperature=0,
                ignore_eos=True,
                logprobs=20,
                return_tokens_as_token_ids=True,
                return_token_ids=True,
            ),
            timeout=600,
        )
        response.raise_for_status()
        return {"name": name, "prompt": prompts[name], "response": response.json()}

    for order in (("A",), ("B",), ("A",), ("B",), ("A", "B"), ("B", "A")):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(order)) as pool:
            records.append({"submission_order": order, "responses": list(pool.map(request, order))})
        args.output.with_suffix(".partial.json").write_text(json.dumps(records, indent=2) + "\n")
    references = {}
    comparisons = []
    for batch in records:
        for row in batch["responses"]:
            choice = row["response"]["choices"][0]
            values = choice["logprobs"]["top_logprobs"][0]
            reference = references.setdefault(row["name"], values)
            comparisons.append(
                {
                    "name": row["name"],
                    "submission_order": batch["submission_order"],
                    "exact": values == reference,
                    "same_tokens": values.keys() == reference.keys(),
                    "max_abs_diff": max(abs(values[k] - reference[k]) for k in values.keys() & reference.keys()),
                }
            )
    result = dict(
        scope="Full model through live vLLM; optional CPU logprob compatibility mode",
        records=records,
        comparisons=comparisons,
        all_exact=all(x["exact"] for x in comparisons),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["all_exact"], comparisons


if __name__ == "__main__":
    main()
