# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native seeded continuation across concurrent request admission and departure."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    prompts = [
        "Explain why the Moon has phases. Use a clear example involving sunlight.",
        "Describe how to bake bread from flour, water, yeast and salt, in order.",
    ]

    def request(job):
        index, count = job
        payload = dict(
            model=model,
            messages=[{"role": "user", "content": prompts[index]}],
            max_tokens=count,
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            seed=42 + index,
            ignore_eos=True,
            return_token_ids=True,
        )
        response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=900)
        response.raise_for_status()
        data = response.json()
        assert data["usage"]["completion_tokens"] == count, data
        token_ids = data["choices"][0]["token_ids"]
        assert len(token_ids) == count, token_ids
        return dict(prompt_id=index, request=payload, response=data, token_ids=token_ids)

    rows = [request((0, 100)), request((1, 47))]
    for jobs in ([(1, 33), (0, 100)], [(0, 100), (1, 47)]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows.extend(pool.map(request, jobs))
    baseline = {row["prompt_id"]: row["token_ids"] for row in rows[:2]}
    matches = [baseline[row["prompt_id"]][: row["request"]["max_tokens"]] == row["token_ids"] for row in rows]
    args.output.write_text(
        json.dumps(
            dict(
                mode="native top_k5 temperature0.7 top_p0.9 seeds42/43; compatibility must not force host",
                scheduler_row_placement="not forced; companion admission/departure varied by HTTP workloads",
                rows=rows,
                exact_continuation_matches=matches,
            ),
            indent=2,
        )
        + "\n"
    )
    assert all(matches), "Seeded continuation changed with companions; inspect saved responses"
    print("NATIVE_SEED_CONTINUITY_PASS", matches)


if __name__ == "__main__":
    main()
