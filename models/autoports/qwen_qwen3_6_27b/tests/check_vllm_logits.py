# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit host-logprob diagnostic, separate from device-sampled performance."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standalone", type=Path)
    args = parser.parse_args()
    model = "Qwen/Qwen3.6-27B"
    revision = "8f33b98937c6031bb49e713dee3e8e185aafd26f"
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    source = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = [p.strip() for p in source.read_text().split("\n\n") if p.strip()]
    ids = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [dict(role="user", content=p)], tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            add_special_tokens=False,
        )
        for p in prompts
    ]

    def request(index):
        response = requests.post(
            args.server_url + "/v1/completions",
            json=dict(
                model=model,
                prompt=ids[index],
                temperature=0.0,
                max_tokens=1,
                logprobs=20,
                return_tokens_as_token_ids=True,
                return_token_ids=True,
            ),
            timeout=900,
        )
        response.raise_for_status()
        raw = response.json()
        choice = raw["choices"][0]
        scores = choice["logprobs"]["top_logprobs"][0]
        assert len(scores) >= 20, raw
        return dict(
            prompt_id=index,
            prompt_token_ids=ids[index],
            response=raw,
            top_logprobs={key.removeprefix("token_id:"): value for key, value in scores.items()},
        )

    runs = [dict(kind="solo", rows=[request(i) for i in range(len(prompts))])]
    for order in (list(reversed(range(len(prompts)))), list(range(len(prompts)))):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(order)) as pool:
            rows = list(pool.map(request, order))
        runs.append(dict(kind="concurrent", order=order, rows=rows))
    baseline = {r["prompt_id"]: r for r in runs[0]["rows"]}
    comparisons = []
    for run in runs[1:]:
        for row in run["rows"]:
            ref = baseline[row["prompt_id"]]["top_logprobs"]
            actual = row["top_logprobs"]
            assert actual.keys() == ref.keys(), (row["prompt_id"], actual, ref)
            error = max(abs(actual[key] - ref[key]) for key in ref)
            assert error <= 1e-5, (row["prompt_id"], error)
            comparisons.append(dict(prompt_id=row["prompt_id"], max_abs_logprob_error=error))
    standalone = []
    if args.standalone:
        controls = json.loads(args.standalone.read_text())["outputs"]
        for control in controls:
            if control["sampled"]:
                continue
            row = baseline[control["prompt_id"]]
            assert row["prompt_token_ids"] == control["prompt_token_ids"]
            actual, ref = row["top_logprobs"], control["prefill_top_logprobs"]
            shared = set(actual) & set(ref)
            assert max(actual, key=actual.get) == max(ref, key=ref.get)
            error = max(abs(actual[key] - ref[key]) for key in shared)
            assert len(shared) >= 19 and error < 0.05, (control["prompt_id"], error, len(shared))
            standalone.append(
                dict(prompt_id=control["prompt_id"], shared_top20=len(shared), max_abs_logprob_error=error)
            )
    result = dict(
        passed=True,
        mode="explicit optional host-logprob diagnostic",
        model=model,
        revision=revision,
        thinking=False,
        runs=runs,
        comparisons=comparisons,
        standalone_path=str(args.standalone) if args.standalone else None,
        standalone_comparisons=standalone,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))


if __name__ == "__main__":
    main()
