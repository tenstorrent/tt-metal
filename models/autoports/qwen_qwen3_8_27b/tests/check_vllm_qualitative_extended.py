# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Extend selected shared qualitative prompts while retaining exact request metadata."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", default="0,2,3,5")
    parser.add_argument("--modes", default="greedy,sampled")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "readiness_vllm/qualitative_prompt_format.json").read_text())
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    rows = []
    report = dict(
        prompt_format_artifact="qualitative_prompt_format.json",
        sampled_scope="New reproducible seed42 control; not an exact continuation of the unseeded shared request",
        rows=rows,
    )
    for index in map(int, args.prompt_ids.split(",")):
        prompt = metadata["rows"][index]
        for mode in args.modes.split(","):
            assert mode in ("greedy", "sampled"), mode
            payload = dict(
                model=model,
                messages=prompt["messages"],
                max_tokens=2048 if index == 2 else 1024,
                temperature=0.0 if mode == "greedy" else 0.7,
                return_token_ids=True,
            )
            if mode == "sampled":
                payload.update(top_k=20, top_p=0.9, seed=42)
            response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=900)
            response.raise_for_status()
            data = response.json()
            rows.append(dict(prompt_id=index, mode=mode, request=payload, response=data))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(index, mode, data["usage"], data["choices"][0]["finish_reason"], flush=True)
    assert all(
        row["response"]["choices"][0]["finish_reason"] == "stop" for row in rows
    ), "Review budget-limited outputs"


if __name__ == "__main__":
    main()
