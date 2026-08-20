# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the accepted no-thinking chat qualitative suite against a vLLM server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_PROMPT_SOURCE = MODEL_DIR / "readiness_vllm/vllm_chat_no_think_qualitative_outputs.json"
DEFAULT_OUTPUT = MODEL_DIR / "readiness_vllm/vllm_chat_no_think_qualitative_outputs.json"


def _load_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        prompts: list[str] = []
        seen = set()
        for record in data["records"]:
            prompt = str(record["prompt"])
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
        if prompts:
            return prompts
    raise RuntimeError(f"Could not load qualitative prompts from {path}")


def _request(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    profile: str,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
    }
    if profile == "greedy":
        payload["temperature"] = 0
    elif profile == "sampled":
        payload.update({"temperature": 0.7, "top_p": 0.9, "seed": seed})
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    started = time.time()
    response = requests.post(url, json=payload, timeout=900)
    elapsed_s = time.time() - started
    response_json = response.json()
    return {
        "payload": payload,
        "status_code": response.status_code,
        "elapsed_s": elapsed_s,
        "text": response_json["choices"][0]["message"]["content"],
        "response": response_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8011")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--sample-seed-base", type=int, default=2000)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt_source)
    endpoint = f"{args.server_url.rstrip('/')}/v1/chat/completions"
    records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts, 1):
        for profile in ("greedy", "sampled"):
            record = _request(
                url=endpoint,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                profile=profile,
                seed=args.sample_seed_base + prompt_index,
            )
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "profile": profile,
                    **record,
                }
            )
            preview = record["text"].replace("\n", " ")[:96]
            print(f"{prompt_index} {profile}: {preview!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"url": endpoint, "model": args.model, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
