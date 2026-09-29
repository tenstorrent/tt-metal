# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Concurrent greedy replay of the pinned six-prompt chat suite."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ttft_qualitative_replay import MODEL_ID, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    suite_bytes = args.suite.read_bytes()
    suite = json.loads(suite_bytes)
    assert len(suite) == 6 and all(row["prompt_mode"] == "chat" for row in suite)
    assert args.output.resolve() != args.suite.resolve()
    import httpx
    from openai import OpenAI
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    assert tokenizer.chat_template
    server_root = args.base_url.rstrip("/").removesuffix("/v1")
    response = httpx.get(server_root + "/server_info", timeout=30)
    response.raise_for_status()
    report = {
        "scope": "Concurrent correctness-only greedy suite replay; not a throughput measurement",
        "model": MODEL_ID,
        "revision": REVISION,
        "endpoint": "/v1/chat/completions",
        "prompt_mode": "chat",
        "chat_template": tokenizer.chat_template,
        "tokenizer_class": type(tokenizer).__name__,
        "suite": str(args.suite),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "server_info": response.json(),
        "concurrency": 18,
        "repeats": 3,
        "request_params": {"model": MODEL_ID, "temperature": 0, "max_tokens": 256, "seed": 0},
        "prompts": [],
        "responses": [],
        "passed": False,
    }
    for index, row in enumerate(suite):
        messages = [{"role": "user", "content": row["prompt"]}]
        report["prompts"].append(
            {
                "index": index,
                "messages": messages,
                "expected_greedy": row["greedy_completion"],
                "rendered_prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "prompt_token_ids": tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    with OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900, max_retries=0) as client:

        def request(index, repeat):
            record = report["prompts"][index]
            completion = client.chat.completions.create(messages=record["messages"], **report["request_params"])
            choice = completion.choices[0]
            return {
                "index": index,
                "repeat": repeat,
                "response_id": completion.id,
                "response_model": completion.model,
                "text": choice.message.content,
                "finish_reason": choice.finish_reason,
                "usage": None if completion.usage is None else completion.usage.model_dump(),
                "matches_suite": choice.message.content == record["expected_greedy"],
            }

        try:
            with ThreadPoolExecutor(max_workers=18) as executor:
                futures = [executor.submit(request, index, repeat) for repeat in range(3) for index in range(6)]
                for future in as_completed(futures):
                    report["responses"].append(future.result())
                    save()
            report["passed"] = len(report["responses"]) == 18 and all(
                row["matches_suite"] for row in report["responses"]
            )
            assert report["passed"], "Concurrent qualitative mismatch; review all saved texts"
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
