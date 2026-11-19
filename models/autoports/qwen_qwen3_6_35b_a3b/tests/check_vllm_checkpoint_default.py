# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify omitted chat-template kwargs preserve checkpoint thinking behavior."""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    messages = [{"role": "user", "content": "What is 2 + 2? Give a short answer."}]
    result = {"status": "RUNNING", "model": MODEL, "revision": REVISION, "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for mode in ("default", "on", "off", "default_after_off"):
        kwargs = {} if mode.startswith("default") else {"enable_thinking": mode == "on"}
        payload = dict(
            model=MODEL, messages=messages, temperature=0, top_k=20, seed=0, max_tokens=256, return_token_ids=True
        )
        tokenize_payload = dict(model=MODEL, messages=messages, add_generation_prompt=True)
        if kwargs:
            payload["chat_template_kwargs"] = kwargs
            tokenize_payload["chat_template_kwargs"] = kwargs
        response = requests.post(args.url + "/tokenize", json=tokenize_payload, timeout=30)
        response.raise_for_status()
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        expected_tokens = tokenizer.encode(rendered, add_special_tokens=False)
        assert response.json()["tokens"] == expected_tokens
        response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=600)
        response.raise_for_status()
        result["rows"].append(
            dict(
                mode=mode,
                request=payload,
                prompt_tokens=expected_tokens,
                rendered_prompt=rendered,
                response=response.json(),
            )
        )
        save()
    choices = [row["response"]["choices"][0] for row in result["rows"]]
    for index in (0, 1, 3):
        assert choices[index]["message"].get("reasoning") or choices[index]["message"].get("reasoning_content")
        assert choices[index]["token_ids"] == choices[0]["token_ids"]
        assert choices[index]["finish_reason"] == choices[0]["finish_reason"]
    off = choices[2]["message"]
    assert not (off.get("reasoning") or off.get("reasoning_content"))
    assert "4" in off["content"]
    result["status"] = "PASS"
    result["checks"] = [
        "all four HF/server tokenizations exact",
        "omitted equals explicit on",
        "explicit off returns direct content",
        "default preserved after off request",
    ]
    save()
    print("CHECKPOINT_DEFAULT_PASS", flush=True)


if __name__ == "__main__":
    main()
