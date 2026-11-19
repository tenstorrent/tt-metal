# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve reasoning and content from the shared six-prompt serving suite.

The packaged collector reads only message.content and raises when a reasoning
model is still thinking at its 256-token cap. This collector keeps both fields,
the rendered prompt and the complete response; it does not change the prompt or
the generation budget. Output requires an explicit qualitative review.
"""

import argparse
import json
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen3.6-35B-A3B"
    revision = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    template_kwargs = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
    metadata = dict(
        model=model,
        revision=revision,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=bool(tokenizer.chat_template),
        prompt_mode="chat",
        endpoint="/v1/chat/completions",
        suite=str(suite),
        max_tokens=256,
        thinking=args.thinking,
        chat_template_kwargs=template_kwargs,
        sampled=dict(temperature=0.7, top_p=0.9, top_k="checkpoint default 20"),
        greedy=dict(temperature=0.0),
    )
    (args.output_dir / "qualitative_prompt_format.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    for index, prompt in enumerate(filter(None, (p.strip() for p in suite.read_text().split("\n\n"))), 1):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        row = dict(
            prompt_id=f"q{index:02}",
            prompt=prompt,
            prompt_mode="chat",
            messages=messages,
            rendered_prompt=rendered,
            prompt_tokens=tokenizer.encode(rendered, add_special_tokens=False),
        )
        for mode, params in (("greedy", dict(temperature=0.0)), ("sampled", dict(temperature=0.7, top_p=0.9))):
            payload = dict(model=model, messages=messages, max_tokens=256, seed=0, top_k=20, **params)
            if template_kwargs:
                payload["chat_template_kwargs"] = template_kwargs
            response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=1800)
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            content = message.get("content") or ""
            if not reasoning and not content:
                raise AssertionError("Neither reasoning nor content was returned")
            row[mode + "_request"] = payload
            row[mode + "_response"] = body
            row[mode + "_completion"] = reasoning + ("\n</think>\n" if reasoning and content else "") + content
            print(f"{row['prompt_id']} {mode}: {body['choices'][0]['finish_reason']}, {body['usage']}", flush=True)
        results.append(row)
        (args.output_dir / "vllm_qualitative_outputs.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
