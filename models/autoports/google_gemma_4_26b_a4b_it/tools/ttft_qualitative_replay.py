# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Separate correctness replay of three consecutive greedy calls per suite prompt."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--sampled-seed", type=int, default=71)
    args = parser.parse_args()
    suite_bytes = args.suite.read_bytes()
    suite = json.loads(suite_bytes)
    if not isinstance(suite, list) or len(suite) != 6:
        parser.error("Expected the official six-prompt qualitative output list")
    if any(
        entry.get("prompt_mode") != "chat"
        or not isinstance(entry.get("prompt"), str)
        or not isinstance(entry.get("greedy_completion"), str)
        for entry in suite
    ):
        parser.error("Each suite entry must contain a chat prompt and greedy_completion")
    if args.output.resolve() == args.suite.resolve():
        parser.error("Replay output must not overwrite the official suite artifact")

    import httpx
    from openai import OpenAI
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    assert tokenizer.chat_template, "Pinned tokenizer must provide the official chat template"
    request_params = dict(model=MODEL_ID, temperature=0, max_tokens=256)
    report = dict(
        scope="Correctness-only sequential greedy replay; not final performance measurement or proof of trace reuse",
        model=MODEL_ID,
        revision=REVISION,
        tokenizer_class=type(tokenizer).__name__,
        prompt_mode="chat",
        suite=str(args.suite.resolve()),
        suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),
        base_url=args.base_url,
        endpoint="/v1/chat/completions",
        chat_template_present=True,
        chat_template=tokenizer.chat_template,
        request_params=request_params,
        repeats=3,
        sampled_control_params=dict(temperature=0.7, top_p=0.9, seed=args.sampled_seed, max_tokens=256),
        prompts=[],
        passed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    try:
        server_root = args.base_url.rstrip("/").removesuffix("/v1")
        response = httpx.get(server_root + "/server_info", timeout=30, headers={"Authorization": "Bearer EMPTY"})
        response.raise_for_status()
        report["server_info"] = response.json()
        with OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600, max_retries=0) as client:
            for index, entry in enumerate(suite):
                messages = [{"role": "user", "content": entry["prompt"]}]
                record = dict(
                    index=index,
                    messages=messages,
                    rendered_prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    prompt_token_ids=tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=False
                    ),
                    expected_greedy=entry["greedy_completion"],
                    responses=[],
                )
                report["prompts"].append(record)
                save()
                for repeat in range(3):
                    completion = client.chat.completions.create(messages=messages, **request_params)
                    assert len(completion.choices) == 1, "Expected one greedy completion"
                    choice = completion.choices[0]
                    text = choice.message.content
                    result = dict(
                        repeat=repeat,
                        response_id=completion.id,
                        response_model=completion.model,
                        text=text,
                        finish_reason=choice.finish_reason,
                        usage=None if completion.usage is None else completion.usage.model_dump(),
                        matches_suite=text == entry["greedy_completion"],
                        matches_first_repeat=repeat == 0 or text == record["responses"][0]["text"],
                    )
                    record["responses"].append(result)
                    save()
                    assert isinstance(text, str), "Chat completion returned no text content"
                record["passed"] = all(
                    result["matches_suite"] and result["matches_first_repeat"] for result in record["responses"]
                )
                sampled = client.chat.completions.create(
                    messages=messages, model=MODEL_ID, **report["sampled_control_params"]
                )
                choice = sampled.choices[0]
                record["sampled_control"] = dict(
                    text=choice.message.content,
                    finish_reason=choice.finish_reason,
                    usage=None if sampled.usage is None else sampled.usage.model_dump(),
                )
                assert isinstance(choice.message.content, str), "Seeded sampled control returned no text"
                save()
        report["passed"] = all(record["passed"] for record in report["prompts"])
        assert report["passed"], "Greedy replay mismatch; inspect saved exact texts"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
