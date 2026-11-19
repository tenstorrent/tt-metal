# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blind paired controls after the same pinned response parser on both sides."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--http", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh blind artifact")
    sources = {"hf": json.loads(args.hf.read_text()), "http": json.loads(args.http.read_text())}
    left, right = sources.values()
    for field in ("model", "revision", "messages", "prompt_tokens", "max_tokens", "generation"):
        assert left[field] == right[field], field
    assert left.get("presence_penalty", 0.0) == right.get("presence_penalty", 0.0)
    assert all(s["status"] == "PASS" and len(s["outputs"]) == 10 for s in sources.values())
    tokenizer = AutoTokenizer.from_pretrained(left["model"], revision=left["revision"], local_files_only=True)
    rng = random.Random(77305)
    rows, key, audit = [], {}, []
    for backend, source in sources.items():
        kwargs = source.get("chat_template_kwargs", {})
        request = ChatCompletionRequest(model=source["model"], messages=source["messages"], chat_template_kwargs=kwargs)
        for row in source["outputs"]:
            tokens = row["tokens"] if backend == "hf" else row["response"]["choices"][0]["token_ids"]
            decoded = tokenizer.decode(tokens, skip_special_tokens=False).replace("<|im_end|>", "")
            response_parser = Qwen3Parser(tokenizer, chat_template_kwargs=kwargs)
            response_parser.adjust_initial_state_from_prompt(source["prompt_tokens"])
            reasoning, content, tool_calls = response_parser.parse(
                decoded, request, enable_auto_tools=True, model_output_token_ids=tokens
            )
            assert not tool_calls
            if backend == "http":
                actual = row["response"]["choices"][0]["message"]
                # Compare edge whitespace separately from substantive content.
                assert (content or "").strip() == (actual.get("content") or "").strip(), row["seed"]
                assert (reasoning or "").strip() == (actual.get("reasoning") or "").strip(), row["seed"]
                audit.append(dict(seed=row["seed"], parsed_matches_api=True))
            identity = f"response_{rng.randrange(10**8):08d}"
            key[identity] = dict(
                backend=backend, seed=row["seed"], artifact=str(args.hf if backend == "hf" else args.http)
            )
            rows.append(
                dict(
                    id=identity,
                    prompt=source["messages"][0]["content"],
                    max_tokens=source["max_tokens"],
                    reasoning=reasoning,
                    content=content,
                )
            )
    rng.shuffle(rows)
    result = dict(
        rubric="Score actual parsed reasoning and final content separately. Empty final content is an availability failure. For secondary quality, flag clear sustained looping, gibberish, unrelated content or contamination. Coherent truncated reasoning is not a secondary failure. Assess coherence, topic, repetition, gibberish, language drift and request contamination. No backend labels are provided.",
        comparison_rule="Serving fails when its failure rate exceeds native by more than 20 percentage points. Preserve earlier raw-token review separately; this review assesses parsed response fields.",
        outputs=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".key.json").write_text(json.dumps(key, indent=2) + "\n")
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(rows)} blinded responses; {len(audit)} actual API comparisons passed")


if __name__ == "__main__":
    main()
