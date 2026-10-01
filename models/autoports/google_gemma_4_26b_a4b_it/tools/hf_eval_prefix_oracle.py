# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Bounded CPU-only HF next-action control; invoke in a memory-limited container.

The caller supplies saved trajectory JSON on stdin. This is a qualitative
control, not a speed comparison or an evaluation-score measurement.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from replay_eval_requests import TOOLS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


class ProgressStreamer(BaseStreamer):
    def __init__(self, tokenizer, path, started):
        self.tokenizer, self.path, self.started = tokenizer, path, started
        self.prompt_seen = False
        self.tokens = []

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.tokens.extend(value.reshape(-1).tolist())
        if len(self.tokens) == 1 or len(self.tokens) % 16 == 0:
            self.save()

    def save(self):
        report = {
            "event": "hf_generation_progress",
            "tokens": len(self.tokens),
            "seconds": time.monotonic() - self.started,
            "output_token_ids": self.tokens,
            "completion": self.tokenizer.decode(self.tokens, skip_special_tokens=False),
            "scope": "partial diagnostic output, not a completed reference",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("event", "tokens", "seconds")}), flush=True)

    def end(self):
        self.save()


def main():
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    saved = json.load(sys.stdin)["messages"][: args.messages]
    messages = [{k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}} for m in saved]
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=True, return_dict=False
    )
    print(
        json.dumps({"event": "prompt_ready", "tokens": len(tokens), "seconds": time.monotonic() - started}), flush=True
    )
    token_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    if len(tokens) != args.expected_prompt_tokens or token_hash != args.expected_prompt_sha256:
        raise ValueError(f"HF prompt differs from serving request: count={len(tokens)} sha256={token_hash}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    loaded = time.monotonic()
    print(json.dumps({"event": "model_loaded", "seconds": loaded - started, "dtype": str(model.dtype)}), flush=True)
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([tokens]),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=ProgressStreamer(tokenizer, args.output.with_suffix(".progress.json"), loaded),
        )
    generated = output[0, len(tokens) :].tolist()
    report = {
        "model": MODEL,
        "revision": REVISION,
        "prompt_tokens": len(tokens),
        "prompt_sha256": token_hash,
        "thinking": True,
        "dtype": str(model.dtype),
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": loaded - started,
        "generation_seconds": time.monotonic() - loaded,
        "output_token_ids": generated,
        "completion": tokenizer.decode(generated, skip_special_tokens=False),
        "scope": "greedy next-action control only; no tool execution or solve/reward claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
