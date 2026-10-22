#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Create exact-length chat input from text; no reference model or KV data."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from transformers import AutoTokenizer


def create_prompt(tok, tokens, model_path):
    nonce = str(time.time_ns())

    def encode(repeats):
        prompt = f"Unique request {nonce}. " + "hello " * repeats + "Briefly summarize the preceding text."
        messages = [{"role": "user", "content": prompt}]
        result = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ids = result["input_ids"] if hasattr(result, "keys") else result
        return messages, list(ids)

    lo, hi = 0, tokens + 1024
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(encode(mid)[1]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    messages, ids = encode(lo)
    if len(ids) != tokens:
        raise ValueError(f"exact token length required: requested {tokens}, got {len(ids)}")
    payload = dict(
        model="gpt-oss-120b-4x2-chat",
        messages=messages,
        max_tokens=32,
        stream=True,
        stream_options={"include_usage": True},
    )
    return dict(
        input_tokens=len(ids),
        token_ids=ids,
        payload=payload,
        tokenizer_path=model_path,
        sha256_token_ids=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        template_sha256=hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),
        nonce=nonce,
        source="fresh text tokenization; no golden",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    record = create_prompt(tok, args.tokens, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(record, f)
        f.write("\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("token_ids", "payload")}))


if __name__ == "__main__":
    main()
