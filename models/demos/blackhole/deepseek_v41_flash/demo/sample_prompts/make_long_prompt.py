# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate a long-context prompt file (json list with one {"prompt": ...}) of EXACTLY ``--isl`` tokens (DeepSeek-V4.1 tokenizer, chat template off,
BOS included) by tiling real text: the cached Gutenberg book of the tt_transformers long-prompt files plus the GSM8K questions.

    python make_long_prompt.py --isl 262144 --out input_data_long_256k.json
"""
import argparse
import json
import os

CKPT = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")
BOOK = "models/tt_transformers/demo/context_cache/4c1a705addfd44fe41b8ce06d83e3fd5"
GSM = "/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl"
QUESTION = "\n\nExplicitly state the quotes directly taken from the book inside double quotes literally, and summarise the story."


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CKPT)
    text = open(BOOK).read() if os.path.exists(BOOK) else ""
    text += "\n".join(json.loads(l)["question"] + "\n" + json.loads(l)["answer"] for l in open(GSM))
    ids = tok.encode(text, add_special_tokens=False)
    q = tok.encode(QUESTION, add_special_tokens=False)
    body = []
    while len(body) < a.isl:
        body += ids
    n = a.isl - 1 - len(q)  # BOS + body + question = isl
    prompt_ids = body[:n] + q
    prompt = tok.decode(prompt_ids)
    re = len(tok.encode(tok.bos_token + prompt, add_special_tokens=False))
    print(f"isl requested {a.isl}, re-encoded {re}")
    json.dump([{"prompt": prompt}], open(a.out, "w"))


if __name__ == "__main__":
    main()
