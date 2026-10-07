# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the decode tests: GSM8K prompts in the non-thinking chat template, row/slot assignment."""
import json

GSM8K = "/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl"


def gsm_prompts(n, tok, start=0):
    """first n GSM8K test rows -> list of dict(idx, question, prompt_ids, gold) using the chat template with enable_thinking=False ('<think></think>' after the assistant tag)"""
    from models.demos.blackhole.mimo_v26.reference.score_gsm8k import norm

    rows = [json.loads(l) for l in open(GSM8K)][start : start + n]
    items = []
    for i, r in enumerate(rows):
        txt = tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        items.append(
            dict(
                idx=start + i,
                question=r["question"],
                prompt=txt,
                prompt_ids=tok(txt, add_special_tokens=False)["input_ids"],
                gold=norm(r["answer"].split("####")[-1].strip()),
            )
        )
    return items
