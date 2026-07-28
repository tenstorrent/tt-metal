#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compute the exact ``DG_UPFRONT_PREFILL_WARMUP_LENS`` whitelist for an lm_eval task.

The up-front traced path pre-enumerates the prompt lengths it will admit, and rejects anything else
at request time. That whitelist is therefore a function of the task prompt, the chat template, the
tokenizer, and whether thinking mode is on — ``run_upfront_gpqa.sh`` says as much and tells you to
recompute it when any of those change. Switching from ``r1_gpqa_diamond`` to
``gpqa_diamond_cot_zeroshot`` changes the task prompt, so the old list is wrong and every prompt
whose padded length is missing from it would abort the run.

This does not estimate the lengths. It renders the prompts through **lm_eval's own task object**, so
the strings are the ones the eval will actually send, then applies the model's chat template exactly
as ``--apply_chat_template`` does and tokenizes with the checkpoint's tokenizer.

Usage (on the device box, where the tokenizer and the cached dataset live)::

    python compute_prefill_whitelist.py --task gpqa_diamond_cot_zeroshot --thinking 1
    python compute_prefill_whitelist.py --task gpqa_diamond_cot_zeroshot --thinking 0

Prints the comma-separated whitelist plus the distribution, and warns when the longest prompt would
not fit the served ``--max-model-len``.
"""
from __future__ import annotations

import argparse
import sys

TILE = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gpqa_diamond_cot_zeroshot")
    ap.add_argument("--checkpoint", default="/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    ap.add_argument("--thinking", type=int, choices=(0, 1), default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--canvas", type=int, default=256)
    args = ap.parse_args()

    from lm_eval.tasks import TaskManager, get_task_dict
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    tasks = get_task_dict([args.task], TaskManager())
    task = tasks[args.task]
    while isinstance(task, dict):  # grouped tasks nest one level
        task = next(iter(task.values()))

    docs = list(task.eval_docs)
    print(f"task={args.task} docs={len(docs)} thinking={args.thinking}", file=sys.stderr)

    lengths = []
    for doc in docs:
        text = task.doc_to_text(doc)
        # --apply_chat_template sends a single user turn and asks for the generation prompt, which is
        # where the thinking template is injected.
        out = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(args.thinking),
        )
        # apply_chat_template(tokenize=True) returns a BatchEncoding on current transformers, so
        # len() on it counts DICT KEYS (2) rather than tokens. Getting this wrong silently produced
        # a one-entry whitelist of "32" for all 198 prompts.
        if hasattr(out, "keys") or isinstance(out, dict):
            ids = out["input_ids"]
        else:
            ids = out
        while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]  # unwrap a batch dimension
        lengths.append(len(ids))

    # The serving path right-pads each prompt to a tile before prefill, so the whitelist is over the
    # PADDED lengths, not the raw ones.
    padded = sorted({max(TILE, -(-n // TILE) * TILE) for n in lengths})
    print()
    print('DEFAULT_PREFILL_WARMUP_LENS="' + ",".join(str(n) for n in padded) + '"')
    print()
    print(
        f"distinct padded lengths: {len(padded)}  raw min/median/max: "
        f"{min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}"
    )

    # A prompt whose raw length is already a 32-multiple has ZERO margin: one extra token from a
    # template tweak, a tokenizer bump or a reworded task pushes it into the next tile, which is not
    # in this list. On the 07-28 CoT set doc 127 tokenizes to exactly 2432 -- dead on the largest
    # entry. That no longer aborts a run (the generator rejects the request and keeps serving) but it
    # still silently costs that question, so it is worth knowing before the run rather than after.
    exact = sorted(n for n in lengths if n % TILE == 0)
    if exact:
        print()
        print(f"NOTE: {len(exact)} prompt(s) sit exactly on a {TILE}-token boundary: {exact}")
        print("      A +1-token change to the template or tokenizer moves each into the next tile,")
        print("      which this list does not contain. Recompute whenever either changes.")

    budget = args.max_model_len - args.canvas
    over = [n for n in padded if n > budget]
    if over:
        print()
        print(f"WARNING: {len(over)} padded length(s) exceed max_model_len - canvas = {budget}: {over}")
        print("         Those requests cannot be served at this --max-model-len; raise it or the")
        print("         run will reject them at admission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
