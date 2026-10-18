#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GSM8K pass@k evaluator for Qwen3 checkpoints on tt-transformers.

Generates with :class:`TTTRolloutSampler`, the same on-device path the GRPO
rollout rank uses, booted with real HuggingFace weights for each ``--models``
entry. Prints one pass@k table per criterion: correct answer, tags present,
tags exactly once, and the <think>/<answer> format regex. Rows follow the order
of ``--models``.

Every ``--models`` entry must be a HuggingFace-format checkpoint directory or
repo id; ttml GRPO checkpoints are not directly loadable here.

Usage:
    python eval_gsm8k.py \\
        --models Qwen/Qwen3-0.6B-Base=base ichovpanTT/qwen3-0.6b-base-think-sft=sft \\
        --n-samples 16 --temperature 1.0 --limit 200
"""

from __future__ import annotations

import argparse
import gc
import os
from math import comb
from typing import Callable, List, Sequence, Tuple

import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.trainers.grpo_trainer.remote_rollout.ttt_presets import bf16_attn_bfp8_mlp_optimizations, stop_and_pad
from ttml.trainers.grpo_trainer.remote_rollout.ttt_rollout_sampler import TTTRolloutSampler

from gsm8k_utils import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    FORMAT_RE,
    THINK_CLOSE,
    THINK_OPEN,
    extract_hash_answer,
    extract_last_number,
    extract_tag_answer,
    format_prompt,
)

TAGS = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, eq. 1)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def score_sample(text: str, gold: str) -> dict:
    return {
        "correct_answer": gold in (extract_tag_answer(text), extract_last_number(text)),
        "tags_present": all(t in text for t in TAGS),
        "tags_exactly_once": all(text.count(t) == 1 for t in TAGS),
        "format_regex": bool(FORMAT_RE.search(text)),
    }


CRITERIA: Tuple[Tuple[str, Callable[[dict], bool]], ...] = (
    ("correct answer", lambda s: s["correct_answer"]),
    ("tags present", lambda s: s["tags_present"]),
    ("tags exactly once", lambda s: s["tags_exactly_once"]),
    ("format-regex", lambda s: s["format_regex"]),
)


def generate_on_tt(
    model_path: str,
    prompts: Sequence[Sequence[int]],
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    max_batch_size: int,
    max_seq_len: int,
    mesh_shape: Sequence[int],
) -> List[List[List[int]]]:
    """Return ``completions[q][s]`` for one checkpoint, on a fresh sampler and mesh."""
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), offset=ttnn.MeshCoordinate(0, 0))
    sampler = None
    try:
        stop_ids, pad_id = stop_and_pad(model_path)
        sampler = TTTRolloutSampler(
            mesh_device=mesh,
            model_source=model_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            completions_per_prompt=n_samples,
            max_completion_length=max_new_tokens,
            temperature=temperature,
            dummy_weights=False,
        )
        prompts_per_call = max(1, max_batch_size * mesh.get_num_devices() // n_samples)
        result: List[List[List[int]]] = []
        for start in range(0, len(prompts), prompts_per_call):
            chunk = [list(p) for p in prompts[start : start + prompts_per_call]]
            print(
                f"[eval_gsm8k] {os.path.basename(model_path)}: prompts {start + len(chunk)}/{len(prompts)}", flush=True
            )
            completions = sampler.generate(chunk).completions
            result.extend(completions[i * n_samples : (i + 1) * n_samples] for i in range(len(chunk)))
        return result
    finally:
        if sampler is not None:
            sampler.close()
        sampler = None
        gc.collect()
        ttnn.close_mesh_device(mesh)


def parse_models(entries: Sequence[str]) -> List[Tuple[str, str]]:
    """``path=display_name``; the display name defaults to the path."""
    return [tuple(s.strip() for s in e.split("=", 1)) if "=" in e else (e, e) for e in entries]


def print_tables(rows: List[Tuple[str, List[List[List[bool]]]]], ks: Sequence[int]) -> None:
    """``rows[i] = (display, hits)`` with ``hits[criterion][question][sample]``."""
    header = "".join(f"{'pass@' + str(k):>9}" for k in ks)
    width = max((len(r[0]) for r in rows), default=10) + 2
    for crit_i, (crit_name, _) in enumerate(CRITERIA):
        print(f"\n{crit_name}")
        print(f"  {'checkpoint':<{width}}" + header)
        for display, hits in rows:
            per_question = hits[crit_i]
            n = len(per_question[0]) if per_question else 0
            correct = [sum(q) for q in per_question]
            cells = "".join(
                f"{100 * sum(pass_at_k(n, c, k) for c in correct) / max(len(correct), 1):>8.1f}%" for k in ks
            )
            print(f"  {display:<{width}}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True, help="path_or_repo[=display_name] entries, in row order.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=200, help="Evaluate the first N questions; 0 runs the full split.")
    ap.add_argument("--n-samples", "-n", type=int, default=16, help="Samples per question (the pass@k horizon).")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; must be > 0.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-batch-size", type=int, default=32, help="Per-device tt-transformers batch size.")
    ap.add_argument("--max-seq-len", type=int, default=1024, help="Must cover the longest prompt + --max-new-tokens.")
    ap.add_argument("--mesh-shape", type=int, nargs=2, default=[1, 1], help="Mesh rows and columns.")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer repo/path; defaults to the first --models entry.")
    ap.add_argument("--show", type=int, default=0, help="Print the first sample of this many questions per model.")
    ap.add_argument("--show-chars", type=int, default=1200, help="Truncate printed completions to this many chars.")
    args = ap.parse_args()
    if args.temperature <= 0.0:
        ap.error("--temperature must be > 0: tt-transformers emits no log-probs on its greedy path.")

    models = parse_models(args.models)

    ds = load_dataset("openai/gsm8k", "main")[args.split]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    questions = [ex["question"] for ex in ds]
    golds = [extract_hash_answer(ex["answer"]) for ex in ds]
    print(f"{len(ds)} questions x {args.n_samples} samples per model")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or models[0][0], trust_remote_code=True)
    prompts = [tokenizer.encode(format_prompt(q), add_special_tokens=False) for q in questions]

    gen_kwargs = dict(
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        mesh_shape=args.mesh_shape,
    )

    # Boot every model once first, so a model that fails to load does so before the long generation runs.
    for path, display in models:
        print(f"[preflight] {display} ({path})")
        generate_on_tt(path, [], **gen_kwargs)

    ks = [k for k in (1, 2, 4, 8, 16) if k <= args.n_samples]
    rows: List[Tuple[str, List[List[List[bool]]]]] = []
    for path, display in models:
        print(f"\n=== Evaluating {display} ({path}) ===")
        completions = generate_on_tt(path, prompts, **gen_kwargs)
        texts = [[tokenizer.decode(s, skip_special_tokens=True) for s in q] for q in completions]

        hits: List[List[List[bool]]] = [[] for _ in CRITERIA]
        for samples, gold in zip(texts, golds):
            scores = [score_sample(t, gold) for t in samples]
            for c_idx, (_, criterion) in enumerate(CRITERIA):
                hits[c_idx].append([criterion(s) for s in scores])
        rows.append((display, hits))

        for q_idx in range(min(args.show, len(texts))):
            text = texts[q_idx][0]
            print("-" * 72)
            print(f" [{q_idx}] gold={golds[q_idx]} answer={extract_tag_answer(text)}")
            print(f" Q: {questions[q_idx]}")
            print(f" A:\n{text[: args.show_chars].strip()}")

    print_tables(rows, ks)


if __name__ == "__main__":
    main()
