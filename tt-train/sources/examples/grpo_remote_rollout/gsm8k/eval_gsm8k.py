#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone GSM8K evaluator for a sequence of Qwen3 checkpoints.

Companion to :mod:`gsm8k_training_example` (the ttml-based GRPO trainer). Runs
on a CUDA host with vLLM -- no tt-metal dependency -- so the same script can
score the base model, the SFT'd starting point, and the post-GRPO checkpoints
side by side.

Output: one table per criterion, matching the multi-checkpoint layout of the
project's reference GPU/TRL eval, minus the ``solved`` column:

    correct answer
      checkpoint            pass@1  pass@2  pass@4  pass@8  pass@16
      Qwen3-0.6B-Base        33.0%   50.6%   67.6%   80.1%    88.5%
      sft                    32.7%   49.3%   65.6%   78.3%    86.0%
      grpo-checkpoint-100    ...
    ...

Four criteria, in the requested order: ``correct answer``, ``tags present``,
``tags exactly once``, ``format-regex`` (renamed from "strict format").

Prompts use the tokenizer's own chat template with ``enable_thinking=False``,
untouched -- no override of ``chat_template`` or ``pad_token``. The default tag
pair is ``<think>...</think>`` + ``<answer>...</answer>``, matching what the
SFT model was trained against; pass ``--tags reasoning`` to swap.

Notes / prerequisites:
  - Ttml GRPO checkpoint directories written by ``gsm8k_training_example.py``
    are not directly HuggingFace-consumable. Converting them to a
    ``from_pretrained``-ready layout is out of scope for this script; if you
    point ``--models`` at a raw ttml checkpoint you will get a load error from
    vLLM. Use the HF-format directory produced by whatever conversion step
    lands next.

Usage:
    python eval_gsm8k.py \\
        --models Qwen/Qwen3-0.6B-Base=Qwen3-0.6B-Base \\
                 ichovpanTT/qwen3-0.6b-base-think-sft=sft \\
                 outputs_grpo/checkpoint-100=grpo-checkpoint-100 \\
                 outputs_grpo/checkpoint-200=grpo-checkpoint-200 \\
                 outputs_grpo/checkpoint-300=grpo-checkpoint-300 \\
        --n-samples 16 --temperature 1.0 --limit 200
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from math import comb
from typing import Callable, List, Optional, Sequence, Tuple

# vLLM tries to JIT-build FlashInfer's sampling kernels at startup. On sm120
# (RTX 50-series) that needs a CUDA >= 12.9 toolkit; with the 12.8 toolkit the
# JIT aborts and takes engine init down with it. The native sampler is fine,
# so opt out before vllm is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Tags and prompt
# ---------------------------------------------------------------------------
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# Tag pairs the script can be pointed at. Default is ``think`` because the SFT
# model uploaded by this project is trained on <think>...</think>; ``reasoning``
# is provided for parity with the reference GPU/TRL eval script.
TAG_PAIRS: dict[str, Tuple[str, str]] = {
    "think": ("<think>", "</think>"),
    "reasoning": ("<reasoning>", "</reasoning>"),
}


def system_prompt(reasoning_open: str, reasoning_close: str) -> str:
    return (
        "Respond in the following format:\n"
        f"{reasoning_open}\n...\n{reasoning_close}\n"
        f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
    )


def strict_format_re(reasoning_open: str, reasoning_close: str) -> re.Pattern:
    return re.compile(
        re.escape(reasoning_open)
        + r".*?"
        + re.escape(reasoning_close)
        + r"\s*"
        + re.escape(ANSWER_OPEN)
        + r".*?"
        + re.escape(ANSWER_CLOSE),
        re.DOTALL,
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return s
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_hash_answer(gold: str) -> str:
    """GSM8K gold answers look like '<reasoning>\\n#### 72'."""
    return normalize_number(gold.split("####")[-1])


def extract_tag_answer(text: str) -> Optional[str]:
    """Content of the last ``<answer>...</answer>`` block, then last number in it."""
    if ANSWER_OPEN not in text:
        return None
    body = text.split(ANSWER_OPEN)[-1].split(ANSWER_CLOSE)[0]
    nums = _NUM_RE.findall(body)
    return normalize_number(nums[-1]) if nums else None


def extract_last_number(text: str, reasoning_open: str, reasoning_close: str) -> Optional[str]:
    """Fallback: last number anywhere in the completion, ignoring the reasoning
    block so we don't grab intermediate scratch numbers when the model uses tags."""
    cleaned = re.sub(
        re.escape(reasoning_open) + r".*?" + re.escape(reasoning_close),
        "",
        text,
        flags=re.DOTALL,
    )
    nums = _NUM_RE.findall(cleaned) or _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# pass@k -- unbiased estimator (Codex paper, Chen et al. 2021, eq. 1)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_sample(
    text: str,
    gold: str,
    tag_strings: Sequence[str],
    reasoning_open: str,
    reasoning_close: str,
    fmt_re: re.Pattern,
) -> dict:
    tag_ans = extract_tag_answer(text)
    fb_ans = extract_last_number(text, reasoning_open, reasoning_close)
    tag_ok = tag_ans is not None and tag_ans == gold
    fb_ok = fb_ans is not None and fb_ans == gold
    return {
        "correct_answer": tag_ok or fb_ok,
        "tags_present": all(t in text for t in tag_strings),
        "tags_exactly_once": all(text.count(t) == 1 for t in tag_strings),
        "format_regex": bool(fmt_re.search(text)),
    }


# In requested display order:
CRITERIA: Tuple[Tuple[str, Callable[[dict], bool]], ...] = (
    ("correct answer", lambda s: s["correct_answer"]),
    ("tags present", lambda s: s["tags_present"]),
    ("tags exactly once", lambda s: s["tags_exactly_once"]),
    ("format-regex", lambda s: s["format_regex"]),
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_prompts(tokenizer, questions: Sequence[str], reasoning_open: str, reasoning_close: str) -> List[str]:
    sysp = system_prompt(reasoning_open, reasoning_close)
    prompts: list[str] = []
    for q in questions:
        messages = [
            {"role": "system", "content": sysp},
            {"role": "user", "content": q},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        )
    return prompts


def generate_vllm(
    model_path: str,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    n_samples: int,
    gpu_mem_util: float,
    max_model_len: int,
) -> List[List[str]]:
    """One-shot generation for a single checkpoint. Instantiates the engine,
    runs, and destroys it before returning so the next checkpoint can load."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.95,
        max_tokens=max_tokens,
        n=n_samples,
    )
    outs = llm.generate(list(prompts), sp)
    result = [[o.text for o in out.outputs] for out in outs]

    # vLLM holds sizeable GPU state; drop refs and force a GC before the next
    # engine is constructed so memory doesn't accumulate across checkpoints.
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return result


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
def _parse_models(entries: Sequence[str]) -> List[Tuple[str, str]]:
    """``path=display_name`` (falls back to ``path`` if no ``=`` is present)."""
    parsed: list[tuple[str, str]] = []
    for e in entries:
        if "=" in e:
            path, name = e.split("=", 1)
        else:
            path, name = e, e
        parsed.append((path.strip(), name.strip()))
    return parsed


def print_tables(rows: List[Tuple[str, List[List[List[bool]]]]], ks: Sequence[int]) -> None:
    """rows[i] = (display, criterion_hits) where criterion_hits[c][q][s] is a bool."""
    header = "".join(f"{'pass@' + str(k):>9}" for k in ks)
    display_width = max((len(r[0]) for r in rows), default=10) + 2

    for crit_i, (crit_name, _) in enumerate(CRITERIA):
        print(f"\n{crit_name}")
        print(f"  {'checkpoint':<{display_width}}" + header)
        for display, per_crit in rows:
            hits = per_crit[crit_i]
            n_q = len(hits)
            ns = len(hits[0]) if n_q else 0
            cs = [sum(1 for x in q if x) for q in hits]
            cells = "".join(f"{100 * sum(pass_at_k(ns, c, k) for c in cs) / max(n_q, 1):>8.1f}%" for k in ks)
            print(f"  {display:<{display_width}}{cells}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more path_or_repo[=display_name] entries. Evaluation runs "
        "each model in the given order, and that order is also the row order "
        "in the printed tables.",
    )
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Evaluate only the first N examples. 200 matches the reference "
        "table shape; pass 0 to run the full split.",
    )
    ap.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=16,
        help="Samples per question (drives the pass@k horizon).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="0.0 => greedy; must be > 0 when --n-samples > 1.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max generation length per sample.",
    )
    ap.add_argument(
        "--tags",
        default="think",
        choices=list(TAG_PAIRS),
        help="Reasoning tag pair. Default 'think' matches the SFT model; "
        "'reasoning' matches the reference GPU/TRL eval script.",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer repo/path. Defaults to the first --models entry so the "
        "chat template comes from the family's canonical checkpoint.",
    )
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    if args.n_samples > 1 and args.temperature == 0.0:
        ap.error(
            "--n-samples > 1 with --temperature 0.0 would return identical "
            "greedy samples; pass e.g. --temperature 1.0."
        )

    models = _parse_models(args.models)
    reasoning_open, reasoning_close = TAG_PAIRS[args.tags]
    tag_strings = (reasoning_open, reasoning_close, ANSWER_OPEN, ANSWER_CLOSE)
    fmt_re = strict_format_re(reasoning_open, reasoning_close)

    # -- dataset --
    print(f"Loading GSM8K {args.split} split...")
    ds = load_dataset("openai/gsm8k", "main")[args.split]
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    questions = [ex["question"] for ex in ds]
    golds = [extract_hash_answer(ex["answer"]) for ex in ds]
    n_q = len(ds)
    print(f"  {n_q} questions x {args.n_samples} samples = {n_q * args.n_samples} completions per model")

    # -- tokenizer / prompts (shared across models: the SFT'd repo carries a
    #    chat template Qwen3-Base does not, so default to the first model
    #    entry unless the user overrides) --
    tokenizer_source = args.tokenizer or models[0][0]
    print(f"Building prompts with tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    prompts = build_prompts(tokenizer, questions, reasoning_open, reasoning_close)

    # -- generate + score per model --
    ks = [k for k in (1, 2, 4, 8, 16) if k <= args.n_samples]
    per_model_rows: list[tuple[str, list[list[list[bool]]]]] = []
    # per_model_rows[i] = (display, [criterion_hits_per_question])
    # criterion_hits_per_question[c][q] = list of per-sample bools

    for path, display in models:
        print(f"\n=== Evaluating {display} ({path}) ===")
        completions = generate_vllm(
            model_path=path,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n_samples=args.n_samples,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
        )

        # criterion_hits[c] is [per-question list of per-sample bools]
        criterion_hits: list[list[list[bool]]] = [[] for _ in CRITERIA]
        for samples, gold in zip(completions, golds):
            per_sample_scores = [
                score_sample(t, gold, tag_strings, reasoning_open, reasoning_close, fmt_re) for t in samples
            ]
            for c_idx, (_, criterion) in enumerate(CRITERIA):
                criterion_hits[c_idx].append([bool(criterion(s)) for s in per_sample_scores])

        per_model_rows.append((display, criterion_hits))

    # -- print tables --
    # For each criterion, print a table row per model. The print_tables helper
    # expects rows of the shape [(display, criterion_hits)], which we already have.
    print_tables(per_model_rows, ks)


if __name__ == "__main__":
    main()
