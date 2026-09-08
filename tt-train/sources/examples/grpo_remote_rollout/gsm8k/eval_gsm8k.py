#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone GSM8K evaluator for a sequence of Qwen3 checkpoints, running on
tt-metal hardware via :class:`TttGenerationWorker`.

Companion to :mod:`gsm8k_training_example` (the ttml-based GRPO trainer). This
script drives the same on-device generation path as the training-time TTT
rollout worker, so accuracy numbers are produced by the exact stack the
trainer will consume during rollouts -- not a separate GPU inference engine.

For every ``--models path[=display_name]`` entry a fresh worker is booted with
``dummy_weights=False``, which routes HF weight loading through the standard
tt-transformers path (``ModelArgs.load_state_dict`` -> tile-layout upload).
Each checkpoint therefore reuses the training-side Q/K permutation, KV-cache
setup and on-device sampling exactly as the trainer would.

Output: one table per criterion, in the same layout the reference GPU/TRL
eval printed (minus the ``solved`` column):

    correct answer
      checkpoint            pass@1  pass@2  pass@4  pass@8  pass@16
      Qwen3-0.6B-Base        33.0%   50.6%   67.6%   80.1%    88.5%
      sft                    32.7%   49.3%   65.6%   78.3%    86.0%
      grpo-checkpoint-100    ...

Four criteria: ``correct answer``, ``tags present``, ``tags exactly once``,
``format-regex`` (renamed from "strict format"). Rows follow the order of the
``--models`` argument.

Prompts use the plain "Question:/Answer:" completion layout that the reference
GPU/TRL scripts SFT'd and GRPO-trained on -- no ChatML, no
``apply_chat_template``. The Qwen3-0.6B-Base checkpoint has untrained ChatML
control-token embeddings (``<|im_start|>`` / ``<|im_end|>`` land at the same
norm as unused reserved specials) and the SFT run in this project was done on
the plain layout, so a chat-templated prompt gives garbage rollouts on the base
and off-distribution rollouts on the SFT / GRPO checkpoints.

Default tag pair is ``<think>...</think>`` + ``<answer>...</answer>``; pass
``--tags reasoning`` to swap.

Prerequisites:
  - This runs on tt-metal, not CUDA. It opens a [1, 1] parent mesh on the
    default Blackhole device; ensure ``TT_METAL_HOME`` is set and the process
    has the card visible.
  - Every ``--models`` entry must be a HuggingFace-format checkpoint directory
    or repo id. Ttml GRPO checkpoints written by ``gsm8k_training_example.py``
    are NOT directly HF-consumable and must be converted first (a separate
    step, not implemented here).
  - Every checkpoint referenced from ``--models`` must be registered in
    ``models/tt_transformers/tt/model_config.py::ModelArgs.LOCAL_HF_PARAMS``
    (needs a ``config.json`` for architecture bootstrap). Otherwise the worker
    fails at boot.

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
import sys
from math import comb
from typing import Any, Callable, List, Optional, Sequence, Tuple

# Make ``utils.*`` importable when the file is run directly (needed before
# any ``from utils.* import ...`` at module scope).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from utils.ttt_generation_worker import TttGenerationWorker


# ---------------------------------------------------------------------------
# Tags and prompt
# ---------------------------------------------------------------------------
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

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
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt_texts(
    questions: Sequence[str],
    reasoning_open: str,
    reasoning_close: str,
) -> List[str]:
    """Plain "Question:/Answer:" completion layout, matching the reference
    training / SFT / eval scripts. No chat template, no control tokens -- that
    is the point on a *-Base checkpoint whose ChatML tokens are untrained."""
    sysp = system_prompt(reasoning_open, reasoning_close)
    return [f"{sysp}\nQuestion: {q}\nAnswer:" for q in questions]


# ---------------------------------------------------------------------------
# On-device generation
# ---------------------------------------------------------------------------
def generate_on_tt(
    model_path: str,
    prompts_token_ids: Sequence[Sequence[int]],
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    max_batch_size: int,
    max_seq_len: int,
) -> List[List[List[int]]]:
    """Run generation for one checkpoint on a fresh TttGenerationWorker.

    Returns ``completions[q][s]`` = list of token IDs (stop-token stripped by
    the worker). One worker is booted with ``dummy_weights=False`` so HF
    weights are loaded through the standard tt-transformers path; on the way
    out the mesh device is closed so the next checkpoint can boot cleanly.

    Sampling params are baked into the worker's decode trace at first capture
    (temperature/top_k/top_p/seed). ``seed=None`` here keeps sampling
    non-deterministic, so calling ``generate()`` multiple times per prompt
    yields distinct samples -- that is what makes pass@k > 1 meaningful.
    """
    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    worker: Any = None
    try:
        stop_ids, pad_id = qwen3_stop_and_pad(model_path)
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=model_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            seed=None,
            dummy_weights=False,  # load real HF weights, not the fast-boot random shim
        )

        # Duplicate each prompt n_samples times: same prompt goes into n_samples
        # slots so one worker call returns n_samples independent rollouts (the
        # baked-in seed=None makes each slot sample its own trajectory).
        expanded: list[list[int]] = []
        for p in prompts_token_ids:
            for _ in range(n_samples):
                expanded.append(list(p))

        # Batch up to ``max_batch_size`` per generate() call. The worker itself
        # pads short batches to the global size, so passing anything up to
        # max_batch_size is safe.
        flat_completions: list[list[int]] = []
        total = len(expanded)
        for start in range(0, total, max_batch_size):
            batch = expanded[start : start + max_batch_size]
            print(
                f"[eval_gsm8k] {os.path.basename(model_path)}: "
                f"generate() {start // max_batch_size + 1}/"
                f"{(total + max_batch_size - 1) // max_batch_size} "
                f"(batch of {len(batch)}, max_new_tokens={max_new_tokens})",
                flush=True,
            )
            flat_completions.extend(worker.generate(batch, max_new_tokens=max_new_tokens))

        # Reshape [total] -> [n_questions][n_samples].
        assert len(flat_completions) == len(prompts_token_ids) * n_samples, (
            f"expected {len(prompts_token_ids) * n_samples} completions, " f"got {len(flat_completions)}"
        )
        result: list[list[list[int]]] = []
        for q_idx in range(len(prompts_token_ids)):
            base = q_idx * n_samples
            result.append([list(flat_completions[base + s]) for s in range(n_samples)])
        return result
    finally:
        worker = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


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
        help="Sampling temperature, baked into the worker's decode trace. "
        "0.0 => greedy; must be > 0 when --n-samples > 1.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max generation length per sample. Matches the training config's " "max_completion_length by default.",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Per-generate batch cap. Matches the training config's "
        "remote_rollout_config.max_batch_size by default.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="KV-cache horizon. Must be >= max prompt tokens + --max-new-tokens.",
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
    ap.add_argument(
        "--show",
        type=int,
        default=0,
        help="Print this many sample completions per model for debugging.",
    )
    ap.add_argument(
        "--show-chars",
        type=int,
        default=1200,
        help="Truncate each printed completion to this many characters.",
    )
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
    print(f"  {n_q} questions x {args.n_samples} samples = " f"{n_q * args.n_samples} completions per model")

    # -- tokenizer / prompts (shared across models: same Qwen3 vocab across
    #    base / SFT / GRPO checkpoints; the tokenizer is only used to tokenize
    #    the pre-assembled plain-format string and to decode completions, not
    #    to build the prompt) --
    tokenizer_source = args.tokenizer or models[0][0]
    print(f"Loading tokenizer from {tokenizer_source} (for tokenize/decode only; prompts use plain layout)")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    prompt_texts = build_prompt_texts(questions, reasoning_open, reasoning_close)
    prompts_token_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in prompt_texts]

    # Print the first assembled prompt once so a prompt-layout mismatch (wrong
    # system prompt, missing "Question:" / "Answer:" markers, stray control
    # tokens) is visible without digging into the model outputs.
    if prompt_texts:
        print("\n--- First assembled prompt (verbatim, tokens NOT decoded) ---")
        print(prompt_texts[0])
        print(f"--- ({len(prompts_token_ids[0])} tokens) ---")

    # -- preflight: boot every worker once with empty prompts so a bad model
    #    (missing LOCAL_HF_PARAMS entry, unreachable repo, incompatible config)
    #    fails immediately instead of after 15+ min of generation on earlier
    #    models. Reuses generate_on_tt with prompts_token_ids=[] -- the worker
    #    is constructed (weights load, tt cache is populated) and torn down;
    #    no generate() call happens. Subsequent full-eval reloads pick up the
    #    already-materialized tt weight cache, so the extra cost is small.
    print("\n=== Preflight: loading every model once to catch boot failures early ===")
    for path, display in models:
        print(f"[preflight] {display} ({path})")
        generate_on_tt(
            model_path=path,
            prompts_token_ids=[],
            n_samples=1,
            max_new_tokens=1,
            temperature=args.temperature,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
        )
    print("[preflight] all models loaded successfully")

    # -- generate + score per model --
    ks = [k for k in (1, 2, 4, 8, 16) if k <= args.n_samples]
    # per_model_rows[i] = (display, criterion_hits) where
    # criterion_hits[c][q] = list of per-sample bools
    per_model_rows: list[tuple[str, list[list[list[bool]]]]] = []

    for path, display in models:
        print(f"\n=== Evaluating {display} ({path}) ===")
        completions_ids = generate_on_tt(
            model_path=path,
            prompts_token_ids=prompts_token_ids,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
        )
        completions_text = [[tokenizer.decode(s, skip_special_tokens=True) for s in q] for q in completions_ids]

        # criterion_hits[c] is [per-question list of per-sample bools]
        criterion_hits: list[list[list[bool]]] = [[] for _ in CRITERIA]
        for samples, gold in zip(completions_text, golds):
            per_sample_scores = [
                score_sample(t, gold, tag_strings, reasoning_open, reasoning_close, fmt_re) for t in samples
            ]
            for c_idx, (_, criterion) in enumerate(CRITERIA):
                criterion_hits[c_idx].append([bool(criterion(s)) for s in per_sample_scores])

        # Optional sample printout: first args.show questions, first sample only,
        # so a tokenizer / chat-template / decoding mismatch is easy to spot.
        if args.show > 0:
            print(
                f"\n--- Sample completions from {display} (first {min(args.show, len(completions_text))} questions, sample 0) ---"
            )
            for q_idx in range(min(args.show, len(completions_text))):
                gold = golds[q_idx]
                text = completions_text[q_idx][0]
                clipped = text[: args.show_chars]
                tag_ans = extract_tag_answer(text)
                fb_ans = extract_last_number(text, reasoning_open, reasoning_close)
                verdict = "CORRECT" if (tag_ans == gold or fb_ans == gold) else "WRONG"
                print("-" * 72)
                print(f" [{q_idx}] gold={gold} tag_answer={tag_ans} fallback_answer={fb_ans} -> {verdict}")
                print(f" Q: {questions[q_idx]}")
                print(f" A:\n{clipped.strip()}")
                if len(text) > args.show_chars:
                    print(f" ... [truncated {len(text) - args.show_chars} more chars]")

        per_model_rows.append((display, criterion_hits))

    print_tables(per_model_rows, ks)


if __name__ == "__main__":
    main()
