#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is `test_penalties[presence_penalty-1.2-repeat_trap]` measuring the model, or the metric?

The tt-inference-server chat-completions conformance test asserts

    unique_ratio(with presence_penalty) >= unique_ratio(baseline) * 0.90

where ``unique_ratio = len(set(words)) / len(words)`` — a type-token ratio.  A
type-token ratio falls as text gets longer even when the text gets *more*
varied, so any model that answers a presence penalty by writing more can fail
the assertion while behaving exactly as the penalty intends.

This is the control for that claim, and it is deliberately as far from the
autoport as possible: a different model (``HuggingFaceTB/SmolLM2-1.7B-Instruct``),
a different runtime (plain `transformers` greedy decoding on CPU), no
Tenstorrent device involved at all, and a presence penalty implemented directly
from the OpenAI definition — subtract the penalty from the logit of every token
already generated.  Same prompt, same comparison, same statistics function as
the test.

If the control moves the same way — more unique words, higher entropy, longer
text, *lower* type-token ratio — then the assertion is measuring length, not the
Tenstorrent implementation.

Usage::

    python presence_penalty_control.py --out doc/tti_release/presence_penalty_control.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

CONTROL_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
PROMPT = "Write a very repetitive story."
MAX_NEW_TOKENS = 400
PRESENCE_PENALTY = 1.2


def tokenize(text: str):
    """Verbatim from tt-inference-server llm_module/test_vllm_chat_completions.py."""
    return text.lower().split()


def repetition_stats(text: str):
    """Verbatim from the same file, plus the sentence count, for reporting."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else 0
    return {
        "len": total,
        "unique": len(set(tokens)),
        "unique_ratio": len(set(tokens)) / total if total else 0,
        "most_common": counts.most_common(3),
        "entropy": round(entropy, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=CONTROL_MODEL)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    class PresencePenalty(LogitsProcessor):
        """OpenAI presence_penalty: a flat subtraction on any already-seen token."""

        def __init__(self, penalty: float, prompt_len: int):
            self.penalty = penalty
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.prompt_len :]
            for row in range(scores.shape[0]):
                seen = torch.unique(generated[row])
                scores[row, seen] -= self.penalty
            return scores

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    def generate(processors):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    base_text = generate(None)
    test_text = generate([PresencePenalty(PRESENCE_PENALTY, prompt_len)])

    base, test = repetition_stats(base_text), repetition_stats(test_text)
    ratio = test["unique_ratio"] / base["unique_ratio"] if base["unique_ratio"] else 0
    doc = {
        "_what": (
            "control for the failing presence-penalty conformance row: a different model, "
            "on CPU, through plain transformers, with no Tenstorrent device involved"
        ),
        "control_model": args.model,
        "runtime": "transformers greedy decoding, float32, CPU",
        "prompt": PROMPT,
        "presence_penalty": PRESENCE_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "assertion_under_test": "unique_ratio(penalty) >= unique_ratio(base) * 0.90",
        "base_no_penalty": base,
        "test_presence_penalty": test,
        "observed_ratio": ratio,
        "assertion_would_pass": ratio >= 0.90,
        "base_text": base_text,
        "test_text": test_text,
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)

    print(f"control model : {args.model} (CPU, transformers, greedy)")
    print(
        f"base          : len={base['len']} unique={base['unique']} "
        f"ratio={base['unique_ratio']:.4f} entropy={base['entropy']}"
    )
    print(
        f"presence 1.2  : len={test['len']} unique={test['unique']} "
        f"ratio={test['unique_ratio']:.4f} entropy={test['entropy']}"
    )
    print(f"ratio         : {ratio:.4f}  (assertion needs >= 0.90) -> " f"{'PASS' if ratio >= 0.90 else 'FAIL'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
