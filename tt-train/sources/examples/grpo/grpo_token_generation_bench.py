#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Time how long it takes to answer a single GSM8K question with Llama-3.2-1B.

Runs ``WARMUP`` warmup iterations (to populate the program cache and amortise
first-iteration compilation), then ``ITERS`` measured iterations and reports:

  - per-iteration wall-clock time
  - per-iteration completion length (in tokens)
  - aggregate tokens/sec across the measured iterations

This is intended as a baseline wall-clock measurement run *without* tracy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ttml
import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config

from utils.llama_completer_composite import LlamaCompletionCtx, LlamaGRPOCompleter

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = (
    "You are a careful math tutor. Read the problem, explain your reasoning "
    "step by step in clear English, then on the final line write four hash "
    "characters followed by a single space and the final numerical answer, "
    "for example: '#### 42'. Do not write anything after the final answer."
)

QUESTION_INDEX = 0
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 0.0
NUM_GENERATIONS = 32

WARMUP = 1
ITERS = 5

_CONFIG_PATH = Path(__file__).with_name("boolq_accuracy_example.yaml")
_RAW_CONFIG = load_config(str(_CONFIG_PATH))
_TRAINING_CONFIG = TrainingConfig(_RAW_CONFIG)
DEVICE_CONFIG = DeviceConfig(_RAW_CONFIG)
assert _TRAINING_CONFIG.model_config, "training_config.model_config must be set"
TRANSFORMER_CONFIG = get_model_config(_TRAINING_CONFIG.model_config)


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    example = dataset[QUESTION_INDEX]
    question = example["question"]
    golden = example["answer"]
    print(f"GSM8K question #{QUESTION_INDEX}:\n{question}\n")
    print(f"Golden answer:\n{golden}\n")

    prompt_str = build_prompt(tokenizer, question)
    prompt_tokens = tokenizer.encode(prompt_str)
    prompts = [prompt_tokens]
    print(f"Prompt token length: {len(prompt_tokens)}")

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=MAX_COMPLETION_LENGTH,
            temperature=TEMPERATURE,
            completions_per_prompt=NUM_GENERATIONS,
        ),
        transformer_config=TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=MODEL_ID,
    )

    device = ttml.autograd.AutoContext.get_instance().get_device()

    print(f"\nWarming up ({WARMUP} iter)...")
    for _ in range(WARMUP):
        completer.generate(prompts)

    print(f"Measuring ({ITERS} iter)...")
    per_iter_seconds: list[float] = []
    per_iter_tokens: list[int] = []
    last_completion_text: str = ""

    t0_total = time.perf_counter()
    for it in range(ITERS):
        t0 = time.perf_counter()
        out = completer.generate(prompts)
        # generate() already blocks via to_numpy(), but sync defensively in case
        # that ever changes.
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()

        n_tokens = sum(len(c) for c in out)
        per_iter_seconds.append(t1 - t0)
        per_iter_tokens.append(n_tokens)
        last_completion_text = tokenizer.decode(out[0], skip_special_tokens=False)

        print(f"  iter {it}: {t1 - t0:.3f}s, {n_tokens} tokens, " f"{n_tokens / (t1 - t0):.1f} tok/s")
    t1_total = time.perf_counter()

    total_tokens = sum(per_iter_tokens)
    total_seconds = t1_total - t0_total
    print("\nLast completion:\n" + last_completion_text + "\n")
    print(
        f"Aggregate over {ITERS} iters: "
        f"{total_tokens} tokens in {total_seconds:.3f}s "
        f"-> {total_tokens / total_seconds:.1f} tok/s"
    )
    print(
        "Per-iter mean: "
        f"{sum(per_iter_seconds) / ITERS:.3f}s, "
        f"min={min(per_iter_seconds):.3f}s, max={max(per_iter_seconds):.3f}s"
    )
