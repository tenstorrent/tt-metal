#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Memory profile of one GRPO training step on TinyLlama_v1.1_math_code + GSM8K.

Wraps :class:`utils.grpo_trainer_memprof.GRPOTrainer` (a memprof-instrumented
copy of ``ttml.trainers.GRPOTrainer``) around a TinyLlama completer and the
GSM8K train split.  Runs *exactly one* optimizer step then prints a
``MemoryUsageTracker`` report broken down by phase:

* ``MODEL_LOAD``           - LlamaGRPOCompleter constructor (device open + weights upload)
* ``OPTIMIZER_CREATION``   - AdamW state allocation
* ``GENERATION``           - one ``completer.generate`` call (KV cache + decode activations)
* ``ADVANTAGES``           - group-relative advantages tensor
* ``NLOG_OLD_COMPUTED``    - old-policy nlog-prob pass (no_grad)
* ``FIRST_FWD_BWD``        - one micro-batch forward + backward (new-policy activations + grads)
* ``FIRST_OPTIMIZER_STEP`` - one optimizer.step()

We also print a live ``ttnn.get_memory_view`` reading right after the
completer constructor returns: that cross-checks the ``MODEL_LOAD`` segment
against the allocator's own bookkeeping.  If the two disagree, the graph
capture isn't seeing some allocations and the tracker numbers must be
interpreted with care.

Run with:

    python tt-train/sources/examples/grpo/tinyllama_gsm8k_memory_usage.py
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import ttml
import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer

from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from utils.grpo_trainer_memprof import GRPOTrainer, get_grpo_config
from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

MODEL_ID = "TinyLlama/TinyLlama_v1.1_math_code"


def format_gsm8k(example):
    # TinyLlama v1.1 is a base completion model; plain Q/A formatting works
    # better than a chat template.  GSM8K stores the final integer/decimal
    # answer after ``####`` in the original 'answer' field.
    gold_str = str(example["answer"]).split("####")[-1].strip().replace(",", "")
    return {
        "prompt": f"Q: {example['question']}\nA:",
        "answer": gold_str,
    }


def gsm8k_reward(completions, answer, **kwargs):
    """Reward = +2 if the last number in the completion matches the gold
    answer, else -1, plus a tiny brevity penalty.  Robust to commas and
    chain-of-thought style outputs.
    """
    rewards = []
    number_re = re.compile(r"-?\d+\.?\d*")
    for text, gt in zip(completions, answer):
        cleaned = text.replace(",", "")
        nums = number_re.findall(cleaned)
        guess = nums[-1] if nums else None
        try:
            correct = guess is not None and float(guess) == float(gt)
        except ValueError:
            correct = False
        brevity = -0.0005 * len(text)
        rewards.append((2.0 if correct else -1.0) + brevity)
    return rewards


def _print_live_dram(label: str) -> None:
    mesh = ttml.autograd.AutoContext.get_instance().get_device()
    ttnn.synchronize_device(mesh)
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    total_mib = v.total_bytes_per_bank * v.num_banks / (1024 * 1024)
    allocated_mib = v.total_bytes_allocated_per_bank * v.num_banks / (1024 * 1024)
    free_mib = v.total_bytes_free_per_bank * v.num_banks / (1024 * 1024)
    print(
        f"[live DRAM] {label}: allocated={allocated_mib:.2f} MiB, "
        f"free={free_mib:.2f} MiB, total={total_mib:.2f} MiB ({v.num_banks} banks)"
    )


def main() -> None:
    config_path = Path(__file__).with_suffix(".yaml")
    raw = load_config(str(config_path))
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    assert training_config.model_config, "training_config.model_config must be set"
    transformer_config = get_model_config(training_config.model_config)
    optimizer_dict = raw["training_config"]["optimizer"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Loading GSM8K train split...")
    dataset = load_dataset("gsm8k", "main", split="train").map(format_gsm8k)

    tt_metal_root = get_tt_metal_runtime_root()
    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_memprof_runs",
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    grpo_config = get_grpo_config(raw, output_dir=output_dir)

    print("\nStarting MemoryUsageTracker capture (RunMode.NORMAL).")
    memory_guard = MemoryUsageTracker.begin_capture()
    try:
        print(f"\nLoading model: {MODEL_ID}")
        completer = LlamaGRPOCompleter(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            device_config=device_config,
            model_source=MODEL_ID,
        )
        _print_live_dram("after LlamaGRPOCompleter()")
        MemoryUsageTracker.snapshot("MODEL_LOAD")

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=gsm8k_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[],
            model_source=MODEL_ID,
            track_memory=True,
        )

        print("\nRunning one GRPO training step (track_memory=True forces grad_accum=1)...")
        trainer.train()
        _print_live_dram("after trainer.train() returns")
    finally:
        # Trainer already calls clear() + end_capture() on the happy path;
        # the guard.release() prevents the C++ ScopeGuard destructor from
        # trying to end an already-ended capture.
        try:
            MemoryUsageTracker.clear()
        except Exception:
            pass
        memory_guard.release()


if __name__ == "__main__":
    main()
