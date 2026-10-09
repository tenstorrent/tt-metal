#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO training of Llama-3.2-1B-Instruct on BoolQ across two tt-run ranks
(rank 0 TTML policy/training, rank 1 TTT generation). Requires HF_TOKEN.

Launch with ``boolq/runner.sh``. Both ranks run this script; GRPOTrainer picks
each rank's role from the config's ``rollout_mode``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, get_model_config, load_config
from ttml.trainers import GRPOTrainer, get_grpo_config

CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml"

REPO_ROOT = Path(__file__).resolve().parents[5]

SYSTEM_PROMPT = "You are a wordy professor. Explain in 3 long sentences before saying Yes or No."


def accuracy_reward(completions, answer, **kwargs):
    """+2 if the completion begins with the correct Yes/No token, -1 otherwise."""
    return [2.0 if text.strip().lower().startswith(gt.lower()) else -1.0 for text, gt in zip(completions, answer)]


def brevity_reward(completions, **kwargs):
    """Quadratic length penalty in characters, discouraging runaway completions."""
    return [-0.1 * (len(text) / 20) ** 2 for text in completions]


def make_dataset_func(model_source: str):
    def make_dataset():
        tokenizer = AutoTokenizer.from_pretrained(model_source)

        def format_boolq(example):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
            ]
            return {
                "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": "yes" if example["answer"] else "no",
            }

        return load_dataset("google/boolq", split="train").shuffle(seed=42).map(format_boolq)

    return make_dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    model_source = raw["training_config"]["model_source"]
    output_dir = os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )

    trainer = GRPOTrainer(
        transformer_config=get_model_config(raw["training_config"]["model_config"]),
        device_config=DeviceConfig(raw),
        model_source=model_source,
        dataset_func=make_dataset_func(model_source),
        config=get_grpo_config(raw, output_dir=output_dir),
        reward_funcs=[accuracy_reward, brevity_reward],
        optimizer_dict=raw["training_config"]["optimizer"],
    )
    trainer.train()
