#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from datetime import datetime, timezone

from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.utils import get_tt_metal_runtime_root
from ttml.trainers import TrainerCallback
from ttml.trainers import GRPOTrainer
from utils.config import read_yaml
from utils.llama_completer import LlamaCompletionCtx
from utils.llama_completer import LlamaGRPOCompleter


class GRPOMonitor(TrainerCallback):
    def __init__(self, output_dir):
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "avg_length"])

    def on_step_end(self, trainer, step, **kwargs):
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        print(f"Step {step} | Reward: {reward:.4f} | Len: {length:.2f} tokens")
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length])

    def on_train_end(self, trainer):
        print("Training complete.")


def boolq_reward(completions, answer, **kwargs):
    rewards = []
    for text, ground_truth in zip(completions, answer):
        clean = text.strip().lower()
        accuracy = 2.0 if clean.startswith(ground_truth.lower()) else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards.append(accuracy + brevity)
    return rewards


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    system_prompt = "You are a wordy professor. Explain in 3 long sentences before saying Yes or No."

    def format_boolq(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": "yes" if example["answer"] else "no",
        }

    dataset = load_dataset("google/boolq", split="train").shuffle(seed=42).map(format_boolq)

    config_path = os.path.join(
        get_tt_metal_runtime_root(),
        "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml",
    )
    transformer_config, device_config, optimizer_config, grpo_config = read_yaml(config_path)

    output_dir = (
        get_tt_metal_runtime_root()
        + "/generated/tt-train/grpo_run/"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    grpo_config.output_dir = output_dir

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=grpo_config.max_completion_length,
            temperature=grpo_config.temperature,
            completions_per_prompt=grpo_config.num_generations,
        ),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=model_id,
    )

    grpo_trainer = GRPOTrainer(
        completer=completer,
        dataset=dataset,
        config=grpo_config,
        reward_func=boolq_reward,
        optimizer_config=optimizer_config,
        callbacks=[GRPOMonitor(output_dir)],
        model_source=model_id,
    )
    grpo_trainer.train()
