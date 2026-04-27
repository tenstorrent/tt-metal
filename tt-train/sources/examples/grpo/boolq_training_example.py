#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from datetime import datetime, timezone

from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from ttml.trainers import GRPOTrainer, TrainerCallback, get_grpo_config
from utils.llama_completer import LlamaCompletionCtx
from utils.llama_completer import LlamaGRPOCompleter


class GRPOMonitor(TrainerCallback):
    def __init__(self, output_dir):
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "avg_length", "step_time_s", "generation_time_s"])

    def on_step_end(self, trainer, step, **kwargs):
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        min_length = kwargs["min_completion_len"]
        max_length = kwargs["max_completion_len"]
        step_time_s = kwargs.get("step_time_s", float("nan"))
        generation_time_s = kwargs.get("generation_time_s", float("nan"))
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Step {step} | Reward: {reward:.4f} "
            f"| Len: {length:.2f} (min {min_length}, max {max_length}) tokens "
            f"| Step: {step_time_s:.2f}s | Gen: {generation_time_s:.2f}s"
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length, step_time_s, generation_time_s])

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

    tt_metal_root = get_tt_metal_runtime_root()
    config_path = os.path.join(
        tt_metal_root,
        "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml",
    )
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    assert training_config.model_config, "training_config.model_config must be set"
    transformer_config = get_model_config(training_config.model_config)
    optimizer_dict = raw["training_config"]["optimizer"]

    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    grpo_config = get_grpo_config(raw, output_dir=output_dir)

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
        optimizer_dict=optimizer_dict,
        callbacks=[GRPOMonitor(output_dir)],
        model_source=model_id,
    )
    grpo_trainer.train()
