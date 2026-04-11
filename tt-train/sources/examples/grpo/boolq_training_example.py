#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.utils import get_tt_metal_runtime_root
from utils.grpo_trainer import GrpoConfig, GrpoTrainer, TrainerCallback


class GRPOMonitor(TrainerCallback):
    def __init__(self, output_dir):
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "avg_length"])

    def on_step_end(self, trainer, step, metrics):
        reward = metrics["reward_mean"]
        length = metrics["mean_completion_len"]
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

    transformer_config = {
        "model_type": "llama",
        "num_heads": 32,
        "num_groups": 8,
        "embedding_dim": 2048,
        "intermediate_dim": 8192,
        "dropout_prob": 0.0,
        "num_blocks": 16,
        "weight_tying": "enabled",
        "vocab_size": 32000,
        "max_sequence_length": 1024,
        "runner_type": "memory_efficient",
        "theta": 500000.0,
        "rope_scaling": {
            "scaling_factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_context_length": 8192,
        },
    }

    device_config = {
        "enable_ddp": True,
        "mesh_shape": [1, 2],
    }

    optimizer_config = {
        "type": "MorehAdamW",
        "lr": 5.0e-6,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1.0e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "kahan_summation": False,
    }

    from datetime import datetime, timezone

    output_dir = (
        get_tt_metal_runtime_root()
        + "/generated/tt-train/grpo_run/"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )

    config = GrpoConfig(
        epsilon=0.2,
        batch_size=4,
        micro_batch_size=32,
        num_iterations=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        output_dir=output_dir,
        checkpointing=False,
        checkpoint_interval=50,
        prompts_to_train=1600,
        temperature=1.5,
        max_completion_length=256,
        num_generations=8,
        warmup_steps=0,
    )

    grpo_trainer = GrpoTrainer(
        model_source=model_id,
        dataset=dataset,
        config=config,
        reward_func=boolq_reward,
        transformer_config=transformer_config,
        optimizer_config=optimizer_config,
        device_config=device_config,
        callbacks=[GRPOMonitor(output_dir)],
    )
    grpo_trainer.train()
