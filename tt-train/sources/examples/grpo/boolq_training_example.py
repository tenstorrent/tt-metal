#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, TransformerConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from ttml.trainers import GRPOTrainer, TrainerCallback, get_grpo_config
from utils.llama_completer import LlamaCompletionCtx
from utils.llama_completer import LlamaGRPOCompleter
from utils.qwen3_completer import Qwen3CompletionCtx
from utils.qwen3_completer import Qwen3GRPOCompleter

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False
    logging.warning("'wandb' package not installed; --wandb will be a no-op.")


class GRPOMonitor(TrainerCallback):
    def __init__(self, output_dir, wandb_enabled=False):
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        self.wandb_enabled = wandb_enabled and _WANDB_AVAILABLE
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
        # The logging format already prepends a timestamp (%(asctime)s).
        logging.info(
            "Step %d | Reward: %.4f | Len: %.2f (min %d, max %d) tokens | Step: %.2fs | Gen: %.2fs",
            step,
            reward,
            length,
            min_length,
            max_length,
            step_time_s,
            generation_time_s,
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length, step_time_s, generation_time_s])

        if self.wandb_enabled:
            wandb.log(
                {
                    "grpo/reward": reward,
                    "grpo/avg_length": length,
                    "grpo/min_length": min_length,
                    "grpo/max_length": max_length,
                    "grpo/step_time_s": step_time_s,
                    "grpo/generation_time_s": generation_time_s,
                },
                step=step,
            )

    def on_train_end(self, trainer):
        logging.info("Training complete.")
        if self.wandb_enabled:
            wandb.finish()


def boolq_reward(completions, answer, **kwargs):
    rewards = []
    correct_flags = []
    brevities = []
    char_lens = []
    for text, ground_truth in zip(completions, answer):
        clean = text.strip().lower()
        correct = clean.startswith(ground_truth.lower())
        accuracy = 2.0 if correct else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards.append(accuracy + brevity)
        correct_flags.append(1.0 if correct else 0.0)
        brevities.append(brevity)
        char_lens.append(len(text))

    # Decompose what GRPO is actually optimizing: task correctness (frac_correct
    # -- THE metric that says the model is learning the task) vs verbosity
    # (mean_brevity / mean_chars).
    n = max(len(rewards), 1)
    frac_correct = sum(correct_flags) / n
    logging.info(
        "[reward] frac_correct=%.3f mean_brevity=%.2f mean_chars=%.1f mean_reward=%.2f",
        frac_correct,
        sum(brevities) / n,
        sum(char_lens) / n,
        sum(rewards) / n,
    )
    # Log first generation for the FIRST prompt.
    if completions:
        logging.info("[reward] first-prompt gt=%r", answer[0])
        preview = completions[0].strip().replace("\n", " ")[:300]
        logging.info("[reward]   gen[%d] = %r", 0, preview)

    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO BoolQ training example")
    parser.add_argument(
        "--config",
        default="tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml",
        help=(
            "Training config path, relative to TT_METAL_RUNTIME_ROOT or absolute. "
            "Its device_config section (enable_ddp, mesh_shape) selects single-device vs DDP."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible runs. If omitted, seed defaults to 42.",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging of GRPO metrics",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="grpo-boolq",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated by wandb)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity / team",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode. If unset, uses WANDB_MODE env var or wandb default.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-1b",
        choices=["llama-1b", "qwen3"],
        help="Which model family to train: 'llama-1b' (single device, default) "
        "or 'qwen3' (ttml Qwen3 sharded with FSDP).",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        default=None,
        help="HuggingFace model ID or local path. Overrides the per-model default " "(qwen3 default: Qwen/Qwen3-32B).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Max sequence length for the qwen3 path (bounds the generation horizon).",
    )
    parser.add_argument(
        "--memory_efficient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Qwen3 runner mode. When set (the default), use gradient checkpointing "
        "(RunnerType.MemoryEfficient): per-block activations are recomputed in the "
        "backward pass to keep within DRAM at large micro-batch / sequence lengths. "
        "Pass --no-memory_efficient for the retain-activations runner (RunnerType.Default): "
        "faster backward, much higher peak memory.",
    )
    # Accept (and ignore) any extra flags passed by launch scripts so they
    # don't crash argument parsing.
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # Configure the root logger before any library (W&B, datasets, ...) can
    # claim it. INFO surfaces the per-generate summaries; set GRPO_LOGLEVEL=DEBUG
    # to also see the per-chunk decode progress in qwen3_completer. force=True
    # wins regardless of import order.
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_qwen3 = args.model == "qwen3"

    if is_qwen3:
        model_id = args.model_source or "Qwen/Qwen3-32B"
    else:
        model_id = args.model_source or "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    system_prompt = "Answer the question. Your answer should begin with either a Yes or a No. Then, explain why you answered Yes or No."

    def format_boolq(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
        ]
        # Qwen3 tokenizers expose a `enable_thinking` flag; disable it so the
        # model answers directly instead of emitting a <think> block.
        template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            ),
            "answer": "yes" if example["answer"] else "no",
        }

    dataset = load_dataset("google/boolq", split="train").shuffle(seed=args.seed).map(format_boolq)

    tt_metal_root = get_tt_metal_runtime_root()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(tt_metal_root, args.config)
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    logging.info(
        "Loaded config %s | enable_ddp=%s mesh_shape=%s (total_devices=%s)",
        config_path,
        device_config.enable_ddp,
        device_config.mesh_shape,
        device_config.total_devices(),
    )
    if is_qwen3:
        # Qwen3 architecture is read from the HF config inside the completer;
        # only max_sequence_length is consulted here (to bound generation).
        if training_config.model_config:
            transformer_config = get_model_config(training_config.model_config)
        else:
            transformer_config = TransformerConfig({"transformer_config": {"max_sequence_length": args.max_seq_len}})
    else:
        assert training_config.model_config, "training_config.model_config must be set"
        transformer_config = get_model_config(training_config.model_config)
    optimizer_dict = raw["training_config"]["optimizer"]

    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    grpo_config = get_grpo_config(raw, output_dir=output_dir)

    wandb_enabled = False
    if args.wandb:
        if not _WANDB_AVAILABLE:
            logging.warning("--wandb specified but the 'wandb' package is not installed; skipping W&B logging.")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config={
                    "model_id": model_id,
                    "max_completion_length": grpo_config.max_completion_length,
                    "num_generations": grpo_config.num_generations,
                    "temperature": grpo_config.temperature,
                    **optimizer_dict,
                },
            )
            wandb_enabled = True
            logging.info(
                "W&B logging enabled (project=%s, run=%s)",
                args.wandb_project,
                args.wandb_run_name or "<auto>",
            )

    if is_qwen3:
        completer = Qwen3GRPOCompleter(
            ctx=Qwen3CompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            device_config=device_config,
            model_source=model_id,
            memory_efficient=args.memory_efficient,
        )
    else:
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
        callbacks=[GRPOMonitor(output_dir, wandb_enabled=wandb_enabled)],
        model_source=model_id,
    )
    grpo_trainer.train()
    logging.info("BOOLQ GRPO TRAINING COMPLETE")
