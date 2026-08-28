#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# `utils` lives one level up, in the shared examples/grpo directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from ttml.trainers import GRPOTrainer, get_grpo_config
from utils.llama_completer import LlamaCompletionCtx
from utils.llama_completer import LlamaGRPOCompleter
from utils.qwen3_completer import Qwen3CompletionCtx
from utils.qwen3_completer import Qwen3GRPOCompleter

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = (
    "Answer the question. Your answer should begin with either a Yes or a No. "
    "Then, explain why you answered Yes or No."
)


def accuracy_reward(completions, answer, **kwargs):
    """+2 if the completion begins with the correct Yes/No token, -1 otherwise."""
    return [2.0 if text.strip().lower().startswith(gt.lower()) else -1.0 for text, gt in zip(completions, answer)]


def brevity_reward(completions, **kwargs):
    """Quadratic length penalty in characters, discouraging runaway completions."""
    return [-0.1 * (len(text) / 20) ** 2 for text in completions]


def make_format_boolq(tokenizer, is_qwen3):
    # Qwen3 tokenizers expose an `enable_thinking` flag; disable it so the
    # model answers directly instead of emitting a <think> block.
    template_kwargs = {"enable_thinking": False} if is_qwen3 else {}

    def format_boolq(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            ),
            "answer": "yes" if example["answer"] else "no",
        }

    return format_boolq


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO BoolQ training example")
    parser.add_argument(
        "--config",
        default="tt-train/configs/training_configs/grpo_boolq_llama_1b_1dev.yaml",
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
        "--model",
        type=str,
        default="llama-1b",
        choices=["llama-1b", "qwen3"],
        help="Which model family to train: 'llama-1b' (single device, default) "
        "or 'qwen3' (ttml Qwen3 sharded with FSDP).",
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
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
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

    model_id = raw["training_config"].get("model_source") or DEFAULT_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dataset = (
        load_dataset("google/boolq", split="train").shuffle(seed=args.seed).map(make_format_boolq(tokenizer, is_qwen3))
    )

    assert training_config.model_config, "training_config.model_config must be set"
    transformer_config = get_model_config(training_config.model_config)
    optimizer_dict = raw["training_config"]["optimizer"]

    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    grpo_config = get_grpo_config(raw, output_dir=output_dir)

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
        reward_funcs=[accuracy_reward, brevity_reward],
        optimizer_dict=optimizer_dict,
        model_source=model_id,
    )
    grpo_trainer.train()
    logging.info("BOOLQ GRPO TRAINING COMPLETE")
