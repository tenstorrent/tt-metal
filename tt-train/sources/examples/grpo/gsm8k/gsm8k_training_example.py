#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import List

from datasets import load_dataset
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root, set_seed
from ttml.trainers import GRPOTrainer, get_grpo_config

from gsm8k_utils import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    FORMAT_RE,
    STRICT_FORMAT_RE,
    THINK_CLOSE,
    THINK_OPEN,
    extract_hash_answer,
    extract_tag_answer,
    format_prompt,
)


def correctness_reward(completions, answer, **kwargs) -> List[float]:
    """+2 when the number in <answer> equals the gold answer."""
    return [2.0 if extract_tag_answer(c) == gold else 0.0 for c, gold in zip(completions, answer)]


def xmlcount_reward(completions, **kwargs) -> List[float]:
    """+0.125 per tag present exactly once, minus a small penalty for text after </answer>."""

    def score(text: str) -> float:
        s = 0.0
        for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN):
            if text.count(tag) == 1:
                s += 0.125
        if text.count(ANSWER_CLOSE) == 1:
            s += 0.125
            s -= len(text.split(ANSWER_CLOSE)[-1].strip()) * 0.001
        return s

    return [score(c) for c in completions]


def soft_format_reward(completions, **kwargs) -> List[float]:
    return [0.5 if FORMAT_RE.search(c) else 0.0 for c in completions]


def strict_format_reward(completions, **kwargs) -> List[float]:
    return [0.5 if STRICT_FORMAT_RE.match(c) else 0.0 for c in completions]


def int_reward(completions, **kwargs) -> List[float]:
    """+0.5 when the answer is an integer."""

    def score(text: str) -> float:
        p = extract_tag_answer(text)
        return 0.5 if p is not None and p.lstrip("-").isdigit() else 0.0

    return [score(c) for c in completions]


REWARD_FUNCS = [correctness_reward, xmlcount_reward, soft_format_reward, strict_format_reward, int_reward]


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO GSM8K training example")
    parser.add_argument(
        "--config",
        default="tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_remote_rollout.yaml",
        help="Training config path, relative to TT_METAL_RUNTIME_ROOT or absolute.",
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

    tt_metal_root = get_tt_metal_runtime_root()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(tt_metal_root, args.config)
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    set_seed(training_config.seed)
    device_config = DeviceConfig(raw)

    def make_dataset():
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return ds.shuffle(seed=training_config.seed).map(
            lambda row: {"prompt": format_prompt(row["question"]), "answer": extract_hash_answer(row["answer"])},
            remove_columns=ds.column_names,
        )

    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_gsm8k_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )

    grpo_trainer = GRPOTrainer(
        transformer_config=get_model_config(training_config.model_config),
        device_config=device_config,
        model_source=raw["training_config"]["model_source"],
        dataset_func=make_dataset,
        config=get_grpo_config(raw, output_dir=output_dir),
        reward_funcs=REWARD_FUNCS,
        optimizer_dict=raw["training_config"]["optimizer"],
    )
    grpo_trainer.train()
    logging.info("GSM8K GRPO TRAINING COMPLETE")
