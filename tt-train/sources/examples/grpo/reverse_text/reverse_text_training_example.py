#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on the reverse-text task, ported from the prime-rl / verifiers example.

Trains Qwen3-0.6B to reverse text character-by-character on a single p150.
Rollouts are generated on device by ``TTMLRolloutSampler``.

Run:
    python3 reverse_text/reverse_text_training_example.py
"""

import argparse
import logging
import os
import random
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from ttml.trainers import GRPOTrainer, TrainerCallback, get_grpo_config

MODEL_SOURCE = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
DATASET = "PrimeIntellect/Reverse-Text-RL"
DATASET_SPLIT = "train"
DEFAULT_CONFIG = "tt-train/configs/training_configs/grpo_reverse_text_qwen3_p6b_1dev.yaml"

SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."

# Prompts held out of training and used by the greedy eval.
EVAL_SIZE = 64

TAG_RE = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


def parse_reversed_text(completion: str) -> str:
    """Return the last ``<reversed_text>`` block, or "" when the format is missing."""
    matches = TAG_RE.findall(completion)
    return matches[-1].strip() if matches else ""


def build_dataset(tokenizer, seed):
    """Return (train, eval) datasets with a templated "prompt" and an "answer"."""
    ds = load_dataset(DATASET, split=DATASET_SPLIT)

    def to_example(row):
        text = row["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        return {
            # Qwen3 tokenizers expose an `enable_thinking` flag; disable it so the
            # model answers directly instead of emitting a <think> block.
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            "answer": text[::-1],
        }

    ds = ds.map(to_example, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=seed)
    return split["train"], split["test"]


def similarity_reward(completions, answer, **kwargs):
    """``difflib`` similarity ratio between the parsed answer and the truth.

    The TRL reference calls this ``lcs_reward``, but ``SequenceMatcher.ratio()``
    scores contiguous matching blocks (2 * matches / total length), not a longest
    common subsequence.

    This is the only weighted reward in the TRL reference run; exact-match and
    format rates are logged as diagnostics so the trailing signals stay visible
    without entering the objective. Per-completion samples are printed by the
    built-in ``GRPOMonitor`` when ``log_completions=True`` in the YAML config.
    """
    parsed = [parse_reversed_text(c) for c in completions]
    rewards = [SequenceMatcher(None, got, truth).ratio() for got, truth in zip(parsed, answer)]

    n = max(len(rewards), 1)
    frac_exact = sum(1.0 for got, truth in zip(parsed, answer) if got == truth) / n
    frac_format = sum(1.0 for got in parsed if got) / n
    logging.info(
        "[reward] mean_similarity=%.3f frac_exact=%.3f frac_format=%.3f",
        sum(rewards) / n,
        frac_exact,
        frac_format,
    )
    return rewards


class EvalCallback(TrainerCallback):
    """Greedy eval on the held-out split, before training and after every step.

    Generation parameters live on the trainer's rollout sampler, so greedy
    decoding is a temporary change of its ``temperature`` and
    ``completions_per_prompt``: ``temperature == 0.0`` takes the pure-argmax
    path in ``ttnn_fixed::sample``.

    Writes the three eval scalars (``eval_similarity`` / ``eval_chars`` /
    ``eval_format``) into ``trainer.metrics`` so the built-in ``GRPOMonitor``
    picks them up as CSV columns in the same step's row.
    """

    def __init__(self, dataset, num_examples):
        rows = dataset.select(range(min(num_examples, len(dataset))))
        self.prompts = list(rows["prompt"])
        self.answers = list(rows["answer"])
        self.latest: dict[str, float] = {}

    def on_train_begin(self, trainer):
        self._evaluate(trainer, 0)
        trainer.metrics.update(self.latest)

    def on_step_end(self, trainer, step, **kwargs):
        self._evaluate(trainer, step)
        trainer.metrics.update(self.latest)

    def _evaluate(self, trainer, step):
        sampler = trainer.rollout_sampler
        tokenizer = trainer.tokenizer
        saved = (sampler.temperature, sampler.completions_per_prompt)
        sampler.temperature, sampler.completions_per_prompt = 0.0, 1
        try:
            batch = sampler.generate([tokenizer.encode(p) for p in self.prompts])
        finally:
            sampler.temperature, sampler.completions_per_prompt = saved
        texts = [tokenizer.decode(c) for c in batch.completions]

        similarities, char_fracs, formats = [], [], []
        for text, answer in zip(texts, self.answers):
            got = parse_reversed_text(text)
            matcher = SequenceMatcher(None, got, answer)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            similarities.append(matcher.ratio())
            char_fracs.append(matched / len(answer) if answer else 0.0)
            formats.append(1.0 if got else 0.0)

        n = max(len(similarities), 1)
        self.latest = {
            "eval_similarity": sum(similarities) / n,
            "eval_chars": sum(char_fracs) / n,
            "eval_format": sum(formats) / n,
        }
        logging.info(
            "[eval] step %d | similarity %.3f | chars %.1f%% | format %.1f%%",
            step,
            self.latest["eval_similarity"],
            100.0 * self.latest["eval_chars"],
            100.0 * self.latest["eval_format"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO reverse-text training example")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Training config path, relative to TT_METAL_RUNTIME_ROOT or absolute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible runs, and the train/eval split seed.",
    )
    parser.add_argument(
        "--eval_examples",
        type=int,
        default=16,
        help="Number of held-out prompts decoded greedily by the eval callback each step.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Override the model config's max_sequence_length (bounds the generation horizon "
        "and the decode KV cache).",
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

    tt_metal_root = get_tt_metal_runtime_root()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(tt_metal_root, args.config)
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)

    model_source = raw["training_config"].get("model_source") or MODEL_SOURCE
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    train_dataset, eval_dataset = build_dataset(tokenizer, args.seed)

    assert training_config.model_config, "training_config.model_config must be set"
    transformer_config = get_model_config(training_config.model_config)
    if args.max_seq_len is not None:
        transformer_config.max_sequence_length = args.max_seq_len
    optimizer_dict = raw["training_config"]["optimizer"]

    output_dir = os.path.join(
        tt_metal_root,
        "generated/tt-train/grpo_reverse_text_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    grpo_config = get_grpo_config(raw, output_dir=output_dir)
    logging.info(
        "Loaded config %s | mesh_shape=%s (total_devices=%s) | max_sequence_length=%d | "
        "%d completions/prompt x %d prompts/batch",
        config_path,
        device_config.mesh_shape,
        device_config.total_devices(),
        transformer_config.max_sequence_length,
        grpo_config.num_generations,
        grpo_config.gradient_accumulation_steps,
    )

    grpo_trainer = GRPOTrainer(
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=model_source,
        dataset=train_dataset,
        config=grpo_config,
        reward_func=similarity_reward,
        optimizer_dict=optimizer_dict,
        callbacks=[EvalCallback(eval_dataset, args.eval_examples)],
    )
    grpo_trainer.train()
    logging.info("REVERSE TEXT GRPO TRAINING COMPLETE")
