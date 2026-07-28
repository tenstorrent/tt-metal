#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Iterator

from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
from ttml.common.utils import get_tt_metal_runtime_root
from utils.llama_completer import LlamaCompletionCtx
from utils.llama_completer import LlamaGRPOCompleter

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a concise assistant that outputs short sentences. Print Yes or No in the first sentence. Make sure your Yes/No answer is factually correct."
BATCH_SIZE = 16
MAX_COMPLETION_LENGTH = 256
TEMPERATURE = 0.0
NUM_GENERATIONS = 1
PROMPTS_TO_VALIDATE = 20

_CONFIG_PATH = Path(__file__).with_suffix(".yaml")
_RAW_CONFIG = load_config(str(_CONFIG_PATH))
_TRAINING_CONFIG = TrainingConfig(_RAW_CONFIG)
DEVICE_CONFIG = DeviceConfig(_RAW_CONFIG)
assert _TRAINING_CONFIG.model_config, "training_config.model_config must be set"
TRANSFORMER_CONFIG = get_model_config(_TRAINING_CONFIG.model_config)


def iter_generated_completions(
    llama: LlamaGRPOCompleter,
    prompts: Sequence[str],
    batch_size: int = 32,
    num_generations: int = 1,
) -> Iterator[tuple[int, str, str]]:
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])
        batch_completions = llama.generate_str(prompt_batch)
        if num_generations != 1:
            raise ValueError(f"Expected num_generations=1, got {num_generations}")
        for offset, completion in enumerate(batch_completions):
            i = start + offset
            yield i, prompts[i], completion


def compare_boolq_answers(completion, golden_answer) -> tuple[bool, str]:
    clean = completion.strip().lower()
    model_answer = clean.split()[0] if clean else "[EMPTY]"
    correct = model_answer.startswith(golden_answer.lower())
    return correct, model_answer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def format_boolq(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": "yes" if example["answer"] else "no",
        }

    dataset = load_dataset("google/boolq", split="validation").map(format_boolq)
    prompts = list(dataset["prompt"])
    answers = list(dataset["answer"])

    output_dir = os.path.join(
        get_tt_metal_runtime_root(),
        "generated/tt-train/grpo_model_accuracy_runs",
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "question_idx",
                "correct",
                "golden_answer",
                "model_answer",
                "correct_so_far",
                "total_so_far",
                "running_accuracy",
            ],
        )
        csv_writer.writeheader()

        llama = LlamaGRPOCompleter(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=MAX_COMPLETION_LENGTH,
                temperature=TEMPERATURE,
                completions_per_prompt=NUM_GENERATIONS,
            ),
            transformer_config=TRANSFORMER_CONFIG,
            device_config=DEVICE_CONFIG,
            model_source=MODEL_ID,
        )

        correct_answers = 0
        wrong_answers = 0
        total_chars = 0
        start_time = time.perf_counter()

        for i, prompt, completion in iter_generated_completions(
            llama, prompts[:PROMPTS_TO_VALIDATE], batch_size=BATCH_SIZE
        ):
            correct, model_answer = compare_boolq_answers(completion, answers[i])
            total_chars += len(completion)
            if correct:
                correct_answers += 1
            else:
                wrong_answers += 1

            total = correct_answers + wrong_answers
            accuracy = correct_answers / total
            status = "CORRECT" if correct else "WRONG"
            print(
                f"Q{i}: {status} | model={model_answer}, golden={answers[i]} | Accuracy: {accuracy:.4f} ({correct_answers}/{total})"
            )

            csv_writer.writerow(
                {
                    "question_idx": i,
                    "correct": correct,
                    "golden_answer": answers[i],
                    "model_answer": model_answer,
                    "correct_so_far": correct_answers,
                    "total_so_far": total,
                    "running_accuracy": f"{accuracy:.4f}",
                }
            )
            csv_file.flush()

        total_answered = correct_answers + wrong_answers
        elapsed = time.perf_counter() - start_time
        avg_chars = total_chars / total_answered if total_answered > 0 else 0
        print(
            f"Done: correct={correct_answers}, wrong={wrong_answers}, "
            f"total={total_answered}, "
            f"accuracy={correct_answers / total_answered:.4f}, "
            f"avg_response_chars={avg_chars:.2f}, "
            f"elapsed={elapsed:.1f}s"
        )
