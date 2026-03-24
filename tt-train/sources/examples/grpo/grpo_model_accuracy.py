#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Sequence, Iterator

from utils.setup import InferenceCtx, setup_inference, setup_grpo_config
from utils.gsm8k import extract_hash_answer, get_gsm8k
from utils.inference import generate_answers_multiple_prompts


def iter_generated_completions(
    ctx: InferenceCtx,
    prompts: Sequence[str],
    batch_size: int = 32,
) -> Iterator[tuple[int, str, str]]:
    """
    Yields aligned results one-by-one:
      i, prompts[i], completions[i]
    """
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])

        batch_completions = generate_answers_multiple_prompts(ctx, prompt_batch)

        if ctx.group_size != 1:
            raise ValueError(f"Expected group_size=1 for 1:1 mapping, got {ctx.group_size}. ")

        for offset, completion in enumerate(batch_completions):
            i = start + offset
            yield i, prompts[i], completion


def compare_numeric_answers(completion, golden_answer) -> bool:
    global correct_answers, wrong_answers

    completion_answer_num = float(extract_hash_answer(completion) or "nan")
    golden_answer_num = float(golden_answer or "nan")

    if abs(completion_answer_num - golden_answer_num) < 0.001:
        print(f"Completion answer is correct. {completion_answer_num=}, {golden_answer_num=}")
        correct = True
    else:
        print(f"Completion answer is wrong. {completion_answer_num=}, {golden_answer_num=}")
        correct = False

    print(f"{completion=}")
    print(f"{extract_hash_answer(completion)=}")
    print(f"{golden_answer=}")

    return correct


if __name__ == "__main__":
    start_time = time.perf_counter()

    yaml_config_path = "tt-train/sources/examples/grpo/grpo_model_accuracy.yaml"

    ctx: InferenceCtx = setup_inference(
        yaml_config_path,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )
    grpo_cfg = setup_grpo_config(yaml_config_path)

    split = "test"
    print(f"{split=}")

    prompts, answers = get_gsm8k(ctx, split=split)

    for a in answers:
        assert a is not None

    start_time = time.perf_counter()

    prompts_to_test = len(prompts)
    print(f"{prompts_to_test=}")
    print(f"{ctx.group_size=}")

    correct_answers = 0
    wrong_answers = 0

    for i, prompt, completion in iter_generated_completions(
        ctx,
        prompts[:prompts_to_test],
        batch_size=grpo_cfg.completions_batch_size,
    ):
        print(f"{i=}")
        correct: bool = compare_numeric_answers(completion, answers[i])
        if correct:
            correct_answers += 1
        else:
            wrong_answers += 1

    print(f"{correct_answers=}, {wrong_answers=}, total_answers={correct_answers + wrong_answers}")
    end_time = time.perf_counter()
    print(f"Completions done. Took {end_time - start_time} s to complete")
