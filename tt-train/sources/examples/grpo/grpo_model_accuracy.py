#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Sequence, Iterator

from utils.setup import InferenceCtx, setup_inference, setup_grpo_config
from utils.gsm8k import extract_hash_answer, get_gsm8k
from utils.inference import generate_answers_multiple_prompts
from utils.bookkeeping import setup_accuracy_run, AccuracyMetricsTracker


def iter_generated_completions(
    ctx: InferenceCtx,
    prompts: Sequence[str],
    batch_size: int = 32,
) -> Iterator[tuple[int, str, str]]:
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])
        batch_completions = generate_answers_multiple_prompts(ctx, prompt_batch)
        if ctx.group_size != 1:
            raise ValueError(f"Expected group_size=1, got {ctx.group_size}")
        for offset, completion in enumerate(batch_completions):
            i = start + offset
            yield i, prompts[i], completion


def compare_numeric_answers(logger, completion, golden_answer) -> tuple[bool, str]:
    completion_answer = extract_hash_answer(completion)
    completion_answer_num = float(completion_answer or "nan")
    golden_answer_num = float(golden_answer or "nan")

    correct = abs(completion_answer_num - golden_answer_num) < 0.001

    if correct:
        logger.info(f"CORRECT: model={completion_answer_num}, golden={golden_answer_num}")
    else:
        logger.info(f"WRONG:   model={completion_answer_num}, golden={golden_answer_num}")

    logger.info(f"  completion={completion}")
    logger.info(f"  extracted={completion_answer}")
    logger.info(f"  golden={golden_answer}")

    return correct, str(completion_answer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts to test (default: all)")
    accuracy_args, _ = parser.parse_known_args()

    run = setup_accuracy_run()
    tracker = AccuracyMetricsTracker(run.output_dir)

    yaml_config_path = "tt-train/sources/examples/grpo/grpo_model_accuracy.yaml"
    ctx = setup_inference(
        yaml_config_path,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )
    grpo_cfg = setup_grpo_config(yaml_config_path)

    split = "test"
    prompts, answers = get_gsm8k(ctx, split=split)
    prompts_to_test = accuracy_args.num_prompts or len(prompts)
    run.logger.info(f"{split=}, {prompts_to_test=}, {ctx.group_size=}")

    correct_answers = 0
    wrong_answers = 0
    start_time = time.perf_counter()

    for i, prompt, completion in iter_generated_completions(
        ctx,
        prompts[:prompts_to_test],
        batch_size=grpo_cfg.completions_batch_size,
    ):
        run.logger.info(f"--- Question {i} ---")
        run.logger.info(f"{prompt=}")
        correct, model_answer = compare_numeric_answers(run.logger, completion, answers[i])
        if correct:
            correct_answers += 1
        else:
            wrong_answers += 1

        total = correct_answers + wrong_answers
        tracker.log_result(
            question_idx=i,
            prompt=prompt,
            correct=correct,
            golden_answer=answers[i],
            model_answer=model_answer,
            correct_so_far=correct_answers,
            total_so_far=total,
            running_accuracy=correct_answers / total,
        )

    elapsed = time.perf_counter() - start_time
    run.logger.info(
        f"Done: {correct_answers=}, {wrong_answers=}, "
        f"total={correct_answers + wrong_answers}, "
        f"accuracy={correct_answers / (correct_answers + wrong_answers):.4f}, "
        f"elapsed={elapsed:.1f}s"
    )
    tracker.close()
