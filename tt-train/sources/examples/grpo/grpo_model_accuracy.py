#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Sequence, Iterator

from utils.setup import InferenceCtx, setup_inference, setup_grpo_config
from utils.gsm8k import extract_hash_answer, get_gsm8k
from utils.boolq import get_boolq
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


def compare_boolq_answers(logger, completion, golden_answer) -> tuple[bool, str]:
    clean = completion.strip().lower()
    model_answer = clean.split()[0] if clean else "[EMPTY]"
    correct = model_answer.startswith(golden_answer.lower())
    char_count = len(completion)

    if correct:
        logger.info(f"CORRECT: model={model_answer.upper()}, golden={golden_answer.upper()} | Chars: {char_count}")
    else:
        logger.info(f"WRONG:   model={model_answer.upper()}, golden={golden_answer.upper()} | Chars: {char_count}")

    logger.info(f"  completion={completion}")

    return correct, model_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts to test (default: all)")
    parser.add_argument(
        "--yaml-path",
        type=str,
        default="tt-train/sources/examples/grpo/grpo_model_accuracy.yaml",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "boolq"],
        help="Dataset to evaluate on (default: gsm8k)",
    )

    accuracy_args, _ = parser.parse_known_args()

    run = setup_accuracy_run(output_dir=accuracy_args.output_dir)
    tracker = AccuracyMetricsTracker(run.output_dir)

    run.logger.info(f"args: {vars(accuracy_args)}")

    yaml_config_path = accuracy_args.yaml_path
    ctx = setup_inference(
        yaml_config_path,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        checkpoint_path=accuracy_args.checkpoint_path,
    )
    grpo_cfg = setup_grpo_config(yaml_config_path)

    if accuracy_args.dataset == "boolq":
        split = "validation"
        prompts, answers = get_boolq(ctx, split=split)
        compare_fn = compare_boolq_answers
    else:
        split = "test"
        prompts, answers = get_gsm8k(ctx, split=split)
        compare_fn = compare_numeric_answers

    prompts_to_test = accuracy_args.num_prompts or len(prompts)
    run.logger.info(f"dataset={accuracy_args.dataset}, {split=}, {prompts_to_test=}, {ctx.group_size=}")

    correct_answers = 0
    wrong_answers = 0
    total_chars = 0
    start_time = time.perf_counter()

    for i, prompt, completion in iter_generated_completions(
        ctx,
        prompts[:prompts_to_test],
        batch_size=grpo_cfg.completions_batch_size,
    ):
        run.logger.info(f"--- Question {i} ---")
        run.logger.info(f"{prompt=}")
        correct, model_answer = compare_fn(run.logger, completion, answers[i])
        total_chars += len(completion)
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

    total_answered = correct_answers + wrong_answers
    elapsed = time.perf_counter() - start_time
    avg_chars = total_chars / total_answered if total_answered > 0 else 0
    run.logger.info(
        f"Done: {correct_answers=}, {wrong_answers=}, "
        f"total={total_answered}, "
        f"accuracy={correct_answers / total_answered:.4f}, "
        f"avg_response_chars={avg_chars:.2f}, "
        f"elapsed={elapsed:.1f}s"
    )
    tracker.close()
