#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
from typing import List, Sequence, Iterator

from utils.setup import InferenceCtx, setup_inference, setup_grpo_config
from utils.boolq import get_boolq, _correctness_reward
from utils.inference import completion_batched_multiple_prompts
from utils.bookkeeping import setup_accuracy_run


def iter_batched_completions(
    ctx: InferenceCtx,
    prompts: Sequence[List[int]],
    batch_size: int = 32,
) -> Iterator[tuple[List[List[int]], List[List[int]]]]:
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])

        completions_batch = completion_batched_multiple_prompts(ctx, prompt_batch)

        prompt_batch_yield = [item for item in prompt_batch for _ in range(ctx.group_size)]
        assert len(prompt_batch_yield) == len(completions_batch)
        yield prompt_batch_yield, completions_batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--yaml-path", type=str, default="tt-train/sources/examples/grpo/boolq_example.yaml")
    args, _ = parser.parse_known_args()

    run = setup_accuracy_run()

    ctx = setup_inference(
        args.yaml_path,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )
    grpo_cfg = setup_grpo_config(args.yaml_path)

    split = "validation"
    prompts, answers = get_boolq(ctx, split=split, shuffle_seed=42)
    prompts_to_test = min(args.num_prompts, len(prompts))
    run.logger.info(f"{split=}, {prompts_to_test=}, {ctx.group_size=}, batch_size={grpo_cfg.completions_batch_size}")

    prompts_raw = [ctx.tokenizer.encode(s) for s in prompts[:prompts_to_test]]

    total_completions = 0
    all_completions = []
    start_time = time.perf_counter()

    for batch_idx, (prompt_batch, completions_batch) in enumerate(
        iter_batched_completions(ctx, prompts_raw, batch_size=grpo_cfg.completions_batch_size)
    ):
        total_completions += len(completions_batch)
        run.logger.info(
            f"batch {batch_idx}: {len(completions_batch)} completions done "
            f"({total_completions}/{prompts_to_test * ctx.group_size} total)"
        )

        global_offset = len(all_completions)
        for i, completion in enumerate(completions_batch):
            q_idx = (global_offset + i) // ctx.group_size
            g_idx = (global_offset + i) % ctx.group_size
            completion_str = ctx.tokenizer.decode(completion, skip_special_tokens=False)
            run.logger.info(f"  Q{q_idx} G{g_idx}: len={len(completion)} tokens, {len(completion_str)} chars")

        all_completions.extend(completions_batch)

    elapsed = time.perf_counter() - start_time
    run.logger.info(f"All done: {total_completions} completions in {elapsed:.2f}s")

    run.logger.info("Completions:")
    for i, completion in enumerate(all_completions):
        q_idx = i // ctx.group_size
        g_idx = i % ctx.group_size
        completion_str = ctx.tokenizer.decode(completion, skip_special_tokens=False)
        run.logger.info(f"--- Question {q_idx}, Completion {g_idx} ---")
        run.logger.info(completion_str)
