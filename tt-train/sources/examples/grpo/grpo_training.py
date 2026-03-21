#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttml.common.utils import (
    no_grad,
)
from typing import List, Sequence, Iterator
from eval_utils import setup, get_gsm8k
from batched_inference import completion_batched_multiple_prompts, compute_nlog_probs, InferenceCtx

system_prompt = "You are a helpful math tutor. Show your reasoning step by step and end with exactly one final line in this format: #### <number>"
user_prompt_template_str = """Question: There are 48 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 64 trees. How many trees did the grove workers plant today?
Answer: There are 48 trees originally.
Then there were 64 trees after some more were planted.
So there must have been 64 - 48 = 16.
#### 16

Question: If there are 13 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 13 cars.
2 more cars arrive.
13 + 2 = 15.
#### 15

Question: $question
Answer:
"""


def iter_batched_completions(
    ctx: InferenceCtx,
    prompts: Sequence[List[int]],
    batch_size: int = 32,
) -> Iterator[tuple[int, int, List[List[int]], List[List[int]]]]:
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])

        completions_batch = completion_batched_multiple_prompts(ctx, prompt_batch)

        yield start, end, prompt_batch, completions_batch


def iter_micro_batch(prompt_len, micro_batch_size=2):
    for start in range(0, prompt_len, micro_batch_size):
        end = min(start + micro_batch_size, len(prompts))

        yield start, end


def train_grpo():
    ctx = setup(
        yaml_config_path="training_grpo_accuracy_unsloth_llama_3_2_1b_instruct.yaml",
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )

    prompts, answers = get_gsm8k(ctx, system_prompt, user_prompt_template_str, split="train")
    prompts = [ctx.tokenizer.encode(s) for s in prompts]

    prompts_to_train = 2
    mini_batches = 2

    for start, end, prompts_batch, completions_batch in iter_batched_completions(
        ctx, prompts[:prompts_to_train], batch_size=32
    ):
        ctx.tt_model.eval()
        with no_grad():
            nlog_probs_old = compute_nlog_probs(ctx, prompts_batch, completions_batch)

        # for mini_batch in range(mini_batches):

        # ctx.tt_model.eval()
        # nlog_probs_old = compute_nlog_probs()

        # ctx.tt_model.train()
        # for mini_batch in range(mini_batches):
        #     nlog_probs_cur = compute_nlog_probs()


if __name__ == "__main__":
    train_grpo()
