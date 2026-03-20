#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    initialize_device,
    set_seed,
    get_tt_metal_home,
)
from ttml.common.config import load_config
from datasets import load_dataset
import time
from typing import List, Tuple
from batched_inference import (
    InferenceCtx,
    generate_answers_multiple_prompts,
    generate_answers_one_prompt,
)

from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from llama_overrides import LlamaCompositeKV
from typing import Iterator, Sequence
from string import Template

from eval_utils import setup, extract_hash_answer, get_gsm8k


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

    ctx: InferenceCtx = setup(
        yaml_config_path="training_grpo_accuracy_unsloth_llama_3_2_1b_instruct.yaml",
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )

    split = "test"
    print(f"{split=}")

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
    prompts, answers = get_gsm8k(ctx, system_prompt, user_prompt_template_str, split=split)

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
        batch_size=64,
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
