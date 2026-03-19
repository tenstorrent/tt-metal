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


def setup(yaml_config_path, hf_model_id, load_pretrained) -> InferenceCtx:
    set_seed(42)

    yaml_config = load_config(yaml_config_path, f"{get_tt_metal_home()}/tt-train/configs/training_configs")

    # training_config -> model_config path
    model_config = load_config(yaml_config["training_config"]["model_config"])

    temperature = float(yaml_config["eval_config"]["temperature"])

    # GRPO runtime knobs from training yaml
    grpo_cfg = yaml_config.get("grpo_config", {})
    max_tokens_to_complete = int(grpo_cfg["max_tokens_to_complete"])
    group_size = int(grpo_cfg["group_size"])

    initialize_device(yaml_config)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = len(tokenizer)
    transformer_config = tt_model_factory.transformer_config

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(transformer_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(transformer_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(transformer_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(transformer_config, "original_context_length", 0) or 0,
    )

    runner_type = RunnerType.from_string(str(transformer_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if transformer_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(transformer_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=transformer_config.embedding_dim,
        intermediate_size=transformer_config.intermediate_dim,
        num_hidden_layers=transformer_config.num_blocks,
        num_attention_heads=transformer_config.num_heads,
        num_key_value_heads=transformer_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=transformer_config.max_sequence_length,
        rope_theta=transformer_config.theta or 10000.0,
        attention_dropout=transformer_config.dropout_prob,
        mlp_dropout=transformer_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    tt_model = LlamaCompositeKV(llama_cfg)
    if load_pretrained:
        model_repo_path = snapshot_download(
            repo_id=hf_model_id,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        load_from_safetensors(tt_model, model_repo_path, llama_cfg)

    ctx = InferenceCtx(
        tt_model=tt_model,
        tokenizer=tokenizer,
        pad_token=pad_token,
        temperature=temperature,
        max_tokens_to_complete=max_tokens_to_complete,
        transformer_config=transformer_config,
        group_size=group_size,
        tile_size=32,
        sample_seed=42,
    )

    return ctx


def extract_hash_answer(text: str) -> float | None:
    if "####" not in text:
        return None
    s = text.split("####")[1].strip()

    if s is None:
        return float("nan")

    import re

    number_re = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")

    if not number_re.fullmatch(s):
        return float("nan")

    return float(s.replace(",", ""))


def get_gsm8k(ctx: InferenceCtx, system_prompt, user_prompt_template_str, split="train") -> Tuple[List[str], List[str]]:
    data = load_dataset("openai/gsm8k", "main")[split]
    dataset = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    questions = [ex["question"] for ex in dataset]
    answers = [ex["answer"] for ex in dataset]

    t = Template(user_prompt_template_str)

    prompts = [
        ctx.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": t.substitute(question=q)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    return prompts, answers


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
        batch_size=32,
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
