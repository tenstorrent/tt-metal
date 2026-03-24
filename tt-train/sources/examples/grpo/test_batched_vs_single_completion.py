# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from utils.setup import setup_inference
from utils.inference import generate_answers_multiple_prompts, generate_answers_one_prompt

HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
YAML_CONFIG = "tt-train/sources/examples/grpo/test_batched_vs_single_completion.yaml"

SYSTEM_PROMPT = (
    "You are a precise geography assistant.\n"
    "Given a country, reply with exactly one word: its capital city in English.\n"
    "Feel free to describe the capital city or the country."
)


def to_chat_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.slow
def test_capitals_one_by_one_equals_single_batch():
    ctx = setup_inference(
        yaml_config_path=YAML_CONFIG,
        hf_model_id=HF_MODEL_ID,
        load_pretrained=True,
    )

    assert ctx.group_size == 1, "This test expects group_size=1 for 1:1 comparison."
    assert ctx.temperature == 0.0, "This test expects greedy decoding (temperature=0)."

    # Trying different token lengths to make sure padding works
    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom of Great Britain and Northern Ireland is",
        "The capital of Czech Republic is",
    ]
    prompts = [to_chat_prompt(ctx.tokenizer, p) for p in user_prompts]

    # One-by-one
    single_outputs = []
    for prompt in prompts:
        completions = generate_answers_one_prompt(ctx, prompt)
        assert len(completions) == 1
        single_outputs.append(completions[0])

    # One batch
    batched_outputs = generate_answers_multiple_prompts(ctx, prompts)
    assert len(batched_outputs) == len(prompts)

    # Exact string equality as requested
    assert batched_outputs == single_outputs, (
        "Mismatch between one-by-one and batched outputs.\n" f"single={single_outputs}\n" f"batch={batched_outputs}"
    )
