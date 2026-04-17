# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from utils.llama_completion import LlamaCompletionCtx
from utils.llama_completion import LlamaCompleter

HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE = 0.0
MAX_COMPLETION_LENGTH = 256
NUM_GENERATIONS = 1

TRANSFORMER_CONFIG = {
    "model_type": "llama",
    "num_heads": 32,
    "num_groups": 8,
    "embedding_dim": 2048,
    "intermediate_dim": 8192,
    "dropout_prob": 0.0,
    "num_blocks": 16,
    "weight_tying": "enabled",
    "vocab_size": 32000,
    "max_sequence_length": 1024,
    "runner_type": "memory_efficient",
    "theta": 500000.0,
    "rope_scaling": {
        "scaling_factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_context_length": 8192,
    },
}

DEVICE_CONFIG = {
    "enable_ddp": False,
    "mesh_shape": [1, 1],
}

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
    llama = LlamaCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=MAX_COMPLETION_LENGTH,
            temperature=TEMPERATURE,
            completions_per_prompt=NUM_GENERATIONS,
        ),
        transformer_config=TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_MODEL_ID,
    )

    assert NUM_GENERATIONS == 1, "This test expects num_generations=1 for 1:1 comparison."
    assert TEMPERATURE == 0.0, "This test expects greedy decoding (temperature=0)."

    tokenizer = llama.tokenizer
    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom is",
        "The capital of Czech Republic is",
    ]
    prompts = [to_chat_prompt(tokenizer, p) for p in user_prompts]

    single_outputs = []
    for prompt in prompts:
        completions = llama.generate_str([prompt])
        assert len(completions) == 1
        single_outputs.append(completions[0])

    batched_outputs = llama.generate_str(prompts)
    assert len(batched_outputs) == len(prompts)

    assert batched_outputs == single_outputs, (
        "Mismatch between one-by-one and batched outputs.\n" f"single={single_outputs}\n" f"batch={batched_outputs}"
    )
