# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttml

from ttml.common.config import DeviceConfig, TransformerConfig
from utils.inference import (
    setup_inference,
    generate_answers_multiple_prompts,
    generate_answers_one_prompt,
)

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
    transformer_config = TransformerConfig({"transformer_config": TRANSFORMER_CONFIG})
    device_config = DeviceConfig({"device_config": DEVICE_CONFIG})

    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    ttml.autograd.AutoContext.get_instance().open_device(device_config.mesh_shape, device_config.device_ids)

    ctx = setup_inference(
        TEMPERATURE, MAX_COMPLETION_LENGTH, NUM_GENERATIONS, transformer_config, device_config, HF_MODEL_ID
    )

    assert ctx.group_size == 1, "This test expects group_size=1 for 1:1 comparison."
    assert ctx.temperature == 0.0, "This test expects greedy decoding (temperature=0)."

    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom is",
        "The capital of Czech Republic is",
    ]
    prompts = [to_chat_prompt(ctx.tokenizer, p) for p in user_prompts]

    single_outputs = []
    for prompt in prompts:
        completions = generate_answers_one_prompt(ctx, prompt)
        assert len(completions) == 1
        single_outputs.append(completions[0])

    batched_outputs = generate_answers_multiple_prompts(ctx, prompts)
    assert len(batched_outputs) == len(prompts)

    assert batched_outputs == single_outputs, (
        "Mismatch between one-by-one and batched outputs.\n" f"single={single_outputs}\n" f"batch={batched_outputs}"
    )
