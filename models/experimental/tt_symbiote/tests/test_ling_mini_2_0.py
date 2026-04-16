# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Ling-mini-2.0 with TTNN backend."""

import os

import pytest

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.models.ling import (
    DEFAULT_MODEL_NAME,
    load_model,
)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_ling_mini_2_0(mesh_device):
    """Test Ling-mini-2.0 model with TTNN acceleration."""

    model, tokenizer, paged_cache = load_model(mesh_device, DEFAULT_MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    print("Running inference with paged attention...")

    # Warmup run without trace
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_cache)
    paged_cache.reset()
    # Actual run with trace
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True, past_key_values=paged_cache)
    paged_cache.reset()

    DispatchManager.clear_timings()
    model._ttnn_causal_lm_wrapper.reset_perf_stats()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=paged_cache)

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Ling-mini-2.0 PAGED ATTENTION OUTPUT: {decoded}")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    model._ttnn_causal_lm_wrapper.print_perf_stats()
    DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
    TracedRun.release_all()
