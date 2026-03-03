# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GLM 5 model with TTNN backend."""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
import transformers

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


MESH_DEVICE_MAP = {
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
}


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128tok"])
def test_glm(mesh_device, max_new_tokens):
    """Test GLM model with TTNN acceleration."""

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5")

    # Load config only (no weights download) and set to 5 decoder layers
    config = AutoConfig.from_pretrained("zai-org/GLM-5")
    config.num_hidden_layers = 5

    # Create model from config with random initialization (no weights download)
    model = AutoModelForCausalLM.from_config(config)
    nn_to_ttnn = {}
    nn_to_ttnn2 = {}

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
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    set_device(model, mesh_device)
    print("Running inference...")
    model.eval()
    torch.set_grad_enabled(False)
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
    DispatchManager.save_stats_to_file("glm_5_timing_stats.csv")
