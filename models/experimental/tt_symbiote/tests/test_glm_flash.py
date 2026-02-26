# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GLM 4.7 Flash model with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
import transformers
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.moe import TTNNMoE
from models.experimental.tt_symbiote.modules.attention import (
    TTNNGlm4MoeLiteAttention,
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create a fresh TTNNPagedAttentionKVCache sized for the model."""
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32)
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim_k=model_config.qk_head_dim,
        head_dim_v=model_config.v_head_dim,
        paged_config=paged_config,
        device=None,
        batch_size=batch_size,
    ).to_device(device)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
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
def test_glm(mesh_device):
    """Test GLM model with TTNN acceleration."""

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
    model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash")
    nn_to_ttnn = {
        model.model.layers[0].self_attn.__class__: TTNNGlm4MoeLiteAttention,  # Add TTNNGlm4MoeLiteAttention module
        model.model.layers[1].mlp.__class__: TTNNMoE,  # Add TTNNMoE module
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,  # TTNNLinearLLamaIColShardedWRowSharded
    }

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
    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    print("Running inference...")
    model.eval()
    torch.set_grad_enabled(False)

    paged_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_cache)
    DispatchManager.clear_timings()

    paged_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=paged_cache)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
    DispatchManager.save_stats_to_file("glm_timing_stats.csv")
    TracedRun.release_all()
