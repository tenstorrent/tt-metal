# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Ling-mini-2.0 with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayer
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create a paged attention KV cache for Ling-mini-2.0.

    Args:
        model_config: Model configuration
        device: TTNN device
        batch_size: Batch size

    Returns:
        TTNNPagedAttentionKVCache instance
    """
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        config=config,
        device=None,
    ).to_device(device)


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

    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayer,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
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
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    print("Running inference with paged attention...")
    model.eval()
    torch.set_grad_enabled(False)

    # Warmup run
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_cache)

    # Reset cache for main run — must also release all traces since they
    # reference the old cache's device buffers which will be deallocated.
    TracedRun.release_all()
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=paged_cache)

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Ling-mini-2.0 PAGED ATTENTION OUTPUT: {decoded}")

    # Verify output is coherent (non-empty generated text)
    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
    TracedRun.release_all()
