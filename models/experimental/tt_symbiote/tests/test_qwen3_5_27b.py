# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3.5-27B-FP8 model with TTNN backend.

This model uses a HYBRID attention architecture with:
- Linear attention layers (48/64): Qwen3_5GatedDeltaNet (DeltaNet-style)
- Full attention layers (16/64): GatedAttention (GQA with 4 KV heads)
- Dense SwiGLU MLP (NO MoE)
- FP8 quantized weights (cast to bfloat16 at load time)
"""

import os

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
import transformers
from models.experimental.tt_symbiote.core.run_config import TracedRun

# Qwen-specific modules for hybrid attention
from models.experimental.tt_symbiote.modules.qwen35_decoder_layer import TTNNQwen35DecoderLayer
from models.experimental.tt_symbiote.modules.qwen35_gated_deltanet import TTNNQwen35GatedDeltaNet
from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


def reset_ttnn_states(all_modules):
    """Reset internal states of all TTNN modules between generate() calls.

    - TTNNQwen35FullAttention: reset device-side KV cache
    - TTNNQwen35GatedDeltaNet: reset conv and recurrence states
    """
    for name, module in all_modules.items():
        if isinstance(module, TTNNQwen35DecoderLayer):
            attn = module.attention
            if isinstance(attn, TTNNQwen35GatedDeltaNet):
                attn.conv_states = None
                attn.rec_states = None
                attn.rec_output = None
            # Full attention layers use external paged KV cache (re-created after reset)


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create paged KV cache for ONLY full attention layers.

    Qwen3.5-27B uses hybrid attention:
    - 48 linear attention layers (DeltaNet) do NOT use KV cache
    - 16 full attention layers (GQA) use paged KV cache

    Layer pattern repeats: [linear, linear, linear, full] x 16 = 64 layers
    So we have 16 full attention layers that need KV cache.
    """
    layer_types = getattr(
        model_config,
        "layer_types",
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    num_layers_in_pattern = len(layer_types)
    num_repeats = model_config.num_hidden_layers // num_layers_in_pattern

    # Compute which model layer indices are full attention layers
    full_attn_layer_indices = []
    for repeat_idx in range(num_repeats):
        for idx_in_pattern, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                full_attn_layer_indices.append(repeat_idx * num_layers_in_pattern + idx_in_pattern)

    num_full_attn_layers = len(full_attn_layer_indices)

    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNQwenPagedAttentionKVCache(
        num_layers=num_full_attn_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads),
        config=config,
        device=None,
        layer_indices=full_attn_layer_indices,
    ).to_device(device)


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
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128tok"])
def test_qwen3_5_27b(mesh_device, max_new_tokens):
    """Test Qwen3.5-27B-FP8 model with TTNN acceleration.

    This model has a hybrid attention architecture:
    - 48 linear attention layers (DeltaNet-style, no KV cache)
    - 16 full attention layers (GQA with 4 KV heads, uses paged KV cache)
    - Dense SwiGLU MLP (no MoE)
    - FP8 quantized weights (cast to bfloat16 at load time)
    """

    model_name = "Qwen/Qwen3.5-27B-FP8"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # FP8 models: load with proper dequantization via shared helper
    from models.experimental.tt_symbiote.tests.test_qwen3_5_27b_modules import _dequantize_fp8_weights

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    _dequantize_fp8_weights(model, model_name)
    model = model.to(torch.bfloat16)

    # Replace entire decoder layer — handles attention (GDN or GQA), MLP, and RMSNorm
    # internally, keeping residual adds on-device to eliminate host round-trips.
    nn_to_ttnn = {
        Qwen3_5DecoderLayer: TTNNQwen35DecoderLayer,
    }

    print(f"Module mappings: {nn_to_ttnn}")

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

    all_modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, mesh_device)

    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items(), desc="Preprocessing weights"):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Convert ModuleList to plain list to avoid TypeError when HF slices
    # self.layers[:num_hidden_layers] (TTNNModule is not an nn.Module subclass).
    # Must bypass nn.Module.__setattr__ which rejects non-Module assignments.
    layers_list = list(model.model.layers)
    model.model._modules.pop("layers", None)
    object.__setattr__(model.model, "layers", layers_list)

    print("Running inference...")
    model.eval()
    torch.set_grad_enabled(False)

    # Create paged KV cache for full attention layers (on-device, no CPU round-trips)
    # GDN layers manage their own conv/recurrence state internally.
    paged_kv_cache = create_paged_kv_cache(model.config, mesh_device)

    # Warmup run
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_kv_cache)
    reset_ttnn_states(all_modules)
    paged_kv_cache = create_paged_kv_cache(model.config, mesh_device)
    # Second warmup
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True, past_key_values=paged_kv_cache)
    reset_ttnn_states(all_modules)
    paged_kv_cache = create_paged_kv_cache(model.config, mesh_device)

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, past_key_values=paged_kv_cache)
    ttnn.synchronize_device(mesh_device)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Qwen3.5-27B-FP8 OUTPUT: {generated_text}")

    DispatchManager.save_stats_to_file("qwen3_5_27b_fp8_timing_stats.csv")
    TracedRun.release_all()
