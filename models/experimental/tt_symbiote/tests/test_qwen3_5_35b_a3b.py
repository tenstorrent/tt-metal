# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3.5-35B-A3B model with TTNN backend.

This model uses a HYBRID attention architecture with:
- Linear attention layers (30/40): Qwen3_5MoeGatedDeltaNet (DeltaNet/Mamba-style)
- Full attention layers (10/40): Qwen3_5MoeAttention (GQA with 2 KV heads)
- MoE layers (all 40): 256 experts, top-8 routing
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

# FIXED IMPORTS: Use Qwen-specific modules from their dedicated files
from models.experimental.tt_symbiote.modules.qwen_moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
from models.experimental.tt_symbiote.modules.qwen_attention import (
    TTNNQwen3LinearAttention,
    TTNNQwen3FullAttention,
    TTNNQwenPagedAttentionKVCache,
)

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create paged KV cache for ONLY full attention layers.

    Qwen3.5-35B-A3B uses hybrid attention:
    - Linear attention layers (DeltaNet) do NOT use KV cache
    - Full attention layers (GQA) use paged KV cache

    Layer pattern repeats: [linear, linear, linear, full] * 10 = 40 layers
    So we have 10 full attention layers that need KV cache.
    """
    # Count full_attention layers from layer_types config
    layer_types = getattr(
        model_config, "layer_types", ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    )
    num_layers_in_pattern = len(layer_types)

    # Compute which model layer indices are full attention layers
    # For pattern [linear, linear, linear, full], full attention is at index 3, 7, 11, 15, ...
    full_attn_layer_indices = []
    for pattern_repeat in range(model_config.num_hidden_layers // num_layers_in_pattern):
        for idx_in_pattern, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                full_attn_layer_indices.append(pattern_repeat * num_layers_in_pattern + idx_in_pattern)

    num_full_attn_layers = len(full_attn_layer_indices)

    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    # FIXED: Use TTNNQwenPagedAttentionKVCache which supports layer_indices mapping
    return TTNNQwenPagedAttentionKVCache(
        num_layers=num_full_attn_layers,  # Only full attention layers need KV cache
        num_kv_heads=model_config.num_key_value_heads,  # 2 KV heads
        head_dim=model_config.head_dim,  # 256 for Qwen3.5
        config=config,
        device=None,
        layer_indices=full_attn_layer_indices,  # Map layer_idx -> cache_idx
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
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("use_paged_attention", [True], ids=["paged"])
@pytest.mark.parametrize("max_new_tokens", [128], ids=["8tok"])
def test_qwen3_5_35b_a3b(mesh_device, use_paged_attention, max_new_tokens):
    """Test Qwen3.5-35B-A3B model with TTNN acceleration.

    This model has a hybrid attention architecture:
    - 30 linear attention layers (DeltaNet/Mamba-style, no KV cache)
    - 10 full attention layers (GQA with 2 KV heads, uses paged KV cache)
    - 256 experts per MoE layer, top-8 routing
    """

    model_name = "Qwen/Qwen3.5-35B-A3B"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Identify layer classes dynamically
    # The model has BOTH linear and full attention layers
    # Layer types alternate: linear, linear, linear, full (repeat)
    linear_attn_class = None
    full_attn_class = None
    moe_class = None

    for layer in model.model.layers:
        layer_type = getattr(layer, "layer_type", None)

        # Try to identify attention class from layer_type attribute
        # Note: linear attention layers use 'linear_attn', full attention uses 'self_attn'
        if layer_type == "linear_attention" and linear_attn_class is None:
            linear_attn_class = layer.linear_attn.__class__
            print(f"Found linear attention class: {linear_attn_class.__name__}")
        elif layer_type == "full_attention" and full_attn_class is None:
            full_attn_class = layer.self_attn.__class__
            print(f"Found full attention class: {full_attn_class.__name__}")

        # Identify MoE class from mlp attribute
        if moe_class is None and hasattr(layer, "mlp"):
            moe_class = layer.mlp.__class__
            print(f"Found MoE class: {moe_class.__name__}")

        # Stop once we've found all classes
        if linear_attn_class and full_attn_class and moe_class:
            break

    # Fallback: if layer_type attribute doesn't exist, inspect class names
    # Linear attention layers use 'linear_attn', full attention uses 'self_attn'
    if linear_attn_class is None or full_attn_class is None:
        for layer in model.model.layers:
            # Check for linear attention (linear_attn attribute)
            if hasattr(layer, "linear_attn") and linear_attn_class is None:
                attn_class = layer.linear_attn.__class__
                class_name = attn_class.__name__
                if "DeltaNet" in class_name or "GatedDelta" in class_name:
                    linear_attn_class = attn_class
                    print(f"Found linear attention class (by name): {class_name}")

            # Check for full attention (self_attn attribute)
            if hasattr(layer, "self_attn") and full_attn_class is None:
                attn_class = layer.self_attn.__class__
                class_name = attn_class.__name__
                if "Attention" in class_name and attn_class != linear_attn_class:
                    full_attn_class = attn_class
                    print(f"Found full attention class (by name): {class_name}")

            if linear_attn_class and full_attn_class:
                break

    # Build module replacement dict
    # Check if we should skip TTNN for linear attention (useful for debugging)
    use_cpu_linear_attn = os.environ.get("TT_QWEN_CPU_LINEAR_ATTN", "0").lower() in ("1", "true", "yes")

    nn_to_ttnn = {}
    if linear_attn_class:
        if not use_cpu_linear_attn:
            nn_to_ttnn[linear_attn_class] = TTNNQwen3LinearAttention
        else:
            print("[DEBUG] TT_QWEN_CPU_LINEAR_ATTN=1: Using PyTorch for linear attention layers")
    if full_attn_class:
        nn_to_ttnn[full_attn_class] = TTNNQwen3FullAttention
    # FIXED: Added NNZ check in TTNNExperts.forward() to handle devices with no tokens routed
    # When sparsity_sum == 0, we return zeros instead of calling sparse_matmul
    if moe_class:
        nn_to_ttnn[moe_class] = TTNNQwen3MoE

    print(f"Module mappings: {nn_to_ttnn}")

    # TODO: Selective linear replacement - don't replace small gating layers
    # The generic nn.Linear replacement breaks small linear layers like shared_expert_gate
    # which have output_size=1 and shouldn't use tensor parallelism with reduce-scatter
    nn_to_ttnn2 = {
        # nn.Linear: TTNNLinearIColShardedWRowSharded,
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

    # In CPU mode or CPU linear attention fallback, don't use custom paged attention cache
    # Qwen's linear attention layers need a cache with has_previous_state, conv_states, recurrent_states
    # which TTNNQwenPagedAttentionKVCache doesn't fully provide for native DeltaNet execution
    run_mode = os.environ.get("TT_SYMBIOTE_RUN_MODE", "NORMAL")
    # Also check TTNN_LINEAR_ATTN_PROJECTIONS - when disabled, fallback to PyTorch needs native cache
    use_ttnn_projections = os.environ.get("TTNN_LINEAR_ATTN_PROJECTIONS", "1") == "1"
    use_custom_cache = use_paged_attention and run_mode != "CPU" and not use_cpu_linear_attn and use_ttnn_projections

    cache_kwargs = {}
    if use_custom_cache:
        cache_kwargs["past_key_values"] = create_paged_kv_cache(model.config, mesh_device)

    # Warmup run
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, **cache_kwargs)

    # Reset cache for actual run
    cache_kwargs = {}
    if use_custom_cache:
        cache_kwargs["past_key_values"] = create_paged_kv_cache(model.config, mesh_device)

    prompt_tokens = inputs["input_ids"].shape[-1]

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, **cache_kwargs)
    ttnn.synchronize_device(mesh_device)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Qwen3.5-35B-A3B OUTPUT: {generated_text}")

    DispatchManager.save_stats_to_file("qwen3_5_35b_a3b_timing_stats.csv")
    TracedRun.release_all()
