# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mandatory ttnn-advise capture target for the rewritten dense 70B block."""

from __future__ import annotations

import os
import sys

import torch

import ttnn

MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/home/mvasiljevic/tt-metal")
CONFIG_DIR = os.environ.get(
    "SHARD_ADVISE_CONFIG_DIR",
    "/home/mvasiljevic/tt-metal/models/tt_transformers/model_params/Llama-3.1-70B-Instruct",
)
LAYER_IDX = int(os.environ.get("SHARD_ADVISE_LAYER", "39"))
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_CACHE_LEN = int(os.environ.get("SHARD_ADVISE_CACHE_LEN", "128"))
CURRENT_POS = int(os.environ.get("SHARD_ADVISE_CURRENT_POS", "18"))


def _synthetic_state_dict(config, layer_idx: int):
    prefix = f"model.layers.{layer_idx}."
    hidden = config.hidden_size
    head_dim = int(getattr(config, "head_dim", hidden // config.num_attention_heads))
    kv_width = config.num_key_value_heads * head_dim
    intermediate = config.intermediate_size

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": zeros(hidden, hidden),
        prefix + "self_attn.k_proj.weight": zeros(kv_width, hidden),
        prefix + "self_attn.v_proj.weight": zeros(kv_width, hidden),
        prefix + "self_attn.o_proj.weight": zeros(hidden, hidden),
        prefix + "mlp.gate_proj.weight": zeros(intermediate, hidden),
        prefix + "mlp.up_proj.weight": zeros(intermediate, hidden),
        prefix + "mlp.down_proj.weight": zeros(hidden, intermediate),
    }


def _build(device):
    # Append so the advisor environment's installed ttnn remains authoritative.
    if MODEL_DIR not in sys.path:
        sys.path.append(MODEL_DIR)

    from transformers import AutoConfig

    from models.autoports.meta_llama_llama_3_1_70b_instruct.tt.optimized_decoder import (
        OptimizationConfig,
        OptimizedDecoder,
    )

    config = AutoConfig.from_pretrained(CONFIG_DIR, local_files_only=True)
    state = _synthetic_state_dict(config, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        # The advisor interception layer does not model paged_fused_update_cache.
        # Capturing the equivalent two paged_update_cache ops preserves the dense
        # attention+MLP topology that the mandatory sharding pass must analyze.
        # Freeze the exact capture policy rather than inheriting the live stage
        # default.  The saved IR is the advisor geometry seed: BFP8 attention
        # and down projections, BFP4 gate/up, and the original down in0 block.
        # Precision and the down block were swept on hardware after capture.
        optimization_config=OptimizationConfig(
            attention_weight_dtype=ttnn.bfloat8_b,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            gate_up_weight_dtype=ttnn.bfloat4_b,
            gate_up_math_fidelity=ttnn.MathFidelity.LoFi,
            down_weight_dtype=ttnn.bfloat8_b,
            down_math_fidelity=ttnn.MathFidelity.HiFi2,
            decode_matmul_strategy="advisor_1d",
            advisor_down_in0_block_w=2,
            fused_cache_update=False,
        ),
    )

    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    hidden = ttnn.from_torch(
        torch.zeros((1, BATCH, 1, config.hidden_size), dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_shape = (BATCH, config.num_key_value_heads, MAX_CACHE_LEN, head_dim)
    key_cache = ttnn.from_torch(
        torch.zeros(cache_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value_cache = ttnn.from_torch(
        torch.zeros(cache_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return decoder, hidden, key_cache, value_cache


_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        _KEY_CACHE,
        _VALUE_CACHE,
        current_pos=CURRENT_POS,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE
    _DECODER, hidden, _KEY_CACHE, _VALUE_CACHE = _build(device)
    return (hidden,)
