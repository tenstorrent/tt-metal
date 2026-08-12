# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for the optimized Falcon3 dense decoder block."""

from __future__ import annotations

import os
import sys

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("FALCON3_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
HF_MODEL = "tiiuae/Falcon3-7B-Base"
LAYER_IDX = 14
BATCH = 32
MAX_CACHE_LEN = 128
POSITION = 17


def _synthetic_state_dict(config, layer_idx: int):
    generator = torch.Generator().manual_seed(20260716)

    def weight(shape):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=0.02, generator=generator)

    prefix = f"model.layers.{layer_idx}."
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    intermediate = config.intermediate_size
    return {
        prefix + "self_attn.q_proj.weight": weight((q_width, hidden)),
        prefix + "self_attn.k_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.v_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.o_proj.weight": weight((hidden, hidden)),
        prefix + "mlp.gate_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.up_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.down_proj.weight": weight((hidden, intermediate)),
        prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _build(device):
    if TT_METAL_ROOT not in sys.path:
        sys.path.append(TT_METAL_ROOT)

    from transformers import AutoConfig

    from models.autoports.tiiuae_falcon3_7b_base.tt.optimized_decoder import OptimizedDecoder

    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    state_dict = _synthetic_state_dict(config, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        decode_matmul_mode="shard_advisor",
        max_cache_len=MAX_CACHE_LEN,
        precision_policy="bfp8_hifi2",
        use_packed_mlp=False,
        use_explicit_decode_mask=False,
    )
    key_cache, value_cache = decoder.allocate_kv_cache()
    hidden = torch.randn(BATCH, 1, config.hidden_size, dtype=torch.bfloat16)
    tt_hidden = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.full((BATCH,), POSITION, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return decoder, key_cache, value_cache, cache_position, tt_hidden


_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_CACHE_POSITION = None


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        cache_position=_CACHE_POSITION,
        position_index=POSITION,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _CACHE_POSITION
    _DECODER, _KEY_CACHE, _VALUE_CACHE, _CACHE_POSITION, hidden = _build(device)
    return (hidden,)
