# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for North-Mini's dense optimized block."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("NORTH_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # The advisor environment's real ttnn package must retain precedence.
    sys.path.append(TT_METAL_ROOT)

from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder

BATCH = int(os.environ.get("NORTH_ADVISE_BATCH", "1"))
MAX_CACHE_LEN = 32
_DECODER = None
_KWARGS = None


def _config():
    return SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        rms_norm_eps=1e-5,
        layer_types=["full_attention"],
        mlp_layer_types=["dense"],
        sliding_window=4096,
        prefix_dense_sliding_window_pattern=1,
        prefix_dense_intermediate_size=3072,
        intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        num_hidden_layers=1,
        max_position_embeddings=500_000,
        rope_parameters={"rope_theta": 1_000_000.0},
    )


def _synthetic_state(config):
    generator = torch.Generator().manual_seed(23001)
    prefix = "model.layers.0."

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.k_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.v_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.o_proj.weight": randn(config.hidden_size, config.num_attention_heads * config.head_dim),
        prefix + "mlp.gate_proj.weight": randn(config.prefix_dense_intermediate_size, config.hidden_size),
        prefix + "mlp.up_proj.weight": randn(config.prefix_dense_intermediate_size, config.hidden_size),
        prefix + "mlp.down_proj.weight": randn(config.hidden_size, config.prefix_dense_intermediate_size),
    }


def _to_tt(tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(device):
    config = _config()
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
    )
    hidden = _to_tt(
        torch.zeros(1, BATCH, 1, config.hidden_size, dtype=torch.bfloat16),
        device,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current = _to_tt(
        torch.zeros(BATCH, dtype=torch.int32),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows([0] * BATCH, hf_config=config, decode=True)
    cos = _to_tt(cos, device)
    sin = _to_tt(sin, device)
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current,
        "position_cos": cos,
        "position_sin": sin,
    }
    return decoder, kwargs, hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
