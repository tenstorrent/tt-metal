# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shard-advisor capture target for the shipped Phi-3.5 decoder."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Keep the advisor environment's ttnn ahead of the tt-metal source tree.
    sys.path.append(TT_METAL_ROOT)

BATCH = 32
MAX_CONTEXT = 128
_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_PAGE_TABLE = None
_POSITIONS = None


def _build(device):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
        OptimizationPolicy,
        OptimizedDecoder,
    )

    class CaptureDecoder(OptimizedDecoder):
        def _decode_rope(self, query, key, current_positions, *, use_long_rope):
            cos_table = self.long_cos if use_long_rope else self.short_cos
            sin_table = self.long_sin if use_long_rope else self.short_sin
            rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
            cos = ttnn.reshape(
                ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
                [1, 1, self.batch, self.head_dim],
            )
            sin = ttnn.reshape(
                ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
                [1, 1, self.batch, self.head_dim],
            )
            query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
            key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
            required = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            return ttnn.to_memory_config(query, required), ttnn.to_memory_config(key, required)

    config = SimpleNamespace(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_scaling={"short_factor": [1.0] * 48, "long_factor": [1.0] * 48},
    )
    prefix = "model.layers.0."
    state = {
        prefix + "input_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
        prefix + "self_attn.qkv_proj.weight": torch.zeros(9216, 3072, dtype=torch.bfloat16),
        prefix + "self_attn.o_proj.weight": torch.zeros(3072, 3072, dtype=torch.bfloat16),
        prefix + "mlp.gate_up_proj.weight": torch.zeros(16384, 3072, dtype=torch.bfloat16),
        prefix + "mlp.down_proj.weight": torch.zeros(3072, 8192, dtype=torch.bfloat16),
    }
    policy = OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
    )
    decoder = CaptureDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        optimization_policy=policy,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
    ).to(torch.bfloat16)
    page_table = ttnn.from_torch(
        torch.arange(BATCH * 4, dtype=torch.int32).reshape(BATCH, 4).flip(-1),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    positions = ttnn.from_torch(
        torch.zeros(BATCH, dtype=torch.int32),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_hidden = ttnn.from_torch(
        hidden.transpose(0, 1).unsqueeze(0),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return (
        decoder,
        key_cache,
        value_cache,
        page_table,
        positions,
        tt_hidden,
    )


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        page_table=_PAGE_TABLE,
        current_positions=_POSITIONS,
        use_long_rope=False,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _POSITIONS
    _DECODER, _KEY_CACHE, _VALUE_CACHE, _PAGE_TABLE, _POSITIONS, hidden = _build(device)
    return (hidden,)
