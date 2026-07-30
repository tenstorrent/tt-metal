# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shipped-policy advisor capture target for Phi-3.5 Mini."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from types import MethodType

import torch
import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

MODEL_ROOT = os.path.join(TT_METAL_ROOT, "models/autoports/microsoft_phi_3_5_mini_instruct")
INCUMBENT = os.path.join(MODEL_ROOT, "doc/advisor_challenger/incumbent.json")
HF_CONFIG = (
    "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/"
    "snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77/config.json"
)
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

    with open(INCUMBENT) as fh:
        incumbent = json.load(fh)
    shipped = incumbent["shipped_policy"]
    policy = replace(
        OptimizationPolicy(),
        attention_weight_dtype=ttnn.bfloat4_b,
        gate_up_weight_dtype=ttnn.bfloat4_b,
        down_weight_dtype=ttnn.bfloat4_b,
        kv_cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        decode_core_grid=tuple(shipped["decode_core_grid"]),
        qkv_in0_block_w=shipped["qkv_in0_block_w"],
        o_proj_in0_block_w=shipped["o_proj_in0_block_w"],
        gate_up_in0_block_w=shipped["gate_up_in0_block_w"],
        down_in0_block_w=shipped["down_in0_block_w"],
        gate_up_split_interleaved=shipped["gate_up_split_interleaved"],
        separate_gate_up_projections=shipped["separate_gate_up_projections"],
        fused_paged_cache_update=shipped["fused_paged_cache_update"],
        explicit_decode_sdpa=shipped["explicit_decode_sdpa"],
        fused_rope=shipped["fused_rope"],
    )
    with open(HF_CONFIG) as fh:
        config = SimpleNamespace(**json.load(fh))
    generator = torch.Generator().manual_seed(20260728)

    def sample(shape, mean, std):
        return (torch.randn(*shape, generator=generator) * std + mean).to(torch.bfloat16)

    prefix = "model.layers.0."
    state = {
        prefix + "input_layernorm.weight": sample((config.hidden_size,), 0.00829245, 0.02295496),
        prefix + "post_attention_layernorm.weight": sample((config.hidden_size,), 0.03923744, 0.00945584),
        prefix + "self_attn.qkv_proj.weight": sample(
            (3 * config.hidden_size, config.hidden_size), 0.00000262, 0.02379715
        ),
        prefix + "self_attn.o_proj.weight": sample(
            (config.hidden_size, config.hidden_size), -0.00000081, 0.01751270
        ),
        prefix + "mlp.gate_up_proj.weight": sample(
            (2 * config.intermediate_size, config.hidden_size), -0.00001401, 0.03248470
        ),
        prefix + "mlp.down_proj.weight": sample(
            (config.hidden_size, config.intermediate_size), 0.00000275, 0.03603584
        ),
    }
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        policy=policy,
    )
    # The pinned advisor tracer has a handler for paged_update_cache but not
    # for the newer paged_fused_update_cache op used by the shipped decoder.
    # Expose the two semantically equivalent cache updates so tracing can
    # continue to the O projection and MLP. No matmul/layout/precision field
    # changes, and production code is not modified.
    decoder.policy = replace(decoder.policy, fused_paged_cache_update=False)

    # The production helper preserves the incoming Q/K layouts by querying
    # tensor.memory_config(). During tracing that layout is intentionally
    # unknown until the advisor assigns it. The nlp_create_qkv_heads_decode
    # call immediately before RoPE declares the same 32-core height-sharded
    # layout for Q and K, so spell that existing contract statically here.
    def decode_rope_capture(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos_decode if use_long_rope else self.short_cos_decode
        sin_table = self.long_sin_decode if use_long_rope else self.short_sin_decode
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.reshape(
            ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        sin = ttnn.reshape(
            ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        cos = ttnn.transpose(cos, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.transpose(sin, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qk_memory_config = self._decode_concat_memory_config()
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, qk_memory_config),
            ttnn.to_memory_config(key, qk_memory_config),
        )

    decoder._decode_rope = MethodType(decode_rope_capture, decoder)
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
    ).to(torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(MAX_CONTEXT / 32)
    page_table = torch.arange(BATCH * blocks, dtype=torch.int32).reshape(BATCH, blocks).flip(-1)
    page_table = ttnn.from_torch(
        page_table,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    positions = ttnn.from_torch(
        torch.full((BATCH,), 127, dtype=torch.int32),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hidden = ttnn.from_torch(
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
        hidden,
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
