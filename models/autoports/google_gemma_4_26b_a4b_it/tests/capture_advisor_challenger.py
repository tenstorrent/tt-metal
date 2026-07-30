# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 shard-advisor capture target for the shipped Gemma-4 decoder.

This target intentionally spells out the selected policy instead of relying on
``OptimizedDecoder.from_state_dict`` defaults.  The values are frozen in
``doc/advisor_challenger/incumbent.json`` from the executed final profiles.
"""

from __future__ import annotations

import os
import sys

import torch
import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

LAYER_KIND = os.environ.get("CHALLENGER_LAYER_KIND", "sliding_attention")
LAYER_IDX = {"sliding_attention": 0, "full_attention": 5}[LAYER_KIND]
BATCH = 1
CURRENT_POS = 32

_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder import (
        HIDDEN_SIZE,
        Gemma4TextRotaryEmbedding,
        _as_tt,
        _cache_shape,
        _load_layer_state,
        _load_text_config,
    )
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    cfg = _load_text_config()
    assert cfg.layer_types[LAYER_IDX] == LAYER_KIND
    state = _load_layer_state(LAYER_IDX)
    attention_dtype = ttnn.bfloat16 if LAYER_KIND == "sliding_attention" else ttnn.bfloat8_b
    dram_roles = ("packed_mlp_gate_up", "mlp_down")
    if LAYER_KIND == "full_attention":
        dram_roles = ("o_proj", *dram_roles)

    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        weight_dtype=ttnn.bfloat16,
        attention_weight_dtype=attention_dtype,
        mlp_weight_dtype=ttnn.bfloat8_b,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
        prefill_expert_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        full_attention_math_fidelity=ttnn.MathFidelity.LoFi,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_gate_in0_block_w=11,
        expert_down_in0_block_w=11,
        expert_gate_per_core_n=2,
        expert_down_per_core_n=2,
        expert_decode_input_l1=True,
        packed_dense_gate_up=True,
        dram_in0_block_w=11,
        dram_sharded_roles=dram_roles,
        residual_shard_cores=0,
    )

    torch.manual_seed(20260730 + LAYER_IDX)
    hidden = torch.randn(BATCH, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.full((BATCH, 1), CURRENT_POS, dtype=torch.long)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=LAYER_KIND)
    if LAYER_KIND == "sliding_attention":
        tt_cos, tt_sin = cos.unsqueeze(0), sin.unsqueeze(0)
        shared_physical = True
    else:
        tt_cos, tt_sin = cos.transpose(0, 1).unsqueeze(0), sin.transpose(0, 1).unsqueeze(0)
        shared_physical = False
    blocks_per_user, num_heads, block_size, head_dim = _cache_shape(
        LAYER_KIND,
        shared_physical=shared_physical,
        token_capacity=CURRENT_POS + 1,
    )
    page_table = _as_tt(
        device,
        torch.arange(blocks_per_user, dtype=torch.int32).view(BATCH, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (BATCH * blocks_per_user, num_heads, block_size, head_dim)
    kv_cache = (
        _as_tt(device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    kwargs = {
        "position_cos": _as_tt(device, tt_cos),
        "position_sin": _as_tt(device, tt_sin),
        "current_pos": _as_tt(
            device,
            torch.full((BATCH,), CURRENT_POS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    tt_hidden = _as_tt(device, hidden.transpose(0, 1).unsqueeze(0))
    return decoder, kwargs, tt_hidden


def decode(hidden):
    # Mirror the shipped decode graph through every capturable operation.
    # Routed expert gate/down are ttnn.sparse_matmul, a terminal operation in
    # the advisor tracer, so the query ends at that documented boundary.
    residual = hidden
    attn_in = _DECODER._rms_norm(hidden, _DECODER.weights.input_ln)
    attn_out = _DECODER._attention_decode(attn_in, cache_position_modulo=None, **_KWARGS)
    attn_out = _DECODER._rms_norm(attn_out, _DECODER.weights.post_attn_ln)
    hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    residual = hidden
    mlp_in = _DECODER._rms_norm(hidden, _DECODER.weights.pre_ff_ln)
    mlp_out = _DECODER._dense_mlp(mlp_in)
    hidden_1 = _DECODER._rms_norm(mlp_out, _DECODER.weights.post_ff_ln_1)
    router_weights = _DECODER._router_weights(residual)
    # Keep the router in the captured graph while returning one tensor.
    return ttnn.add(hidden_1, ttnn.sum(router_weights), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
