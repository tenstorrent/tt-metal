# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shard-advisor capture target for the shipped Qwen3.6 decoder."""

from __future__ import annotations

import os
import sys
import math
import types
from types import SimpleNamespace

import torch

import ttnn
from transformers import AutoConfig

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

BATCH = 32
MAX_CONTEXT = 64
PAGE_SIZE = 64
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]

_DECODER = None
_PAGE_TABLE = None
_POSITIONS = None


def _config():
    return SimpleNamespace(
        hidden_size=5120,
        intermediate_size=17408,
        head_dim=256,
        num_attention_heads=24,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"),
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        max_position_embeddings=262144,
        num_hidden_layers=64,
    )


def _state(config, layer_idx):
    prefix = f"model.language_model.layers.{layer_idx}."
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    common = {
        prefix + "input_layernorm.weight": torch.zeros(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.zeros(hidden, dtype=torch.bfloat16),
        prefix + "mlp.gate_proj.weight": torch.zeros(intermediate, hidden, dtype=torch.bfloat16),
        prefix + "mlp.up_proj.weight": torch.zeros(intermediate, hidden, dtype=torch.bfloat16),
        prefix + "mlp.down_proj.weight": torch.zeros(hidden, intermediate, dtype=torch.bfloat16),
    }
    if layer_idx == 3:
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        common.update(
            {
                prefix + "self_attn.q_proj.weight": torch.zeros(2 * q_width, hidden, dtype=torch.bfloat16),
                prefix + "self_attn.k_proj.weight": torch.zeros(kv_width, hidden, dtype=torch.bfloat16),
                prefix + "self_attn.v_proj.weight": torch.zeros(kv_width, hidden, dtype=torch.bfloat16),
                prefix + "self_attn.o_proj.weight": torch.zeros(hidden, q_width, dtype=torch.bfloat16),
                prefix + "self_attn.q_norm.weight": torch.zeros(config.head_dim, dtype=torch.bfloat16),
                prefix + "self_attn.k_norm.weight": torch.zeros(config.head_dim, dtype=torch.bfloat16),
            }
        )
    else:
        key_width = config.linear_num_key_heads * config.linear_key_head_dim
        value_width = config.linear_num_value_heads * config.linear_value_head_dim
        conv_width = 2 * key_width + value_width
        common.update(
            {
                prefix + "linear_attn.in_proj_qkv.weight": torch.zeros(conv_width, hidden, dtype=torch.bfloat16),
                prefix + "linear_attn.in_proj_z.weight": torch.zeros(value_width, hidden, dtype=torch.bfloat16),
                prefix + "linear_attn.in_proj_b.weight": torch.zeros(
                    config.linear_num_value_heads, hidden, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.in_proj_a.weight": torch.zeros(
                    config.linear_num_value_heads, hidden, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.conv1d.weight": torch.zeros(
                    conv_width, 1, config.linear_conv_kernel_dim, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.dt_bias": torch.zeros(
                    config.linear_num_value_heads, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.A_log": torch.full(
                    (config.linear_num_value_heads,), math.log(0.5), dtype=torch.float32
                ),
                prefix + "linear_attn.norm.weight": torch.zeros(
                    config.linear_value_head_dim, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.out_proj.weight": torch.zeros(hidden, value_width, dtype=torch.bfloat16),
            }
        )
    return common


def decode(hidden):
    # This spells the capturable projection envelope without runtime
    # tensor-property queries, which TracedTensor intentionally omits. The
    # full-attention SDPA/cache core and linear gated-delta core do not change
    # the projection shapes; the latter is terminal in the tracer.
    residual_memory_config = _DECODER._decode_residual_memory_config()
    residual = ttnn.to_memory_config(hidden, residual_memory_config)
    hidden = _DECODER._rms_norm_decode_sharded(residual, "input_norm")
    if LAYER_KIND == "full_attention":
        hidden = _full_attention_decode_tracer_safe(hidden)
    else:
        projected = _DECODER._optimized_decode_linear(
            hidden, _DECODER.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        value_width = (
            _DECODER.hf_config.linear_num_value_heads * _DECODER.hf_config.linear_value_head_dim
        )
        # The gated-delta core is terminal in the tracer. Its validated output
        # contract is value_width, so preserve the shipped out-projection
        # shape while recording that the intervening core is uncapturable.
        hidden = ttnn.slice(
            projected,
            (0, 0, 0, 2 * 2048),
            (1, 1, BATCH, 2 * 2048 + value_width),
        )
        hidden = _DECODER._optimized_decode_linear(
            hidden, _DECODER.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    hidden = ttnn.add(residual, hidden, memory_config=residual_memory_config)
    residual = hidden
    hidden = _DECODER._rms_norm_decode_sharded(hidden, "post_attention_norm")
    hidden = _DECODER._mlp(hidden)
    return ttnn.add(residual, hidden, memory_config=residual_memory_config)


def _per_head_norm_tracer_safe(tensor, weight_name, heads):
    tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    flat = ttnn.reshape(tensor, (1, 1, BATCH * heads, _DECODER.head_dim))
    flat = ttnn.rms_norm(
        flat,
        epsilon=_DECODER.eps,
        weight=_DECODER.weights[weight_name],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.reshape(flat, (1, BATCH, heads, _DECODER.head_dim))


def _partial_rope_decode_tracer_safe(tensor, current_positions, heads):
    rotary_dim = int(_DECODER.head_dim * float(_DECODER.hf_config.partial_rotary_factor))
    rotary = ttnn.slice(tensor, (0, 0, 0, 0), (1, BATCH, heads, rotary_dim))
    passthrough = ttnn.slice(
        tensor,
        (0, 0, 0, rotary_dim),
        (1, BATCH, heads, _DECODER.head_dim),
    )
    cos = ttnn.embedding(current_positions, _DECODER.rope["cos"], layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(current_positions, _DECODER.rope["sin"], layout=ttnn.TILE_LAYOUT)
    cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
    sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
    cos = ttnn.slice(cos, (0, 0, 0, 0), (1, BATCH, 1, rotary_dim))
    sin = ttnn.slice(sin, (0, 0, 0, 0), (1, BATCH, 1, rotary_dim))
    cos = ttnn.repeat(cos, ttnn.Shape([1, 1, heads, 1]))
    sin = ttnn.repeat(sin, ttnn.Shape([1, 1, heads, 1]))
    half = rotary_dim // 2
    first = ttnn.slice(rotary, (0, 0, 0, 0), (1, BATCH, heads, half))
    second = ttnn.slice(rotary, (0, 0, 0, half), (1, BATCH, heads, rotary_dim))
    rotated_half = ttnn.concat([ttnn.neg(second), first], dim=-1)
    rotary = ttnn.add(ttnn.multiply(rotary, cos), ttnn.multiply(rotated_half, sin))
    return ttnn.to_memory_config(
        ttnn.concat([rotary, passthrough], dim=-1),
        _DECODER.decode_attention_memory_config,
    )


def _full_attention_decode_tracer_safe(hidden):
    cache_positions = ttnn.typecast(_POSITIONS, ttnn.int32)
    projected = _DECODER._optimized_decode_linear(
        hidden, _DECODER.weights["packed_qkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q_width = _DECODER.num_heads * _DECODER.head_dim
    kv_width = _DECODER.num_kv_heads * _DECODER.head_dim
    q = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, BATCH, q_width))
    gate = ttnn.slice(projected, (0, 0, 0, q_width), (1, 1, BATCH, 2 * q_width))
    k = ttnn.slice(
        projected,
        (0, 0, 0, 2 * q_width),
        (1, 1, BATCH, 2 * q_width + kv_width),
    )
    v = ttnn.slice(
        projected,
        (0, 0, 0, 2 * q_width + kv_width),
        (1, 1, BATCH, 2 * q_width + 2 * kv_width),
    )
    fused_qkv = _DECODER._optimized_decode_concat(
        [q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        fused_qkv,
        num_heads=_DECODER.num_heads,
        num_kv_heads=_DECODER.num_kv_heads,
        memory_config=_DECODER.decode_attention_memory_config,
    )
    q = _per_head_norm_tracer_safe(q, "q_norm", _DECODER.num_heads)
    k = _per_head_norm_tracer_safe(k, "k_norm", _DECODER.num_kv_heads)
    q = _partial_rope_decode_tracer_safe(q, _POSITIONS, _DECODER.num_heads)
    k = _partial_rope_decode_tracer_safe(k, _POSITIONS, _DECODER.num_kv_heads)
    ttnn.experimental.paged_update_cache(
        _DECODER.caches["key"],
        k,
        update_idxs_tensor=cache_positions,
        page_table=_PAGE_TABLE,
    )
    ttnn.experimental.paged_update_cache(
        _DECODER.caches["value"],
        v,
        update_idxs_tensor=cache_positions,
        page_table=_PAGE_TABLE,
    )
    attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        _DECODER.caches["key"],
        _DECODER.caches["value"],
        cur_pos_tensor=cache_positions,
        page_table_tensor=_PAGE_TABLE,
        scale=_DECODER.head_dim**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attention = ttnn.to_memory_config(attention, _DECODER.decode_attention_memory_config)
    attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=_DECODER.num_heads)
    attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
    attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
    attention = _DECODER._optimized_decode_linear(
        attention, _DECODER.weights["o_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.reshape(attention, (1, 1, BATCH, _DECODER.hidden_size))


def _rms_norm_decode_sharded_no_query(self, hidden, name):
    memory_config = self._decode_residual_memory_config()
    return ttnn.rms_norm(
        hidden,
        epsilon=self.eps,
        weight=self.weights[name],
        memory_config=memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 1],
            subblock_w=4,
            block_h=1,
            block_w=self.hidden_size // 8 // 32,
            inplace=False,
        ),
    )


def _optimized_decode_linear_no_query(self, activation, weight, **kwargs):
    weight_name = next(name for name, value in self.weights.items() if value is weight)
    activation = ttnn.to_memory_config(activation, self.decode_input_memory_configs[weight_name])
    kwargs["memory_config"] = self.decode_output_memory_configs[weight_name]
    kwargs["program_config"] = self.decode_program_configs[weight_name]
    kwargs["compute_kernel_config"] = self.compute_kernel_config
    kwargs["dtype"] = ttnn.bfloat16
    output = ttnn.linear(activation, weight, **kwargs)
    if weight_name == "packed_linear_inputs":
        output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
    return output


def _optimized_decode_concat_no_query(self, tensors, *args, **kwargs):
    tensors = [ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG) for tensor in tensors]
    kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.concat(tensors, *args, **kwargs)


def make_inputs(device):
    global _DECODER, _PAGE_TABLE, _POSITIONS
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True).text_config
    if LAYER_KIND == "full_attention":
        layer_idx = 3
    elif LAYER_KIND == "linear_attention":
        layer_idx = 0
    else:
        raise ValueError(f"unknown CHALLENGER_LAYER_KIND={LAYER_KIND!r}")
    state = _state(config, layer_idx)

    # This is the policy frozen in incumbent.json and evidenced by the
    # incumbent tt-perf-report CSVs. Never omit this argument.
    _DECODER = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        page_size=PAGE_SIZE,
        optimization_policy="bfp4_all_dram_w8",
    )
    _DECODER._rms_norm_decode_sharded = types.MethodType(_rms_norm_decode_sharded_no_query, _DECODER)
    _DECODER._optimized_decode_linear = types.MethodType(_optimized_decode_linear_no_query, _DECODER)
    _DECODER._optimized_decode_concat = types.MethodType(_optimized_decode_concat_no_query, _DECODER)
    hidden = _to_device(
        torch.zeros((1, 1, BATCH, config.hidden_size), dtype=torch.bfloat16),
        mesh_device=device,
    )
    _PAGE_TABLE = _to_device(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    _POSITIONS = _to_device(
        torch.zeros(BATCH, dtype=torch.uint32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    return (hidden,)
