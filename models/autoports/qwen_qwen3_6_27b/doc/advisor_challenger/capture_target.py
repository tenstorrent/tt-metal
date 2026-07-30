# SPDX-License-Identifier: Apache-2.0
"""Batch-32 shipped-policy capture target for the advisor-challenger stage."""

from __future__ import annotations

import os
import sys
import math
from types import MethodType, SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import _to_device
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
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
        num_hidden_layers=64,
        max_position_embeddings=262144,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def _diagonal(rows, columns, scale):
    value = torch.zeros(rows, columns, dtype=torch.bfloat16)
    count = min(rows, columns)
    value[torch.arange(count), torch.arange(count)] = scale
    return value


def _state(config, layer_idx):
    prefix = f"model.language_model.layers.{layer_idx}."
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    common = {
        prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "mlp.gate_proj.weight": _diagonal(intermediate, hidden, 0.1),
        prefix + "mlp.up_proj.weight": _diagonal(intermediate, hidden, 0.08),
        prefix + "mlp.down_proj.weight": _diagonal(hidden, intermediate, 0.12),
    }
    if config.layer_types[layer_idx] == "full_attention":
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        common.update(
            {
                prefix + "self_attn.q_proj.weight": _diagonal(2 * q_width, hidden, 0.25),
                prefix + "self_attn.k_proj.weight": _diagonal(kv_width, hidden, 0.2),
                prefix + "self_attn.v_proj.weight": _diagonal(kv_width, hidden, 0.15),
                prefix + "self_attn.o_proj.weight": _diagonal(hidden, q_width, 0.2),
                prefix + "self_attn.q_norm.weight": torch.ones(config.head_dim, dtype=torch.bfloat16),
                prefix + "self_attn.k_norm.weight": torch.ones(config.head_dim, dtype=torch.bfloat16),
            }
        )
    else:
        key_width = config.linear_num_key_heads * config.linear_key_head_dim
        value_width = config.linear_num_value_heads * config.linear_value_head_dim
        conv_width = 2 * key_width + value_width
        conv = torch.zeros(conv_width, 1, config.linear_conv_kernel_dim, dtype=torch.bfloat16)
        conv[:, 0, -1] = 0.5
        common.update(
            {
                prefix + "linear_attn.in_proj_qkv.weight": _diagonal(conv_width, hidden, 0.2),
                prefix + "linear_attn.in_proj_z.weight": _diagonal(value_width, hidden, 0.15),
                prefix + "linear_attn.in_proj_b.weight": _diagonal(config.linear_num_value_heads, hidden, 0.1),
                prefix + "linear_attn.in_proj_a.weight": _diagonal(config.linear_num_value_heads, hidden, 0.08),
                prefix + "linear_attn.conv1d.weight": conv,
                prefix + "linear_attn.dt_bias": torch.full(
                    (config.linear_num_value_heads,), 0.1, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.A_log": torch.full(
                    (config.linear_num_value_heads,), math.log(0.5), dtype=torch.float32
                ),
                prefix + "linear_attn.norm.weight": torch.ones(
                    config.linear_value_head_dim, dtype=torch.bfloat16
                ),
                prefix + "linear_attn.out_proj.weight": _diagonal(hidden, value_width, 0.2),
            }
        )
    return common


def decode(hidden):
    # Spell the shipped decode shell explicitly because TracedTensor cannot
    # answer runtime ``is_sharded()`` queries during layout assignment.
    memory_config = _DECODER._decode_residual_memory_config()
    residual = ttnn.to_memory_config(hidden, memory_config)
    hidden = _DECODER._rms_norm_decode_sharded(residual, "input_norm")
    # Capture the projection/layout skeleton around the attention core. The
    # linear-attention recurrence is terminal in the current tracer
    # (softplus/copy); its uncapturable share is recorded in report metadata.
    if LAYER_KIND == "full_attention":
        projected = _DECODER._linear(hidden, "packed_qkv")
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
        fused_qkv = ttnn.concat([q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, _ = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=_DECODER.num_heads,
            num_kv_heads=_DECODER.num_kv_heads,
            memory_config=_DECODER.decode_attention_memory_config,
        )
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.reshape(q, (1, 1, BATCH * _DECODER.num_heads, _DECODER.head_dim))
        q = ttnn.rms_norm(
            q,
            epsilon=_DECODER.eps,
            weight=_DECODER.weights["q_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, (1, 1, BATCH * _DECODER.num_kv_heads, _DECODER.head_dim))
        k = ttnn.rms_norm(
            k,
            epsilon=_DECODER.eps,
            weight=_DECODER.weights["k_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Q/K then enter RoPE and the paged-attention core. For layout recall,
        # preserve both norm chains and use normalized Q as the projection
        # continuation; compute fidelity from this traced state is ignored.
        hidden = ttnn.reshape(q, (1, 1, BATCH, q_width))
        hidden = ttnn.multiply(hidden, ttnn.sigmoid(gate))
        hidden = _DECODER._linear(hidden, "o_proj")
    else:
        projected = _DECODER._linear(hidden, "packed_linear_inputs")
        value_width = (
            int(_DECODER.hf_config.linear_num_value_heads)
            * int(_DECODER.hf_config.linear_value_head_dim)
        )
        hidden = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, BATCH, value_width))
        hidden = _DECODER._linear(hidden, "out_proj")
    hidden = ttnn.to_memory_config(hidden, memory_config)
    hidden = ttnn.add(residual, hidden, memory_config=memory_config)
    residual = hidden
    hidden = _DECODER._rms_norm_decode_sharded(hidden, "post_attention_norm")
    hidden = _DECODER._mlp(hidden)
    return ttnn.add(residual, hidden, memory_config=memory_config)


def _capture_rms_norm(self, hidden, name):
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


def _capture_decode_linear(self, activation, weight, **kwargs):
    weight_name = next(name for name, value in self.weights.items() if value is weight)
    program_config = self.decode_program_configs.get(weight_name)
    if program_config is not None:
        activation = ttnn.to_memory_config(activation, self.decode_input_memory_configs[weight_name])
        kwargs["memory_config"] = self.decode_output_memory_configs[weight_name]
        kwargs["program_config"] = program_config
    kwargs["compute_kernel_config"] = self.compute_kernel_config
    kwargs["dtype"] = ttnn.bfloat16
    output = ttnn.linear(activation, weight, **kwargs)
    if weight_name == "packed_linear_inputs":
        output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
    return output


def _capture_decode_concat(self, tensors, *args, **kwargs):
    tensors = [ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG) for tensor in tensors]
    kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.concat(tensors, *args, **kwargs)


def make_inputs(device):
    global _DECODER, _PAGE_TABLE, _POSITIONS
    config = _config()
    if LAYER_KIND == "full_attention":
        layer_idx = 3
    elif LAYER_KIND == "linear_attention":
        layer_idx = 0
    else:
        raise ValueError(f"unknown CHALLENGER_LAYER_KIND={LAYER_KIND!r}")
    state = _state(config, layer_idx)

    # This explicit argument is the policy frozen in incumbent.json. It is not
    # inferred from constructor defaults.
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
    _DECODER._rms_norm_decode_sharded = MethodType(_capture_rms_norm, _DECODER)
    _DECODER._optimized_decode_linear = MethodType(_capture_decode_linear, _DECODER)
    _DECODER._optimized_decode_concat = MethodType(_capture_decode_concat, _DECODER)
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
