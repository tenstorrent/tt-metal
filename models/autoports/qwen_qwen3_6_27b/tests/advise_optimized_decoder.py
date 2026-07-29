# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for the rewritten dense full-attention block."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("SHARD_ADVISE_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # The advisor's installed ttnn must retain precedence.
    sys.path.append(TT_METAL_ROOT)

_DECODER = None
_PAGE_TABLE = None
_POSITIONS = None


def _build(device):
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder, _to_device

    config = SimpleNamespace(
        layer_types=["full_attention"],
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )
    policy = POLICIES["default"]
    batch = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))

    def tt_zeros(shape, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return _to_device(torch.zeros(shape, dtype=torch.bfloat16), mesh_device=device, dtype=dtype, layout=layout)

    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    weights = {
        "input_norm": tt_zeros((config.hidden_size,)),
        "post_attention_norm": tt_zeros((config.hidden_size,)),
        "packed_qkv": tt_zeros(
            (config.hidden_size, 2 * q_width + 2 * kv_width),
            dtype=policy.attention_weight_dtype,
        ),
        "o_proj": tt_zeros((q_width, config.hidden_size), dtype=policy.attention_weight_dtype),
        "q_norm": tt_zeros((config.head_dim,)),
        "k_norm": tt_zeros((config.head_dim,)),
        "mlp_gate_up": tt_zeros(
            (config.hidden_size, 2 * config.intermediate_size),
            dtype=policy.mlp_gate_up_dtype,
        ),
        "mlp_down": tt_zeros(
            (config.intermediate_size, config.hidden_size),
            dtype=policy.mlp_down_dtype,
        ),
    }
    caches = {
        "key": tt_zeros((1, config.num_key_value_heads, 64, config.head_dim), dtype=policy.cache_dtype),
        "value": tt_zeros((1, config.num_key_value_heads, 64, config.head_dim), dtype=policy.cache_dtype),
        "batch_indices": _to_device(
            torch.zeros(1, dtype=torch.int32),
            mesh_device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    }
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    rope = {
        "cos": tt_zeros((64, rotary_dim), layout=ttnn.ROW_MAJOR_LAYOUT),
        "sin": tt_zeros((64, rotary_dim), layout=ttnn.ROW_MAJOR_LAYOUT),
    }
    attention_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, config.head_dim),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decoder = OptimizedDecoder(
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=batch,
        max_context=64,
        page_size=64,
        weights=weights,
        caches=caches,
        rope=rope,
        policy=policy,
        candidate="default",
        decode_attention_memory_config=attention_memcfg,
    )
    hidden = _to_device(
        torch.randn(1, 1, batch, config.hidden_size, generator=torch.Generator().manual_seed(20260729)).bfloat16(),
        mesh_device=device,
    )
    page_table = _to_device(
        torch.zeros(batch, 1, dtype=torch.int32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(batch, dtype=torch.uint32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    return decoder, page_table, positions, hidden


def decode(hidden):
    # Capture the rewritten dense projection/residual skeleton directly. The
    # pinned tracer cannot subscript TracedTensor in the cache/head portion,
    # while the advisor only owns dense linear layouts/program configs.
    normed = _DECODER._rms_norm(hidden, "input_norm")
    qkv = ttnn.linear(
        normed,
        _DECODER.weights["packed_qkv"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.attention_compute_kernel_config,
    )
    q_width = _DECODER.num_heads * _DECODER.head_dim
    attention = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, _DECODER.batch, q_width))
    attention = ttnn.linear(
        attention,
        _DECODER.weights["o_proj"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.attention_compute_kernel_config,
    )
    residual = ttnn.add(hidden, attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    normed = _DECODER._rms_norm(residual, "post_attention_norm")
    gate_up = ttnn.linear(
        normed,
        _DECODER.weights["mlp_gate_up"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.mlp_compute_kernel_config,
    )
    gate = ttnn.slice(
        gate_up,
        (0, 0, 0, 0),
        (1, 1, _DECODER.batch, _DECODER.intermediate_size),
    )
    up = ttnn.slice(
        gate_up,
        (0, 0, 0, _DECODER.intermediate_size),
        (1, 1, _DECODER.batch, 2 * _DECODER.intermediate_size),
    )
    activated = ttnn.multiply(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mlp = ttnn.linear(
        activated,
        _DECODER.weights["mlp_down"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.mlp_compute_kernel_config,
    )
    return ttnn.add(residual, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(device):
    global _DECODER, _PAGE_TABLE, _POSITIONS
    _DECODER, _PAGE_TABLE, _POSITIONS, hidden = _build(device)
    return (hidden,)
