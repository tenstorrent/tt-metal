# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for Qwen3.6-27B's rewritten dense block."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder, _to_device

BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_CONTEXT = 64
PAGE_SIZE = 64
LAYER_IDX = 3


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
    )


def _tensor(device, shape, dtype=ttnn.bfloat16):
    return _to_device(torch.zeros(shape, dtype=torch.bfloat16), mesh_device=device, dtype=dtype)


_DECODER = None
_PAGE_TABLE = None
_POSITIONS = None
_DENSE_WEIGHTS = None
_CONFIG = None
_POLICY = None


def decode(hidden):
    residual = hidden
    hidden = ttnn.rms_norm(
        hidden,
        epsilon=_CONFIG.rms_norm_eps,
        weight=_DENSE_WEIGHTS["input_norm"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    packed = ttnn.linear(
        hidden,
        _DENSE_WEIGHTS["qkv_proj"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.attention_compute_kernel_config,
    )
    q_width = _CONFIG.num_attention_heads * _CONFIG.head_dim
    q = ttnn.slice(packed, (0, 0, 0, 0), (1, 1, BATCH, q_width))
    gate = ttnn.slice(packed, (0, 0, 0, q_width), (1, 1, BATCH, 2 * q_width))
    attention = ttnn.multiply(q, ttnn.sigmoid(gate))
    attention = ttnn.linear(
        attention,
        _DENSE_WEIGHTS["o_proj"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.attention_compute_kernel_config,
    )
    hidden = ttnn.add(residual, attention)
    residual = hidden
    hidden = ttnn.rms_norm(
        hidden,
        epsilon=_CONFIG.rms_norm_eps,
        weight=_DENSE_WEIGHTS["post_attention_norm"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate_up = ttnn.linear(
        hidden,
        _DENSE_WEIGHTS["mlp_gate_up"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.mlp_compute_kernel_config,
    )
    gate = ttnn.slice(
        gate_up,
        (0, 0, 0, 0),
        (1, 1, BATCH, _CONFIG.intermediate_size),
    )
    up = ttnn.slice(
        gate_up,
        (0, 0, 0, _CONFIG.intermediate_size),
        (1, 1, BATCH, 2 * _CONFIG.intermediate_size),
    )
    hidden = ttnn.multiply(ttnn.silu(gate), up)
    hidden = ttnn.linear(
        hidden,
        _DENSE_WEIGHTS["mlp_down"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_DECODER.mlp_compute_kernel_config,
    )
    return ttnn.add(residual, hidden)


def make_inputs(device):
    global _DECODER, _PAGE_TABLE, _POSITIONS, _DENSE_WEIGHTS, _CONFIG, _POLICY
    config = _config()
    policy = POLICIES["default"]
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    weights = {
        "input_norm": _tensor(device, (config.hidden_size,)),
        "post_attention_norm": _tensor(device, (config.hidden_size,)),
        "mlp_gate_up": _tensor(
            device,
            (config.hidden_size, 2 * config.intermediate_size),
            policy.mlp_gate_up_dtype,
        ),
        "mlp_down": _tensor(
            device,
            (config.intermediate_size, config.hidden_size),
            policy.mlp_down_dtype,
        ),
        "qkv_proj": _tensor(
            device,
            (config.hidden_size, 2 * q_width + 2 * kv_width),
            policy.attention_weight_dtype,
        ),
        "o_proj": _tensor(device, (q_width, config.hidden_size), policy.attention_weight_dtype),
        "q_norm": _tensor(device, (config.head_dim,)),
        "k_norm": _tensor(device, (config.head_dim,)),
    }
    caches = {
        "key": _tensor(device, (BATCH, config.num_key_value_heads, PAGE_SIZE, config.head_dim), policy.cache_dtype),
        "value": _tensor(device, (BATCH, config.num_key_value_heads, PAGE_SIZE, config.head_dim), policy.cache_dtype),
        "batch_indices": _to_device(
            torch.arange(BATCH, dtype=torch.int32),
            mesh_device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        ),
    }
    rope = {
        "cos": _to_device(
            torch.zeros((MAX_CONTEXT, int(config.head_dim * config.partial_rotary_factor)), dtype=torch.bfloat16),
            mesh_device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "sin": _to_device(
            torch.zeros((MAX_CONTEXT, int(config.head_dim * config.partial_rotary_factor)), dtype=torch.bfloat16),
            mesh_device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    }
    grid_x = min(BATCH, device.compute_with_storage_grid_size().x)
    while BATCH % grid_x or BATCH // grid_x > device.compute_with_storage_grid_size().y:
        grid_x -= 1
    decode_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, config.head_dim),
        core_grid=ttnn.CoreGrid(y=BATCH // grid_x, x=grid_x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    _DECODER = OptimizedDecoder(
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        page_size=PAGE_SIZE,
        weights=weights,
        caches=caches,
        rope=rope,
        policy=policy,
        candidate="default",
        decode_attention_memory_config=decode_memcfg,
    )
    _DENSE_WEIGHTS = weights
    _CONFIG = config
    _POLICY = policy
    hidden = _to_device(
        torch.zeros((1, 1, BATCH, config.hidden_size), dtype=torch.bfloat16),
        mesh_device=device,
    )
    pages_per_row = MAX_CONTEXT // PAGE_SIZE
    _PAGE_TABLE = _to_device(
        torch.arange(BATCH * pages_per_row, dtype=torch.int32).reshape(BATCH, pages_per_row),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    _POSITIONS = _to_device(
        torch.zeros((BATCH,), dtype=torch.int32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    return (hidden,)
