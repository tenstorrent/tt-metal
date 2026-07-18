# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture for the final rewritten TP4-rank Falcon3 decode.

This is the post-graph-rewrite stack-native path: packed QKV, dedicated rotary
embedding, paged SDPA, separate gate/up, and a residual output that remains in
the decoder-stack contract. The mesh CCL itself cannot be represented by the
single-device advisor, so each row-parallel partial is modeled at the exact
post-CCL residual shape and its following residual add. The production
candidate separately restores the explicit TP4 async-all-reduce boundary.
"""

from __future__ import annotations

import torch

import ttnn

BATCH = 32
HIDDEN = 3072
LOCAL_HIDDEN = 768
LOCAL_Q_HEADS = 3
LOCAL_KV_HEADS = 1
HEAD_DIM = 256
LOCAL_QKV = 1280
LOCAL_MLP = 6144
MAX_CACHE_LEN = 128
POSITION = 17


def _tensor(host, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _weight(shape, device, generator):
    host = torch.empty(shape, dtype=torch.bfloat16)
    host.normal_(mean=0.0, std=0.02, generator=generator)
    return _tensor(host, device, dtype=ttnn.bfloat4_b)


def _dedicated_rope(tensor, cos, sin):
    """Mirror ``MultichipDecoder._apply_decode_rope_dedicated`` exactly."""
    transposed = ttnn.transpose(tensor, 1, 2)
    rotated = ttnn.experimental.rotary_embedding(
        transposed,
        cos,
        sin,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.transpose(rotated, 1, 2)


_STATE = None


def decode(hidden):
    (
        qkv_weight,
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        norm1,
        norm2,
        key_cache,
        value_cache,
        pos,
        cos,
        sin,
    ) = _STATE
    residual = ttnn.reshape(hidden, (1, 1, BATCH, HIDDEN))
    normed = ttnn.rms_norm(residual, epsilon=1e-6, weight=norm1)
    qkv = ttnn.matmul(normed, qkv_weight, dtype=ttnn.bfloat16)
    qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
    query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv,
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=LOCAL_KV_HEADS,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
    key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
    query = _dedicated_rope(query, cos, sin)
    key = _dedicated_rope(key, cos, sin)
    key_for_cache = ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    ttnn.experimental.paged_update_cache(value_cache, value, update_idxs_tensor=pos, share_cache=False)
    ttnn.experimental.paged_update_cache(key_cache, key_for_cache, update_idxs_tensor=pos, share_cache=False)
    attention = ttnn.transformer.scaled_dot_product_attention_decode(
        query,
        key_cache,
        value_cache,
        cur_pos_tensor=pos,
        is_causal=True,
        scale=HEAD_DIM**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attention = ttnn.to_memory_config(attention, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=LOCAL_Q_HEADS)
    projected = ttnn.matmul(attention, o_weight, dtype=ttnn.bfloat16)
    residual = ttnn.add(residual, projected)
    normed = ttnn.rms_norm(residual, epsilon=1e-6, weight=norm2)
    gate = ttnn.matmul(normed, gate_weight, dtype=ttnn.bfloat16)
    up = ttnn.matmul(normed, up_weight, dtype=ttnn.bfloat16)
    gated = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    down = ttnn.matmul(gated, down_weight, dtype=ttnn.bfloat16)
    residual = ttnn.add(residual, down)
    # Final stack-native boundary: keep [1,1,32,3072] for the next decoder.
    # Host/public DRAM materialization is an outer boundary, not part of the
    # optimized multi-layer decoder path.
    return residual


def make_inputs(device):
    global _STATE
    generator = torch.Generator().manual_seed(20260718)
    qkv_weight = _weight((HIDDEN, LOCAL_QKV), device, generator)
    o_weight = _weight((LOCAL_HIDDEN, HIDDEN), device, generator)
    gate_weight = _weight((HIDDEN, LOCAL_MLP), device, generator)
    up_weight = _weight((HIDDEN, LOCAL_MLP), device, generator)
    down_weight = _weight((LOCAL_MLP, HIDDEN), device, generator)
    norm1 = _tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    norm2 = _tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    key_cache = _tensor(
        torch.zeros(BATCH, LOCAL_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16),
        device,
        dtype=ttnn.bfloat8_b,
    )
    value_cache = _tensor(
        torch.zeros(BATCH, LOCAL_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16),
        device,
        dtype=ttnn.bfloat8_b,
    )
    pos = _tensor(
        torch.full((BATCH,), POSITION, dtype=torch.int32),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    inv_freq = 1.0 / (1000042.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    angles = torch.cat([torch.outer(torch.full((BATCH,), float(POSITION)), inv_freq)] * 2, dim=-1)
    # Match `_decode_rotary_positions`: embedding returns [1,BATCH,HEAD_DIM]
    # and `unsqueeze_to_4D` produces [1,1,BATCH,HEAD_DIM].
    cos = _tensor(angles.cos().to(torch.bfloat16).reshape(1, 1, BATCH, HEAD_DIM), device)
    sin = _tensor(angles.sin().to(torch.bfloat16).reshape(1, 1, BATCH, HEAD_DIM), device)
    hidden = _tensor(torch.randn(1, BATCH, 1, HIDDEN, dtype=torch.bfloat16), device)
    _STATE = (
        qkv_weight,
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        norm1,
        norm2,
        key_cache,
        value_cache,
        pos,
        cos,
        sin,
    )
    return (hidden,)
