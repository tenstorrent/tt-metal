# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for one Falcon3 TP4 decoder rank.

The advisor does not model CCL.  This target therefore captures the exact
per-rank dense decode graph around the two collectives: packed local QKV,
local GQA/SDPA, row-parallel O partial, split gate/up, and row-parallel down
partial.  Shapes, padding, cache dtype, activation dtype, and projection dtype
match the selected ``MultichipDecoder`` batch-32 path.
"""

from __future__ import annotations

import torch

import ttnn

BATCH = 32
POSITION = 17
MAX_CACHE_LEN = 128
HIDDEN = 3072
HEAD_DIM = 256
LOCAL_Q_HEADS = 3
LOCAL_KV_HEADS = 1
LOCAL_QKV = 1280
LOCAL_HIDDEN = 768
LOCAL_MLP_PADDED = 6144


def _weight(shape, device):
    host = torch.randn(shape, dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tensor(host, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


_INPUT_NORM = None
_QKV = None
_COS = None
_SIN = None
_KEY_CACHE = None
_VALUE_CACHE = None
_CACHE_POSITION = None
_O = None
_POST_NORM = None
_GATE = None
_UP = None
_DOWN = None


def decode(hidden):
    residual = ttnn.reshape(hidden, (1, 1, BATCH, HIDDEN))
    normed = ttnn.rms_norm(residual, epsilon=1e-6, weight=_INPUT_NORM)
    qkv = ttnn.matmul(normed, _QKV, dtype=ttnn.bfloat16)
    query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv,
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=LOCAL_KV_HEADS,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    key = ttnn.experimental.rotary_embedding(
        key, _COS, _SIN, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    query = ttnn.experimental.rotary_embedding(
        query, _COS, _SIN, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    ttnn.experimental.paged_update_cache(
        _VALUE_CACHE,
        value,
        update_idxs_tensor=_CACHE_POSITION,
        share_cache=False,
        page_table=None,
    )
    ttnn.experimental.paged_update_cache(
        _KEY_CACHE,
        key,
        update_idxs_tensor=_CACHE_POSITION,
        share_cache=False,
        page_table=None,
    )
    attention = ttnn.transformer.scaled_dot_product_attention_decode(
        query,
        _KEY_CACHE,
        _VALUE_CACHE,
        cur_pos_tensor=_CACHE_POSITION,
        is_causal=True,
        scale=1.0 / 16.0,
    )
    attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=LOCAL_Q_HEADS)
    attention = ttnn.reshape(attention, (1, 1, BATCH, LOCAL_HIDDEN))
    attention_partial = ttnn.matmul(attention, _O, dtype=ttnn.bfloat16)

    # The real path all-reduces this partial before the replicated add.  CCL is
    # intentionally absent here because shard-advise only owns local L1 layout.
    residual = ttnn.add(residual, attention_partial)
    normed = ttnn.rms_norm(residual, epsilon=1e-6, weight=_POST_NORM)
    gate = ttnn.matmul(normed, _GATE, dtype=ttnn.bfloat16)
    up = ttnn.matmul(normed, _UP, dtype=ttnn.bfloat16)
    gated = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    down_partial = ttnn.matmul(gated, _DOWN, dtype=ttnn.bfloat16)

    # The production graph performs its second all-reduce before this add.
    residual = ttnn.add(residual, down_partial)
    return ttnn.reshape(residual, (1, BATCH, 1, HIDDEN))


def make_inputs(device):
    global _INPUT_NORM, _QKV, _COS, _SIN, _KEY_CACHE, _VALUE_CACHE
    global _CACHE_POSITION, _O, _POST_NORM, _GATE, _UP, _DOWN

    _INPUT_NORM = _tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    _QKV = _weight((HIDDEN, LOCAL_QKV), device)
    _COS = _tensor(torch.ones((1, 1, 1, HEAD_DIM), dtype=torch.bfloat16), device)
    _SIN = _tensor(torch.zeros((1, 1, 1, HEAD_DIM), dtype=torch.bfloat16), device)
    cache_shape = (BATCH, LOCAL_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM)
    _KEY_CACHE = _tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device, dtype=ttnn.bfloat8_b)
    _VALUE_CACHE = _tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device, dtype=ttnn.bfloat8_b)
    _CACHE_POSITION = _tensor(
        torch.full((BATCH,), POSITION, dtype=torch.int32),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    _O = _weight((LOCAL_HIDDEN, HIDDEN), device)
    _POST_NORM = _tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    _GATE = _weight((HIDDEN, LOCAL_MLP_PADDED), device)
    _UP = _weight((HIDDEN, LOCAL_MLP_PADDED), device)
    _DOWN = _weight((LOCAL_MLP_PADDED, HIDDEN), device)
    hidden = _tensor(torch.randn((1, BATCH, 1, HIDDEN), dtype=torch.bfloat16), device)
    return (hidden,)
