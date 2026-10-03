# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Experiment switch: keep Gemma4 decode intermediates in fp32.

``GEMMA4_FP32_ACTIVATIONS=1`` keeps every intermediate tensor of the decode path in fp32
(the residual stream, norm outputs, matmul outputs, attention, the router and the experts,
the KV cache and the RoPE tables) and runs every matmul / norm / SDPA with HiFi4 and fp32
accumulation. Weights stay at their loaded dtype (bf16 checkpoint values). Unset, every
helper returns the caller's existing value, so the default path is unchanged.
"""

import os

import ttnn

ENABLED = os.environ.get("GEMMA4_FP32_ACTIVATIONS", "0") == "1"


def act_dtype(default=ttnn.bfloat16):
    """dtype for an intermediate tensor: fp32 when the switch is on, else ``default``."""
    return ttnn.float32 if ENABLED else default


def compute_config(default=None):
    """Compute kernel config: HiFi4 + fp32 accumulation when the switch is on, else ``default``."""
    if not ENABLED:
        return default
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def to_fp32(t):
    """Typecast ``t`` to fp32 when the switch is on and it is not fp32 already."""
    if ENABLED and t.dtype != ttnn.float32:
        return ttnn.typecast(t, ttnn.float32)
    return t


class SplitTable:
    """A 2-D RoPE table stored as two bf16 tables, ``hi + lo``, for ttnn.embedding (bf16-only).

    hi = bf16(value), lo = bf16(value - hi): the sum carries ~16 significant bits instead of
    bf16's 8, so the decode RoPE lookup keeps near-fp32 cos/sin values.
    """

    def __init__(self, hi, lo):
        self.hi, self.lo = hi, lo
        self.shape = hi.shape


def rope_gather(position_idx, table, layout=ttnn.TILE_LAYOUT):
    """``ttnn.embedding(position_idx, table)``; a ``SplitTable`` is gathered as hi + lo in fp32."""
    if isinstance(table, SplitTable):
        hi = ttnn.typecast(ttnn.embedding(position_idx, table.hi, layout=layout), ttnn.float32)
        lo = ttnn.typecast(ttnn.embedding(position_idx, table.lo, layout=layout), ttnn.float32)
        return ttnn.add(hi, lo)
    return ttnn.embedding(position_idx, table, layout=layout)


def sdpa_compute_config(default=None):
    """SDPA decode compute config: HiFi4 WITHOUT fp32 dest accumulation when the switch is on.

    ttnn.transformer.(paged_)scaled_dot_product_attention_decode with fp32_dest_acc_en=True returns
    wrong results once cur_pos >= k_chunk_size (64): PCC ~0.2 vs torch, while the default and
    HiFi4 + bf16 accumulation stay at PCC 0.9996+ (generated/scratch_token_accuracy/sdpa_decode_check.py,
    2026-10-03). HiFi4 alone keeps the improvement without the bug.
    """
    if not ENABLED:
        return default
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


# Separate switch: pick the router's top-k experts from the raw scores instead of from
# ttnn.softmax over all experts (whose probabilities carry up to ~0.02 absolute error, enough
# to swap near-tied experts). Same math as HF: top-k probs / their sum == softmax over the
# top-k scores.
ROUTER_TOPK_ON_SCORES = os.environ.get("GEMMA4_ROUTER_TOPK_ON_SCORES", "0") == "1"


# Separate switch (needs GEMMA4_FP32_ACTIVATIONS=1): attention in fp32 from basic ops instead of
# sdpa_decode (bf16-only inputs). The KV cache is kept as two bf16 paged caches, hi + lo
# (paged_update_cache into an fp32 cache rounds to bf16), so cached K/V carry ~16 significant bits.
# Experiment scope: one user, identity page table, positions < sliding window.
ATTENTION_FP32 = ENABLED and os.environ.get("GEMMA4_FP32_ATTENTION", "0") == "1"
_LO_CACHES = {}
_ARANGE = {}


def _lo_cache_for(cache):
    key = id(cache)
    if key not in _LO_CACHES:
        _LO_CACHES[key] = ttnn.zeros_like(cache)
    return _LO_CACHES[key]


def _split_hi_lo(x):
    hi = ttnn.typecast(x, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.sub(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


def _cache_fp32(cache):
    """[blocks, nkv, block, hd] bf16 hi cache + its lo cache -> [1, nkv, blocks*block, hd] fp32."""
    full = ttnn.add(ttnn.typecast(cache, ttnn.float32), ttnn.typecast(_lo_cache_for(cache), ttnn.float32))
    nb, nkv, bs, hd = full.shape
    full = ttnn.permute(full, (1, 0, 2, 3))
    return ttnn.reshape(full, (1, nkv, nb * bs, hd))


def fp32_attention_decode(tt_q, tt_k, tt_v, k_cache, v_cache, cache_pos, page_table, sharded_mem, update_kwargs, num_kv_heads, write_kv=True):
    """Write K/V (hi/lo) into the paged caches and return attention output [1, 1, heads, hd] fp32."""
    if write_kv:
        for t, cache in ((tt_k, k_cache), (tt_v, v_cache)):
            hi, lo = _split_hi_lo(t)
            hi = ttnn.to_memory_config(hi, sharded_mem)
            lo = ttnn.to_memory_config(lo, sharded_mem)
            ttnn.experimental.paged_update_cache(cache, hi, update_idxs_tensor=cache_pos, page_table=page_table, **update_kwargs)
            ttnn.experimental.paged_update_cache(_lo_cache_for(cache), lo, update_idxs_tensor=cache_pos, page_table=page_table, **update_kwargs)
    K = _cache_fp32(k_cache)  # [1, nkv, S, hd]
    V = _cache_fp32(v_cache)
    S = K.shape[2]
    heads, hd = tt_q.shape[2], tt_q.shape[3]
    groups = heads // num_kv_heads
    q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    q = ttnn.reshape(q, (1, num_kv_heads, groups, hd))  # GQA: q head h uses kv head h // groups
    cfg = compute_config()
    scores = ttnn.matmul(q, ttnn.transpose(K, -2, -1), compute_kernel_config=cfg)  # [1, nkv, groups, S], scale 1.0
    if S not in _ARANGE:
        import torch

        _ARANGE[S] = ttnn.from_torch(
            torch.arange(S, dtype=torch.float32).reshape(1, 1, 1, S),
            device=tt_q.device(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(tt_q.device()),
        )
    cur = ttnn.typecast(ttnn.to_layout(ttnn.reshape(cache_pos, (1, 1, 1, 1)), ttnn.TILE_LAYOUT), ttnn.float32)
    future = ttnn.gt(_ARANGE[S], cur)  # 1.0 where position > current
    scores = ttnn.add(scores, ttnn.mul(future, -1e9))
    probs = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=cfg)
    out = ttnn.matmul(probs, V, compute_kernel_config=cfg)  # [1, nkv, groups, hd]
    return ttnn.reshape(out, (1, 1, heads, hd))
