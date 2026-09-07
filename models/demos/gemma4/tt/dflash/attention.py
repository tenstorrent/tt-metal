# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter attention: the one genuinely DFlash-specific op. Query comes only from
the noise/draft block; key and value are the concatenation of the (fixed) context and the
noise block's own key/value -- see models/demos/gemma4/docs/dflash_design.md section 1.

Head reshape/RoPE/SDPA call pattern mirrors gated_attention_forward_ttnn
(models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py), already
proven on real T3K hardware for the architecturally-similar Qwen3-style Qwen3.6 MTP head --
plain reshape+transpose for heads (no nlp_create_qkv_heads needed), plain (non-zero-centered)
ttnn.rms_norm for q_norm/k_norm, and ttnn.transformer.scaled_dot_product_attention handling
GQA internally (no manual KV-head repeat needed).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4.tt.attention.weights import AttentionWeights
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import rotate_half_ttnn


def build_attention_mask_additive(
    ctx_len: int, q_len: int, is_causal: bool, sliding_window: int | None
) -> torch.Tensor:
    """Host torch additive mask [1,1,q_len,ctx_len+q_len], ported from the reference
    Qwen3DFlashAttention's _attention_mask (dflash.py). 0.0 where visible, -inf where not.

    Query row i is the (ctx_len+i)-th absolute position (queries are the tail of the
    combined [context, noise] sequence) -- so context columns (< ctx_len) are always
    visible under causal masking, and only the noise-vs-noise sub-block is restricted.
    """
    total = ctx_len + q_len
    query_position = (total - q_len) + torch.arange(q_len)[:, None]
    key_position = torch.arange(total)[None, :]
    visible = torch.ones((q_len, total), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= (query_position - key_position) < sliding_window
        if not is_causal:
            visible &= (key_position - query_position) < sliding_window
    # -1e4 (not -inf): matches the established bf16-safe convention elsewhere in this repo
    # (ttnn_gated_attention.py's segmented_attn_mask) -- large enough to zero out via softmax
    # without the NaN risk -inf carries in bfloat16.
    mask = torch.where(visible, torch.zeros(1), torch.full((1,), -1e4))
    return mask.unsqueeze(0).unsqueeze(0)  # [1,1,q_len,total]


def build_attention_mask_additive_device(
    mesh_device, ctx_len: int, q_len: int, is_causal: bool, sliding_window: int | None
) -> ttnn.Tensor:
    """On-device equivalent of ``build_attention_mask_additive`` -- same formula, built
    entirely with ttnn ops (``ttnn.arange``/``le``/``lt``/``logical_and``/``where``), no
    host torch computation or upload. Confirmed exact match (bf16) against the host-torch
    version, cast to bf16 the same way the caller would (``ttnn.from_torch(...,
    dtype=bfloat16)``), for every (ctx_len, q_len, is_causal, sliding_window) combination
    this pipeline actually uses. Returns [1,1,q_len,ctx_len+q_len] bf16, replicated."""
    total = ctx_len + q_len
    query_position = ttnn.arange(total - q_len, total, 1, device=mesh_device, dtype=ttnn.int32)
    key_position = ttnn.arange(0, total, 1, device=mesh_device, dtype=ttnn.int32)
    query_col = ttnn.reshape(query_position, [q_len, 1])
    key_row = ttnn.reshape(key_position, [1, total])
    query_full = ttnn.repeat(query_col, ttnn.Shape([1, total]))
    key_full = ttnn.repeat(key_row, ttnn.Shape([q_len, 1]))

    visible = ttnn.ones([q_len, total], dtype=ttnn.int32, device=mesh_device)
    if is_causal:
        visible = ttnn.logical_and(visible, ttnn.le(key_full, query_full))
    if sliding_window is not None:
        visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(query_full, key_full), sliding_window))
        if not is_causal:
            visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(key_full, query_full), sliding_window))

    visible = ttnn.to_layout(ttnn.typecast(visible, ttnn.bfloat16), ttnn.TILE_LAYOUT)
    zero = ttnn.zeros([q_len, total], dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    neg = ttnn.full([q_len, total], fill_value=-1e4, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    mask = ttnn.where(visible, zero, neg)
    return ttnn.unsqueeze_to_4D(ttnn.unsqueeze_to_4D(mask))  # [1,1,q_len,total]


def build_attention_mask_additive_device_dynamic(
    mesh_device,
    ctx_len: int,
    q_len: int,
    is_causal: bool,
    sliding_window: int | None,
    context_valid_len_tt: ttnn.Tensor,
) -> ttnn.Tensor:
    """Same as ``build_attention_mask_additive_device``, plus one more masking condition:
    context columns whose index is >= ``context_valid_len_tt`` are ALSO marked invisible.

    For a fixed-size context window (``ctx_len`` always the drafter's block_size, padded
    with the tail ``block_size - produced`` rows being meaningless -- see generate.py),
    this masks out exactly the padding rows while leaving the real ``produced`` rows (the
    FIRST ``context_valid_len`` columns, matching how the context accumulator's real
    content occupies the earliest rows) visible under the ordinary causal/sliding rules.
    Noise columns (index >= ctx_len) are never affected by this condition.

    ``context_valid_len_tt`` is an ON-DEVICE scalar tensor (``[1,1]`` int32), not a Python
    int -- its VALUE can differ every call without changing the mask's SHAPE, which is
    exactly what makes this trace-replay-compatible (a captured trace can't have its op
    graph depend on a Python-int branch, but it CAN depend on a tensor's contents).

    NOT trace-safe by itself: builds the static grids from scratch every call via
    ``ttnn.arange``/``ones``/``zeros``/``full``, each of which does a host->device WRITE
    internally (confirmed: ``creation.cpp``'s ``arange_impl``/``full_impl`` build a host
    ``std::vector`` and upload it via ``Tensor::from_vector``) -- disallowed mid-capture
    (``TT_FATAL: Writes are not supported during trace capture``). Eager (non-traced)
    callers only. For the traced steady-state loop, use
    ``build_attention_mask_static_parts`` (once, outside capture) +
    ``combine_attention_mask_dynamic`` (inside the captured body, per replay) instead --
    identical result, split so only the ``context_valid_len_tt``-dependent combine step
    re-runs on every replay."""
    total = ctx_len + q_len
    query_position = ttnn.arange(total - q_len, total, 1, device=mesh_device, dtype=ttnn.int32)
    key_position = ttnn.arange(0, total, 1, device=mesh_device, dtype=ttnn.int32)
    query_col = ttnn.reshape(query_position, [q_len, 1])
    key_row = ttnn.reshape(key_position, [1, total])
    query_full = ttnn.repeat(query_col, ttnn.Shape([1, total]))
    key_full = ttnn.repeat(key_row, ttnn.Shape([q_len, 1]))

    visible = ttnn.ones([q_len, total], dtype=ttnn.int32, device=mesh_device)
    if is_causal:
        visible = ttnn.logical_and(visible, ttnn.le(key_full, query_full))
    if sliding_window is not None:
        visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(query_full, key_full), sliding_window))
        if not is_causal:
            visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(key_full, query_full), sliding_window))

    valid_len_col = ttnn.repeat(ttnn.reshape(context_valid_len_tt, [1, 1]), ttnn.Shape([q_len, 1]))
    valid_len_full = ttnn.repeat(valid_len_col, ttnn.Shape([1, total]))
    is_valid_context_col = ttnn.lt(key_full, valid_len_full)  # key_position < context_valid_len
    is_noise_col = ttnn.ge(key_full, ctx_len)  # noise columns are unaffected by context padding
    visible = ttnn.logical_and(visible, ttnn.logical_or(is_valid_context_col, is_noise_col))

    visible = ttnn.to_layout(ttnn.typecast(visible, ttnn.bfloat16), ttnn.TILE_LAYOUT)
    zero = ttnn.zeros([q_len, total], dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    neg = ttnn.full([q_len, total], fill_value=-1e4, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    mask = ttnn.where(visible, zero, neg)
    return ttnn.unsqueeze_to_4D(ttnn.unsqueeze_to_4D(mask))  # [1,1,q_len,total]


class DynamicMaskStaticParts:
    """Everything about ``build_attention_mask_additive_device_dynamic`` that does NOT
    depend on ``context_valid_len_tt``'s current value -- built ONCE (via
    ``build_attention_mask_static_parts``) outside a trace capture, since ``ctx_len``,
    ``q_len``, ``is_causal`` and ``sliding_window`` are all fixed for the whole steady-state
    loop. Holds device tensors only (no host state), safe to reference from inside a
    captured trace body."""

    __slots__ = ("key_full", "visible_base", "is_noise_col", "zero", "neg", "q_len", "total")

    def __init__(self, key_full, visible_base, is_noise_col, zero, neg, q_len, total):
        self.key_full = key_full
        self.visible_base = visible_base
        self.is_noise_col = is_noise_col
        self.zero = zero
        self.neg = neg
        self.q_len = q_len
        self.total = total


def build_attention_mask_static_parts(
    mesh_device, ctx_len: int, q_len: int, is_causal: bool, sliding_window: int | None
) -> DynamicMaskStaticParts:
    """One-time setup (call OUTSIDE ``begin_trace_capture``): builds every piece of
    ``build_attention_mask_additive_device_dynamic`` that's independent of
    ``context_valid_len_tt``'s value -- the position grids, the causal/sliding base
    visibility, the noise-column exemption, and the bf16 zero/neg constants. Uses
    ``ttnn.arange``/``ones``/``zeros``/``full`` (host-write ops), which is fine here since
    this runs once, before capture begins -- see ``combine_attention_mask_dynamic`` for the
    per-replay half that's actually inside the trace."""
    total = ctx_len + q_len
    query_position = ttnn.arange(total - q_len, total, 1, device=mesh_device, dtype=ttnn.int32)
    key_position = ttnn.arange(0, total, 1, device=mesh_device, dtype=ttnn.int32)
    query_col = ttnn.reshape(query_position, [q_len, 1])
    key_row = ttnn.reshape(key_position, [1, total])
    query_full = ttnn.repeat(query_col, ttnn.Shape([1, total]))
    key_full = ttnn.repeat(key_row, ttnn.Shape([q_len, 1]))

    visible_base = ttnn.ones([q_len, total], dtype=ttnn.int32, device=mesh_device)
    if is_causal:
        visible_base = ttnn.logical_and(visible_base, ttnn.le(key_full, query_full))
    if sliding_window is not None:
        visible_base = ttnn.logical_and(visible_base, ttnn.lt(ttnn.subtract(query_full, key_full), sliding_window))
        if not is_causal:
            visible_base = ttnn.logical_and(visible_base, ttnn.lt(ttnn.subtract(key_full, query_full), sliding_window))

    is_noise_col = ttnn.ge(key_full, ctx_len)  # noise columns are unaffected by context padding

    zero = ttnn.zeros([q_len, total], dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    neg = ttnn.full([q_len, total], fill_value=-1e4, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    return DynamicMaskStaticParts(key_full, visible_base, is_noise_col, zero, neg, q_len, total)


def combine_attention_mask_dynamic(static: DynamicMaskStaticParts, context_valid_len_tt: ttnn.Tensor) -> ttnn.Tensor:
    """The per-replay half: ONLY elementwise ops on already-existing device tensors
    (``reshape``/``repeat``/``lt``/``logical_and``/``logical_or``/``where``/``typecast``/
    ``to_layout``) -- no ``arange``/``ones``/``zeros``/``full``, so no host->device write,
    safe to call from inside a captured trace body. Recomputes exactly the
    ``context_valid_len_tt``-dependent part of
    ``build_attention_mask_additive_device_dynamic``, using ``static``'s precomputed,
    value-independent pieces for everything else. Confirmed bit-for-bit identical to that
    function's output for the same inputs."""
    q_len, total = static.q_len, static.total
    valid_len_col = ttnn.repeat(ttnn.reshape(context_valid_len_tt, [1, 1]), ttnn.Shape([q_len, 1]))
    valid_len_full = ttnn.repeat(valid_len_col, ttnn.Shape([1, total]))
    is_valid_context_col = ttnn.lt(static.key_full, valid_len_full)
    visible = ttnn.logical_and(static.visible_base, ttnn.logical_or(is_valid_context_col, static.is_noise_col))
    visible = ttnn.to_layout(ttnn.typecast(visible, ttnn.bfloat16), ttnn.TILE_LAYOUT)
    mask = ttnn.where(visible, static.zero, static.neg)
    return ttnn.unsqueeze_to_4D(ttnn.unsqueeze_to_4D(mask))  # [1,1,q_len,total]


def _apply_rope_single(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotate_half_ttnn(x), sin))


def dflash_attention_forward(
    context: ttnn.Tensor,  # [1,1,ctx_len,hidden], full-width replicated
    noise: ttnn.Tensor,  # [1,1,q_len,hidden], full-width replicated (already input_layernorm'd)
    weights: AttentionWeights,
    cos_full: ttnn.Tensor,  # [1,1,ctx_len+q_len,head_dim], replicated
    sin_full: ttnn.Tensor,
    attn_mask: ttnn.Tensor,  # [1,1,q_len,ctx_len+q_len], replicated (build_attention_mask_additive, uploaded)
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
) -> ttnn.Tensor:
    scale = head_dim**-0.5
    q_w = num_local_heads * head_dim
    kv_w = num_local_kv_heads * head_dim

    q_len = noise.shape[-2]
    ctx_len = context.shape[-2]

    qkv_noise = ttnn.linear(noise, weights.wqkv)  # [1,1,q_len, q_w+2*kv_w] per device
    qkv_ctx = ttnn.linear(context, weights.wqkv)  # [1,1,ctx_len, q_w+2*kv_w] per device

    q = ttnn.slice(qkv_noise, [0, 0, 0, 0], [1, 1, q_len, q_w])
    k_noise = ttnn.slice(qkv_noise, [0, 0, 0, q_w], [1, 1, q_len, q_w + kv_w])
    v_noise = ttnn.slice(qkv_noise, [0, 0, 0, q_w + kv_w], [1, 1, q_len, q_w + 2 * kv_w])
    ttnn.deallocate(qkv_noise)

    k_ctx = ttnn.slice(qkv_ctx, [0, 0, 0, q_w], [1, 1, ctx_len, q_w + kv_w])
    v_ctx = ttnn.slice(qkv_ctx, [0, 0, 0, q_w + kv_w], [1, 1, ctx_len, q_w + 2 * kv_w])
    ttnn.deallocate(qkv_ctx)

    k = ttnn.concat([k_ctx, k_noise], dim=-2)  # [1,1,ctx_len+q_len, kv_w]
    v = ttnn.concat([v_ctx, v_noise], dim=-2)
    ttnn.deallocate(k_ctx)
    ttnn.deallocate(k_noise)
    ttnn.deallocate(v_ctx)
    ttnn.deallocate(v_noise)

    # heads: [1,1,seq,W] -> [1,seq,heads,head_dim] -> norm (last axis) -> transpose -> [1,heads,seq,head_dim]
    q = ttnn.reshape(q, [1, q_len, num_local_heads, head_dim])
    q = ttnn.rms_norm(q, weight=weights.q_norm_weight, epsilon=eps)
    q = ttnn.transpose(q, 1, 2)

    k = ttnn.reshape(k, [1, ctx_len + q_len, num_local_kv_heads, head_dim])
    k = ttnn.rms_norm(k, weight=weights.k_norm_weight, epsilon=eps)
    k = ttnn.transpose(k, 1, 2)

    v = ttnn.reshape(v, [1, ctx_len + q_len, num_local_kv_heads, head_dim])
    v = ttnn.transpose(v, 1, 2)

    # RoPE: q only ever needs the LAST q_len positions' rotation; k spans the full range.
    total = ctx_len + q_len
    cos_q = ttnn.slice(cos_full, [0, 0, total - q_len, 0], [1, 1, total, head_dim])
    sin_q = ttnn.slice(sin_full, [0, 0, total - q_len, 0], [1, 1, total, head_dim])
    q = _apply_rope_single(q, cos_q, sin_q)
    k = _apply_rope_single(k, cos_full, sin_full)

    attn_out = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, scale=scale)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, [1, q_len, num_local_heads * head_dim])

    out = ttnn.linear(attn_out, weights.o_proj)
    ttnn.deallocate(attn_out)
    out = ccl_allreduce(out, mesh_config, ccl_manager)
    return out
