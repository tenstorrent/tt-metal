# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local chunked (bounded-memory) long-context prefill (#47466).

Why this module exists
----------------------
The shared gemma4 backbone does **not** honor the vLLM multi-chunk prefill
contract. In ``models/demos/gemma4/tt/model.py`` both ``chunk_start_idx`` and
``chunk_page_table`` are accepted for signature-compat and then ``del``'d
(``prepare_inputs_prefill`` line ~1298, ``ttnn_prefill_forward`` line ~1436).
Consequences:

* **Wrong RoPE offset** — prefill RoPE is always sliced ``cos[:, :, :seq_len]``
  (``_get_rope_mats``), i.e. positions ``0..seq_len-1``. A second chunk's tokens
  get positions ``0..L-1`` instead of ``chunk_start_idx..chunk_start_idx+L-1``.
* **No cross-chunk attention** — a chunk's SDPA never reads the KV written by a
  prior chunk, so attention is broken past a single chunk.
* **Unbounded memory** — the single-chunk prefill projects / RoPEs / fills the
  *whole* prompt at once, so prefill L1+DRAM scales with the full prompt length
  (OOM past ~64k).

The shared backbone must not be edited (``git diff main -- models/demos/gemma4``
stays empty), so this module **composes over** it: it copies the gemma4
single-user prefill-attention routine (``attention/prefill.py::_prefill_forward_single``)
and fixes the three defects locally, then drives the *unmodified* backbone graph
one bounded chunk at a time.

The correct contract (mirrors ``models/tt_transformers/tt/attention.py`` +
``generator.py``, the reference that already implements chunked prefill):

* ``page_table``       — the **full** per-user page table (logical blocks
  ``0 .. chunk_end``). Passed to the SDPA op so a chunk's queries attend the
  *entire* KV prefix, including all prior chunks.
* ``chunk_page_table`` — this chunk's blocks
  (``page_table[:, chunk_start_block:chunk_end_block]``). Passed to
  ``paged_fill_cache`` so only this chunk's K/V is written.
* ``chunk_start_idx``  — absolute start position of the chunk. Drives **both**
  the per-chunk RoPE slice offset and the SDPA causal-mask offset
  (``chunked_scaled_dot_product_attention(chunk_start_idx=...)``).

Because only one chunk's activations are resident at a time (prior chunks live
in the paged KV cache and are read directly by the SDPA op — never
materialized), prefill memory is bounded by ``O(chunk_size)`` instead of
``O(prompt_len)``.

Scope (prototype, gated by ``DG_CHUNKED_PREFILL``, default OFF)
--------------------------------------------------------------
* Single-user (``batch_size == 1``) bounded-memory prefill. Batched chunked
  prefill is the #47557 batched-canvas / #47488 paged-ownership follow-up.
* Full-attention layers get true cross-chunk attention via the paged
  ``chunked_scaled_dot_product_attention`` op, reading all prior chunks straight
  from the paged KV cache. Memory bound: ``O(chunk_size)`` prefill scratch (prior
  chunks live in the paged cache, never materialized).
* Sliding-window layers work for prompts of *any* length — including *longer*
  than the sliding window (1024). The paged chunked SDPA op is **causal-only**
  (no window mask), so sliding layers cannot use it once total context exceeds
  the window (over-attention). Instead each sliding layer threads a **bounded
  rolling in-memory K/V window buffer** (``<= sliding_window + chunk_size``
  positions) across chunks: every chunk appends its RoPE'd K/V to the buffer,
  runs the overlapping-window causal+sliding SDPA (gemma4
  ``chunked_prefill_sdpa_sliding``, copied below) over the buffer for this
  chunk's queries, then trims the buffer back to the last ``sliding_window``
  positions. A sliding query at absolute pos ``p`` only depends on K/V in
  ``(p - window, p]``, so the trimmed buffer always contains every key any of the
  chunk's queries can attend — making the bounded result identical to a single
  full-length sliding-window prefill. Memory bound: ``O(chunk_size + window)``.

Both attention kinds therefore stay bounded-memory, so prompts far past the
single-chunk ``O(prompt_len)`` OOM cliff (>64k) prefill in a fixed footprint.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import ttnn
import models.demos.gemma4.tt.attention as _gemma4_attn
from models.demos.gemma4.tt.attention.operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    effective_block_size,
    split_qkv_heads_prefill,
)

FLAG = "DG_CHUNKED_PREFILL"


def chunked_prefill_enabled() -> bool:
    """True when the ``DG_CHUNKED_PREFILL`` flag opts in to the chunked path.

    Default OFF — callers fall back to the stock single-chunk gemma4 prefill.
    """
    return os.environ.get(FLAG, "0") == "1"


# ── per-sliding-layer rolling K/V window buffer (threaded across chunks) ──────
@dataclass
class _SlidingWindowState:
    """Bounded rolling in-memory K/V window buffers, one per sliding-window layer.

    Keyed by ``id(weights)`` (the per-layer ``AttentionWeights`` object is unique
    and alive for the whole chunked-prefill call, so its id is a stable per-layer
    key). Each value is ``(k_buf, v_buf)`` device tensors holding RoPE'd K/V for a
    contiguous run of absolute positions ending at the last-appended chunk; the
    run is trimmed to ``<= sliding_window`` positions after each chunk so peak
    residency is ``<= sliding_window + chunk_size``.
    """

    buffers: dict  # id(weights) -> (k_buf, v_buf)

    def release(self):
        for k_buf, v_buf in self.buffers.values():
            k_buf.deallocate(True)
            v_buf.deallocate(True)
        self.buffers.clear()


# ── per-chunk context (read by the swapped attention-prefill routine) ────────
@dataclass
class _ChunkContext:
    chunk_start_idx: int  # absolute start position of the active chunk
    chunk_page_table: object  # ttnn.Tensor: this chunk's blocks (fill target)
    sliding_state: _SlidingWindowState  # per-sliding-layer rolling K/V window buffers


_CHUNK_CTX: _ChunkContext | None = None


def _default_chunk_size() -> int:
    """Chunk length in tokens (``DG_CHUNKED_PREFILL_CHUNK``, default 256).

    Must be a multiple of 128 (the ``chunked_scaled_dot_product_attention``
    ``q_chunk_size``) and of the tile height (32); 256 satisfies both and matches
    the DiffusionGemma canvas granularity.
    """
    return int(os.environ.get("DG_CHUNKED_PREFILL_CHUNK", "256"))


def _chunked_sdpa_program_config(head_dim: int) -> "ttnn.SDPAProgramConfig":
    # Mirrors gemma4 operations.chunked_prefill_sdpa: head_dim=512 needs more
    # L1/core, so a smaller grid; sliding head_dim uses the full grid.
    grid = ttnn.CoreCoord(8, 4) if head_dim >= 512 else ttnn.CoreCoord(8, 8)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )


# ── sliding-window SDPA over the buffer (adapted from gemma4 chunked_prefill_sdpa_sliding) ──
def _sliding_window_square_sdpa(tt_q, tt_k, tt_v, sliding_window, scale=1.0):
    """One causal+sliding-window SDPA over a square (Q.s == K.s) window slice.

    This is the core of gemma4 ``operations.chunked_prefill_sdpa_sliding`` (the
    per-stride SDPA), lifted DG-local. The paged ``chunked_scaled_dot_product_...``
    op is causal-only (no window mask); the non-chunked op supports a window mask
    but requires ``Q.s == K.s`` (a square causal mask, with a 32768 cliff on that
    shared length). A sliding query at absolute pos ``p`` depends only on K/V in
    ``(p - window, p]``, so callers pass a bounded square window slice
    (``<= sliding_window + chunk_size`` positions, well under the cliff) and keep
    only the query rows they care about.

    Unlike the gemma4 routine this does NOT slice-and-deallocate its inputs: the
    caller here passes a *persistent* rolling window buffer that must survive the
    call, and a full-range ``ttnn.slice`` aliases its input (so deallocating the
    slice would free the buffer). The buffer is always a single stride, so one
    direct SDPA call replaces the gemma4 strided loop.

    Args:
        tt_q, tt_k, tt_v: [1, heads/kv_heads, buf_len, head_dim] (TILE), RoPE'd,
            with ``buf_len`` the same for Q and K/V (square).
        sliding_window: window size W (Gemma4 sliding layers: 1024, tile-aligned).
    Returns:
        [1, num_heads, buf_len, head_dim] attention output (TILE layout).
    """
    # HiFi4 + FP32 dest-acc: match the non-chunked prefill SDPA's softmax-reduce
    # precision (the reference single-prefill path uses the same fidelity).
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tt_q.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        scale=scale,
        sliding_window_size=sliding_window,
        compute_kernel_config=compute_kernel_config,
    )


def _bounded_sliding_sdpa(ctx, key, tt_q, tt_k, tt_v, sliding_window, head_dim):
    """Sliding-window SDPA for ONE chunk over a bounded rolling K/V window buffer.

    This is the fix for "the paged chunked SDPA op is causal-only, so sliding
    layers over-attend once total context exceeds the window". Instead of the
    paged op, each sliding layer keeps a small rolling buffer of RoPE'd K/V:

    1. **Append** this chunk's ``(tt_k, tt_v)`` to the layer's buffer (via
       ``clone``/``concat`` so it is independent of the caller's ``tt_k``/``tt_v``,
       which are deallocated after this call). The buffer now holds absolute
       positions ``[buf_start, chunk_end)`` where
       ``buf_start = max(0, chunk_start - window)`` (from the prior trim).
    2. **Attend** the buffer for this chunk's queries via the square causal+sliding
       SDPA (:func:`_sliding_window_square_sdpa`). ``scaled_dot_product_attention``
       with ``is_causal`` requires ``Q.s == K.s``, so the chunk's queries are
       front-aligned to the buffer's tail (``hist_len`` zero query rows in front);
       only the tail ``chunk_len`` output rows (the real queries) are kept. A
       sliding query at pos ``p`` attends only ``(p - window, p]`` — all inside the
       buffer — so the result equals a single full-length sliding-window prefill.
    3. **Trim** the buffer back to the last ``sliding_window`` positions so peak
       residency stays ``<= sliding_window + chunk_size`` (bounded memory).

    Returns the chunk's attention output ``[1, num_heads, chunk_len, head_dim]``.
    """
    state = ctx.sliding_state
    nh = tt_q.shape[1]
    nkv = tt_k.shape[1]
    chunk_len = tt_q.shape[-2]

    # 1. append this chunk's K/V to the rolling buffer (independent of tt_k/tt_v).
    prev = state.buffers.get(key)
    if prev is None:
        k_buf = ttnn.clone(tt_k)
        v_buf = ttnn.clone(tt_v)
    else:
        pk, pv = prev
        k_buf = ttnn.concat([pk, tt_k], dim=2)
        v_buf = ttnn.concat([pv, tt_v], dim=2)
        pk.deallocate(True)
        pv.deallocate(True)

    buf_len = k_buf.shape[-2]
    hist_len = buf_len - chunk_len  # history positions preceding this chunk in the buffer

    # 2. square Q: front-pad this chunk's queries with hist_len zero rows so the
    #    causal+sliding SDPA aligns query row (hist_len + j) with buffer position
    #    j's absolute pos. hist_len is a multiple of 32 (window + chunk are
    #    tile-aligned), so the concat/slice stay tile-aligned. The zero front rows
    #    produce outputs that are discarded (attention rows are independent, so
    #    they cannot affect the real query rows). The SDPA op reads but does not
    #    deallocate its inputs, so the persistent buffer is untouched.
    if hist_len > 0:
        zeros_q = ttnn.zeros(
            [1, nh, hist_len, head_dim], dtype=tt_q.dtype, layout=ttnn.TILE_LAYOUT, device=tt_q.device()
        )
        q_square = ttnn.concat([zeros_q, tt_q], dim=2)
        zeros_q.deallocate(True)
    else:
        q_square = tt_q

    o_full = _sliding_window_square_sdpa(q_square, k_buf, v_buf, sliding_window, scale=1.0)
    if q_square is not tt_q:
        q_square.deallocate(True)

    if hist_len > 0:
        out = ttnn.slice(o_full, [0, 0, hist_len, 0], [1, nh, buf_len, head_dim])
        o_full.deallocate(True)
    else:
        out = o_full

    # 3. trim the buffer to the last `sliding_window` positions (bounded memory).
    #    ttnn.slice over a strict sub-range returns an independent copy, so the
    #    trimmed buffer survives the original buffer's deallocation.
    if buf_len > sliding_window:
        trim = buf_len - sliding_window
        k_trim = ttnn.slice(k_buf, [0, 0, trim, 0], [1, nkv, buf_len, head_dim])
        v_trim = ttnn.slice(v_buf, [0, 0, trim, 0], [1, nkv, buf_len, head_dim])
        k_buf.deallocate(True)
        v_buf.deallocate(True)
        k_buf, v_buf = k_trim, v_trim
    state.buffers[key] = (k_buf, v_buf)

    return out


# ── copied + fixed gemma4 _prefill_forward_single ────────────────────────────
def chunked_prefill_attention_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    page_table=None,
    user_id=0,
    ccl_manager=None,
    shared_kv=None,
    keep_kv=False,
    batch_size=1,
    valid_seq_len=None,
):
    """Single-user prefill attention for ONE bounded chunk.

    Signature-compatible with ``gemma4 attention.prefill.prefill_forward`` so it
    can transparently stand in for it (see :func:`_swap_prefill_attention`). The
    three chunked-prefill fixes vs the gemma4 original:

    1. RoPE ``cos_cache``/``sin_cache`` arrive **already sliced** to the chunk's
       absolute positions (the driver passes a ``rope_mats`` dict offset by
       ``chunk_start_idx``), so ``apply_rope`` rotates the chunk to the right
       positions.
    2. K/V is written with ``_CHUNK_CTX.chunk_page_table`` (this chunk's blocks),
       not the full ``page_table``.
    3. SDPA runs ``chunked_scaled_dot_product_attention`` over the **full**
       ``page_table`` with ``chunk_start_idx`` — so the chunk's queries attend
       the entire KV prefix (all prior chunks) under a correctly-offset causal
       mask, reading prior chunks straight from the paged cache (bounded memory).
    """
    ctx = _CHUNK_CTX
    if ctx is None:
        # Not inside a chunked-prefill driver call — defer to the stock backbone.
        return _ORIG_PREFILL_FORWARD(
            hidden_states,
            cos_cache,
            sin_cache,
            weights,
            kv_cache,
            config,
            mesh_config,
            mesh_device,
            page_table=page_table,
            user_id=user_id,
            ccl_manager=ccl_manager,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            batch_size=batch_size,
            valid_seq_len=valid_seq_len,
        )

    if batch_size > 1:
        raise NotImplementedError("chunked prefill prototype is single-user (batch_size==1)")
    if page_table is None or kv_cache is None:
        raise ValueError("chunked prefill requires a paged kv_cache + full page_table")

    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(hidden_states, weights)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # RoPE — cos/sin are pre-sliced to [chunk_start_idx : chunk_start_idx + L].
    tt_q = apply_rope(tt_q, cos_cache, sin_cache)
    if shared_kv is None:
        tt_k = apply_rope(tt_k, cos_cache, sin_cache)

    # ── FIX 2: write THIS chunk's K/V using chunk_page_table ─────────────────
    if kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
        paged_modulo_kwargs = (
            {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
        )
        ttnn.experimental.paged_fill_cache(
            k_cache, tt_k, ctx.chunk_page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
        )
        ttnn.experimental.paged_fill_cache(
            v_cache, tt_v, ctx.chunk_page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
        )

    # ── FIX 3: attend the correct prefix under each layer's mask ─────────────
    #   full-attention: the ENTIRE KV prefix (all prior chunks) via the paged
    #     causal chunked SDPA op with chunk_start_idx — prior chunks are read
    #     straight from the paged cache (never materialized → O(chunk) memory).
    #   sliding-window: the paged chunked op is CAUSAL-ONLY (no window mask), so
    #     it would over-attend once total context exceeds the window. Instead
    #     thread a bounded rolling in-memory K/V window buffer per sliding layer
    #     (:func:`_bounded_sliding_sdpa`) → O(chunk + window) memory. This is what
    #     makes sliding-window chunked prefill correct for prompts LONGER than the
    #     sliding window.
    nh = tt_q.shape[1]
    head_dim = config.head_dim

    if config.is_sliding and config.sliding_window is not None:
        tt_sdpa = _bounded_sliding_sdpa(ctx, id(weights), tt_q, tt_k, tt_v, config.sliding_window, head_dim)
    else:
        k_cache, v_cache = kv_cache
        num_pages = page_table.shape[-1]
        if page_table.shape[0] > 1:
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, num_pages])
            owns_user_pt = True
        else:
            user_pt = page_table
            owns_user_pt = False

        q_len = tt_q.shape[-2]
        # chunked SDPA q_chunk_size=128 needs q_len % 128 == 0; pad the tail and slice back.
        pad = (-q_len) % 128
        q_in = tt_q
        if pad:
            q_in = ttnn.pad(tt_q, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            tt_q.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        tt_sdpa = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_in,
            k_cache,
            v_cache,
            user_pt,
            chunk_start_idx=ctx.chunk_start_idx,
            scale=1.0,
            program_config=_chunked_sdpa_program_config(head_dim),
            compute_kernel_config=compute_kernel_config,
        )
        if pad:
            q_in.deallocate(True)
            sdpa_unpadded = ttnn.slice(tt_sdpa, [0, 0, 0, 0], [1, nh, q_len, head_dim])
            tt_sdpa.deallocate(True)
            tt_sdpa = sdpa_unpadded
        if owns_user_pt:
            user_pt.deallocate(True)

    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out, kept_kv


# The gemma4 name the swapped attention defers to for the non-chunked path.
_ORIG_PREFILL_FORWARD = _gemma4_attn.prefill_forward


@contextmanager
def _swap_prefill_attention():
    """Route ``Gemma4Attention.__call__``'s prefill through the fixed routine.

    ``Gemma4Attention.__call__`` resolves ``prefill_forward`` from the
    ``models.demos.gemma4.tt.attention`` package globals. Rebinding that name for
    the duration of a chunked-prefill call swaps in the fixed attention **without
    editing any gemma4 file** (this is runtime composition, not a source edit —
    the diff against gemma4 stays empty). The whole backbone graph (layers, MoE,
    KV-sharing, norms, lm_head) is otherwise the *real* unmodified backbone, so a
    chunked-vs-single comparison is apples-to-apples. Restored on exit.
    """
    saved = _gemma4_attn.prefill_forward
    _gemma4_attn.prefill_forward = chunked_prefill_attention_forward
    try:
        yield
    finally:
        _gemma4_attn.prefill_forward = saved


# ── page-table helpers (mirror tt_transformers generator chunk math) ─────────
def _blocks_in(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def make_reference_page_table(num_blocks: int, *, mesh_device) -> torch.Tensor:
    """Identity logical→physical page table for a single contiguous sequence.

    Real serving hands the model a vLLM-owned page table; for the standalone
    correctness check the sequence owns blocks ``0..num_blocks-1`` contiguously.
    Returned as a host torch tensor ``[1, num_blocks]`` (int32).
    """
    return torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)


def _to_device_page_table(page_table_torch, mesh_device):
    is_mesh = hasattr(mesh_device, "shape")
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    return ttnn.from_torch(
        page_table_torch,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )


def _chunk_rope_mats(model, chunk_start_idx: int, chunk_len: int):
    """Per-layer-type RoPE (cos, sin) sliced to this chunk's absolute positions.

    Returns a dict ``{layer_type: (cos, sin)}`` that ``Gemma4Model.__call__``
    consumes directly (bypassing its ``_get_rope_mats``, which always slices from
    position 0). This is the RoPE-offset fix — without it the chunk would be
    rotated as if it started at position 0.
    """
    rope = {}
    for layer_type, (cos, sin) in model.rope_caches.items():
        end = chunk_start_idx + chunk_len
        if end > cos.shape[-2]:
            raise ValueError(f"chunk RoPE slice [{chunk_start_idx}:{end}] exceeds cache length {cos.shape[-2]}")
        rope[layer_type] = (
            cos[:, :, chunk_start_idx:end, :],
            sin[:, :, chunk_start_idx:end, :],
        )
    return rope


def chunked_prefill(
    model,
    prompt_embeds,
    *,
    input_ids_torch,
    embeds_torch,
    kv_cache,
    page_table_torch,
    block_size: int,
    chunk_size: int | None = None,
    user_id: int = 0,
    return_last_logits: bool = True,
):
    """Prefill ``prompt_embeds`` in bounded-memory chunks over a paged KV cache.

    Args:
        model: a ``DiffusionGemma4Model`` / ``Gemma4Model`` (unmodified backbone).
        prompt_embeds: ``[1, 1, S, hidden]`` tile-laid prompt embeddings (S is a
            tile multiple; the caller pads).
        input_ids_torch, embeds_torch: host token ids / embeddings for the whole
            prompt (used by the backbone's per-layer-input / MoE-free path); sliced
            per chunk here.
        kv_cache: list of ``[k_cache, v_cache]`` paged caches per layer.
        page_table_torch: host ``[1, num_blocks]`` full page table for the sequence.
        block_size: paged cache block size (tokens per block).
        chunk_size: chunk length in tokens (default from :func:`_default_chunk_size`).
        return_last_logits: when True, only the final chunk runs lm_head and its
            logits ``[1, 1, chunk_len, vocab]`` are returned; earlier chunks only
            write KV.

    Returns:
        The final chunk's logits (device tensor) when ``return_last_logits``,
        else ``None``. Callers slice the last-token row.
    """
    chunk_size = chunk_size or _default_chunk_size()
    if chunk_size % 128 != 0:
        raise ValueError(f"chunk_size {chunk_size} must be a multiple of 128 (chunked SDPA q_chunk_size)")
    if chunk_size % block_size != 0:
        raise ValueError(f"chunk_size {chunk_size} must be a multiple of block_size {block_size}")

    seq_len = prompt_embeds.shape[-2]
    hidden = prompt_embeds.shape[-1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"prompt seq_len {seq_len} must be a multiple of chunk_size {chunk_size} (pad the caller)")
    num_chunks = seq_len // chunk_size

    full_pt_dev = _to_device_page_table(page_table_torch, model.mesh_device)

    # Rolling K/V window buffers for sliding-window layers, threaded across all
    # chunks (one per sliding layer). Built here so it lives for the whole prefill
    # and is released at the end — bounded to O(window + chunk_size) per layer.
    sliding_state = _SlidingWindowState(buffers={})

    logits = None
    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size
        is_last = c == num_chunks - 1

        # This chunk's blocks: page_table[:, start_block:end_block].
        start_block = start // block_size
        end_block = _blocks_in(end, block_size)
        chunk_pt_torch = page_table_torch[:, start_block:end_block]
        chunk_pt_dev = _to_device_page_table(chunk_pt_torch, model.mesh_device)

        # This chunk's embeddings [1,1,chunk_size,hidden] (bounded: only one chunk resident).
        chunk_embeds = ttnn.slice(prompt_embeds, [0, 0, start, 0], [1, 1, end, hidden])

        rope = _chunk_rope_mats(model, start, chunk_size)

        # Only the final chunk needs full logits (lm_head over the whole chunk).
        # Earlier chunks write KV only; get_last_token=0 slices a single 32-row
        # tile before lm_head (get_last_token=None would crash the last-token
        # slice), keeping the vocab matmul cheap. (get_last_token != -1 both here
        # and there — the None sentinel is never passed to the backbone.)
        want_logits = is_last and return_last_logits
        get_last = -1 if want_logits else 0

        global _CHUNK_CTX
        _CHUNK_CTX = _ChunkContext(chunk_start_idx=start, chunk_page_table=chunk_pt_dev, sliding_state=sliding_state)
        try:
            with _swap_prefill_attention():
                out = model(
                    chunk_embeds,
                    rope_mats=rope,
                    is_decode=False,
                    page_table=full_pt_dev,
                    kv_caches=kv_cache,
                    input_ids_torch=input_ids_torch[:, start:end],
                    embeds_torch=embeds_torch[:, start:end, :] if embeds_torch is not None else None,
                    get_last_token=get_last,
                    batch_size=1,
                    user_id=user_id,
                )
        finally:
            _CHUNK_CTX = None
            chunk_pt_dev.deallocate(True)
            chunk_embeds.deallocate(True)

        if want_logits:
            logits = out
        elif out is not None:
            out.deallocate(True)

    sliding_state.release()
    full_pt_dev.deallocate(True)
    return logits
