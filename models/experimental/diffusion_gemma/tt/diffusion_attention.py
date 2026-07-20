# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local denoise attention.

The DiffusionGemma denoise pass is **bidirectional and read-only on the KV cache**:
canvas queries attend to a frozen prompt prefix (concatenated in front of the
canvas K/V) plus the canvas itself, and nothing is written back into the
committed cache. That write discipline is the ``DENOISE_READONLY`` phase (see
:mod:`models.experimental.diffusion_gemma.kv_phase`).

Rather than thread non-causal / prefix-KV / RoPE-offset knobs through the shared
``models.demos.gemma4`` attention op (which would touch the production causal
path), this module owns the denoise SDPA itself and **reuses Gemma4's importable
building blocks** (QKV projection, head split, per-head norm, RoPE, head concat,
output projection, all-reduce). The prefill-write and commit-append phases are
plain Gemma4 prefill/decode and need no diffusion-specific code.

It operates on a live ``Gemma4Attention`` instance, reading its ``weights`` /
``config`` / ``mesh_config`` / ``ccl_manager`` — so the decoder weights are shared
with the backbone, only the attention math differs.
"""

from __future__ import annotations

import os

from loguru import logger
import ttnn

from models.demos.gemma4.tt.attention.operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    concat_heads,
    split_qkv_heads_prefill,
)

TILE_SIZE = 32

# The staged-GQA fallback is expected on QB2 (the ttnn SDPA kernel misses L1 by
# less than one tile on the first real adapter layer). The C++ side still emits a
# caught ``TT_THROW`` before the Python fallback engages, so we log once per
# process to label that noise as benign rather than a real failure.
_FALLBACK_WARNED = False
_FALLBACK_COUNTS = {}


def reset_sdpa_fallback_counts():
    _FALLBACK_COUNTS.clear()


def get_sdpa_fallback_counts():
    return dict(_FALLBACK_COUNTS)


def validate_q_rope_offset(q_rope_offset: int) -> None:
    if q_rope_offset % TILE_SIZE != 0:
        raise ValueError(f"q_rope_offset must be a multiple of {TILE_SIZE}, got {q_rope_offset}")


def _largest_tile_divisor(length, preferred):
    """Pick a tile-multiple chunk size that divides ``length``."""
    start = (min(preferred, length) // TILE_SIZE) * TILE_SIZE
    for candidate in range(start, TILE_SIZE - 1, -TILE_SIZE):
        if length % candidate == 0:
            return candidate
    return TILE_SIZE


def _denoise_sdpa_program_config(head_dim, q_seq_len, k_seq_len):
    """SDPAProgramConfig for the rectangular non-causal denoise SDPA.

    Mirrors the Gemma4 prefill tuning (head_dim-dependent grid + default chunks)
    but allows ``q_seq_len != k_seq_len`` (canvas Q vs prompt+canvas K), so chunk
    sizes must independently divide each axis.
    """
    if head_dim >= 512:
        grid = ttnn.CoreCoord(8, 1)
        dq, dk = 32, 32
    else:
        grid = ttnn.CoreCoord(8, 1)
        dq, dk = 32, 32
    q_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_QCHUNK", dq))
    k_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_KCHUNK", dk))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=_largest_tile_divisor(q_seq_len, q_chunk),
        k_chunk_size=_largest_tile_divisor(k_seq_len, k_chunk),
        exp_approx_mode=False,
    )


def _slice_rope_cache(cache, start, length):
    if start % TILE_SIZE != 0:
        raise ValueError(f"RoPE cache start must be a multiple of {TILE_SIZE}, got {start}")
    if start + length > cache.shape[-2]:
        raise ValueError(f"RoPE cache slice [{start}, {start + length}) exceeds cache length {cache.shape[-2]}")
    if start == 0 and cache.shape[-2] == length:
        return cache
    return ttnn.slice(cache, [0, 0, start, 0], [cache.shape[0], cache.shape[1], start + length, cache.shape[3]])


def _apply_rope_chunked(
    tensor,
    cos_cache,
    sin_cache,
    *,
    start_offset: int,
    chunk_size: int = TILE_SIZE,
    head_chunk_size: int = 1,
):
    def apply_rope_dram(chunk, token_index):
        half_dim = chunk.shape[-1] // 2
        cos = _slice_rope_cache(cos_cache, token_index, chunk.shape[-2])
        sin = _slice_rope_cache(sin_cache, token_index, chunk.shape[-2])
        x1 = ttnn.slice(
            chunk,
            [0, 0, 0, 0],
            [chunk.shape[0], chunk.shape[1], chunk.shape[2], half_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x2 = ttnn.slice(
            chunk,
            [0, 0, 0, half_dim],
            [chunk.shape[0], chunk.shape[1], chunk.shape[2], chunk.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        neg_x2 = ttnn.mul(x2, -1.0)
        rotated = ttnn.concat([neg_x2, x1], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_cos = ttnn.mul(chunk, cos)
        r_sin = ttnn.mul(rotated, sin)
        out = ttnn.add(x_cos, r_sin)
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)
        rotated.deallocate(True)
        x_cos.deallocate(True)
        r_sin.deallocate(True)
        if cos is not cos_cache:
            cos.deallocate(True)
        if sin is not sin_cache:
            sin.deallocate(True)
        return out

    seq_len = tensor.shape[-2]
    num_heads = tensor.shape[1]
    if seq_len <= chunk_size and num_heads <= head_chunk_size:
        return apply_rope_dram(tensor, start_offset)

    # Full-canvas RoPE in a single pass. ``apply_rope_dram`` is shape-agnostic, so
    # applying it to the whole ``[1, H, C, hd]`` tensor is bit-identical to the
    # per-(seq,head)-chunk path (verified torch.equal) while replacing ~H*(C/32)
    # tiny slice/concat ops per call with one — a large trace-size + dispatch cut
    # for the 256-token denoise canvas. Opt out with DG_ROPE_FULLCANVAS=0.
    if os.environ.get("DG_ROPE_FULLCANVAS", "1").strip().lower() not in ("0", "false", "no", "off"):
        out = apply_rope_dram(tensor, start_offset)
        tensor.deallocate(True)
        return out

    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        head_chunks = []
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            chunk = ttnn.slice(
                tensor,
                [0, head_start, start, 0],
                [tensor.shape[0], head_end, end, tensor.shape[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            head_chunks.append(apply_rope_dram(chunk, start_offset + start))
            chunk.deallocate(True)
        if len(head_chunks) == 1:
            chunks.append(head_chunks[0])
        else:
            chunks.append(ttnn.concat(head_chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            for chunk in head_chunks:
                chunk.deallocate(True)
    if len(chunks) == 1:
        out = chunks[0]
    else:
        out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            chunk.deallocate(True)
    tensor.deallocate(True)
    return out


def _sdpa_q_chunked(tt_q, tt_k, tt_v, *, attn_mask=None, head_dim, chunk_size: int = TILE_SIZE, layer_idx=None):
    q_seq_len = tt_q.shape[-2]
    k_seq_len = tt_k.shape[-2]
    # Single fused-SDPA call over the full canvas instead of ceil(C/32) python-level
    # q-slices + per-chunk SDPA + concat. The SDPA op still chunks internally via its
    # program_config, so the result is bit-identical while cutting the per-step op
    # count / dispatch. Opt out with DG_SDPA_FULLCANVAS=0.
    if q_seq_len > chunk_size and os.environ.get("DG_SDPA_FULLCANVAS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        chunk_size = q_seq_len
    if q_seq_len <= chunk_size:
        program_config = _denoise_sdpa_program_config(head_dim, q_seq_len, k_seq_len)
        kwargs = {
            "is_causal": False,
            "scale": 1.0,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": program_config,
        }
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        try:
            return ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, **kwargs)
        except RuntimeError as exc:
            if attn_mask is None and _is_sdpa_l1_cb_clash(exc):
                key = int(layer_idx) if layer_idx is not None else -1
                _FALLBACK_COUNTS[key] = _FALLBACK_COUNTS.get(key, 0) + 1
                _warn_sdpa_fallback_once()
                return _manual_gqa_attention(tt_q, tt_k, tt_v)
            raise

    chunks = []
    for start in range(0, q_seq_len, chunk_size):
        end = min(start + chunk_size, q_seq_len)
        q_chunk = ttnn.slice(
            tt_q,
            [0, 0, start, 0],
            [tt_q.shape[0], tt_q.shape[1], end, tt_q.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_chunk = None
        if attn_mask is not None:
            mask_chunk = ttnn.slice(
                attn_mask,
                [0, 0, start, 0],
                [attn_mask.shape[0], attn_mask.shape[1], end, attn_mask.shape[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        chunks.append(
            _sdpa_q_chunked(
                q_chunk,
                tt_k,
                tt_v,
                attn_mask=mask_chunk,
                head_dim=head_dim,
                chunk_size=chunk_size,
                layer_idx=layer_idx,
            )
        )
        q_chunk.deallocate(True)
        if mask_chunk is not None:
            mask_chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    return out


def _is_sdpa_l1_cb_clash(exc: RuntimeError) -> bool:
    message = str(exc)
    return "Statically allocated circular buffers" in message and "clash with L1 buffers" in message


def _warn_sdpa_fallback_once() -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    logger.warning(
        "denoise SDPA hit the known L1 CB clash; using the staged GQA fallback. "
        "The preceding TT_THROW is caught and expected on QB2 (logged once)."
    )


def _manual_gqa_attention(tt_q, tt_k, tt_v):
    """Staged non-causal GQA fallback for small denoise chunks.

    The SDPA kernel can miss L1 by less than one tile on the first real adapter
    layer. This fallback uses ordinary TTNN ops for the same 32-token Q chunk.
    """
    q_heads = tt_q.shape[1]
    kv_heads = tt_k.shape[1]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(f"unsupported GQA shape q_heads={q_heads}, kv_heads={kv_heads}")

    q_heads_per_kv = q_heads // kv_heads
    outputs = []
    for kv_head in range(kv_heads):
        q_start = kv_head * q_heads_per_kv
        q_end = q_start + q_heads_per_kv
        q_group = ttnn.slice(
            tt_q,
            [0, q_start, 0, 0],
            [tt_q.shape[0], q_end, tt_q.shape[2], tt_q.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_head = ttnn.slice(
            tt_k,
            [0, kv_head, 0, 0],
            [tt_k.shape[0], kv_head + 1, tt_k.shape[2], tt_k.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_head = ttnn.slice(
            tt_v,
            [0, kv_head, 0, 0],
            [tt_v.shape[0], kv_head + 1, tt_v.shape[2], tt_v.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if q_heads_per_kv > 1:
            k_heads = [k_head] + [
                ttnn.clone(k_head, memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(q_heads_per_kv - 1)
            ]
            v_heads = [v_head] + [
                ttnn.clone(v_head, memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(q_heads_per_kv - 1)
            ]
            k_group = ttnn.concat(k_heads, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_group = ttnn.concat(v_heads, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in k_heads[1:]:
                tensor.deallocate(True)
            for tensor in v_heads[1:]:
                tensor.deallocate(True)
            owns_k_group = True
            owns_v_group = True
        else:
            k_group = k_head
            v_group = v_head
            owns_k_group = False
            owns_v_group = False

        # QKt via fused transpose_b (bit-exact vs explicit permute; drops the standalone
        # Kt-materialization Permute op that was ~97% of per-layer device-fw, see #47465).
        scores = ttnn.matmul(q_group, k_group, transpose_b=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        outputs.append(ttnn.matmul(probs, v_group, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        q_group.deallocate(True)
        if owns_k_group:
            k_group.deallocate(True)
        if owns_v_group:
            v_group.deallocate(True)
        scores.deallocate(True)
        probs.deallocate(True)

    if len(outputs) == 1:
        out = outputs[0]
    else:
        out = ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for tensor in outputs:
            tensor.deallocate(True)
    return out


def denoise_attention(
    attn,
    hidden_states,
    *,
    rope_mats,
    attn_mask=None,
    kv_hidden_states=None,
    prefix_kv=None,
    q_rope_offset=0,
):
    """Read-only bidirectional denoise attention for one decoder layer.

    Args:
        attn: a ``Gemma4Attention`` instance (shared backbone weights/config).
        hidden_states: canvas hidden states ``[1, 1, C, H]`` (already normed by
            the caller's ``input_layernorm``).
        rope_mats: ``(cos_cache, sin_cache)`` for this layer, sliced/offset by the
            caller to cover ``q_rope_offset + C``.
        attn_mask: optional additive ``[1, 1, C, P+C]`` mask. ``None`` = all-attend.
        kv_hidden_states: optional ``[1, 1, P+C, H]`` source to recompute K/V from
            (legacy hidden-state shim); mutually exclusive with ``prefix_kv``.
        prefix_kv: optional projected frozen prompt ``(K, V)`` heads concatenated in
            front of the canvas K/V (the cache-shaped production path).
        q_rope_offset: absolute RoPE position of the first canvas token (``prompt_len``).

    Returns:
        Attention output for the canvas positions ``[1, 1, C, H]``.
    """
    if kv_hidden_states is not None and prefix_kv is not None:
        raise ValueError("pass at most one of kv_hidden_states or prefix_kv")
    validate_q_rope_offset(q_rope_offset)

    weights = attn.weights
    config = attn.config
    mesh_config = attn.mesh_config
    ccl_manager = attn.ccl_manager
    cos_cache, sin_cache = rope_mats
    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )
    xqkv.deallocate(True)
    if kv_hidden_states is not None:
        xqkv_kv = apply_qkv_projection(kv_hidden_states, weights)
        tt_kv_q, tt_k_from_kv, tt_v_from_kv = split_qkv_heads_prefill(
            xqkv_kv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        xqkv_kv.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_kv_q.deallocate(True)
        tt_k, tt_v = tt_k_from_kv, tt_v_from_kv

    raw_q, raw_k, raw_v = tt_q, tt_k, tt_v
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
    raw_q.deallocate(True)
    raw_k.deallocate(True)
    raw_v.deallocate(True)

    tt_q = _apply_rope_chunked(tt_q, cos_cache, sin_cache, start_offset=q_rope_offset)

    # When K is just the canvas (kv_hidden recomputes the full prompt+canvas, so its
    # RoPE starts at 0), only the canvas-only K path needs the prompt_len offset.
    k_rope_offset = q_rope_offset if prefix_kv is not None else 0
    tt_k = _apply_rope_chunked(tt_k, cos_cache, sin_cache, start_offset=k_rope_offset)

    if prefix_kv is not None:
        prefix_k, prefix_v = prefix_kv
        canvas_k, canvas_v = tt_k, tt_v
        prefix_k_concat = ttnn.to_memory_config(prefix_k, canvas_k.memory_config())
        prefix_v_concat = ttnn.to_memory_config(prefix_v, canvas_v.memory_config())
        tt_k = ttnn.concat([prefix_k_concat, canvas_k], dim=2)
        tt_v = ttnn.concat([prefix_v_concat, canvas_v], dim=2)
        if prefix_k_concat is not prefix_k:
            prefix_k_concat.deallocate(True)
        if prefix_v_concat is not prefix_v:
            prefix_v_concat.deallocate(True)
        canvas_k.deallocate(True)
        canvas_v.deallocate(True)

    q_old, k_old, v_old = tt_q, tt_k, tt_v
    tt_q_dram = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    if tt_q_dram is not tt_q:
        q_old = tt_q
    tt_q = tt_q_dram
    tt_k_dram = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
    if tt_k_dram is not tt_k:
        k_old = tt_k
    tt_k = tt_k_dram
    tt_v_dram = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
    if tt_v_dram is not tt_v:
        v_old = tt_v
    tt_v = tt_v_dram

    tt_sdpa = _sdpa_q_chunked(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=attn_mask,
        head_dim=config.head_dim,
        layer_idx=getattr(attn, "layer_idx", None),
    )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)
    if q_old is not tt_q:
        q_old.deallocate(True)
    if k_old is not tt_k:
        k_old.deallocate(True)
    if v_old is not tt_v:
        v_old.deallocate(True)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_sdpa.deallocate(True)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)
    return tt_out
