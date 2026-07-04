# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched (single-prefill) commit-append for DiffusionGemma (#47557).

The generation loop commits a denoised canvas into the frozen Gemma4 KV cache.
The baseline path (:func:`models.experimental.diffusion_gemma.tt.generate.commit_canvas_tokens`)
does this with **256 sequential single-token decode-appends** — one full 30-layer
decode forward per committed token (~31.5 s / 256-token block on QB2). That is
mathematically a causal prefill of the 256 committed tokens: committed token ``i``
attends to the frozen prefix (prompt + prior blocks) plus canvas tokens ``0..i``,
exactly the ``is_causal`` prefill pattern.

This module collapses those 256 forwards into **one causal masked prefill** over
the whole 256-token canvas, reusing the already-validated denoise attention math
(``diffusion_attention`` building blocks: shared Gemma4 QKV projection, per-head
norm, RoPE at the absolute canvas position, GQA SDPA + the L1 fallback) with two
changes vs the read-only bidirectional denoise pass:

1. a **causal** prefix+canvas mask instead of the all-attend / bidirectional mask;
2. the canvas K/V is **written** into the frozen cache at the committed positions
   (the denoise pass is read-only and never writes).

Design: **write-then-read-from-cache**, per layer, in layer order:
  * compute canvas Q (per-head norm + RoPE at ``start_pos``);
  * for a non-shared layer, compute canvas K/V (per-head norm + RoPE), and write it
    into ``tt_kv_cache[i]`` at seq positions ``start_pos .. start_pos+C-1``;
  * read the full ``[0 : start_pos+C]`` K/V back from the cache (= frozen prefix ++
    freshly-written canvas) and run a causal-masked SDPA;
  * the MLP / MoE / norm tail is byte-identical to the denoise layer body.

Reading the K/V for the SDPA out of the cache (rather than concatenating a
register copy) means cross-layer **KV-sharing** is handled for free: a shared
layer skips its own K/V write, and its earlier source layer has already written
the canvas K/V into the shared cache tensor by the time the shared layer runs.

This path is **opt-in and guarded** (``DG_COMMIT_BATCHED`` / ``commit_fn``); the
sequential path stays the default until this is validated on device
(``doc/optimize_perf/verify_commit_batching.py``). It never edits shared
``models/demos/gemma4`` code — it composes over the importable Gemma4 ops, exactly
like ``tt/commit_decode.py`` and ``tt/diffusion_attention.py``.
"""

from __future__ import annotations

import os

from loguru import logger
import torch
import ttnn

from models.demos.gemma4.tt.attention.operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    concat_heads,
    split_qkv_heads_prefill,
)
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    _chunked_norm_forward,
    _denoise_moe_forward,
)
from models.experimental.diffusion_gemma.tt.diffusion_attention import (
    TILE_SIZE,
    _apply_rope_chunked,
    _denoise_sdpa_program_config,
    _is_sdpa_l1_cb_clash,
    _warn_sdpa_fallback_once,
    validate_q_rope_offset,
)

NEG = -1.0e9

# Default KV-write granularity for the contiguous (page_table=None) cache. 1 =
# per-position writes with the exact op the sequential path uses (proven; the
# safe default while this path is device-unvalidated). >1 uses the batched
# 1-block-paged write (see ``_write_canvas_kv_contiguous``); opt in with
# ``DG_COMMIT_WRITE_BATCH`` once device-validated.
_DEFAULT_WRITE_BATCH = int(os.environ.get("DG_COMMIT_WRITE_BATCH", "1"))


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_commit_causal_mask(
    mesh_device,
    *,
    prefix_len: int,
    canvas_len: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    dtype=ttnn.bfloat16,
):
    """Build the ``[1, 1, C, prefix_len + C]`` causal commit mask on device.

    ``prefix_len`` is the number of frozen positions in front of the canvas
    (``start_pos`` = prompt + all previously committed blocks). The mask is causal:
    canvas query ``i`` attends the whole prefix plus canvas ``0..i`` (and, on a
    sliding layer, only the last ``sliding_window`` positions).
    """
    mask = build_canvas_denoise_mask(
        prefix_len,
        canvas_len,
        layer_type=layer_type,
        sliding_window=sliding_window,
        causal=True,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prefix_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _layer_type_for_commit(tt_model, layer_idx: int) -> str | None:
    layer_types = getattr(getattr(tt_model, "hf_config", None), "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx]
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    return getattr(attn_config, "layer_type", None)


def _sliding_window_for_commit(tt_model, layer_idx: int) -> int | None:
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    window = getattr(attn_config, "sliding_window", None)
    if window is not None:
        return window
    return getattr(getattr(tt_model, "hf_config", None), "sliding_window", None)


def _read_cache_kv(kv_cache, *, end_pos: int):
    """Read the frozen prefix ++ freshly-written canvas K/V ``[0 : end_pos]``.

    ``ttnn.slice`` over the contiguous ``[B, heads, max_seq, head_dim]`` cache; the
    seq bound must be tile-aligned (guaranteed: ``start_pos`` and ``canvas_len`` are
    both multiples of 32, so ``end_pos = start_pos + canvas_len`` is too).
    """
    if end_pos % ttnn.TILE_SIZE != 0:
        raise ValueError(f"cache read end_pos must be a multiple of {ttnn.TILE_SIZE}, got {end_pos}")
    k_cache, v_cache = kv_cache
    starts = [0, 0, 0, 0]
    k_ends = [k_cache.shape[0], k_cache.shape[1], end_pos, k_cache.shape[3]]
    v_ends = [v_cache.shape[0], v_cache.shape[1], end_pos, v_cache.shape[3]]
    return (
        ttnn.slice(k_cache, starts, k_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.slice(v_cache, starts, v_ends, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )


def _write_canvas_kv_contiguous(
    k_cache,
    v_cache,
    canvas_k,
    canvas_v,
    *,
    start_pos: int,
    canvas_len: int,
    mesh_device,
    write_batch: int = _DEFAULT_WRITE_BATCH,
):
    """Write canvas K/V ``[1, nkv, C, hd]`` into a contiguous cache at ``start_pos``.

    ``write_batch == 1`` (default, safe): one ``paged_update_cache`` per committed
    position — the exact non-paged decode-append op the sequential path uses, so
    the write positions and cache layout are provably identical. A single-sequence
    contiguous cache (batch 1) can only address one seq position per non-paged
    update, so this is the only single-op-count option there.

    ``write_batch > 1`` (opt-in, device-unvalidated): treat the contiguous cache
    ``[1, nkv, max_seq, hd]`` as a 1-block paged cache (``block_size = max_seq``) and
    write ``write_batch`` positions per op — each "user" maps to one canvas slot of
    block 0 via ``page_table`` all-zeros and ``update_idxs = start_pos + slot``.
    Fewer dispatches; the semantics (multiple users → distinct slots of one block)
    must be confirmed on device before making this the default.
    """
    nkv = canvas_k.shape[1]
    hd = canvas_k.shape[3]
    # [1, nkv, C, hd] -> [1, C, nkv, hd] so slot dim is the update-cache batch dim.
    k_perm = ttnn.transpose(canvas_k, 1, 2)
    v_perm = ttnn.transpose(canvas_v, 1, 2)

    if write_batch <= 1:
        for t in range(canvas_len):
            kb = ttnn.slice(k_perm, [0, t, 0, 0], [1, t + 1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            vb = ttnn.slice(v_perm, [0, t, 0, 0], [1, t + 1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.paged_update_cache(k_cache, kb, update_idxs=[start_pos + t])
            ttnn.experimental.paged_update_cache(v_cache, vb, update_idxs=[start_pos + t])
            kb.deallocate(True)
            vb.deallocate(True)
        k_perm.deallocate(True)
        v_perm.deallocate(True)
        return

    max_seq = k_cache.shape[2]
    for c0 in range(0, canvas_len, write_batch):
        c1 = min(c0 + write_batch, canvas_len)
        n = c1 - c0
        kb = ttnn.slice(k_perm, [0, c0, 0, 0], [1, c1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vb = ttnn.slice(v_perm, [0, c0, 0, 0], [1, c1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idxs = ttnn.from_torch(
            torch.arange(start_pos + c0, start_pos + c1, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        page_table = ttnn.from_torch(
            torch.zeros((n, 1), dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        ttnn.experimental.paged_update_cache(
            k_cache, kb, update_idxs_tensor=idxs, page_table=page_table, block_size=max_seq
        )
        ttnn.experimental.paged_update_cache(
            v_cache, vb, update_idxs_tensor=idxs, page_table=page_table, block_size=max_seq
        )
        for tensor in (kb, vb, idxs, page_table):
            tensor.deallocate(True)
    k_perm.deallocate(True)
    v_perm.deallocate(True)


def _write_canvas_kv_paged(
    k_cache,
    v_cache,
    canvas_k,
    canvas_v,
    *,
    start_pos: int,
    canvas_len: int,
    page_table,
    config,
    weights,
    tp: int,
):
    """Write canvas K/V into a paged cache with one ``paged_fill_cache`` per K/V.

    ``paged_fill_cache`` fills logical positions ``0 .. C-1`` of the *given* page
    table into the physical blocks it maps, so a chunk page table rolled to the
    block containing ``start_pos`` writes the canvas at absolute positions
    ``start_pos .. start_pos+C-1``. Requires ``start_pos`` to be block-aligned.
    """
    from models.demos.gemma4.tt.attention.operations import effective_block_size

    num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
    eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
    if start_pos % eff_bs != 0:
        raise ValueError(
            f"paged batched commit requires start_pos ({start_pos}) aligned to block_size ({eff_bs}); "
            "use the contiguous write path or fall back to the sequential commit"
        )
    start_block = start_pos // eff_bs
    chunk_page_table = ttnn.slice(
        page_table,
        [0, start_block],
        [page_table.shape[0], page_table.shape[1]],
        memory_config=page_table.memory_config(),
    )
    paged_modulo_kwargs = (
        {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
    )
    ttnn.experimental.paged_fill_cache(
        k_cache, canvas_k, chunk_page_table, batch_idx=0, block_size=eff_bs, **paged_modulo_kwargs
    )
    ttnn.experimental.paged_fill_cache(
        v_cache, canvas_v, chunk_page_table, batch_idx=0, block_size=eff_bs, **paged_modulo_kwargs
    )
    chunk_page_table.deallocate(True)


def _manual_gqa_attention_masked(tt_q, tt_k, tt_v, attn_mask):
    """Staged GQA fallback that honors an additive ``[1, 1, Cq, K]`` mask.

    Mirrors ``diffusion_attention._manual_gqa_attention`` (used when the ttnn SDPA
    kernel misses L1 by < 1 tile), adding the mask to the scores before softmax so
    the commit's causal / sliding visibility is preserved on the fallback path.
    """
    q_heads = tt_q.shape[1]
    kv_heads = tt_k.shape[1]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(f"unsupported GQA shape q_heads={q_heads}, kv_heads={kv_heads}")
    q_heads_per_kv = q_heads // kv_heads
    outputs = []
    for kv_head in range(kv_heads):
        q_start = kv_head * q_heads_per_kv
        q_group = ttnn.slice(
            tt_q,
            [0, q_start, 0, 0],
            [tt_q.shape[0], q_start + q_heads_per_kv, tt_q.shape[2], tt_q.shape[3]],
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
            owns_group = True
        else:
            k_group = k_head
            v_group = v_head
            owns_group = False

        scores = ttnn.matmul(q_group, k_group, transpose_b=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if attn_mask is not None:
            masked = ttnn.add(scores, attn_mask)
            scores.deallocate(True)
            scores = masked
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        outputs.append(ttnn.matmul(probs, v_group, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        q_group.deallocate(True)
        if owns_group:
            k_group.deallocate(True)
            v_group.deallocate(True)
        else:
            k_head.deallocate(True)
            v_head.deallocate(True)
        scores.deallocate(True)
        probs.deallocate(True)

    out = ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for tensor in outputs:
        tensor.deallocate(True)
    return out


def _sdpa_causal_masked(tt_q, tt_k, tt_v, *, attn_mask, head_dim, chunk_size: int = TILE_SIZE):
    """Masked SDPA over canvas Q vs cached K/V, chunked over Q with an L1 fallback.

    Mirrors ``diffusion_attention._sdpa_q_chunked`` but the fallback keeps the
    additive mask (the commit mask is never ``None``), so a sliding / causal layer
    stays correct even when the SDPA kernel hits the known L1 CB clash.
    """
    q_seq_len = tt_q.shape[-2]
    k_seq_len = tt_k.shape[-2]
    if q_seq_len <= chunk_size:
        program_config = _denoise_sdpa_program_config(head_dim, q_seq_len, k_seq_len)
        try:
            return ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=False,
                attn_mask=attn_mask,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program_config,
            )
        except RuntimeError as exc:
            if _is_sdpa_l1_cb_clash(exc):
                _warn_sdpa_fallback_once()
                return _manual_gqa_attention_masked(tt_q, tt_k, tt_v, attn_mask)
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
        mask_chunk = ttnn.slice(
            attn_mask,
            [0, 0, start, 0],
            [attn_mask.shape[0], attn_mask.shape[1], end, attn_mask.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunks.append(
            _sdpa_causal_masked(q_chunk, tt_k, tt_v, attn_mask=mask_chunk, head_dim=head_dim, chunk_size=chunk_size)
        )
        q_chunk.deallocate(True)
        mask_chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    return out


def _commit_attention_batched(
    attn,
    canvas_hidden,
    *,
    rope_mats,
    kv_cache,
    attn_mask,
    start_pos: int,
    canvas_len: int,
    is_kv_shared: bool,
    mesh_device,
    page_table=None,
    write_batch: int = _DEFAULT_WRITE_BATCH,
):
    """Causal masked prefix+canvas attention for one commit layer (writes K/V).

    Reuses the shared Gemma4 building blocks (QKV projection, prefill head split,
    per-head norm, output projection, all-reduce) and the diffusion-local chunked
    RoPE — the same ops as ``diffusion_attention.denoise_attention`` — so the K/V
    written here are computed by the identical projection + norm + RoPE the denoise
    and (per-token) commit paths use.
    """
    validate_q_rope_offset(start_pos)
    weights = attn.weights
    config = attn.config
    mesh_config = attn.mesh_config
    ccl_manager = attn.ccl_manager
    cos_cache, sin_cache = rope_mats
    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(canvas_hidden, weights)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )
    xqkv.deallocate(True)

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_q = _apply_rope_chunked(tt_q, cos_cache, sin_cache, start_offset=start_pos)

    if is_kv_shared:
        # KV-shared layer: the source layer already wrote the canvas K/V into this
        # (shared) cache tensor earlier in the layer loop; do not recompute/write.
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
        tt_k = _apply_rope_chunked(tt_k, cos_cache, sin_cache, start_offset=start_pos)
        k_cache, v_cache = kv_cache
        if page_table is not None:
            _write_canvas_kv_paged(
                k_cache,
                v_cache,
                tt_k,
                tt_v,
                start_pos=start_pos,
                canvas_len=canvas_len,
                page_table=page_table,
                config=config,
                weights=weights,
                tp=tp,
            )
        else:
            _write_canvas_kv_contiguous(
                k_cache,
                v_cache,
                tt_k,
                tt_v,
                start_pos=start_pos,
                canvas_len=canvas_len,
                mesh_device=mesh_device,
                write_batch=write_batch,
            )
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    if page_table is not None:
        raise NotImplementedError(
            "batched commit SDPA read for paged caches is not wired yet; the standalone/serving "
            "RUN path uses the contiguous model-owned cache (page_table=None). Use the sequential "
            "commit for paged/vLLM caches (batched-canvas paged decode is tracked in #47557)."
        )

    # Read the frozen prefix ++ freshly-written canvas out of the cache and run the
    # causal-masked SDPA. Reading from the cache (rather than a register concat)
    # means a KV-shared layer transparently sees the source layer's canvas K/V.
    full_k, full_v = _read_cache_kv(kv_cache, end_pos=start_pos + canvas_len)

    tt_q_dram = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    if tt_q_dram is not tt_q:
        tt_q.deallocate(True)
    tt_q = tt_q_dram

    tt_sdpa = _sdpa_causal_masked(tt_q, full_k, full_v, attn_mask=attn_mask, head_dim=config.head_dim)
    tt_q.deallocate(True)
    full_k.deallocate(True)
    full_v.deallocate(True)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_sdpa.deallocate(True)
    tt_out = apply_output_projection(tt_out, weights)
    return apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)


def _commit_layer_forward_batched(
    tt_model,
    layer_idx,
    hidden_states,
    *,
    kv_cache,
    attn_mask,
    start_pos: int,
    canvas_len: int,
    is_kv_shared: bool,
    page_table=None,
    write_batch: int = _DEFAULT_WRITE_BATCH,
):
    """One commit layer: causal attention (writes K/V) + the denoise MLP/MoE tail.

    The MLP / MoE / norm body is intentionally the exact ``denoise_forward`` layer
    body (same ops, same order) so only the attention differs from the validated
    denoise pass.
    """
    layer = tt_model.layers[layer_idx]
    residual = hidden_states
    normed = _chunked_norm_forward(layer.input_layernorm, hidden_states)
    attn_output = _commit_attention_batched(
        layer.self_attn,
        normed,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=start_pos + canvas_len),
        kv_cache=kv_cache,
        attn_mask=attn_mask,
        start_pos=start_pos,
        canvas_len=canvas_len,
        is_kv_shared=is_kv_shared,
        mesh_device=tt_model.mesh_device,
        page_table=page_table,
        write_batch=write_batch,
    )
    normed.deallocate(True)

    attn_output = _chunked_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _chunked_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    mlp_output = layer.shared_mlp(normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_1, mlp_output)
        mlp_output.deallocate(True)
        expert_input = _chunked_norm_forward(layer.pre_feedforward_layernorm_2, residual)
        expert_output = _denoise_moe_forward(layer.moe, residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_2, expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = _chunked_norm_forward(layer.post_feedforward_layernorm, hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    if layer.layer_scalar != 1.0:
        scaled = ttnn.mul(combined, layer.layer_scalar)
        combined.deallocate(True)
        combined = scaled
    return combined


def commit_hidden_forward_batched(
    tt_model,
    canvas_hidden,
    *,
    start_pos: int,
    kv_caches=None,
    page_table=None,
    write_batch: int = _DEFAULT_WRITE_BATCH,
):
    """Run the full batched commit backbone: append every layer's canvas K/V.

    ``canvas_hidden`` is the ``[1, 1, C, H]`` embedded committed canvas. The K/V
    append happens inside each layer; the returned hidden states are discarded by
    the commit (no final norm / LM head — the commit throws away logits).
    """
    caches = kv_caches or tt_model.tt_kv_cache
    canvas_len = canvas_hidden.shape[-2]
    kv_shared_map = getattr(tt_model, "kv_shared_layer_map", {})
    hidden_states = canvas_hidden
    for layer_idx in range(len(tt_model.layers)):
        layer_type = _layer_type_for_commit(tt_model, layer_idx)
        sliding_window = _sliding_window_for_commit(tt_model, layer_idx) if layer_type == "sliding_attention" else None
        attn_mask = build_device_commit_causal_mask(
            tt_model.mesh_device,
            prefix_len=start_pos,
            canvas_len=canvas_len,
            layer_type=layer_type,
            sliding_window=sliding_window,
        )
        try:
            hidden_states = _commit_layer_forward_batched(
                tt_model,
                layer_idx,
                hidden_states,
                kv_cache=caches[layer_idx] if caches else None,
                attn_mask=attn_mask,
                start_pos=start_pos,
                canvas_len=canvas_len,
                is_kv_shared=layer_idx in kv_shared_map,
                page_table=page_table,
                write_batch=write_batch,
            )
        finally:
            attn_mask.deallocate(True)
    return hidden_states


def commit_canvas_tokens_batched(
    tt_model,
    canvas_tokens: torch.Tensor,
    *,
    start_pos: int,
    page_table=None,
    page_tables_per_layer=None,
    write_batch: int = _DEFAULT_WRITE_BATCH,
) -> None:
    """Append committed canvas token ids to the KV cache in ONE causal prefill.

    Drop-in replacement for
    :func:`models.experimental.diffusion_gemma.tt.generate.commit_canvas_tokens`
    (same signature). Instead of 256 sequential single-token decode-appends, this
    embeds all ``canvas_len`` committed tokens and runs one causal masked prefill
    that writes every layer's K/V at positions ``start_pos .. start_pos+C-1``.

    See the module docstring and ``doc/optimize_perf/`` for the bit-exactness
    argument and the device verify harness. Guarded / opt-in: the sequential path
    stays the default until this is validated on device.
    """
    # Local imports to avoid an import cycle (generate imports this module lazily).
    from models.experimental.diffusion_gemma.tt.generate import (
        _validate_nonnegative_integer_token_tensor,
        _validate_position_span,
        embed_host_tokens,
    )

    _validate_nonnegative_integer_token_tensor(
        canvas_tokens,
        name="canvas_tokens",
        shape_name="[batch, canvas_len]",
    )
    if canvas_tokens.shape[0] != 1:
        raise NotImplementedError("commit_canvas_tokens_batched currently supports batch=1")
    canvas_len = canvas_tokens.shape[1]
    _validate_position_span(start_pos, canvas_len, name="start_pos")
    if start_pos % TILE_SIZE != 0:
        raise ValueError(
            f"batched commit requires start_pos ({start_pos}) to be a multiple of {TILE_SIZE}; "
            "cache_len is padded to 32 and canvas_len is 256, so this holds for the standard run"
        )
    if page_tables_per_layer is not None:
        raise NotImplementedError(
            "batched commit does not support per-layer (hybrid) page tables yet; "
            "use the sequential commit for the vLLM hybrid-cache path"
        )

    # commit_hidden_forward_batched consumes canvas_hidden through the layer stack
    # (layer 0's residual add deallocates it), so the caller does not free it.
    canvas_hidden = embed_host_tokens(tt_model, canvas_tokens)
    hidden = commit_hidden_forward_batched(
        tt_model,
        canvas_hidden,
        start_pos=start_pos,
        page_table=page_table,
        write_batch=write_batch,
    )
    hidden.deallocate(True)


def batched_commit_enabled() -> bool:
    """Whether the opt-in batched commit is enabled via ``DG_COMMIT_BATCHED``."""
    return os.environ.get("DG_COMMIT_BATCHED", "0").lower() in ("1", "true", "yes", "on")


def select_commit_fn(batched: bool | None = None):
    """Return the commit callable: batched when opted in, else sequential.

    ``batched=None`` consults ``DG_COMMIT_BATCHED``. Kept here (not in ``generate``)
    so the default sequential path has no dependency on this module.
    """
    from models.experimental.diffusion_gemma.tt.generate import commit_canvas_tokens

    use_batched = batched_commit_enabled() if batched is None else batched
    if use_batched:
        logger.info("[commit] using batched single-prefill commit (DG_COMMIT_BATCHED)")
        return commit_canvas_tokens_batched
    return commit_canvas_tokens
