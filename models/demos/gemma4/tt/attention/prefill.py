# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import os

import torch

import ttnn

from .operations import (
    PREFILL_SDPA_MAX_SEQ,
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    chunked_prefill_sdpa,
    chunked_prefill_sdpa_sliding,
    concat_heads,
    effective_block_size,
    prefill_sdpa_program_config,
    prefill_short_lived_memcfg,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

TILE_HEIGHT = 32


def _resolve_valid_seq_len_tensor(config, valid_seq_len, padded_seq_len, mesh_device):
    """Resolve the per-request fill length as a device tensor for
    ``paged_fill_cache``'s kernel-side bounded-fill cap, or None to fall back
    to the host-side slice.

    Only meaningful for a bounded (circular) cache.

    * Traced prefill (``valid_seq_len is None`` / ``get_last_token=-1``): use the
      persistent tensor stashed on ``config`` by the model
      (``prefill_valid_len_dev``), refreshed by the generator out-of-trace.
    * Eager opt-in (``GEMMA4_KERNEL_FILL_CAP``): build an inline tensor from the
      known real length. Otherwise the caller host-slices the K/V fill input.
    """
    if config.cache_position_modulo is None:
        return None
    # Prefer the persistent per-request tensor only when the host length is
    # unknown (traced prefill, get_last_token=-1). Eager multi-chunk still uses
    # the host-side slice / inline tensor so each chunk can carry its own length.
    dev = getattr(config, "prefill_valid_len_dev", None)
    if dev is not None and valid_seq_len is None:
        return dev
    if os.environ.get("GEMMA4_KERNEL_FILL_CAP", "0").lower() not in ("1", "true", "yes"):
        return None
    if valid_seq_len is None:
        return None
    # Store the raw real length; the writer kernel rounds up to a whole block.
    real_len = min(valid_seq_len, padded_seq_len)
    if not (0 < real_len < padded_seq_len):
        return None
    return ttnn.from_torch(
        torch.tensor([real_len], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _merge_bounded_boundary_fill(tt_x, valid_seq_len, modulo):
    """Restore wrap-window rows into the newest tile's padding slots.

    Kernel ``skip_tiles`` is tile-granular: when ``V % 32 != 0`` it commits
    padding rows ``[V, ceil_tile)`` into the ring's newest slots and drops the
    oldest ``V % 32`` in-window tokens. Splice those wrap-window rows into the
    padding slots so the filled tile window matches ``[V - modulo, V)``.
    """
    if tt_x is None or valid_seq_len is None or modulo is None:
        return tt_x
    v = int(valid_seq_len)
    mod = int(modulo)
    s = int(tt_x.shape[-2])
    if v <= 0 or v >= s or v % TILE_HEIGHT == 0 or mod <= 0:
        return tt_x
    pad_rows = TILE_HEIGHT - (v % TILE_HEIGHT)
    tile_end = min(v + pad_rows, s)
    pad_rows = tile_end - v
    if pad_rows <= 0:
        return tt_x
    wrap_start = v - mod
    if wrap_start < 0:
        return tt_x
    b, h, _, d = (int(tt_x.shape[i]) for i in range(4))
    head = ttnn.slice(tt_x, [0, 0, 0, 0], [b, h, v, d])
    wrap = ttnn.slice(tt_x, [0, 0, wrap_start, 0], [b, h, wrap_start + pad_rows, d])
    parts = [head, wrap]
    tail = None
    if tile_end < s:
        tail = ttnn.slice(tt_x, [0, 0, tile_end, 0], [b, h, s, d])
        parts.append(tail)
    out = ttnn.concat(parts, dim=2)
    for t in (head, wrap, tail):
        if t is not None:
            t.deallocate(True)
    return out


def flush_deferred_bounded_fills(layers):
    """Merge + ``paged_fill_cache`` for stashed bounded ring fills.

    Must run after lm_head (never mid-layer / between layers and lm_head on TP).
    Intermediate generator chunks leave the stash empty (they skip ring fill).
    """
    for layer in layers:
        cfg = getattr(getattr(layer, "self_attn", None), "config", None)
        if cfg is None:
            continue
        pending = getattr(cfg, "_deferred_bounded_fill", None)
        if not pending:
            continue
        cfg._deferred_bounded_fill = None
        k_fill = pending["k_fill"]
        v_fill = pending["v_fill"]
        k_merged = k_fill
        v_merged = v_fill
        try:
            k_merged = _merge_bounded_boundary_fill(k_fill, pending["valid_seq_len"], pending["modulo"])
            v_merged = _merge_bounded_boundary_fill(v_fill, pending["valid_seq_len"], pending["modulo"])
            ttnn.experimental.paged_fill_cache(
                pending["k_cache"],
                k_merged,
                pending["page_table"],
                batch_idx=pending["user_id"],
                block_size=pending["block_size"],
                **pending["paged_modulo_kwargs"],
            )
            ttnn.experimental.paged_fill_cache(
                pending["v_cache"],
                v_merged,
                pending["page_table"],
                batch_idx=pending["user_id"],
                block_size=pending["block_size"],
                **pending["paged_modulo_kwargs"],
            )
        finally:
            seen = set()
            for t in (k_fill, v_fill, k_merged, v_merged):
                if t is None or id(t) in seen:
                    continue
                seen.add(id(t))
                try:
                    t.deallocate(True)
                except Exception:
                    pass


def _prefill_forward_single(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    page_table=None,
    user_id=0,
    ccl_manager=None,
    shared_kv=None,
    keep_kv=False,
    valid_seq_len=None,
    chunk_start_idx=None,
    chunk_page_table=None,
    sliding_tail_in=None,
):
    """Single-user prefill — matches arg/gemma4_optimizations.

    Generator-level multi-chunk prefill (``chunk_page_table`` not None): the
    current chunk's K/V is written at its absolute blocks via ``chunk_page_table``.
    FULL-attention layers then read the whole prior prefix from the paged cache
    (cross-chunk, via ``chunked_prefill_sdpa`` + ``base_offset``). SLIDING layers
    only look back ``sliding_window`` tokens, so they attend a square
    ``[prev-window tail | current chunk]`` slice: ``sliding_tail_in`` carries the
    previous chunk's last ``sliding_window`` K/V tokens; this call returns the
    current chunk's last ``sliding_window`` K/V as the tail for the next chunk
    (third return value). Single-chunk prefill (``chunk_page_table`` None) is
    unchanged and returns ``sliding_tail_out=None``.

    Returns ``(tt_out, kept_kv, sliding_tail_out)``.
    """
    tp = mesh_config.tp if mesh_config else 1
    is_chunked = chunk_page_table is not None
    # Host int for control-flow (path selection). Device tensor offsets are used
    # for RoPE / chunked SDPA under traced multi-chunk replay.
    if isinstance(chunk_start_idx, ttnn.Tensor):
        chunk_offset = None  # signal: use tensor path; cross-chunk always on
        chunk_offset_tensor = chunk_start_idx
        need_cross_chunk = is_chunked  # sp1 / middle-chunk traced graph
    else:
        chunk_offset = int(chunk_start_idx) if chunk_start_idx is not None else 0
        chunk_offset_tensor = None
        need_cross_chunk = is_chunked and chunk_offset > 0
    # Generator-level chunked prefill on a sliding-window layer (any chunk,
    # including the first). Handled via the in-memory window tail below rather
    # than the full-prefix paged read used for full-attention layers.
    sliding_chunked = is_chunked and config.is_sliding and config.sliding_window is not None
    # KV-shared + generator multi-chunk: current-chunk K/V still arrive via
    # ``shared_kv`` (source layer's keep_kv). Cross-chunk full-attention then
    # reads the source's already-filled paged cache (``need_cross_chunk`` path);
    # sliding layers use the in-memory window tail. Do not hard-error here —
    # forcing single-chunk at 64k+ hangs E2B/E4B warmup on P150x8.
    # Fill the current chunk's K/V at its physical blocks. For a single chunk the
    # chunk table equals the (full) page_table, so behavior is unchanged.
    #
    # Bounded sliding (cache_position_modulo set): the writer wraps absolute
    # positions into the window and looks up page_table[wrapped_block]. The
    # layer's hybrid page_table has the correct physical blocks in that prefix.
    # ``chunk_page_table`` is a slice of the *full-attention* table at absolute
    # offsets — using it here would write sliding K/V into the wrong pool.
    # Full-attention layers still need the absolute chunk slice.
    if config.cache_position_modulo is not None:
        fill_page_table = page_table
    else:
        fill_page_table = chunk_page_table if is_chunked else page_table

    xqkv = apply_qkv_projection(hidden_states, weights)

    # Short-lived prefill activations in L1 when GEMMA4_PREFILL_L1_ACT=1 (Qwen36
    # #48861). o_proj / allreduce stay DRAM (CB clash with CCL).
    act_mc = prefill_short_lived_memcfg()
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv,
        config,
        weights.is_global,
        tp=tp,
        kv_replicated=weights.kv_replicated,
        memory_config=act_mc,
    )

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc)

    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(
            tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc
        )
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=act_mc)

    # RoPE Q (and K, unless KV-shared — then K comes already-RoPE'd from the
    # source layer). A concat(Q,K)->rope->split fusion was evaluated to collapse
    # the two rotary_embedding calls into one, but it adds concat+split device
    # kernels for no throughput benefit (RoPE is ~1% of the step), so Q and K are
    # rotated separately.
    tt_q = apply_rope(tt_q, cos_cache, sin_cache, memory_config=act_mc)
    if shared_kv is None:
        tt_k = apply_rope(tt_k, cos_cache, sin_cache, memory_config=act_mc)

    if kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
            paged_modulo_kwargs = (
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            )
            if config.cache_position_modulo is not None:
                # Bounded ring fill: never merge/where/paged_fill mid-forward on TP
                # (corrupts token-0). Eager last chunk (valid_seq_len known): stash
                # a tile-ceil K/V clone; model flushes after lm_head. Intermediate
                # chunks (valid_seq_len None + chunked): skip — last chunk overwrites
                # the ring. Traced path (valid_seq_len None, non-chunked / kernel
                # cap): may still kernel-cap-fill in-graph.
                if valid_seq_len is not None:
                    v = min(int(valid_seq_len), int(tt_k.shape[-2]))
                    tile_end = ((v + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
                    tile_end = min(tile_end, int(tt_k.shape[-2]))
                    if tile_end <= 0:
                        pass
                    else:
                        # Drop any prior stash (e.g. re-run) before cloning.
                        old = getattr(config, "_deferred_bounded_fill", None)
                        if old:
                            for t in (old.get("k_fill"), old.get("v_fill")):
                                if t is not None:
                                    try:
                                        t.deallocate(True)
                                    except Exception:
                                        pass
                            config._deferred_bounded_fill = None
                        if tile_end < int(tt_k.shape[-2]):
                            k_slice = ttnn.slice(
                                tt_k,
                                [0, 0, 0, 0],
                                [tt_k.shape[0], tt_k.shape[1], tile_end, tt_k.shape[3]],
                            )
                            v_slice = ttnn.slice(
                                tt_v,
                                [0, 0, 0, 0],
                                [tt_v.shape[0], tt_v.shape[1], tile_end, tt_v.shape[3]],
                            )
                            k_stash = ttnn.clone(k_slice)
                            v_stash = ttnn.clone(v_slice)
                            k_slice.deallocate(True)
                            v_slice.deallocate(True)
                        else:
                            k_stash = ttnn.clone(tt_k)
                            v_stash = ttnn.clone(tt_v)
                        config._deferred_bounded_fill = {
                            "k_cache": k_cache,
                            "v_cache": v_cache,
                            "k_fill": k_stash,
                            "v_fill": v_stash,
                            "page_table": fill_page_table,
                            "user_id": user_id,
                            "block_size": eff_bs,
                            "paged_modulo_kwargs": paged_modulo_kwargs,
                            "valid_seq_len": v,
                            "modulo": int(config.cache_position_modulo),
                        }
                elif not is_chunked:
                    # Traced / single-chunk with get_last_token=-1: kernel-cap fill.
                    k_fill, v_fill = tt_k, tt_v
                    fill_kwargs = {}
                    valid_dev = _resolve_valid_seq_len_tensor(config, valid_seq_len, tt_k.shape[-2], k_cache.device())
                    if valid_dev is not None:
                        fill_kwargs["valid_seq_len_tensor"] = valid_dev
                    ttnn.experimental.paged_fill_cache(
                        k_cache,
                        k_fill,
                        fill_page_table,
                        batch_idx=user_id,
                        block_size=eff_bs,
                        **paged_modulo_kwargs,
                        **fill_kwargs,
                    )
                    ttnn.experimental.paged_fill_cache(
                        v_cache,
                        v_fill,
                        fill_page_table,
                        batch_idx=user_id,
                        block_size=eff_bs,
                        **paged_modulo_kwargs,
                        **fill_kwargs,
                    )
                    if valid_dev is not None and valid_dev is not getattr(config, "prefill_valid_len_dev", None):
                        valid_dev.deallocate(True)
                # else: intermediate multi-chunk — skip ring (last chunk overwrites)
            else:
                # Unbounded: previous in-forward fill.
                k_fill, v_fill = tt_k, tt_v
                ttnn.experimental.paged_fill_cache(
                    k_cache,
                    k_fill,
                    fill_page_table,
                    batch_idx=user_id,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
                ttnn.experimental.paged_fill_cache(
                    v_cache,
                    v_fill,
                    fill_page_table,
                    batch_idx=user_id,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # 6. SDPA (causal prefill, scale=1.0)
    # The non-chunked SDPA silently returns WRONG results at seq_len >= 32768
    # (2^15) — generation degrades to garbage. The cliff is INCLUSIVE of 32768:
    # a power-of-2-padded prompt that lands exactly on 32768 is already broken
    # (empirically garbage at 32768, coherent at 16384 and — via chunking — at
    # 65536). So chunk whenever seq_len >= PREFILL_SDPA_MAX_SEQ:
    #   - full-attention layers: chunk Q and attend the full K prefix from the
    #     (already-filled) paged cache via chunked_scaled_dot_product_attention.
    #   - sliding-window layers: that op is causal-only, so use an overlapping
    #     windowed chunking over the in-memory K/V (each slice stays <32768).
    # Both stay correct at/above 32768 and reduce to the non-chunked op below it.
    #
    # KV-shared layers still take the paged chunked path: ``kv_cache`` points at
    # the source layer's already-filled cache. Do NOT require ``shared_kv is None``
    # here — that used to fall through to non-chunked SDPA and produce garbage
    # (e.g. trailing "la la la") on E2B/E4B long-context.
    seq_len = tt_q.shape[-2]
    long_seq = seq_len >= PREFILL_SDPA_MAX_SEQ
    sliding_window = config.sliding_window if config.is_sliding else None
    sliding_tail_out = None
    if sliding_chunked:
        # Generator-level chunked prefill, sliding-window layer. The chunked
        # paged SDPA op has no window mask, so attend a SQUARE [tail | chunk]
        # slice: prepend the previous chunk's last ``sliding_window`` K/V tokens
        # (sliding_tail_in) and run the normal causal + sliding-window SDPA. Q is
        # padded in front by ``hist`` filler rows (a copy of the chunk's leading
        # rows) so Q.seq == K.seq (the op requires square); those rows' outputs
        # are dropped and — being causal — never influence the kept rows. Kept
        # rows [hist, hist+seq_len) are query positions [chunk_offset,
        # chunk_offset+seq_len) with their full window covered. The current
        # chunk's last ``sliding_window`` K/V become next chunk's tail.
        sdpa_ckc = ttnn.init_device_compute_kernel_config(
            tt_q.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        hist = ((sliding_window + 31) // 32) * 32
        use_persistent_tail = isinstance(chunk_start_idx, ttnn.Tensor)
        if sliding_tail_in is not None:
            k_tail, v_tail = sliding_tail_in
            nqh = tt_q.shape[1]
            # Filler Q rows (outputs discarded); reuse the chunk's leading rows.
            q_pad = ttnn.slice(tt_q, [0, 0, 0, 0], [1, nqh, hist, config.head_dim])
            q_cat = ttnn.concat([q_pad, tt_q], dim=2)
            k_cat = ttnn.concat([k_tail, tt_k], dim=2)
            v_cat = ttnn.concat([v_tail, tt_v], dim=2)
            q_pad.deallocate(True)
            sdpa_full = ttnn.transformer.scaled_dot_product_attention(
                q_cat,
                k_cat,
                v_cat,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
                program_config=prefill_sdpa_program_config(config.head_dim, hist + seq_len),
                compute_kernel_config=sdpa_ckc,
            )
            q_cat.deallocate(True)
            k_cat.deallocate(True)
            v_cat.deallocate(True)
            # Persistent ring (traced multi-chunk): keep the buffer addresses so
            # execute_trace can refresh them via ttnn.copy below. Eager path
            # frees the previous chunk's tail.
            if not use_persistent_tail:
                k_tail.deallocate(True)
                v_tail.deallocate(True)
            tt_sdpa = ttnn.slice(sdpa_full, [0, 0, hist, 0], [1, nqh, hist + seq_len, config.head_dim])
            sdpa_full.deallocate(True)
        else:
            # First chunk (chunk_offset==0): no history, window lies inside the
            # chunk (seq_len=chunk_size >= sliding_window). Standard windowed SDPA.
            tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
                program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
                compute_kernel_config=sdpa_ckc,
            )
        # Save this chunk's last ``hist`` K/V tokens as the next chunk's tail.
        kseq = tt_k.shape[-2]
        nkv = tt_k.shape[1]
        tail_start = max(0, kseq - hist)
        k_tail_out = ttnn.slice(tt_k, [0, 0, tail_start, 0], [1, nkv, kseq, config.head_dim])
        v_tail_out = ttnn.slice(tt_v, [0, 0, tail_start, 0], [1, nkv, kseq, config.head_dim])
        if use_persistent_tail:
            # Bind fixed DRAM addresses into the graph so middle-chunk replay
            # sees the previous chunk's window without re-running Python.
            # Prefer the capture-time buffers stashed on config so replay copies
            # into the same addresses the trace recorded.
            persistent = getattr(config, "sliding_prefill_tail_persistent", None)
            if persistent is not None:
                persistent_k, persistent_v = persistent
            elif sliding_tail_in is not None:
                persistent_k, persistent_v = sliding_tail_in
            else:
                persistent_k = ttnn.clone(k_tail_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                persistent_v = ttnn.clone(v_tail_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            config.sliding_prefill_tail_persistent = (persistent_k, persistent_v)
            ttnn.copy(k_tail_out, persistent_k)
            ttnn.copy(v_tail_out, persistent_v)
            k_tail_out.deallocate(True)
            v_tail_out.deallocate(True)
            sliding_tail_out = (persistent_k, persistent_v)
        else:
            sliding_tail_out = (k_tail_out, v_tail_out)
    elif need_cross_chunk:
        # Full-attention chunk N>0: attend the full prefix already filled in the
        # paged cache. base_offset shifts the causal window to this chunk's
        # absolute positions [chunk_offset, chunk_offset+seq_len).
        k_cache, v_cache = kv_cache
        nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        tt_sdpa = chunked_prefill_sdpa(
            tt_q,
            k_cache,
            v_cache,
            page_table,
            user_id,
            config.head_dim,
            scale=1.0,
            base_offset=chunk_offset_tensor if chunk_offset_tensor is not None else chunk_offset,
            num_kv_heads=nkv_local,
        )
    elif long_seq and config.is_sliding and sliding_window is not None:
        tt_sdpa = chunked_prefill_sdpa_sliding(tt_q, tt_k, tt_v, sliding_window, config.head_dim, scale=1.0)
    elif long_seq and not config.is_sliding and page_table is not None and kv_cache is not None:
        # Full-attention long context (incl. KV-shared layers whose kv_cache is
        # the source layer's already-filled pool).
        k_cache, v_cache = kv_cache
        nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        tt_sdpa = chunked_prefill_sdpa(
            tt_q, k_cache, v_cache, page_table, user_id, config.head_dim, scale=1.0, num_kv_heads=nkv_local
        )
    elif long_seq:
        raise RuntimeError(
            f"Gemma4 long-context prefill (seq_len={seq_len} >= {PREFILL_SDPA_MAX_SEQ}) requires "
            f"chunked SDPA, but no valid path was selected "
            f"(is_sliding={config.is_sliding}, page_table={page_table is not None}, "
            f"kv_cache={kv_cache is not None}, shared_kv={shared_kv is not None}). "
            f"Non-chunked SDPA silently returns garbage above this length."
        )
    else:
        # HiFi4 + FP32 dest-acc SDPA: restore the softmax-reduce precision #47311 removed
        # (it dropped the reduce's forced-FP32 accumulation). fp32_dest_acc is safe on the
        # prefill SDPA op (unlike the decode op, where it halves dest for head_dim=512).
        sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            tt_q.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=sliding_window,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
            compute_kernel_config=sdpa_compute_kernel_config,
        )
    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False, memory_config=act_mc)
    # o_proj + allreduce need DRAM activations (CCL / matmul CB pressure).
    if act_mc == ttnn.L1_MEMORY_CONFIG:
        tt_out_l1 = tt_out
        tt_out = ttnn.to_memory_config(tt_out_l1, ttnn.DRAM_MEMORY_CONFIG)
        tt_out_l1.deallocate(True)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out, kept_kv, sliding_tail_out


def prefill_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
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
    chunk_start_idx=None,
    chunk_page_table=None,
    sliding_tail_in=None,
):
    """
    Multi-token prefill attention, fully on device.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] or [B, 1, S, hidden_size] on device
        batch_size: padded batch for batched prefill (1 for single-user / test_full_model)
        chunk_start_idx: absolute start position of this generator-level prefill
            chunk (None/0 for single-chunk prefill). When >0 the K/V of prior
            chunks already sit in the paged cache and cross-chunk attention must
            read them; ``chunk_page_table`` maps the current chunk's tokens to
            their physical blocks for the offset KV fill.
        chunk_page_table: per-user page-table slice for the current chunk's
            blocks (used for the offset ``paged_fill_cache``). None => single chunk.
        sliding_tail_in: previous chunk's last ``sliding_window`` K/V for
            sliding-window layers under generator chunking (None otherwise).

    Returns ``(tt_out, kept_kv, sliding_tail_out)``; the batched path returns
    ``sliding_tail_out=None`` (it does not chunk the sequence).
    """
    if batch_size <= 1:
        return _prefill_forward_single(
            hidden_states,
            cos_cache,
            sin_cache,
            weights,
            kv_cache,
            config,
            mesh_config,
            page_table=page_table,
            user_id=user_id,
            ccl_manager=ccl_manager,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            valid_seq_len=valid_seq_len,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
            sliding_tail_in=sliding_tail_in,
        )

    tp = mesh_config.tp if mesh_config else 1
    hidden_states = ttnn.reshape(
        hidden_states, [1, 1, hidden_states.shape[-2] * hidden_states.shape[-3] * hidden_states.shape[0], -1]
    )

    seq_len = hidden_states.shape[-2]
    original_seq_len = seq_len

    xqkv = apply_qkv_projection(hidden_states, weights)
    ttnn.deallocate(hidden_states)

    xqkv = ttnn.reshape(xqkv, [batch_size, 1, seq_len // batch_size, -1])
    seq_len_per_user = seq_len // batch_size

    act_mc = prefill_short_lived_memcfg()
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv,
        config,
        weights.is_global,
        tp=tp,
        kv_replicated=weights.kv_replicated,
        memory_config=act_mc,
    )
    ttnn.deallocate(xqkv)

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc)

    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(
            tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc
        )
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=act_mc)

    # RoPE Q (and K, unless KV-shared — then K comes already-RoPE'd from the
    # source layer). A concat(Q,K)->rope->split fusion was evaluated to collapse
    # the two rotary_embedding calls into one, but it adds concat+split device
    # kernels for no throughput benefit (RoPE is ~1% of the step), so Q and K are
    # rotated separately.
    tt_q = apply_rope(tt_q, cos_cache, sin_cache, memory_config=act_mc)
    if shared_kv is None:
        tt_k = apply_rope(tt_k, cos_cache, sin_cache, memory_config=act_mc)

    if kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
            paged_modulo_kwargs = (
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            )
            page_len = page_table.shape[1] * eff_bs
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
            for slot_idx in valid_slots:
                k_user = tt_k[slot_idx : slot_idx + 1, :, :, :]
                v_user = tt_v[slot_idx : slot_idx + 1, :, :, :]
                k_user_sliced = k_user[:, :, :page_len, :] if page_len < seq_len_per_user else k_user
                v_user_sliced = v_user[:, :, :page_len, :] if page_len < seq_len_per_user else v_user
                ttnn.experimental.paged_fill_cache(
                    k_cache,
                    k_user_sliced,
                    page_table,
                    batch_idx=slot_idx,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
                ttnn.experimental.paged_fill_cache(
                    v_cache,
                    v_user_sliced,
                    page_table,
                    batch_idx=slot_idx,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
        else:
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
            for slot_idx in valid_slots:
                ttnn.fill_cache(k_cache, tt_k[slot_idx : slot_idx + 1], batch_idx=slot_idx)
                ttnn.fill_cache(v_cache, tt_v[slot_idx : slot_idx + 1], batch_idx=slot_idx)

    sliding_window = config.sliding_window if config.is_sliding else None
    # HiFi4 + FP32 dest-acc SDPA: restore softmax-reduce precision lost after #47311
    # (forced-FP32 reduce accumulation removed).
    sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tt_q.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        scale=1.0,
        sliding_window_size=sliding_window,
        compute_kernel_config=sdpa_compute_kernel_config,
    )
    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False, memory_config=act_mc)
    if act_mc == ttnn.L1_MEMORY_CONFIG:
        tt_out_l1 = tt_out
        tt_out = ttnn.to_memory_config(tt_out_l1, ttnn.DRAM_MEMORY_CONFIG)
        tt_out_l1.deallocate(True)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    tt_out = ttnn.reshape(tt_out, [1, 1, original_seq_len, -1])

    return tt_out, kept_kv, None
