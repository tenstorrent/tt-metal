# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import os

import torch
from loguru import logger

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
    interleave_qkv_if_sharded,
    o_proj_input_memcfg,
    prefill_sdpa_act_memcfg,
    prefill_sdpa_compute_kernel_config,
    prefill_sdpa_program_config,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

TILE_HEIGHT = 32


def _ensure_dram_interleaved(tensor):
    """Park an activation in DRAM interleaved before SDPA (L1 CBs need the room)."""
    if tensor is None:
        return tensor
    try:
        mem = tensor.memory_config()
        is_l1 = mem.buffer_type == ttnn.BufferType.L1
        is_sharded = tensor.is_sharded()
    except Exception:
        return tensor
    if not is_l1 and not is_sharded:
        return tensor
    if is_sharded:
        out = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)
    else:
        out = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    tensor.deallocate(True)
    return out


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


def _left_pad_kv_to_hist(tt_k, tt_v, hist, head_dim, *, deallocate_inputs=False):
    """Left-pad K/V with zeros to ``hist`` rows (causal window right-aligned).

    ``ttnn.pad`` cannot front-pad TILE tensors on device. Used by the sliding
    SDPA consumer when a prior short stash is narrower than ``hist``. Avoid
    calling mid-trace-capture (``ttnn.zeros`` host write). When already
    ``>= hist``, returns the last ``hist`` rows (cloned if truncated).
    """
    if tt_k is None or tt_v is None or hist is None or hist <= 0:
        return tt_k, tt_v
    kseq = int(tt_k.shape[-2])
    nkv = int(tt_k.shape[1])
    if kseq == hist:
        return tt_k, tt_v
    if kseq > hist:
        start = kseq - hist
        k_s = ttnn.slice(tt_k, [0, 0, start, 0], [1, nkv, kseq, head_dim])
        v_s = ttnn.slice(tt_v, [0, 0, start, 0], [1, nkv, kseq, head_dim])
        k_out = ttnn.clone(k_s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_out = ttnn.clone(v_s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_s.deallocate(True)
        v_s.deallocate(True)
        if deallocate_inputs:
            tt_k.deallocate(True)
            tt_v.deallocate(True)
        return k_out, v_out
    pad = hist - kseq
    zero_shape = [1, nkv, pad, head_dim]
    k_zeros = ttnn.zeros(
        zero_shape,
        dtype=tt_k.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=tt_k.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_zeros = ttnn.zeros(
        zero_shape,
        dtype=tt_v.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=tt_v.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_out = ttnn.concat([k_zeros, tt_k], dim=2)
    v_out = ttnn.concat([v_zeros, tt_v], dim=2)
    k_zeros.deallocate(True)
    v_zeros.deallocate(True)
    if deallocate_inputs:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    return k_out, v_out


def _copy_sliding_tail_into_persistent(config, k_tail_out, v_tail_out, head_dim):
    """Copy a (possibly short) sliding-tail stash into the persistent ring buffers.

    Traced short buckets stash unpadded tails (``allow_short_pad=False`` mid-
    capture). A prior full-chunk request may have left a hist-wide persistent
    ring (e.g. 1024) on ``config``. Blind ``ttnn.copy(128 → 1024)`` TT_FATALs
    under vLLM APC remnant / preempt-resume (GPQA conc=32 engine death).

    On width mismatch, **rebind** (deallocate old ring, adopt the new stash).
    Do not left-pad here: ``ttnn.zeros`` is illegal mid-trace-capture. Same-
    width path keeps the captured addresses via ``ttnn.copy``. Returns the
    ``(k, v)`` pair to keep as ``sliding_tail_out``.
    """
    del head_dim  # width checks use tensor shapes; kept for call-site symmetry
    persistent = getattr(config, "sliding_prefill_tail_persistent", None)
    if persistent is None:
        config.sliding_prefill_tail_persistent = (k_tail_out, v_tail_out)
        return (k_tail_out, v_tail_out)
    persistent_k, persistent_v = persistent
    pk_seq = int(persistent_k.shape[-2])
    ko_seq = int(k_tail_out.shape[-2])
    if ko_seq != pk_seq:
        for t in (persistent_k, persistent_v):
            try:
                t.deallocate(True)
            except Exception:
                pass
        config.sliding_prefill_tail_persistent = (k_tail_out, v_tail_out)
        return (k_tail_out, v_tail_out)
    ttnn.copy(k_tail_out, persistent_k)
    ttnn.copy(v_tail_out, persistent_v)
    k_tail_out.deallocate(True)
    v_tail_out.deallocate(True)
    return (persistent_k, persistent_v)


def _clone_sliding_prefill_tail(tt_k, tt_v, hist, head_dim, valid_seq_len=None, *, allow_short_pad=True):
    """Clone the last up-to-``hist`` K/V rows for the next sliding prefill chunk.

    vLLM APC / token-chunked prefill often delivers a first scheduler grant
    shorter than ``sliding_window`` (e.g. ``chunk_start`` remnant 128/384 with
    ``hist=1024``). Skipping the stash when ``kseq < hist`` leaves the next
    continuation without ``sliding_tail_in`` (shield QB2 hang / #51186).

    Always clone at least the available rows (safe mid-trace-capture). Eager
    paths may left-pad to ``hist`` here; traced short buckets keep a short
    clone and the consumer pads via ``_left_pad_kv_to_hist``.
    """
    if tt_k is None or tt_v is None or hist is None or hist <= 0:
        return None
    kseq = int(tt_k.shape[-2])
    if valid_seq_len is not None:
        kseq = min(kseq, max(0, int(valid_seq_len)))
    if kseq <= 0:
        return None
    nkv = int(tt_k.shape[1])
    take = min(kseq, hist)
    tail_start = kseq - take
    k_part = ttnn.slice(tt_k, [0, 0, tail_start, 0], [1, nkv, kseq, head_dim])
    v_part = ttnn.slice(tt_v, [0, 0, tail_start, 0], [1, nkv, kseq, head_dim])
    # ``ttnn.slice`` may share storage with ``tt_k``/``tt_v``. Source layers with
    # ``keep_kv=True`` (E2B/E4B: num_kv_shared_layers) must keep those parents
    # alive for later shared layers. Deallocating the slice (or left-pad of it)
    # was freeing keep_kv tensors → SDPA ``input_tensor.is_allocated()`` on the
    # first KV-shared sliding layer during WH N150 warmup (run 31353872112).
    k_owned = ttnn.clone(k_part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_owned = ttnn.clone(v_part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if take < hist and allow_short_pad:
        # Eager only: traced capture forbids ttnn.zeros (host write). Safe to
        # deallocate the owned clones inside left-pad — not the keep_kv parents.
        return _left_pad_kv_to_hist(k_owned, v_owned, hist, head_dim, deallocate_inputs=True)
    return (k_owned, v_owned)


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
    # Free the QKV in0 (often L1 interleaved from input_layernorm S2I / hoist)
    # before SDPA. Leaving it resident clashes with SDPA's static circular
    # buffers on the full worker grid (same pattern as the batched path below).
    ttnn.deallocate(hidden_states)

    # Short-lived prefill activations in L1 when GEMMA4_PREFILL_L1_ACT=1 (Qwen36
    # #48861). The allreduce input stays DRAM (CCL path); the o_proj matmul writes
    # L1 block-sharded under its tuned config and is interleaved back before CCL.
    # The tuned prefill QKV already writes DRAM interleaved; the guard stays for
    # callers/paths that hand back a sharded projection (head-split needs interleaved).
    act_mc = prefill_sdpa_act_memcfg()
    xqkv = interleave_qkv_if_sharded(xqkv, memory_config=act_mc)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv,
        config,
        weights.is_global,
        tp=tp,
        kv_replicated=weights.kv_replicated,
        memory_config=act_mc,
    )
    # Free the fused projection buffer once the heads are materialized so it is
    # not still resident under SDPA.
    ttnn.deallocate(xqkv)

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=act_mc)

    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        # Do not K→V clone (resync): that produced unicode garbage on LB 12B.
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
                # Unbounded: in-forward fill. When ``valid_seq_len`` is known
                # (last multi-chunk with true last-token index), slice away
                # power-of-2 pad rows before fill. Pad rows otherwise write
                # through extra page-table columns; valid_seq_len caps the fill.
                # Extra columns pad with 0 (vLLM null block).
                k_fill, v_fill = tt_k, tt_v
                if valid_seq_len is not None:
                    v = min(int(valid_seq_len), int(tt_k.shape[-2]))
                    # Tile-ceil so the writer sees a legal RM height; unused
                    # rows inside the last tile sit on pad columns (0).
                    tile_end = ((v + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
                    tile_end = min(tile_end, int(tt_k.shape[-2]))
                    if 0 < tile_end < int(tt_k.shape[-2]):
                        k_fill = ttnn.slice(
                            tt_k,
                            [0, 0, 0, 0],
                            [tt_k.shape[0], tt_k.shape[1], tile_end, tt_k.shape[3]],
                        )
                        v_fill = ttnn.slice(
                            tt_v,
                            [0, 0, 0, 0],
                            [tt_v.shape[0], tt_v.shape[1], tile_end, tt_v.shape[3]],
                        )
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
    # Prefill SDPA's static CBs fill Wormhole L1; any L1 Q/K/V (or residue from
    # an L1 QKV out) clashes. Force DRAM interleaved heads before the op.
    tt_q = _ensure_dram_interleaved(tt_q)
    tt_k = _ensure_dram_interleaved(tt_k)
    tt_v = _ensure_dram_interleaved(tt_v)
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
        sdpa_ckc = prefill_sdpa_compute_kernel_config(tt_q.device(), hidden_size=config.hidden_size)
        hist = ((sliding_window + 31) // 32) * 32
        use_persistent_tail = isinstance(chunk_start_idx, ttnn.Tensor)
        if sliding_tail_in is not None:
            k_tail, v_tail = sliding_tail_in
            # Traced short first-buckets stash an unpadded tail (< hist); pad
            # here so concat stays square. Eager APC often already padded.
            if int(k_tail.shape[-2]) != hist:
                k_tail, v_tail = _left_pad_kv_to_hist(
                    k_tail,
                    v_tail,
                    hist,
                    config.head_dim,
                    # Never free persistent ring buffers; eager stashes are owned here.
                    deallocate_inputs=not use_persistent_tail,
                )
            nqh = int(tt_q.shape[1])
            # Filler Q rows (outputs discarded). Prefer the chunk's leading
            # rows when ``seq_len >= hist``; APC remnant chunks can be shorter
            # than ``hist`` (e.g. 128/384), so left-pad with zeros instead of
            # slicing past ``tt_q`` (TT_FATAL Ends hist > tensor seq).
            if seq_len >= hist:
                q_pad = ttnn.slice(tt_q, [0, 0, 0, 0], [1, nqh, hist, config.head_dim])
            else:
                q_zeros = ttnn.zeros(
                    [1, nqh, hist - seq_len, config.head_dim],
                    dtype=tt_q.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=tt_q.device(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                q_pad = ttnn.concat([q_zeros, tt_q], dim=2)
                q_zeros.deallocate(True)
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
                program_config=prefill_sdpa_program_config(
                    config.head_dim, hist + seq_len, sliding_window=sliding_window
                ),
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
            # No in-memory tail. Correct for the first chunk (chunk_offset==0).
            # Continuation without a tail (e.g. prior scheduler chunk took the
            # single-chunk path and failed to stash — see post-SDPA stash below)
            # silently drops the prior window; log so the ~9k remnant cliff is
            # diagnosable if it regresses.
            if chunk_offset is not None and chunk_offset > 0:
                logger.warning(
                    "Gemma4 sliding prefill: chunk_start={} without sliding_tail_in; "
                    "windowed SDPA will miss prior-chunk K/V (vLLM chunked prefill "
                    "remnant < sliding_window).",
                    chunk_offset,
                )
            tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
                program_config=prefill_sdpa_program_config(config.head_dim, seq_len, sliding_window=sliding_window),
                compute_kernel_config=sdpa_ckc,
            )
        # Save this chunk's last ``hist`` K/V tokens as the next chunk's tail.
        # Slices may be views of tt_k/tt_v; clone (+ left-pad if short) so the
        # tail outlives parent dealloc across vLLM cross-call chunked prefill
        # (#51041) and APC short-first-grant continuations.
        sliding_tail_stash = _clone_sliding_prefill_tail(
            tt_k,
            tt_v,
            hist,
            config.head_dim,
            valid_seq_len=valid_seq_len,
            # Tensor chunk_start_idx ⇒ traced multi-chunk; skip short-pad alloc.
            allow_short_pad=not use_persistent_tail,
        )
        if sliding_tail_stash is None:
            k_tail_out = v_tail_out = None
        else:
            k_tail_out, v_tail_out = sliding_tail_stash
        if use_persistent_tail and k_tail_out is not None:
            # Bind fixed DRAM addresses into the graph so middle-chunk replay
            # sees the previous chunk's window without re-running Python.
            # Prefer the capture-time buffers stashed on config so replay copies
            # into the same addresses the trace recorded. Shape-mismatch safe
            # (short APC remnant vs hist-wide ring from a prior chunk/request).
            if getattr(config, "sliding_prefill_tail_persistent", None) is not None:
                sliding_tail_out = _copy_sliding_tail_into_persistent(config, k_tail_out, v_tail_out, config.head_dim)
            elif sliding_tail_in is not None:
                config.sliding_prefill_tail_persistent = sliding_tail_in
                sliding_tail_out = _copy_sliding_tail_into_persistent(config, k_tail_out, v_tail_out, config.head_dim)
            else:
                # First persistent alloc: clones above already own independent DRAM.
                config.sliding_prefill_tail_persistent = (k_tail_out, v_tail_out)
                sliding_tail_out = (k_tail_out, v_tail_out)
        elif k_tail_out is not None:
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
            hidden_size=config.hidden_size,
        )
    elif long_seq and config.is_sliding and sliding_window is not None:
        tt_sdpa = chunked_prefill_sdpa_sliding(
            tt_q,
            tt_k,
            tt_v,
            sliding_window,
            config.head_dim,
            scale=1.0,
            hidden_size=config.hidden_size,
        )
    elif long_seq and not config.is_sliding and page_table is not None and kv_cache is not None:
        # Full-attention long context (incl. KV-shared layers whose kv_cache is
        # the source layer's already-filled pool).
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
            num_kv_heads=nkv_local,
            hidden_size=config.hidden_size,
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
        # fp32 dest-acc is safe on the prefill SDPA op (unlike the decode op, where
        # it halves dest for head_dim=512). Fidelity policy and the #38306 caveat
        # live in prefill_sdpa_compute_kernel_config.
        sdpa_compute_kernel_config = prefill_sdpa_compute_kernel_config(tt_q.device(), hidden_size=config.hidden_size)
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=sliding_window,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len, sliding_window=sliding_window),
            compute_kernel_config=sdpa_compute_kernel_config,
        )
    # Persist sliding-window K/V tail after *any* sliding prefill, not only
    # generator-level ``sliding_chunked``. vLLM token-chunked prefill
    # (``enable_chunked_prefill``) often runs the first scheduler chunk through
    # the single-chunk path (no ``chunk_page_table``); without this stash the
    # next call's sliding SDPA has no prior-window K/V — including when the
    # first grant is shorter than ``sliding_window`` (APC remnant / short
    # ``max_num_batched_tokens`` slice). Left-pad short tails to ``hist`` so
    # the continuation concat stays square (#51186). Clone so the tail
    # outlives tt_k/tt_v dealloc (#51041). Width-mismatch rebind keeps the
    # stash alive after traced 4k grants (see ``_DEFAULT_TRACE_PREFILL_SEQ_LENS``).
    if (
        sliding_tail_out is None
        and config.is_sliding
        and sliding_window is not None
        and tt_k is not None
        and shared_kv is None
    ):
        hist = ((sliding_window + 31) // 32) * 32
        # Short-pad uses ttnn.zeros (host write) — illegal mid-trace-capture.
        # Cold traced single-chunk prepares chunk_start_idx=None; eager APC /
        # token-chunked prefill passes a host int (incl. 0). Device-tensor
        # offsets are traced multi-chunk (chunk sizes ≥ hist; no short-pad).
        allow_short_pad = chunk_start_idx is not None and not isinstance(chunk_start_idx, ttnn.Tensor)
        sliding_tail_out = _clone_sliding_prefill_tail(
            tt_k,
            tt_v,
            hist,
            config.head_dim,
            valid_seq_len=valid_seq_len,
            allow_short_pad=allow_short_pad,
        )
    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    # The tuned o_proj matmul reads in0 from L1 interleaved — land the head-concat
    # there directly when it fits the L1 budget (else act_mc, i.e. DRAM by default).
    concat_mc = o_proj_input_memcfg(tt_sdpa, config.hidden_size, default_memcfg=act_mc)
    tt_out = concat_heads(tt_sdpa, is_decode_mode=False, memory_config=concat_mc)
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

    # Block-sharded QKV (tuned prefill path) must be interleaved before reshape/split.
    act_mc = prefill_sdpa_act_memcfg()
    xqkv = interleave_qkv_if_sharded(xqkv, memory_config=act_mc)
    xqkv = ttnn.reshape(xqkv, [batch_size, 1, seq_len // batch_size, -1])
    seq_len_per_user = seq_len // batch_size

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
        # Do not K→V clone (resync): that produced unicode garbage on LB 12B.
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
            # Per-slot real lengths (hetero prompts in one pad bucket). Without
            # this, zero-pad rows are written into KV and corrupt decode — B=1
            # already slices via scalar ``valid_seq_len``; batched used to skip it.
            for slot_idx in valid_slots:
                k_user = tt_k[slot_idx : slot_idx + 1, :, :, :]
                v_user = tt_v[slot_idx : slot_idx + 1, :, :, :]
                fill_len = seq_len_per_user
                if isinstance(valid_seq_len, (list, tuple)):
                    if 0 <= int(slot_idx) < len(valid_seq_len) and valid_seq_len[int(slot_idx)] is not None:
                        fill_len = min(fill_len, max(0, int(valid_seq_len[int(slot_idx)])))
                elif valid_seq_len is not None:
                    fill_len = min(fill_len, max(0, int(valid_seq_len)))
                fill_len = min(fill_len, page_len)
                # Tile-ceil so the writer sees a legal RM height (match B=1 path).
                # Zero-length slots must skip fill — tile_end==0 used to fall into
                # the else branch and write the full pad into KV.
                tile_end = ((fill_len + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
                tile_end = min(tile_end, seq_len_per_user, page_len) if fill_len > 0 else 0
                if tile_end <= 0:
                    continue
                if tile_end < seq_len_per_user:
                    k_user_sliced = ttnn.slice(
                        k_user,
                        [0, 0, 0, 0],
                        [1, k_user.shape[1], tile_end, k_user.shape[3]],
                    )
                    v_user_sliced = ttnn.slice(
                        v_user,
                        [0, 0, 0, 0],
                        [1, v_user.shape[1], tile_end, v_user.shape[3]],
                    )
                else:
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
    tt_q = _ensure_dram_interleaved(tt_q)
    tt_k = _ensure_dram_interleaved(tt_k)
    tt_v = _ensure_dram_interleaved(tt_v)
    # Fidelity policy and the #38306 caveat live in
    # prefill_sdpa_compute_kernel_config.
    sdpa_compute_kernel_config = prefill_sdpa_compute_kernel_config(tt_q.device(), hidden_size=config.hidden_size)
    tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        scale=1.0,
        sliding_window_size=sliding_window,
        program_config=prefill_sdpa_program_config(config.head_dim, int(tt_q.shape[-2]), sliding_window=sliding_window),
        compute_kernel_config=sdpa_compute_kernel_config,
    )
    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    concat_mc = o_proj_input_memcfg(tt_sdpa, config.hidden_size, default_memcfg=act_mc)
    tt_out = concat_heads(tt_sdpa, is_decode_mode=False, memory_config=concat_mc)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    tt_out = ttnn.reshape(tt_out, [1, 1, original_seq_len, -1])

    return tt_out, kept_kv, None
