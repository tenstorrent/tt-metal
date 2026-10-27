# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched (single-prefill) commit-append for DiffusionGemma (#47557).

The generation loop commits a denoised canvas into the frozen Gemma4 KV cache.
Committing a 256-token canvas is mathematically a causal prefill: committed
token ``i`` attends the frozen prefix (prompt + prior blocks) plus canvas tokens
``0..i``. This module runs that as ONE causal masked prefill over the whole
canvas — replacing 256 sequential single-token decode-appends — reusing the
denoise attention building blocks with two changes vs the read-only
bidirectional denoise pass:

1. a **causal** prefix+canvas mask instead of the bidirectional mask;
2. the canvas K/V is **written** into the frozen cache at the committed
   positions (the denoise pass never writes).

Design: **write-then-read-from-cache**, per layer, in layer order:
  * compute canvas Q (per-head norm + RoPE at ``start_pos``);
  * for a non-shared layer, compute canvas K/V and write it into
    ``tt_kv_cache[i]`` at seq positions ``start_pos .. start_pos+C-1`` (the
    write span is tile-aligned — see :func:`_write_canvas_kv_contiguous`);
  * read the full ``[0 : start_pos+C]`` K/V back from the cache (= frozen prefix
    ++ freshly-written canvas) and run a causal-masked SDPA;
  * the MLP / MoE / norm tail is identical to the denoise layer body.

Reading the K/V for the SDPA out of the cache (rather than concatenating a
register copy) handles cross-layer **KV-sharing** for free: a shared layer skips
its own K/V write, and its earlier source layer has already written the canvas
K/V into the shared cache tensor by the time the shared layer runs.

The model-owned hybrid paged path preserves the same algebra with two ordering
requirements: sliding layers read their old circular window and attend BEFORE
the bulk write can evict history, while full layers bulk-write complete pages
and then use paged chunked SDPA. Partial 64-token pages are staged with their
untouched rows because prompt prefill guarantees 32-token, not 64-token,
alignment.

This path is the default commit.
"""

from __future__ import annotations

import os

from loguru import logger
import torch
import ttnn
from models.experimental.diffusion_gemma.tt.expert_operations import shared_mlp_forward

from models.demos.gemma4.tt.attention.operations import (
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    concat_heads,
    effective_block_size,
    split_qkv_heads_prefill,
)
from models.experimental.diffusion_gemma.tt.ccl import apply_allreduce, replicate_mapper as _replicate_mapper
from models.experimental.diffusion_gemma.tt.chunked_prefill import (
    _chunked_sdpa_program_config,
    _sliding_window_square_sdpa,
)
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    _chunked_norm_forward,
    _denoise_moe_forward,
    _read_hybrid_sliding_cache,
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

# KV-write mechanism for the contiguous (page_table=None) cache.
#   "fill"     — DEFAULT: one ``ttnn.fill_cache`` per K/V per layer writes the whole
#                canvas at the tile-aligned seq offset ``start_pos``.
#   "position" — per-position ``paged_update_cache`` writes; the reference the
#                "fill" path is verified against (``tests/test_commit.py``) and the
#                automatic fallback when a "fill" precondition does not hold.
_KV_WRITE_FILL = "fill"
_KV_WRITE_POSITION = "position"
_KV_WRITE_MODES = (_KV_WRITE_FILL, _KV_WRITE_POSITION)


def _default_kv_write_mode() -> str:
    """Resolve the default write mechanism from ``DG_COMMIT_KV_WRITE`` (default fill)."""
    mode = os.environ.get("DG_COMMIT_KV_WRITE")
    if mode:
        mode = mode.strip().lower()
        if mode not in _KV_WRITE_MODES:
            raise ValueError(f"DG_COMMIT_KV_WRITE must be one of {_KV_WRITE_MODES}, got {mode!r}")
        return mode
    return _KV_WRITE_FILL


_DEFAULT_KV_WRITE_MODE = _default_kv_write_mode()

_FILL_FALLBACK_WARNED: set[str] = set()
_LOGGED_KV_WRITE_MODES: set[str] = set()


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
    hf_config = getattr(tt_model, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx]
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    return getattr(attn_config, "layer_type", None)


def _sliding_window_for_commit(tt_model, layer_idx: int) -> int | None:
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    window = getattr(attn_config, "sliding_window", None)
    if window is not None:
        return window
    hf_config = getattr(tt_model, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    return getattr(text_config, "sliding_window", None)


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

    def read(cache, ends):
        # A full-span slice (all starts 0, all ends max) short-circuits in ttnn to an
        # ALIAS of the input (``slice.cpp``: `if (no_step && starts_zero && ends_max)`
        # returns the tensor itself once the no-op to_memory_config/to_layout pass
        # through). The caller deallocates what we return, which would free the KV
        # CACHE ITSELF — reachable whenever the committed block ends exactly at
        # max_seq. Clone in that case so the caller always owns a distinct buffer.
        if end_pos == cache.shape[2]:
            return ttnn.clone(cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.slice(cache, starts, ends, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return read(k_cache, k_ends), read(v_cache, v_ends)


def _fill_write_unsupported_reason(
    k_cache,
    v_cache,
    canvas_k,
    canvas_v,
    *,
    start_pos: int,
    canvas_len: int,
    mesh_device,
) -> str | None:
    """Why one ``ttnn.fill_cache`` cannot write this canvas (``None`` ⇒ it can).

    Checked in Python, before any device work, so a geometry the op does not support
    degrades to the per-position write instead of raising mid-layer and leaving a
    half-written cache. The checks mirror the FILL validator
    (``ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp``)
    plus one guard the op does NOT enforce itself (the head-boundary spill below).
    """
    tile = ttnn.TILE_SIZE
    if start_pos % tile != 0:
        return f"start_pos {start_pos} is not a multiple of {tile}"
    if canvas_len % tile != 0:
        return f"canvas_len {canvas_len} is not a multiple of {tile}"
    # NB: cache batch > 1 is deliberately NOT a fallback reason. ``batch_idx=0`` into a
    # multi-slot cache is legal for FILL (the op only requires batch_idx < cache batch)
    # and correct for this single-user commit, whereas the per-position fallback would
    # TT_FATAL on it (non-paged paged_update_cache asserts num_users == cache batch).
    # Falling back there would turn a working write into a crash; the position branch
    # raises its own clear error instead.
    for name, cache, canvas in (("k", k_cache, canvas_k), ("v", v_cache, canvas_v)):
        if canvas.dtype != cache.dtype:
            # FILL is a pure copy and refuses to convert (unlike paged_update_cache).
            return f"{name} dtype mismatch: canvas {canvas.dtype} vs cache {cache.dtype}"
        if canvas.layout != ttnn.TILE_LAYOUT or cache.layout != ttnn.TILE_LAYOUT:
            return f"{name} not TILE_LAYOUT (canvas {canvas.layout}, cache {cache.layout})"
        if canvas.shape[0] != 1:
            return f"{name} canvas batch {canvas.shape[0]} != 1"
        if canvas.shape[1] != cache.shape[1]:
            return f"{name} kv-head mismatch: canvas {canvas.shape[1]} vs cache {cache.shape[1]}"
        if canvas.shape[3] != cache.shape[3]:
            return f"{name} head_dim mismatch: canvas {canvas.shape[3]} vs cache {cache.shape[3]}"
        if canvas.shape[3] % tile != 0:
            return f"{name} head_dim {canvas.shape[3]} is not a multiple of {tile}"
        if canvas.shape[2] != canvas_len:
            return f"{name} canvas seq {canvas.shape[2]} != canvas_len {canvas_len}"
        if start_pos + canvas_len > cache.shape[2]:
            return f"{name} write span [{start_pos}, {start_pos + canvas_len}) exceeds cache seq {cache.shape[2]}"
        if cache.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return f"{name} cache is not INTERLEAVED ({cache.memory_config().memory_layout})"
        if canvas.is_sharded():
            # The sharded-input branch of the factory splits work by shard height; only
            # the interleaved split is audited here.
            return f"{name} canvas is sharded; fill path expects an interleaved canvas"
    # Head-boundary spill: the FILL program factory splits (nkv * C/32) tile-rows over
    # the core grid and each core writes its rows CONTIGUOUSLY from one cache_start_id,
    # assuming no core's range crosses a kv-head boundary
    # (fill_cache_multi_core_program_factory.cpp). That holds whenever every core gets
    # exactly one row, i.e. rows <= cores, and trivially when the input spans the whole
    # cache (C == max_seq ⇒ one contiguous run). Above that the op silently writes some
    # rows to the wrong head. DiffusionGemma is far inside the safe region; this guards
    # future geometries.
    rows = canvas_k.shape[1] * (canvas_len // tile)
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if canvas_len != k_cache.shape[2] and rows > num_cores:
        return (
            f"fill would spill across kv-head boundaries: {rows} tile-rows "
            f"(nkv={canvas_k.shape[1]} x C/{tile}={canvas_len // tile}) > {num_cores} cores"
        )
    return None


def _warn_fill_fallback_once(reason: str) -> None:
    if reason not in _FILL_FALLBACK_WARNED:
        _FILL_FALLBACK_WARNED.add(reason)
        logger.warning(
            f"[commit] one-op fill KV write unavailable ({reason}); falling back to the "
            "per-position write (correct, ~800x more dispatch per layer)"
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
    write_mode: str | None = None,
):
    """Write canvas K/V ``[1, nkv, C, hd]`` into a contiguous cache at ``start_pos``.

    ``write_mode="fill"`` (default): one ``ttnn.fill_cache`` per K/V writes the whole
    canvas at the tile-aligned seq offset ``update_idx=start_pos``. The write span is
    tile-aligned by construction (``start_pos % 32 == 0`` is validated in
    :func:`commit_canvas_tokens_batched`, ``canvas_len`` is a tile multiple), so FILL
    is a pure tile copy — it writes exactly the ``[start_pos, start_pos+C)`` tile-rows
    of each kv head and touches neither the frozen prefix nor the tail. Falls back to
    ``"position"`` (with a warning) if any precondition fails — see
    :func:`_fill_write_unsupported_reason`.

    ``write_mode="position"``: one ``paged_update_cache`` per committed position — the
    same non-paged decode-append op the sequential path uses, so write positions and
    cache layout are identical. A batch-1 contiguous cache can only address one seq
    position per non-paged update, hence the per-position loop. This is the reference
    the fill path is verified against.

    Trace caveat (both write modes): ``start_pos`` lives in the op's runtime args and
    is excluded from the program-cache hash, so a commit captured into a metal trace
    would replay the offset it was captured at. The commit runs eagerly today; tracing
    it needs a per-block re-capture or a device-side index tensor.

    Do not batch ``paged_update_cache`` over multiple positions here: the op is a
    per-TILE read-modify-write and consecutive positions share one 32-row cache tile,
    so concurrent "users" race last-writer-wins — the same failure
    ``gemma4/tt/attention/decode.py`` serializes around (``sequential_kv_write``,
    issue #44923) — and paged mode rejects the op's only serialization
    (``in0_sequential_mode``).
    """
    mode = _DEFAULT_KV_WRITE_MODE if write_mode is None else write_mode.strip().lower()
    if mode not in _KV_WRITE_MODES:
        raise ValueError(f"write_mode must be one of {_KV_WRITE_MODES}, got {mode!r}")
    if mode not in _LOGGED_KV_WRITE_MODES:
        _LOGGED_KV_WRITE_MODES.add(mode)
        logger.info(f'[commit] contiguous KV write mode = "{mode}"')

    if mode == _KV_WRITE_FILL:
        reason = _fill_write_unsupported_reason(
            k_cache,
            v_cache,
            canvas_k,
            canvas_v,
            start_pos=start_pos,
            canvas_len=canvas_len,
            mesh_device=mesh_device,
        )
        if reason is None:
            # Whole canvas, one op per K/V. No transpose (the [1, C, nkv, hd] permute
            # below only exists to satisfy paged_update_cache's user-dim contract), no
            # reshard, and no per-call host tensors.
            ttnn.fill_cache(k_cache, canvas_k, 0, update_idx=start_pos)
            ttnn.fill_cache(v_cache, canvas_v, 0, update_idx=start_pos)
            return
        _warn_fill_fallback_once(reason)
        mode = _KV_WRITE_POSITION

    # ── per-position write (explicit, or the fill path's fallback) ───────────────
    # Still before any device work: the per-position write is NOT a universal safe harbor,
    # so name the two geometries it cannot serve (both of which FILL can) rather than
    # letting them TT_FATAL deep inside the op — or, when we got here by falling back,
    # rather than turning a working write into a crash.
    if k_cache.shape[0] != 1 or v_cache.shape[0] != 1:
        # Non-paged ``paged_update_cache`` asserts num_users == cache batch, and this loop
        # writes one user at a time.
        raise ValueError(
            f"the per-position KV write needs a batch-1 cache (got k={k_cache.shape[0]}, "
            f"v={v_cache.shape[0]}); use write_mode={_KV_WRITE_FILL!r}"
        )
    if canvas_k.dtype not in (ttnn.bfloat16, ttnn.float32) or canvas_v.dtype not in (
        ttnn.bfloat16,
        ttnn.float32,
    ):
        # ``paged_update_cache`` accepts only fp32/bf16 inputs — e.g. a bfloat8_b canvas
        # that FILL would copy happily would TT_FATAL here.
        raise ValueError(
            f"the per-position KV write needs a bfloat16/float32 canvas (got k={canvas_k.dtype}, "
            f"v={canvas_v.dtype}); use write_mode={_KV_WRITE_FILL!r}"
        )

    nkv = canvas_k.shape[1]
    hd = canvas_k.shape[3]
    # [1, nkv, C, hd] -> [1, C, nkv, hd] so slot dim is the update-cache batch dim.
    k_perm = ttnn.transpose(canvas_k, 1, 2)
    v_perm = ttnn.transpose(canvas_v, 1, 2)

    # ``paged_update_cache`` requires the update tensor to be HEIGHT_SHARDED with one core
    # per user (batch), shard width == head_dim, ROW_MAJOR — the exact contract the proven
    # decode single-user write uses (``gemma4/tt/attention/decode.py``
    # ``sequential_kv_write``: ``single_user_mem`` = 1 core, shard shape
    # ``[TILE, head_dim]``). Our per-position slice is ``[1, 1, nkv, hd]`` (num_users=1),
    # so reshard it onto one core with the tile-padded nkv height before the write; a bare
    # DRAM-interleaved slice trips the op's ``is_sharded()`` assert.
    shard_h = ((nkv + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    single_user_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(one_core, [shard_h, hd], ttnn.ShardOrientation.ROW_MAJOR),
    )
    for t in range(canvas_len):
        kb = ttnn.slice(k_perm, [0, t, 0, 0], [1, t + 1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vb = ttnn.slice(v_perm, [0, t, 0, 0], [1, t + 1, nkv, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kb_s = ttnn.to_memory_config(kb, single_user_mem)
        vb_s = ttnn.to_memory_config(vb, single_user_mem)
        ttnn.experimental.paged_update_cache(k_cache, kb_s, update_idxs=[start_pos + t])
        ttnn.experimental.paged_update_cache(v_cache, vb_s, update_idxs=[start_pos + t])
        kb.deallocate(True)
        vb.deallocate(True)
        kb_s.deallocate(True)
        vb_s.deallocate(True)
    k_perm.deallocate(True)
    v_perm.deallocate(True)


def _paged_write_plan(
    *,
    start_pos: int,
    canvas_len: int,
    block_size: int,
    num_blocks: int,
    circular: bool,
) -> tuple[tuple[int, ...], int, int]:
    """Plan an offset paged fill without a per-token update loop.

    ``paged_fill_cache`` always treats input row zero as the beginning of a
    logical block; it has no absolute ``start_pos`` argument.  A commit can start
    half way through a cache block because prompt prefill is only 32-token
    aligned while the model-owned hybrid cache uses 64-token pages.  The caller
    therefore stages the untouched leading/trailing rows around the canvas and
    fills the complete physical blocks listed here.

    Returns ``(physical_block_ids, leading_rows, trailing_rows)``.  Sliding
    layers wrap block ids through their bounded physical pool; full-attention
    layers retain the ordinary identity mapping.
    """

    for name, value in (
        ("start_pos", start_pos),
        ("canvas_len", canvas_len),
        ("block_size", block_size),
        ("num_blocks", num_blocks),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if canvas_len <= 0 or block_size <= 0 or num_blocks <= 0:
        raise ValueError(
            f"canvas_len, block_size, and num_blocks must be positive, got " f"{canvas_len}, {block_size}, {num_blocks}"
        )
    if start_pos % TILE_SIZE != 0 or canvas_len % TILE_SIZE != 0 or block_size % TILE_SIZE != 0:
        raise ValueError(
            f"paged commit geometry must be {TILE_SIZE}-token aligned: "
            f"start_pos={start_pos}, canvas_len={canvas_len}, block_size={block_size}"
        )

    leading = start_pos % block_size
    staged_len = ((leading + canvas_len + block_size - 1) // block_size) * block_size
    trailing = staged_len - leading - canvas_len
    first_block = start_pos // block_size
    block_count = staged_len // block_size
    absolute_ids = tuple(range(first_block, first_block + block_count))
    if circular:
        physical_ids = tuple(block_id % num_blocks for block_id in absolute_ids)
    else:
        if absolute_ids[-1] >= num_blocks:
            raise ValueError(
                f"paged commit block span [{absolute_ids[0]}, {absolute_ids[-1]}] "
                f"exceeds the full-attention pool with {num_blocks} blocks"
            )
        physical_ids = absolute_ids
    return physical_ids, leading, trailing


def _to_device_page_table(mesh_device, block_ids: tuple[int, ...]):
    host = torch.tensor(block_ids, dtype=torch.int32).reshape(1, -1)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _stage_paged_canvas(cache, canvas, *, block_ids: tuple[int, ...], leading: int, trailing: int):
    """Add untouched rows around ``canvas`` so paged_fill_cache can write whole pages."""

    if not block_ids:
        raise ValueError("paged canvas staging needs at least one physical block id")
    parts = []
    owned = []
    if leading:
        prefix = ttnn.slice(
            cache,
            [block_ids[0], 0, 0, 0],
            [block_ids[0] + 1, cache.shape[1], leading, cache.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parts.append(prefix)
        owned.append(prefix)
    parts.append(canvas)
    if trailing:
        suffix_start = int(cache.shape[2]) - trailing
        suffix = ttnn.slice(
            cache,
            [block_ids[-1], 0, suffix_start, 0],
            [block_ids[-1] + 1, cache.shape[1], cache.shape[2], cache.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parts.append(suffix)
        owned.append(suffix)
    if len(parts) == 1:
        return canvas, False
    staged = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for tensor in owned:
        tensor.deallocate(True)
    return staged, True


def _write_canvas_kv_paged(
    k_cache,
    v_cache,
    canvas_k,
    canvas_v,
    *,
    page_table,
    block_ids: tuple[int, ...],
    leading: int,
    trailing: int,
    block_size: int,
    head_dim: int,
    num_kv_heads: int,
):
    """Write one canvas into identity-mapped model-owned paged K/V in two fills."""

    effective = effective_block_size(k_cache, head_dim, num_kv_heads)
    if effective != block_size:
        raise ValueError(
            f"model-owned hybrid commit expected effective block_size={block_size}, got {effective}; "
            "the cache is not in the attached per-layer view"
        )
    k_staged, owns_k = _stage_paged_canvas(k_cache, canvas_k, block_ids=block_ids, leading=leading, trailing=trailing)
    v_staged, owns_v = _stage_paged_canvas(v_cache, canvas_v, block_ids=block_ids, leading=leading, trailing=trailing)
    try:
        ttnn.experimental.paged_fill_cache(
            k_cache,
            k_staged,
            page_table,
            batch_idx=0,
            block_size=effective,
        )
        ttnn.experimental.paged_fill_cache(
            v_cache,
            v_staged,
            page_table,
            batch_idx=0,
            block_size=effective,
        )
    finally:
        if owns_k:
            k_staged.deallocate(True)
        if owns_v:
            v_staged.deallocate(True)


def _sliding_commit_attention_paged(
    tt_q,
    tt_k,
    tt_v,
    *,
    kv_cache,
    start_pos: int,
    sliding_window: int,
    head_dim: int,
):
    """Attend one canvas before its circular-cache write evicts needed history.

    A 256-row bulk write can overwrite prefix rows that the earliest canvas
    queries still need.  Read the previous bounded window first, concatenate the
    current K/V in registers, run the same square causal+sliding SDPA used by
    chunked prefill, and only then let the caller update the circular cache.
    """

    prefix_len = min(start_pos, sliding_window)
    if prefix_len:
        seq_len_start = start_pos - prefix_len
        prefix_k = _read_hybrid_sliding_cache(
            kv_cache[0],
            prompt_len=prefix_len,
            seq_len_start=seq_len_start,
            capacity=sliding_window,
        )
        prefix_v = _read_hybrid_sliding_cache(
            kv_cache[1],
            prompt_len=prefix_len,
            seq_len_start=seq_len_start,
            capacity=sliding_window,
        )
        all_k = ttnn.concat([prefix_k, tt_k], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        all_v = ttnn.concat([prefix_v, tt_v], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prefix_k.deallocate(True)
        prefix_v.deallocate(True)

        zeros_q = ttnn.zeros(
            [1, tt_q.shape[1], prefix_len, head_dim],
            dtype=tt_q.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=tt_q.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_square = ttnn.concat([zeros_q, tt_q], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        zeros_q.deallocate(True)
    else:
        all_k, all_v, q_square = tt_k, tt_v, tt_q

    full_out = _sliding_window_square_sdpa(q_square, all_k, all_v, sliding_window, scale=1.0)
    if prefix_len:
        out = ttnn.slice(
            full_out,
            [0, 0, prefix_len, 0],
            [full_out.shape[0], full_out.shape[1], full_out.shape[2], full_out.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        full_out.deallocate(True)
        q_square.deallocate(True)
        all_k.deallocate(True)
        all_v.deallocate(True)
        return out
    return full_out


def _full_commit_attention_paged(tt_q, *, kv_cache, page_table, start_pos: int, head_dim: int):
    """Paged causal attention with a 128-aligned chunk start.

    The chunked SDPA kernel requires ``chunk_start_idx`` to be divisible by both
    its 128-row Q and K chunks, while prompt prefill only guarantees a 32-aligned
    commit position.  Front-pad Q back to the previous 128 boundary (and pad the
    tail to a whole Q chunk), then discard those synthetic output rows.  Real
    query row ``i`` still lands at absolute position ``start_pos + i`` under the
    kernel's causal mask, without reducing K chunk size for long contexts.
    """

    q_chunk = 128
    front = start_pos % q_chunk
    aligned_start = start_pos - front
    real_len = int(tt_q.shape[-2])
    tail = (-(front + real_len)) % q_chunk
    parts = []
    owned = []
    if front:
        zeros_front = ttnn.zeros(
            [1, tt_q.shape[1], front, head_dim],
            dtype=tt_q.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=tt_q.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parts.append(zeros_front)
        owned.append(zeros_front)
    parts.append(tt_q)
    if tail:
        zeros_tail = ttnn.zeros(
            [1, tt_q.shape[1], tail, head_dim],
            dtype=tt_q.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=tt_q.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parts.append(zeros_tail)
        owned.append(zeros_tail)
    if len(parts) == 1:
        q_in = tt_q
        owns_q_in = False
    else:
        q_in = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owns_q_in = True
        for tensor in owned:
            tensor.deallocate(True)

    k_cache, v_cache = kv_cache
    out_full = ttnn.transformer.chunked_scaled_dot_product_attention(
        q_in,
        k_cache,
        v_cache,
        page_table,
        chunk_start_idx=aligned_start,
        scale=1.0,
        program_config=_chunked_sdpa_program_config(head_dim),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            tt_q.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    if not owns_q_in:
        return out_full

    q_in.deallocate(True)
    out = ttnn.slice(
        out_full,
        [0, 0, front, 0],
        [out_full.shape[0], out_full.shape[1], front + real_len, out_full.shape[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_full.deallocate(True)
    return out


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

    if len(outputs) == 1:
        # ``ttnn.concat([x])`` aliases ``x``; the cleanup below would then
        # deallocate the tensor returned to the caller.
        return outputs[0]

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
        program_config = _denoise_sdpa_program_config(head_dim, q_seq_len, k_seq_len, device=tt_q.device())
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
    paged_write=None,
    write_mode: str | None = None,
):
    """Causal masked prefix+canvas attention for one commit layer (writes K/V).

    Reuses the shared Gemma4 building blocks (QKV projection, prefill head split,
    per-head norm, output projection, all-reduce) and the diffusion-local chunked
    RoPE — the same ops as ``diffusion_attention.denoise_attention`` — so the K/V
    written here are computed by the identical projection + norm + RoPE the denoise
    and (per-token) commit paths use.
    """
    validate_q_rope_offset(start_pos)
    if (page_table is None) != (paged_write is None):
        raise ValueError("paged commit needs both the layer page table and its offset-write plan")
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

    paged_sdpa = None
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
        if page_table is None:
            _write_canvas_kv_contiguous(
                k_cache,
                v_cache,
                tt_k,
                tt_v,
                start_pos=start_pos,
                canvas_len=canvas_len,
                mesh_device=mesh_device,
                write_mode=write_mode,
            )
        else:
            if config.is_sliding and config.sliding_window is not None:
                # Read/attend before the circular bulk write can evict history
                # needed by early queries in this canvas.
                paged_sdpa = _sliding_commit_attention_paged(
                    tt_q,
                    tt_k,
                    tt_v,
                    kv_cache=kv_cache,
                    start_pos=start_pos,
                    sliding_window=config.sliding_window,
                    head_dim=config.head_dim,
                )
            _write_canvas_kv_paged(
                k_cache,
                v_cache,
                tt_k,
                tt_v,
                page_table=paged_write["page_table"],
                block_ids=paged_write["block_ids"],
                leading=paged_write["leading"],
                trailing=paged_write["trailing"],
                block_size=paged_write["block_size"],
                head_dim=config.head_dim,
                num_kv_heads=1 if weights.kv_replicated else config.num_key_value_heads // tp,
            )
            if paged_sdpa is None:
                # Full-attention layers can write first: their pool is append-only,
                # and the paged chunked op reads prefix + fresh canvas directly.
                paged_sdpa = _full_commit_attention_paged(
                    tt_q,
                    kv_cache=kv_cache,
                    page_table=page_table,
                    start_pos=start_pos,
                    head_dim=config.head_dim,
                )
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    if paged_sdpa is not None:
        tt_q.deallocate(True)
        tt_out = concat_heads(paged_sdpa, is_decode_mode=False)
        paged_sdpa.deallocate(True)
        tt_out = apply_output_projection(tt_out, weights)
        return apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    # Read the frozen prefix ++ freshly-written canvas out of the cache and run the
    # causal-masked SDPA. Reading from the cache (rather than a register concat)
    # means a KV-shared layer transparently sees the source layer's canvas K/V.
    full_k, full_v = _read_cache_kv(kv_cache, end_pos=start_pos + canvas_len)

    # tt_q is already DRAM-interleaved (chunked RoPE emits DRAM). Only convert when
    # it is not — an unconditional no-op ``to_memory_config`` returns a fresh, NOT-
    # allocated alias here, so the SDPA input dies. Mirrors the guarded decode path
    # (``commit_decode.py``).
    if tt_q.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tt_q_l1 = tt_q
        tt_q = ttnn.to_memory_config(tt_q_l1, ttnn.DRAM_MEMORY_CONFIG)
        tt_q_l1.deallocate(True)

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
    paged_write=None,
    write_mode: str | None = None,
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
        paged_write=paged_write,
        write_mode=write_mode,
    )
    normed.deallocate(True)

    attn_output = _chunked_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _chunked_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    mlp_output = shared_mlp_forward(layer.shared_mlp, normed)
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
    page_tables_per_layer=None,
    paged_writes=None,
    write_mode: str | None = None,
):
    """Run the full batched commit backbone: append every layer's canvas K/V.

    ``canvas_hidden`` is the ``[1, 1, C, H]`` embedded committed canvas. The K/V
    append happens inside each layer; the returned hidden states are discarded by
    the commit (no final norm / LM head — the commit throws away logits).
    """
    caches = kv_caches or tt_model.tt_kv_cache
    canvas_len = canvas_hidden.shape[-2]
    kv_shared_map = getattr(tt_model, "kv_shared_layer_map", {})

    # Per-layer-input (Gemma-3n E2B/E4B) is applied by the sequential commit but not
    # by the denoise body this mirrors. It is inactive for DiffusionGemma-26B-A4B
    # (an MoE, ``hidden_size_per_layer_input == 0``), so both paths agree. Guard so a
    # PLI-bearing model raises here instead of silently diverging from the sequential
    # commit (flag, don't force).
    if getattr(tt_model, "hidden_size_per_layer_input", 0):
        raise NotImplementedError(
            "batched commit does not apply per-layer inputs (E2B/E4B); this model has "
            "hidden_size_per_layer_input != 0, so use the sequential commit"
        )
    hidden_states = canvas_hidden
    for layer_idx in range(len(tt_model.layers)):
        layer_type = _layer_type_for_commit(tt_model, layer_idx)
        sliding_window = _sliding_window_for_commit(tt_model, layer_idx) if layer_type == "sliding_attention" else None
        layer_page_table = page_tables_per_layer[layer_idx] if page_tables_per_layer is not None else page_table
        paged_write = paged_writes.get(layer_type) if paged_writes is not None else None
        attn_mask = None
        if layer_page_table is None:
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
                page_table=layer_page_table,
                paged_write=paged_write,
                write_mode=write_mode,
            )
        finally:
            if attn_mask is not None:
                attn_mask.deallocate(True)
    return hidden_states


def commit_canvas_tokens_batched(
    tt_model,
    canvas_tokens: torch.Tensor,
    *,
    start_pos: int,
    page_table=None,
    page_tables_per_layer=None,
    write_mode: str | None = None,
) -> None:
    """Append committed canvas token ids to the KV cache in ONE causal prefill.

    Drop-in replacement for
    :func:`models.experimental.diffusion_gemma.tt.generate.commit_canvas_tokens`
    (same signature). Instead of 256 sequential single-token decode-appends, this
    embeds all ``canvas_len`` committed tokens and runs one causal masked prefill
    that writes every layer's K/V at positions ``start_pos .. start_pos+C-1``.

    A contiguous append is one ``ttnn.fill_cache`` per K/V by default; pass
    ``write_mode="position"`` (or ``DG_COMMIT_KV_WRITE=position``) for its
    per-position reference. Model-owned hybrid KV uses offset-staged
    ``paged_fill_cache`` writes plus paged attention. See the module docstring.
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
    if canvas_len % TILE_SIZE != 0:
        # Both ends of the write span must be tile-aligned. Reject here rather than
        # after the writes: the KV write is tile-granular, so a ragged canvas_len would
        # push tile-pad K/V (computed from the zero-padded embedding tail, i.e. not
        # zeros) into the cache past start_pos+canvas_len — and only then would
        # ``_read_cache_kv``'s end_pos check fire, with the cache already dirty.
        raise ValueError(
            f"batched commit requires canvas_len ({canvas_len}) to be a multiple of {TILE_SIZE}; "
            "the standard canvas is 256"
        )
    if page_table is not None:
        raise NotImplementedError(
            "batched commit does not support a legacy/shared page_table; use the sequential "
            "commit unless DG model-owned per-layer hybrid page tables are attached"
        )
    if page_tables_per_layer is None:
        # A hybrid-cache model without explicit tables (demo, tests) commits against
        # its own attached identity page tables; the contiguous writer below survives
        # only for unit tests that validate numerics on the simplest layout.
        page_tables_per_layer = getattr(tt_model, "_dg_hybrid_page_tables_per_layer", None)
    paged_writes = None
    if page_tables_per_layer is not None:
        if not getattr(tt_model, "_dg_model_owned_hybrid_kv", False):
            raise NotImplementedError(
                "batched paged commit currently requires DG's identity-mapped model-owned hybrid cache"
            )
        if len(page_tables_per_layer) != len(tt_model.layers):
            raise ValueError(
                f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"but model has {len(tt_model.layers)} layers"
            )
        if getattr(tt_model, "kv_shared_layer_map", {}):
            raise NotImplementedError("batched model-owned hybrid commit does not yet support KV-shared layers")

        block_size = int(getattr(tt_model, "_dg_hybrid_block_size", 0))
        sliding_window = int(getattr(tt_model, "_dg_hybrid_sliding_window", 0))
        max_seq_len = int(getattr(tt_model, "_dg_hybrid_max_seq_len", 0))
        if not block_size or not sliding_window or not max_seq_len:
            raise RuntimeError("model-owned hybrid cache metadata is incomplete")

        paged_writes = {}
        for layer_type, span, circular in (
            ("sliding_attention", sliding_window, True),
            ("full_attention", max_seq_len, False),
        ):
            num_blocks = span // block_size
            block_ids, leading, trailing = _paged_write_plan(
                start_pos=start_pos,
                canvas_len=canvas_len,
                block_size=block_size,
                num_blocks=num_blocks,
                circular=circular,
            )
            paged_writes[layer_type] = {
                "page_table": _to_device_page_table(tt_model.mesh_device, block_ids),
                "block_ids": block_ids,
                "leading": leading,
                "trailing": trailing,
                "block_size": block_size,
            }

    # commit_hidden_forward_batched consumes canvas_hidden through the layer stack
    # (layer 0's residual add deallocates it), so the caller does not free it.
    try:
        canvas_hidden = embed_host_tokens(tt_model, canvas_tokens)
        hidden = commit_hidden_forward_batched(
            tt_model,
            canvas_hidden,
            start_pos=start_pos,
            page_table=page_table,
            page_tables_per_layer=page_tables_per_layer,
            paged_writes=paged_writes,
            write_mode=write_mode,
        )
        hidden.deallocate(True)
    finally:
        if paged_writes is not None:
            for write in paged_writes.values():
                write["page_table"].deallocate(True)


def select_commit_fn(batched: bool | None = None):
    """Return the commit callable: batched by default.

    ``batched=None`` means batched. Kept here (not in ``generate``) so the commit
    dispatch lives with the batched implementation.
    """
    from models.experimental.diffusion_gemma.tt.generate import commit_canvas_tokens

    use_batched = True if batched is None else batched
    if use_batched:
        logger.info("[commit] using batched single-prefill commit (torch-verified correct, default)")
        return commit_canvas_tokens_batched
    return commit_canvas_tokens
