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
    into ``tt_kv_cache[i]`` at seq positions ``start_pos .. start_pos+C-1`` with ONE
    ``ttnn.fill_cache`` per K/V (the write span is tile-aligned, so it is a pure tile
    copy — see :func:`_write_canvas_kv_contiguous`);
  * read the full ``[0 : start_pos+C]`` K/V back from the cache (= frozen prefix ++
    freshly-written canvas) and run a causal-masked SDPA;
  * the MLP / MoE / norm tail is byte-identical to the denoise layer body.

Reading the K/V for the SDPA out of the cache (rather than concatenating a
register copy) means cross-layer **KV-sharing** is handled for free: a shared
layer skips its own K/V write, and its earlier source layer has already written
the canvas K/V into the shared cache tensor by the time the shared layer runs.

This path is the **default**
— it is both faster and more correct than the sequential decode-append commit, whose
decode-MoE kernel is defective. It never edits shared ``models/demos/gemma4`` code — it
composes over the importable Gemma4 ops, exactly like ``tt/commit_decode.py`` and
``tt/diffusion_attention.py``. See ``doc/optimize_perf/commit_batching.md`` for the
equivalence argument, the device results and the verify harnesses.
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
    split_qkv_heads_prefill,
)
from models.experimental.diffusion_gemma.tt.ccl import apply_allreduce
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

# KV-write mechanism for the contiguous (page_table=None) cache.
#   "fill"     — DEFAULT: ONE ``ttnn.fill_cache`` per K/V per layer writes the whole
#                canvas at the tile-aligned seq offset ``start_pos``. 2 ops/layer.
#   "position" — the legacy per-position write: 256 x (2 slice + 2 reshard + 2
#                ``paged_update_cache``) = ~1536 tiny dispatches/layer. Device-proven,
#                kept as the reference the "fill" path is verified bit-identical
#                against (``tests/test_device_commit_kv_write.py``) and as the
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
    # assuming no core's range crosses a kv-head boundary ("assume that work doesn't
    # spill over to next head" — fill_cache_multi_core_program_factory.cpp). That holds
    # whenever every core gets exactly one row, i.e. rows <= cores (split_work_to_cores:
    # units < max_cores ⇒ target_num_cores = units), and trivially when the input spans
    # the whole cache (C == max_seq ⇒ the destination is one contiguous run). Above that
    # the op silently writes some rows to the wrong head — device-confirmed with
    # nkv=8, C=1024, max_seq=2048 (49106 wrong elements). DiffusionGemma is far inside
    # the safe region (nkv <= 8, C = 256 ⇒ <= 64 rows vs 110 Blackhole cores), so this
    # only guards future geometries.
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

    ``write_mode="fill"`` (**default**): ONE ``ttnn.fill_cache`` per K/V writes the whole
    canvas at the tile-aligned seq offset ``update_idx=start_pos``. The commit's write
    span is tile-aligned by construction (``start_pos % 32 == 0`` is validated in
    :func:`commit_canvas_tokens_batched`, ``canvas_len`` is 256), so no read-modify-write
    of a partial tile is needed and FILL is a pure tile copy (reader → CB → writer, no
    compute kernel) — it writes exactly the ``[start_pos, start_pos+C)`` tile-rows of each
    kv head and touches neither the frozen prefix nor the tail. Device-verified
    bit-identical to the per-position path over the whole cache
    (``tests/test_device_commit_kv_write.py``): 2 ops/layer instead of ~1536, 9.59 ms →
    0.012 ms per layer-write at the 26B-A4B geometry. Falls back to ``"position"`` (with a
    warning) if any precondition fails — see :func:`_fill_write_unsupported_reason`.

    Trace caveat (applies to BOTH write modes, so this is not a regression): ``start_pos``
    lives in the op's runtime args and is excluded from the program-cache hash, so a
    commit captured into a metal trace would replay the offset it was captured at. The
    commit runs eagerly today (``traced_denoise.py`` traces the denoise loop only); tracing
    it later needs a per-block re-capture or a device-side index tensor.

    ``write_mode="position"``: one ``paged_update_cache`` per committed position — the
    exact non-paged decode-append op the sequential path uses, so the write positions and
    cache layout are provably identical. A single-sequence contiguous cache (batch 1) can
    only address one seq position per non-paged update, hence the 256-iteration loop.
    This is the reference the fill path is verified against.

    **Do not reintroduce a batched** ``paged_update_cache`` **here.** The obvious-looking
    speedup for the per-position loop — view the contiguous cache as a 1-block paged cache
    (``block_size = max_seq``, all-zero ``page_table``) and write ``n`` positions per op as
    ``n`` "users" — is racy by construction, and the reshard that makes it *run* is what
    makes it dangerous. ``paged_update_cache`` is a per-TILE read-modify-write, and ``n``
    consecutive positions share ONE 32-row cache tile, so all ``n`` cores compute the same
    ``cache_id`` and last-writer-wins: the same failure ``gemma4/tt/attention/decode.py``
    serializes around (``sequential_kv_write``, issue #44923), with no rescue here because
    the op's only serialization (``in0_sequential_mode``) is wired for ``share_cache``,
    which paged mode rejects outright. Without the reshard it fails loudly on the op's
    ``is_sharded()`` assert; with it, the cache corrupts silently. A race-free variant
    (stride-32 grouping, one user per tile) caps at 8 users for a 256-token canvas →
    ≥258 dispatches/layer and no DRAM saving. The fill write makes all of it moot; the
    experiment is written up in ``doc/optimize_perf/commit_batching.md``.
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
    if page_table is not None:
        # Reject before any device work so we never leave a half-written cache: the
        # paged commit (offset chunk write + SDPA prefix read from the paged pool) is
        # not wired. The standalone / serving RUN path uses the contiguous model-owned
        # cache (page_table=None).
        raise NotImplementedError(
            "batched commit does not support paged caches yet (SDPA prefix read from the paged "
            "pool is unwired); use the sequential commit for paged/vLLM caches (#47557/#47488)"
        )
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
        tt_k.deallocate(True)
        tt_v.deallocate(True)

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
                write_mode=write_mode,
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
    write_mode: str | None = None,
) -> None:
    """Append committed canvas token ids to the KV cache in ONE causal prefill.

    Drop-in replacement for
    :func:`models.experimental.diffusion_gemma.tt.generate.commit_canvas_tokens`
    (same signature). Instead of 256 sequential single-token decode-appends, this
    embeds all ``canvas_len`` committed tokens and runs one causal masked prefill
    that writes every layer's K/V at positions ``start_pos .. start_pos+C-1``.

    Each layer's K/V append is one ``ttnn.fill_cache`` per K/V by default; pass
    ``write_mode="position"`` (or ``DG_COMMIT_KV_WRITE=position``) for the per-position
    reference write. See the module docstring and ``doc/optimize_perf/commit_batching.md``
    for the equivalence argument and the device verify harnesses.
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
    if page_table is not None or page_tables_per_layer is not None:
        raise NotImplementedError(
            "batched commit supports only the contiguous model-owned cache (page_table=None); "
            "use the sequential commit for paged / vLLM hybrid-cache paths (#47557/#47488)"
        )

    # commit_hidden_forward_batched consumes canvas_hidden through the layer stack
    # (layer 0's residual add deallocates it), so the caller does not free it.
    canvas_hidden = embed_host_tokens(tt_model, canvas_tokens)
    hidden = commit_hidden_forward_batched(
        tt_model,
        canvas_hidden,
        start_pos=start_pos,
        page_table=page_table,
        write_mode=write_mode,
    )
    hidden.deallocate(True)


def select_commit_fn(batched: bool | None = None):
    """Return the commit callable: batched (default) unless forced off / paged.

    ``batched=None`` means batched. Kept here (not in ``generate``) so the commit dispatch lives
    with the batched implementation. The ``DG_COMMIT_BATCHED`` env override was deleted
    2026-07-28: the path it forced measures PCC 0.154 vs 0.994 and ~6.3x slower, and the paged case
    it cited is already selected automatically in ``tt/generate.py``.
    """
    from models.experimental.diffusion_gemma.tt.generate import commit_canvas_tokens

    use_batched = True if batched is None else batched
    if use_batched:
        logger.info("[commit] using batched single-prefill commit (torch-verified correct, default)")
        return commit_canvas_tokens_batched
    return commit_canvas_tokens
