# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter attention: the one genuinely DFlash-specific op. Query comes only from
the noise/draft block; key and value are the concatenation of the (fixed) context and the
noise block's own key/value -- see models/demos/gemma4/docs/dflash_design.md section 1.

Context's K/V are a genuinely INCREMENTAL per-layer cache (see
``project_and_cache_context_delta``): projected+normed+RoPE'd ONCE, when each iteration's
newly-committed-token tap arrives, and read as-is (no reprojection) by every later
attention forward call -- matching the reference's own ``past_key_values_draft``
(dflash.py) compute profile, not the earlier (correct but wasteful) approach of
reprojecting the WHOLE context buffer from scratch on every call.

Head reshape/RoPE/SDPA call pattern mirrors gated_attention_forward_ttnn
(models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py), already
proven on real T3K hardware for the architecturally-similar Qwen3-style Qwen3.6 MTP head --
plain reshape+transpose for heads (no nlp_create_qkv_heads needed), plain (non-zero-centered)
ttnn.rms_norm for q_norm/k_norm, and ttnn.transformer.scaled_dot_product_attention handling
GQA internally (no manual KV-head repeat needed).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.gemma4.tt.attention.weights import AttentionWeights
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import rotate_half_ttnn

_TILE_HEIGHT = 32


def _dflash_pad_noise_concat_enabled() -> bool:
    """Pad the noise block's K/V to a TILE_HEIGHT (32) multiple before
    ``ttnn.concat``-ing them onto the (already tile-aligned) context cache,
    instead of concatenating at the drafter's raw ``block_size`` (16) width.

    ``ttnn.concat`` falls back to a generic untilize -> splice -> retile
    composite whenever either operand's row count isn't a multiple of 32 (the
    checkpoint's ``block_size=16`` guarantees this every call) -- confirmed on
    real hardware via an isolated repro matching the real op names/core counts
    seen in ``test_dflash_drafter_tracy.py`` captures (UntilizeWithUnpadding
    x2, Concat, TilizeWithValPadding; ~16.6us + 9.6us + 3.3us + 9.2us =
    ~38.7us/device per concat call). Padding both operands to a 32-row
    boundary first lets ``ttnn.concat`` take its native tile-aligned path
    instead: same isolated repro measured ~3us/device -- roughly 13x less,
    twice per layer (K and V) times every drafter layer.

    The padding rows are never real content -- they must be masked out of
    attention entirely, which is why this flag also changes
    ``build_attention_mask_additive_device``/``_dynamic``/
    ``build_attention_mask_static_parts``: every mask-building call site needs
    the SAME padded width, or the mask's last dim silently no longer matches
    K's sequence length. ``dflash_drafter_forward``'s ``mask_for`` closure is
    the only caller of those builders and passes this same flag through, so
    producer (the K/V pad here) and consumer (the mask width) can't disagree.

    Default ON. Beyond the isolated bit-exact check above, also run end-to-end
    in ``demo/dflash_fused_decoder_demo.py`` alongside
    ``GEMMA4_ROPE_EXPAND_GATHER``/``GEMMA4_KV_FUSED_WRITE`` (66.4 tok/s, 260
    tokens over 39 iterations, mean 5.67 accepted drafts/iteration, coherent
    generated output) — real speculative-decoding traffic through this flag,
    not just the isolated concat repro. That demo run checks liveness and
    output plausibility, not a token-for-token match against the flag
    disabled; a strict bit-exact diff of full-sequence output with this flag
    on vs. off has NOT been run. ``GEMMA4_DFLASH_PAD_NOISE_CONCAT=0`` restores
    the previous unpadded-concat path, as an escape hatch.
    """
    return os.environ.get("GEMMA4_DFLASH_PAD_NOISE_CONCAT", "1").lower() not in ("0", "false", "no")


def _tile_pad_len(n: int, tile: int = _TILE_HEIGHT) -> int:
    return ((n + tile - 1) // tile) * tile


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
    mesh_device, ctx_len: int, q_len: int, is_causal: bool, sliding_window: int | None, q_len_padded: int | None = None
) -> ttnn.Tensor:
    """On-device equivalent of ``build_attention_mask_additive`` -- same formula, built
    entirely with ttnn ops (``ttnn.arange``/``le``/``lt``/``logical_and``/``where``), no
    host torch computation or upload. Confirmed exact match (bf16) against the host-torch
    version, cast to bf16 the same way the caller would (``ttnn.from_torch(...,
    dtype=bfloat16)``), for every (ctx_len, q_len, is_causal, sliding_window) combination
    this pipeline actually uses. Returns [1,1,q_len,ctx_len+q_len] bf16, replicated.

    ``q_len_padded``: when given (> q_len), the mask's key axis extends to
    ``ctx_len + q_len_padded`` instead of ``ctx_len + q_len``, with the extra
    trailing ``q_len_padded - q_len`` columns forced invisible for every query
    row regardless of causal/sliding rules. Matches
    ``_dflash_pad_noise_concat_enabled``'s K/V padding in
    ``dflash_attention_forward`` -- those trailing columns are where the
    padding rows land after ``ttnn.concat``, and they carry no real content."""
    total_real = ctx_len + q_len
    total = ctx_len + (q_len_padded if q_len_padded else q_len)
    query_position = ttnn.arange(total_real - q_len, total_real, 1, device=mesh_device, dtype=ttnn.int32)
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
    if total > total_real:
        visible = ttnn.logical_and(visible, ttnn.lt(key_full, total_real))

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
    q_len_padded: int | None = None,
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

    The noise block's query positions are ``context_valid_len_tt + local_row_index``, NOT
    ``ctx_len + local_row_index`` -- ``context_valid_len_tt``'s value IS the sequence's true
    absolute position where the noise block begins (generate.py's ``context_len``, "always
    the count of real rows accumulated so far, which is exactly the next absolute
    position"), while ``ctx_len`` is only the FIXED buffer width and generally differs from
    it during growth. Using ``ctx_len`` here was a bug (see generate.py's module docstring,
    "KNOWN LATENT LIMITATION" -- now fixed): it left the causal check and the
    context-validity check unaffected (their own reasoning doesn't depend on the query's
    absolute value), but skewed the SLIDING-WINDOW distance check for a noise query
    attending a context key by exactly ``ctx_len - context_valid_len_tt`` -- invisible while
    ``sliding_window`` comfortably exceeds ``ctx_len`` (every config tested before this fix),
    increasingly wrong as ``ctx_len`` approaches or exceeds ``sliding_window``. Verified
    against a host-torch reference using true absolute positions, including configs where
    the old formula demonstrably diverged (test_dflash_sliding_window_mask.py).

    NOT trace-safe by itself: builds the static grids from scratch every call via
    ``ttnn.arange``/``ones``/``zeros``/``full``, each of which does a host->device WRITE
    internally (confirmed: ``creation.cpp``'s ``arange_impl``/``full_impl`` build a host
    ``std::vector`` and upload it via ``Tensor::from_vector``) -- disallowed mid-capture
    (``TT_FATAL: Writes are not supported during trace capture``). Eager (non-traced)
    callers only. For the traced steady-state loop, use
    ``build_attention_mask_static_parts`` (once, outside capture) +
    ``combine_attention_mask_dynamic`` (inside the captured body, per replay) instead --
    identical result, split so only the ``context_valid_len_tt``-dependent combine step
    re-runs on every replay.

    ``q_len_padded``: same meaning as ``build_attention_mask_additive_device`` -- extends
    the key axis to ``ctx_len + q_len_padded`` with the trailing padding columns forced
    invisible, independent of (and applied after) the context-validity condition below."""
    total_real = ctx_len + q_len
    total = ctx_len + (q_len_padded if q_len_padded else q_len)
    local_query_idx = ttnn.arange(0, q_len, 1, device=mesh_device, dtype=ttnn.int32)
    key_position = ttnn.arange(0, total, 1, device=mesh_device, dtype=ttnn.int32)
    local_query_col = ttnn.reshape(local_query_idx, [q_len, 1])
    key_row = ttnn.reshape(key_position, [1, total])
    key_full = ttnn.repeat(key_row, ttnn.Shape([q_len, 1]))

    # True absolute query position = context_valid_len_tt (the sequence's real current
    # length) + local row offset -- NOT ctx_len + local row offset (see docstring).
    start_col = ttnn.repeat(ttnn.reshape(context_valid_len_tt, [1, 1]), ttnn.Shape([q_len, 1]))
    query_col = ttnn.add(local_query_col, start_col)
    query_full = ttnn.repeat(query_col, ttnn.Shape([1, total]))

    visible = ttnn.ones([q_len, total], dtype=ttnn.int32, device=mesh_device)
    if is_causal:
        visible = ttnn.logical_and(visible, ttnn.le(key_full, query_full))
    if sliding_window is not None:
        visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(query_full, key_full), sliding_window))
        if not is_causal:
            visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(key_full, query_full), sliding_window))

    # context_valid_len_tt IS the true start (see above) -- reuse start_col/query_col's
    # broadcast rather than rebuilding it.
    valid_len_full = ttnn.repeat(start_col, ttnn.Shape([1, total]))
    is_valid_context_col = ttnn.lt(key_full, valid_len_full)  # key_position < context_valid_len
    is_noise_col = ttnn.ge(key_full, ctx_len)  # noise columns are unaffected by context padding
    visible = ttnn.logical_and(visible, ttnn.logical_or(is_valid_context_col, is_noise_col))
    if total > total_real:
        visible = ttnn.logical_and(visible, ttnn.lt(key_full, total_real))

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
    captured trace body.

    The causal and sliding-window checks are NOT part of ``visible_base`` (unlike an
    earlier version of this class) -- they depend on the query's TRUE absolute position,
    which is ``context_valid_len_tt``'s per-replay value, not anything fixed at build
    time. ``visible_base`` here holds only the trailing-padding-columns-invisible
    condition, which genuinely is value-independent. ``local_query_full`` is the LOCAL
    (0-based) query row index broadcast over columns; ``combine_attention_mask_dynamic``
    adds the per-replay start to it to get the true query position before running the
    causal/sliding checks."""

    __slots__ = (
        "key_full",
        "local_query_full",
        "visible_base",
        "is_noise_col",
        "zero",
        "neg",
        "q_len",
        "total",
        "is_causal",
        "sliding_window",
    )

    def __init__(
        self, key_full, local_query_full, visible_base, is_noise_col, zero, neg, q_len, total, is_causal, sliding_window
    ):
        self.key_full = key_full
        self.local_query_full = local_query_full
        self.visible_base = visible_base
        self.is_noise_col = is_noise_col
        self.zero = zero
        self.neg = neg
        self.q_len = q_len
        self.total = total
        self.is_causal = is_causal
        self.sliding_window = sliding_window


def build_attention_mask_static_parts(
    mesh_device, ctx_len: int, q_len: int, is_causal: bool, sliding_window: int | None, q_len_padded: int | None = None
) -> DynamicMaskStaticParts:
    """One-time setup (call OUTSIDE ``begin_trace_capture``): builds every piece of
    ``build_attention_mask_additive_device_dynamic`` that's independent of
    ``context_valid_len_tt``'s value -- the position grids, the trailing-padding-columns
    condition, the noise-column exemption, and the bf16 zero/neg constants. Uses
    ``ttnn.arange``/``ones``/``zeros``/``full`` (host-write ops), which is fine here since
    this runs once, before capture begins -- see ``combine_attention_mask_dynamic`` for the
    per-replay half that's actually inside the trace (which now includes the
    causal/sliding-window checks -- see ``DynamicMaskStaticParts``'s docstring for why
    those moved here from ``visible_base``).

    ``q_len_padded``: same meaning as ``build_attention_mask_additive_device``. The
    trailing-padding-columns-invisible condition is independent of
    ``context_valid_len_tt``, so it's folded into ``visible_base`` here rather than
    needing any change in ``combine_attention_mask_dynamic``."""
    total_real = ctx_len + q_len
    total = ctx_len + (q_len_padded if q_len_padded else q_len)
    local_query_idx = ttnn.arange(0, q_len, 1, device=mesh_device, dtype=ttnn.int32)
    key_position = ttnn.arange(0, total, 1, device=mesh_device, dtype=ttnn.int32)
    local_query_col = ttnn.reshape(local_query_idx, [q_len, 1])
    key_row = ttnn.reshape(key_position, [1, total])
    local_query_full = ttnn.repeat(local_query_col, ttnn.Shape([1, total]))
    key_full = ttnn.repeat(key_row, ttnn.Shape([q_len, 1]))

    visible_base = ttnn.ones([q_len, total], dtype=ttnn.int32, device=mesh_device)
    if total > total_real:
        visible_base = ttnn.logical_and(visible_base, ttnn.lt(key_full, total_real))

    is_noise_col = ttnn.ge(key_full, ctx_len)  # noise columns are unaffected by context padding

    zero = ttnn.zeros([q_len, total], dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    neg = ttnn.full([q_len, total], fill_value=-1e4, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    return DynamicMaskStaticParts(
        key_full, local_query_full, visible_base, is_noise_col, zero, neg, q_len, total, is_causal, sliding_window
    )


def combine_attention_mask_dynamic(static: DynamicMaskStaticParts, context_valid_len_tt: ttnn.Tensor) -> ttnn.Tensor:
    """The per-replay half: ONLY elementwise ops on already-existing device tensors
    (``reshape``/``repeat``/``add``/``subtract``/``le``/``lt``/``logical_and``/
    ``logical_or``/``where``/``typecast``/``to_layout``) -- no ``arange``/``ones``/
    ``zeros``/``full``, so no host->device write, safe to call from inside a captured
    trace body. Recomputes exactly the ``context_valid_len_tt``-dependent part of
    ``build_attention_mask_additive_device_dynamic``, using ``static``'s precomputed,
    value-independent pieces for everything else. Confirmed bit-for-bit identical to that
    function's output for the same inputs.

    ``context_valid_len_tt`` IS the true absolute start position (see
    ``build_attention_mask_additive_device_dynamic``'s docstring), so its broadcast is
    reused for both the query-position correction and the context-validity check below --
    they are, numerically, the same quantity used two different ways. The causal and
    sliding-window checks are computed HERE (not in ``static.visible_base``) because they
    depend on the query's true position, which only exists once ``context_valid_len_tt``'s
    value is known -- this is the one part of this function that's genuinely new work per
    replay rather than a value-independent constant, an unavoidable cost of the fix (see
    ``DynamicMaskStaticParts``)."""
    q_len, total = static.q_len, static.total
    start_col = ttnn.repeat(ttnn.reshape(context_valid_len_tt, [1, 1]), ttnn.Shape([q_len, 1]))
    start_full = ttnn.repeat(start_col, ttnn.Shape([1, total]))
    query_full = ttnn.add(static.local_query_full, start_full)

    visible = static.visible_base
    if static.is_causal:
        visible = ttnn.logical_and(visible, ttnn.le(static.key_full, query_full))
    if static.sliding_window is not None:
        visible = ttnn.logical_and(visible, ttnn.lt(ttnn.subtract(query_full, static.key_full), static.sliding_window))
        if not static.is_causal:
            visible = ttnn.logical_and(
                visible, ttnn.lt(ttnn.subtract(static.key_full, query_full), static.sliding_window)
            )

    is_valid_context_col = ttnn.lt(static.key_full, start_full)
    visible = ttnn.logical_and(visible, ttnn.logical_or(is_valid_context_col, static.is_noise_col))
    visible = ttnn.to_layout(ttnn.typecast(visible, ttnn.bfloat16), ttnn.TILE_LAYOUT)
    mask = ttnn.where(visible, static.zero, static.neg)
    return ttnn.unsqueeze_to_4D(ttnn.unsqueeze_to_4D(mask))  # [1,1,q_len,total]


def _apply_rope_single(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotate_half_ttnn(x), sin))


def _dflash_kv_fill_cache_enabled() -> bool:
    """Write the drafter's per-layer K/V cache delta via ``ttnn.fill_cache`` (one
    in-place device write covering only the touched rows) instead of
    ``_write_seq_slice``'s current slice-head + slice-tail + concat-the-whole-buffer +
    copy-the-whole-buffer dance.

    ``_write_seq_slice`` currently reconstructs and rewrites the ENTIRE
    ``[1,heads,max_seq_len,head_dim]`` cache on every single delta commit, no matter how
    small ``length`` is -- O(max_seq_len) data movement (2 slices + 1 concat + 1
    full-buffer copy) to write ``length`` rows. ``ttnn.fill_cache(cache, input,
    batch_idx=0, update_idx=...)`` (``ttnn/cpp/ttnn/operations/kv_cache/kv_cache.cpp``)
    is a purpose-built in-place write that only touches ``[update_idx : update_idx +
    input.shape[-2]]`` -- no read/rewrite of the untouched tail at all -- but its device
    op (``update_cache_device_operation.cpp``) hard-requires ``update_idx % TILE_HEIGHT
    (32) == 0``.

    ``offset`` here is NOT tile-aligned in general: generate.py's ``context_len``
    (this function's ``offset``) advances by ``produced = min(accept + 1, block_size)``
    every drafting iteration -- a variable count of accepted speculative tokens, not a
    32-multiple. So a plain drop-in swap would ``TT_FATAL`` after the first iteration in
    real use (confirmed by reading ``generate.py``'s call sites, not assumed). This
    rounds ``offset`` DOWN to its containing tile boundary and reads back the <=31-row
    in-tile prefix that exposes (``[aligned_start:offset]`` -- already real content,
    rewritten with its own unchanged values, not new data).

    A second, initially-missed constraint applies at the OTHER end too: the device op
    (``fill_cache_multi_core_program_factory.cpp``) sizes its write from the input
    tensor's PADDED row count (``input_tensor.padded_shape()[-2] / TILE_HEIGHT``), which
    rounds ``length`` UP to the next tile multiple -- not the logical ``length`` itself.
    A first version of this that only fixed the start alignment was caught by
    ``test_dflash_kv_fill_cache.py``'s bit-exact check: at ``length=16`` (not a
    32-multiple), the input tensor's tile padding (zero-filled by ``ttnn.from_torch``)
    got written into the next 16 real cache rows past ``offset+length``, corrupting
    them. So this also reads back a <=31-row in-tile SUFFIX (``[offset+length :
    aligned_end]``, ``aligned_end`` = ``offset+length`` rounded up) and appends it
    before the ``fill_cache`` call, so every row actually written -- including the
    tile-padding rows the kernel touches regardless -- has real, correct content.

    Total touched width is bounded by ``31 + length + 31`` (<=78 for DFlash's
    block_size=16), independent of ``max_seq_len`` -- vs. the current path's full
    ``max_seq_len``. Verified bit-exact against ``_write_seq_slice`` for both aligned
    and unaligned offsets, aligned and unaligned lengths, and a write crossing a tile
    boundary (``test_dflash_kv_fill_cache.py``).

    Default OFF (``GEMMA4_DFLASH_KV_FILL_CACHE=1`` to enable) pending bit-exact +
    device-time verification on real hardware, same rollout convention as this file's
    ``GEMMA4_DFLASH_PAD_NOISE_CONCAT`` and ``attention/decode.py``'s
    ``GEMMA4_ROPE_EXPAND_GATHER`` / ``GEMMA4_KV_FUSED_WRITE``.
    """
    return os.environ.get("GEMMA4_DFLASH_KV_FILL_CACHE", "0").lower() in ("1", "true", "yes")


def _write_seq_slice_fill_cache(buf: ttnn.Tensor, new_rows: ttnn.Tensor, offset: int, length: int) -> None:
    b, h, max_len, d = buf.shape
    aligned_start = (offset // _TILE_HEIGHT) * _TILE_HEIGHT
    write_end = offset + length
    aligned_end = min(_tile_pad_len(write_end), max_len)
    prefix_len = offset - aligned_start
    suffix_len = aligned_end - write_end

    pieces = []
    prefix = suffix = None
    if prefix_len > 0:
        prefix = ttnn.slice(buf, [0, 0, aligned_start, 0], [b, h, offset, d])
        pieces.append(prefix)
    pieces.append(new_rows)
    if suffix_len > 0:
        suffix = ttnn.slice(buf, [0, 0, write_end, 0], [b, h, aligned_end, d])
        pieces.append(suffix)

    combined = ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0]
    ttnn.fill_cache(buf, combined, batch_idx=0, update_idx=aligned_start)
    if len(pieces) > 1:
        ttnn.deallocate(combined)
    if prefix is not None:
        ttnn.deallocate(prefix)
    if suffix is not None:
        ttnn.deallocate(suffix)


def _write_seq_slice(buf: ttnn.Tensor, new_rows: ttnn.Tensor, offset: int, length: int) -> None:
    """Write ``new_rows`` (exactly ``length`` rows along the sequence axis, dim=2) into
    ``buf`` at [offset:offset+length], leaving every other row untouched -- an
    in-place-content update (``ttnn.copy``) of a persistent, fixed-shape buffer. Shape
    generic: works for a per-layer K/V cache ([1,heads,seq,head_dim]) or any other
    [B,H,seq,D]-shaped persistent buffer. Runs eagerly (offset/length are only known
    after each iteration's own host-side accept decision), so plain python-int slice
    bounds are fine -- see generate.py's module docstring.

    See ``_dflash_kv_fill_cache_enabled`` for a faster (``ttnn.fill_cache``-based)
    opt-in alternative to this function's default slice/concat/full-buffer-copy path.
    """
    if _dflash_kv_fill_cache_enabled():
        _write_seq_slice_fill_cache(buf, new_rows, offset, length)
        return
    b, h, max_len, d = buf.shape
    pieces = []
    head = tail = None
    if offset > 0:
        head = ttnn.slice(buf, [0, 0, 0, 0], [b, h, offset, d])
        pieces.append(head)
    pieces.append(new_rows)
    tail_start = offset + length
    if tail_start < max_len:
        tail = ttnn.slice(buf, [0, 0, tail_start, 0], [b, h, max_len, d])
        pieces.append(tail)
    updated = ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0]
    ttnn.copy(updated, buf)
    if len(pieces) > 1:
        ttnn.deallocate(updated)
    if head is not None:
        ttnn.deallocate(head)
    if tail is not None:
        ttnn.deallocate(tail)


def project_and_cache_context_delta(
    delta: ttnn.Tensor,  # [1,1,>=length,hidden] -- sliced internally to the first `length` rows
    length: int,
    weights: AttentionWeights,
    cos_delta: ttnn.Tensor,  # [1,1,>=length,head_dim] -- sliced internally to match `length`
    sin_delta: ttnn.Tensor,
    k_cache: ttnn.Tensor,  # [1,num_local_kv_heads,max_seq_len,head_dim], written at [offset:offset+length]
    v_cache: ttnn.Tensor,
    offset: int,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Project this iteration's newly-committed-token context tap (``delta``, already
    ``hidden_norm(fc(...))``'d -- see context.py) through THIS layer's own k_proj/v_proj
    (via the fused ``wqkv`` weight) + k_norm + RoPE (at the delta's own true absolute
    positions), then write the result into ``k_cache``/``v_cache`` -- the persistent,
    incremental analogue of what ``dflash_attention_forward`` used to recompute from
    scratch, over the FULL context buffer, on every single attention forward call.

    RoPE and k_norm are both per-position/per-row operations (no cross-row mixing), so
    projecting+normalizing+rotating just the new delta here and concatenating it with
    already-cached rows at attention time is mathematically identical to reprojecting
    the whole history every call -- confirmed against the whole-buffer-reprojection
    version this replaces. Runs eagerly (offset/length only known after this iteration's
    own host-side accept decision), matching generate.py's write-timing conventions."""
    hidden = delta.shape[-1]
    delta = delta if delta.shape[-2] == length else ttnn.slice(delta, [0, 0, 0, 0], [1, 1, length, hidden])
    if cos_delta.shape[-2] != length:
        cos_delta = ttnn.slice(cos_delta, [0, 0, 0, 0], [1, 1, length, head_dim])
        sin_delta = ttnn.slice(sin_delta, [0, 0, 0, 0], [1, 1, length, head_dim])

    q_w = num_local_heads * head_dim
    kv_w = num_local_kv_heads * head_dim

    qkv_delta = ttnn.linear(delta, weights.wqkv)  # [1,1,length, q_w+2*kv_w] -- the q slice is unused, same as before
    k_delta = ttnn.slice(qkv_delta, [0, 0, 0, q_w], [1, 1, length, q_w + kv_w])
    v_delta = ttnn.slice(qkv_delta, [0, 0, 0, q_w + kv_w], [1, 1, length, q_w + 2 * kv_w])
    ttnn.deallocate(qkv_delta)

    k_delta = ttnn.reshape(k_delta, [1, length, num_local_kv_heads, head_dim])
    k_delta = ttnn.rms_norm(k_delta, weight=weights.k_norm_weight, epsilon=eps)
    k_delta = ttnn.transpose(k_delta, 1, 2)  # [1, num_local_kv_heads, length, head_dim]
    k_delta = _apply_rope_single(k_delta, cos_delta, sin_delta)

    v_delta = ttnn.reshape(v_delta, [1, length, num_local_kv_heads, head_dim])
    v_delta = ttnn.transpose(v_delta, 1, 2)  # v is never RoPE'd, matching the reference

    _write_seq_slice(k_cache, k_delta, offset, length)
    _write_seq_slice(v_cache, v_delta, offset, length)
    ttnn.deallocate(k_delta)
    ttnn.deallocate(v_delta)


def dflash_attention_forward(
    k_cache: ttnn.Tensor,  # [1,num_local_kv_heads,max_seq_len,head_dim], already projected+normed+RoPE'd
    v_cache: ttnn.Tensor,  # [1,num_local_kv_heads,max_seq_len,head_dim], already projected
    noise: ttnn.Tensor,  # [1,1,q_len,hidden], full-width replicated (already input_layernorm'd)
    weights: AttentionWeights,
    cos_noise: ttnn.Tensor,  # [1,1,q_len,head_dim], replicated
    sin_noise: ttnn.Tensor,
    attn_mask: ttnn.Tensor,  # [1,1,q_len,max_seq_len+q_len], replicated
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
) -> ttnn.Tensor:
    """Context's K/V come straight from the persistent, already-projected+RoPE'd
    per-layer cache (no projection here at all -- see ``project_and_cache_context_delta``
    for where/when that happens) -- only the noise block's own Q/K/V are computed live
    here, exactly as every call already did before this cache existed."""
    scale = head_dim**-0.5
    q_w = num_local_heads * head_dim
    kv_w = num_local_kv_heads * head_dim

    q_len = noise.shape[-2]
    ctx_len = k_cache.shape[-2]

    qkv_noise = ttnn.linear(noise, weights.wqkv)  # [1,1,q_len, q_w+2*kv_w] per device

    q = ttnn.slice(qkv_noise, [0, 0, 0, 0], [1, 1, q_len, q_w])
    k_noise = ttnn.slice(qkv_noise, [0, 0, 0, q_w], [1, 1, q_len, q_w + kv_w])
    v_noise = ttnn.slice(qkv_noise, [0, 0, 0, q_w + kv_w], [1, 1, q_len, q_w + 2 * kv_w])
    ttnn.deallocate(qkv_noise)

    # heads: [1,1,seq,W] -> [1,seq,heads,head_dim] -> norm (last axis) -> transpose -> [1,heads,seq,head_dim]
    q = ttnn.reshape(q, [1, q_len, num_local_heads, head_dim])
    q = ttnn.rms_norm(q, weight=weights.q_norm_weight, epsilon=eps)
    q = ttnn.transpose(q, 1, 2)
    q = _apply_rope_single(q, cos_noise, sin_noise)

    k_noise = ttnn.reshape(k_noise, [1, q_len, num_local_kv_heads, head_dim])
    k_noise = ttnn.rms_norm(k_noise, weight=weights.k_norm_weight, epsilon=eps)
    k_noise = ttnn.transpose(k_noise, 1, 2)
    k_noise = _apply_rope_single(k_noise, cos_noise, sin_noise)

    v_noise = ttnn.reshape(v_noise, [1, q_len, num_local_kv_heads, head_dim])
    v_noise = ttnn.transpose(v_noise, 1, 2)

    if _dflash_pad_noise_concat_enabled():
        # See _dflash_pad_noise_concat_enabled: ttnn.concat falls back to an
        # untilize/splice/retile composite whenever either operand's row count
        # isn't a TILE_HEIGHT (32) multiple -- block_size=16 guarantees that
        # every call. Padding both operands up to 32 first lets concat take
        # its native tile-aligned path. The padding rows are never attended
        # to: attn_mask's caller (dflash_drafter_forward's mask_for) must pad
        # its own key axis to the identical width with the same flag, or this
        # concat's output width silently disagrees with attn_mask's last dim.
        pad_len = _tile_pad_len(q_len) - q_len
        if pad_len:
            k_noise = ttnn.pad(k_noise, [(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
            v_noise = ttnn.pad(v_noise, [(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)

    k = ttnn.concat([k_cache, k_noise], dim=2)  # [1,num_local_kv_heads,ctx_len+q_len(_padded),head_dim]
    v = ttnn.concat([v_cache, v_noise], dim=2)
    ttnn.deallocate(k_noise)
    ttnn.deallocate(v_noise)

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
