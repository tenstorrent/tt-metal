# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Inject SigLIP2 vision features into the LLM token sequence (HunyuanImage-3.0
# Instruct / image-to-image input path).
#
# The reference (`HunyuanImage3...instantiate_vit_image_tokens`) does an in-place
# masked scatter of the projected vision embeddings into the `<img>` placeholder
# positions of the text-token embeddings. For the common single-image case the
# `<img>` tokens form a CONTIGUOUS, TILE-aligned span, so the scatter reduces to a
# device-side `[text_pre | image_embeds | text_post]` concat — exactly the scatter
# the T2I gen-image path already uses (see `tt/pipeline.py` `HunyuanTtDenoiseStep._scatter`).
#
# Two entry points, in increasing generality — both pure on-device (no host round-trip):
#   scatter_cond_vision_embeddings        — single contiguous TILE-aligned <img> span
#                                           (TILE-layout concat; the cheapest path)
#   scatter_cond_vision_embeddings_multi  — one or more contiguous spans (multi-image).
#                                           TILE-aligned spans use TILE concat; otherwise
#                                           it falls back to a ROW_MAJOR concat that
#                                           allows arbitrary (non-32) span boundaries —
#                                           still entirely on device.

import ttnn

TILE = 32


def _align_to(tokens: ttnn.Tensor, ref: ttnn.Tensor):
    """Match `tokens` to `ref`'s layout/dtype (concat needs every piece to agree).

    Returns `(aligned, made_copy)` — `made_copy` is True iff a new tensor was
    allocated (so the caller knows whether to deallocate it).
    """
    out = tokens
    if out.layout != ttnn.TILE_LAYOUT:
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    if out.dtype != ref.dtype:
        out = ttnn.typecast(out, ref.dtype)
    return out, (out is not tokens)


def scatter_cond_vision_embeddings(hidden: ttnn.Tensor, image_embeds: ttnn.Tensor, img_slice: slice) -> ttnn.Tensor:
    """Write projected vision features into a contiguous `<img>` span on device.

    Args:
        hidden:       ttnn [B, S, H] TILE — text-token embeddings; the rows in
                      `img_slice` are placeholders (`<img>`) that get overwritten.
        image_embeds: ttnn [B, n_img, H] — SigLIP2 + LightProjector output
                      (`forward_vision_with_aligner`); `n_img` must equal the span.
        img_slice:    contiguous `<img>` span; `start`/`stop` must be TILE-aligned
                      (multiples of 32) so the concat pieces are valid in TILE layout.

    Returns:
        ttnn [B, S, H] TILE — the sequence with vision features injected.
    """
    B, S, H = hidden.shape
    start, stop = img_slice.start, img_slice.stop
    n_img = stop - start
    assert image_embeds.shape[0] == B, f"batch mismatch: hidden {B} vs image_embeds {image_embeds.shape[0]}"
    assert image_embeds.shape[2] == H, f"hidden-size mismatch: hidden {H} vs image_embeds {image_embeds.shape[2]}"
    assert image_embeds.shape[1] == n_img, f"image_embeds has {image_embeds.shape[1]} tokens, span needs {n_img}"
    # TILE-layout slices along the (tiled) sequence dim must start/stop on tile
    # boundaries; the contiguous-span assumption is what makes the concat valid.
    assert start % TILE == 0 and stop % TILE == 0, f"img_slice [{start}:{stop}] must be TILE({TILE})-aligned"

    # Align the vision features to the text-embedding dtype/layout (concat requires
    # all pieces to match), mirroring HunyuanTtDenoiseStep._scatter.
    toks, made_copy = _align_to(image_embeds, hidden)

    pre = ttnn.slice(hidden, [0, 0, 0], [B, start, H]) if start > 0 else None
    post = ttnn.slice(hidden, [0, stop, 0], [B, S, H]) if stop < S else None

    pieces = [p for p in (pre, toks, post) if p is not None]
    seq = ttnn.concat(pieces, dim=1)

    if made_copy:
        ttnn.deallocate(toks)
    if pre is not None:
        ttnn.deallocate(pre)
    if post is not None:
        ttnn.deallocate(post)
    return seq


def _validate_spans(ordered, B, S, H):
    cursor = 0
    for sl, emb in ordered:
        start, stop = sl.start, sl.stop
        assert start >= cursor, f"spans overlap or are out of order at [{start}:{stop}] (cursor {cursor})"
        assert emb.shape[0] == B, f"batch mismatch: hidden {B} vs image_embeds {emb.shape[0]}"
        assert emb.shape[2] == H, f"hidden-size mismatch: hidden {H} vs image_embeds {emb.shape[2]}"
        assert emb.shape[1] == stop - start, f"image_embeds has {emb.shape[1]} tokens, span needs {stop - start}"
        cursor = stop
    assert cursor <= S, f"last span stop {cursor} exceeds S {S}"


def _scatter_concat(hidden, ordered, B, S, H, *, layout):
    """Rebuild [text | emb | text | …] by slicing/concatenating in `layout`.

    TILE layout requires every span boundary to be a multiple of 32; ROW_MAJOR allows
    arbitrary boundaries (used for ragged spans). Either way it is a single device
    `concat` — no host round-trip.
    """
    row_major = layout == ttnn.ROW_MAJOR_LAYOUT
    base = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT) if row_major else hidden
    pieces = []
    to_free = []
    if row_major:
        to_free.append(base)
    cursor = 0
    for sl, emb in ordered:
        start, stop = sl.start, sl.stop
        if start > cursor:
            seg = ttnn.slice(base, [0, cursor, 0], [B, start, H])
            pieces.append(seg)
            to_free.append(seg)
        toks, made_copy = _align_to(emb, hidden)  # match dtype, TILE
        if made_copy:
            to_free.append(toks)
        if row_major:
            rm = ttnn.to_layout(toks, ttnn.ROW_MAJOR_LAYOUT)
            if rm is not toks:
                to_free.append(rm)
            toks = rm
        pieces.append(toks)
        cursor = stop
    if cursor < S:
        seg = ttnn.slice(base, [0, cursor, 0], [B, S, H])
        pieces.append(seg)
        to_free.append(seg)

    seq = ttnn.concat(pieces, dim=1)
    if row_major:
        tiled = ttnn.to_layout(seq, ttnn.TILE_LAYOUT)
        ttnn.deallocate(seq)
        seq = tiled
    for t in to_free:
        ttnn.deallocate(t)
    return seq


def scatter_cond_vision_embeddings_multi(hidden: ttnn.Tensor, spans: list) -> ttnn.Tensor:
    """Inject one or more contiguous `<img>` spans on device (multi-image Instruct input).

    Generalises `scatter_cond_vision_embeddings`: the sequence is rebuilt as
    `[text | emb_0 | text | emb_1 | … | text]` with a single device `concat`. When every
    span is TILE-aligned (the layout the Instruct chat template produces for each
    `<boi> <img>… <eoi>` block — each cond image is a 1024-token, i.e. 32×32, span) it
    uses a fast TILE-layout concat. Spans with arbitrary (non-32) boundaries fall back
    to a ROW_MAJOR concat — still entirely on device, no host round-trip.

    Args:
        hidden: ttnn [B, S, H] TILE — text-token embeddings with `<img>` placeholders.
        spans:  list of `(img_slice, image_embeds)` pairs. `image_embeds` is
                ttnn [B, span_len, H]. Spans must be non-overlapping (any order; sorted
                internally).

    Returns:
        ttnn [B, S, H] TILE — the sequence with every span injected.
    """
    B, S, H = hidden.shape
    ordered = sorted(spans, key=lambda se: se[0].start)
    _validate_spans(ordered, B, S, H)
    tile_aligned = all(sl.start % TILE == 0 and sl.stop % TILE == 0 for sl, _ in ordered)
    layout = ttnn.TILE_LAYOUT if tile_aligned else ttnn.ROW_MAJOR_LAYOUT
    return _scatter_concat(hidden, ordered, B, S, H, layout=layout)
