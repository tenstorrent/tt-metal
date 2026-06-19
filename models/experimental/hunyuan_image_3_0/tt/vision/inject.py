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
# Ragged / multi-image layouts (several non-contiguous `<img>` spans) are NOT
# handled here; they need the host-scatter fallback noted in `tt/pipeline.py`.

import ttnn

TILE = 32


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
    toks = image_embeds
    if toks.layout != ttnn.TILE_LAYOUT:
        toks = ttnn.to_layout(toks, ttnn.TILE_LAYOUT)
    if toks.dtype != hidden.dtype:
        toks = ttnn.typecast(toks, hidden.dtype)

    pre = ttnn.slice(hidden, [0, 0, 0], [B, start, H]) if start > 0 else None
    post = ttnn.slice(hidden, [0, stop, 0], [B, S, H]) if stop < S else None

    pieces = [p for p in (pre, toks, post) if p is not None]
    seq = ttnn.concat(pieces, dim=1)

    if toks is not image_embeds:
        ttnn.deallocate(toks)
    if pre is not None:
        ttnn.deallocate(pre)
    if post is not None:
        ttnn.deallocate(post)
    return seq
