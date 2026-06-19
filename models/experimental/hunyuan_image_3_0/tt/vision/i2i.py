# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Instruct (image-to-image) assembly: tie the validated device pieces together into
# a single host-callable that turns text embeddings + conditioning image(s) into the
# `inputs_embeds` the backbone consumes.
#
# Pieces (all individually PCC-tested):
#   tt/vision/preprocess.py  — PIL -> Siglip2VisionInputs, <img> span lookup
#   tt/vision/siglip2.py     — SigLIP2 tower + LightProjector (forward_vision_with_aligner)
#   tt/vision/inject.py      — scatter projected features into the <img> span(s)
#
# This module only orchestrates; it owns no new math, so its correctness follows by
# composition from the unit PCC tests of the pieces it calls.

import ttnn

from models.experimental.hunyuan_image_3_0.tt.vision.inject import scatter_cond_vision_embeddings_multi
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import forward_vision_with_aligner


def encode_cond_vision(vision, aligner, vision_inputs) -> ttnn.Tensor:
    """Run the SigLIP2 tower + LightProjector: Siglip2VisionInputs -> [B, n_img, 4096].

    Thin alias for `forward_vision_with_aligner` so callers import a single I2I entry
    point. `n_img` is the padded patch count (`max_num_patches`, 1024) — i.e. the full
    `<img>` span length the tokenizer reserves per conditioning image.
    """
    return forward_vision_with_aligner(vision, aligner, vision_inputs)


def inject_cond_vision(hidden, image_embeds, *, img_slices) -> ttnn.Tensor:
    """Scatter projected vision features into the `<img>` span(s), fully on device.

    Splits `image_embeds` (all conditioning images concatenated, in sequence order) per
    span and hands them to `scatter_cond_vision_embeddings_multi`, which picks a TILE
    concat for TILE-aligned spans or a ROW_MAJOR concat for ragged ones — no host
    round-trip either way.

    Args:
        hidden:       ttnn [B, S, H] — text-token embeddings with `<img>` placeholders.
        image_embeds: ttnn [B, n_img_total, H] — projected vision features for ALL
                      conditioning images concatenated in sequence order.
        img_slices:   list[slice] over the `<img>` span(s) (e.g. from
                      `preprocess.find_image_token_spans`).
    """
    Be, _, He = image_embeds.shape
    ordered = sorted(img_slices, key=lambda s: s.start)
    single = len(ordered) == 1
    spans = []
    cursor = 0
    for sl in ordered:
        n = sl.stop - sl.start
        chunk = image_embeds if single else ttnn.slice(image_embeds, [0, cursor, 0], [Be, cursor + n, He])
        spans.append((sl, chunk))
        cursor += n
    out = scatter_cond_vision_embeddings_multi(hidden, spans)
    if not single:
        for _, e in spans:
            ttnn.deallocate(e)
    return out


def build_i2i_inputs_embeds(device, *, vision, aligner, text_embeds, vision_inputs, img_slices) -> ttnn.Tensor:
    """Full I2I embedding assembly: text embeds + cond image(s) -> inputs_embeds.

    Args:
        device:        the (mesh) device the backbone runs on.
        vision:        HunyuanTtSiglip2Vision.
        aligner:       HunyuanTtLightProjector.
        text_embeds:   ttnn [B, S, H] — wte(input_ids) with `<img>` placeholders.
        vision_inputs: Siglip2VisionInputs (from preprocess.to_vision_inputs); may be a
                       single bundle or a list (one per conditioning image).
        img_slices:    list[slice] of `<img>` span(s) (from find_image_token_spans).

    Returns:
        ttnn [B, S, H] — sequence embeddings ready for HunyuanTtModel.forward(inputs_embeds=).
    """
    bundles = vision_inputs if isinstance(vision_inputs, (list, tuple)) else [vision_inputs]
    embeds = [encode_cond_vision(vision, aligner, b) for b in bundles]
    image_embeds = embeds[0] if len(embeds) == 1 else ttnn.concat(embeds, dim=1)
    out = inject_cond_vision(text_embeds, image_embeds, img_slices=img_slices)
    if len(embeds) > 1:
        for e in embeds:
            ttnn.deallocate(e)
    return out
