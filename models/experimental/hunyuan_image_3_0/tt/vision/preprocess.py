# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host glue for the HunyuanImage-3.0 Instruct (image-to-image) input path.
#
# This wraps the reference image preprocessing (`HunyuanImage3ImageProcessor.
# vit_process_image`, which is just `Siglip2ImageProcessorFast` under the hood) and
# bridges its output into the on-device `Siglip2VisionInputs` bundle that the TTNN
# vision tower (`tt/vision/siglip2.py`) consumes. It also locates the `<img>`
# placeholder span(s) in a tokenized sequence so the projected vision features can
# be scattered into the right rows (`tt/vision/inject.py`).
#
# End-to-end Instruct I2I flow (host side):
#     PIL image
#       -> process_cond_image(...)          # pixel_values / spatial_shapes / mask
#       -> to_vision_inputs(device, ...)     # Siglip2VisionInputs (on device)
#       -> forward_vision_with_aligner(...)  # [B, n_img, 4096]  (tt/vision/siglip2)
#       -> find_image_token_spans(input_ids) # <img> slice(s) in the LLM sequence
#       -> scatter_cond_vision_embeddings*   # inject into text embeddings (tt/vision/inject)
#       -> HunyuanTtModel.forward(inputs_embeds=...)
#
# The reference processor pads every image to `max_num_patches` (1024) patches, so a
# single conditioning image occupies a contiguous, TILE-aligned (1024 = 32*32) span
# of `<img>` tokens — exactly what the device concat scatter handles.

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.vision.preprocess import (
    IMAGE_TOKEN_ID,
    build_cond_image_processor,
    vit_process_image,
)
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import Siglip2VisionInputs

# Re-export the host-side reference processor build so callers have one import site.
__all_ref__ = (IMAGE_TOKEN_ID, build_cond_image_processor)


def process_cond_image(processor, image, model_dir=MODEL_DIR):
    """Run a PIL image through the SigLIP2 processor (host).

    Thin pass-through to the reference `vit_process_image`
    (`ref/vision/preprocess.py`, faithful to image_processor.py:422). Preprocessing
    is host-side, so the TT path reuses the golden verbatim rather than reimplementing
    it. Returns `(pixel_values, spatial_shapes_hw, pixel_attention_mask)`.
    """
    return vit_process_image(processor, image)


def to_vision_inputs(device, pixel_values, spatial_shapes_hw, pixel_attention_mask) -> Siglip2VisionInputs:
    """Upload host processor outputs into an on-device `Siglip2VisionInputs` bundle.

    Mirrors the upload pattern in `tests/pcc/siglip2_helpers.py` (bf16 TILE on DRAM); the
    4D additive attention mask is built on device inside the vision tower.
    Accepts either batched ``[B, S, ...]`` / ``((h, w), ...)`` or a single-image
    unbatched ``[S, ...]`` / ``(h, w)`` and normalizes to batched form.
    """
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 2:
        pixel_values = pixel_values.unsqueeze(0)
    if isinstance(pixel_attention_mask, torch.Tensor) and pixel_attention_mask.ndim == 1:
        pixel_attention_mask = pixel_attention_mask.unsqueeze(0)
    # Bare (h, w) -> ((h, w),) for Siglip2VisionInputs / pos-embed resize.
    if (
        isinstance(spatial_shapes_hw, tuple)
        and len(spatial_shapes_hw) == 2
        and not isinstance(spatial_shapes_hw[0], tuple)
    ):
        spatial_shapes_hw = (spatial_shapes_hw,)

    # ttnn.from_torch casts the source dtype (float32 pixels / int32 mask) to bf16.
    pv = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask = ttnn.from_torch(
        pixel_attention_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return Siglip2VisionInputs.create(pv, spatial_shapes_hw, mask)


def find_image_token_spans(input_ids, image_token_id: int = IMAGE_TOKEN_ID) -> list:
    """Locate contiguous runs of `<img>` placeholder tokens in a 1D token sequence.

    Returns a list of `slice(start, stop)` — one per conditioning image — over the
    `<img>` rows. This is the span(s) `scatter_cond_vision_embeddings*` write into,
    and corresponds to the tokenizer's `vit_image_slices`
    (tokenization_hunyuan_image_3.py) without needing to thread the full chat-template
    bookkeeping: the `<boi> <img>…<img> <eoi>` layout makes each image a single
    contiguous run.

    Args:
        input_ids:      1D token ids for one batch row — a torch.Tensor, list, or any
                        sequence/tensor exposing `flatten()`/`tolist()`.
        image_token_id: the `<img>` placeholder id (config.json `image_token_id`).
    """
    if hasattr(input_ids, "flatten") and hasattr(input_ids, "tolist"):
        ids = input_ids.flatten().tolist()  # torch/np tensor -> python list
    else:
        ids = list(input_ids)
    spans = []
    start = None
    for i, tok in enumerate(ids):
        if tok == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            spans.append(slice(start, i))
            start = None
    if start is not None:
        spans.append(slice(start, len(ids)))
    return spans
