# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 Instruct cond-vision sequence
# injection: scattering SigLIP2 image embeddings into the `<img>` placeholder
# positions of the LLM token-embedding sequence. Golden reference for
# `tt/vision/inject.py`.
#
# Extracted / adapted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     HunyuanImage3ForCausalMM.instantiate_vit_image_tokens   lines 1903-1962
#       (index build       line 1924,
#        masked scatter     lines 1939-1943)
#
# The upstream method couples vision encoding (`_forward_vision_encoder`) with the
# scatter and mutates `hidden_states` in place. This reference DECOUPLES the two:
# it takes the already-encoded/projected `image_embeds` and returns a new tensor,
# so the scatter can be validated in isolation (the vision tower + aligner have
# their own reference in `ref/vision/siglip2.py`).

import torch


def scatter_vit_image_tokens(
    hidden_states: torch.Tensor,  # [bsz, seqlen, n_embd]  text-token embeddings
    image_embeds: torch.Tensor,  # [bsz, n_img, n_embd]   projected vision features
    image_masks: torch.Tensor,  # [bsz, seqlen]          bool, True at <img> positions
) -> torch.Tensor:
    """Write projected vision embeddings into the masked `<img>` rows of the sequence.

    Faithful to `instantiate_vit_image_tokens` (modeling_hunyuan_image_3.py:1924,
    1939-1943): build the per-batch `<img>` position index with `masked_select`,
    then `scatter_` the embeddings into those rows (in selection order). Operates on
    pre-encoded `image_embeds` — running the SigLIP2 tower + LightProjector is the
    caller's responsibility. Unlike the in-place upstream `scatter_`, this returns a
    NEW tensor and leaves `hidden_states` untouched.

    The number of True positions in `image_masks` must equal `bsz * n_img`.
    """
    bsz, seqlen, n_embd = hidden_states.shape
    index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
    out = hidden_states.clone()
    out.scatter_(
        dim=1,
        index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
        src=image_embeds.reshape(bsz, -1, n_embd),
    )
    return out


def contiguous_image_mask(bsz: int, seqlen: int, img_slice: slice, device=None) -> torch.Tensor:
    """Build a `[bsz, seqlen]` bool mask True over a single contiguous `<img>` span.

    This is the single-image case the on-device `scatter_cond_vision_embeddings`
    handles (a contiguous, TILE-aligned span). Multi-image / ragged masks are also
    valid inputs to `scatter_vit_image_tokens` but use the host-scatter fallback on
    device.
    """
    mask = torch.zeros(bsz, seqlen, dtype=torch.bool, device=device)
    mask[:, img_slice.start : img_slice.stop] = True
    return mask
