# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SAM3 segmentation head and dot-product scoring for ttnn.

Both operations run on CPU via PyTorch for correctness.
Results are converted to ttnn tensors at the end.

# TODO: Replace CPU ops with ttnn equivalents for on-device execution.
"""

from typing import Dict, List

import torch
import ttnn


def tt_segmentation_head(
    decoder_output: torch.Tensor,
    fpn_features: List[torch.Tensor],
    seg_head_module,
    device,
) -> Dict[str, torch.Tensor]:
    """Run segmentation head on CPU, return masks as ttnn tensors.

    Args:
        decoder_output: torch tensor (num_layers, batch, num_queries, d_model) - decoder queries
                        or (num_queries, batch, d_model) for a single layer.
        fpn_features: list of torch tensors - multi-scale features from FPN neck (NCHW).
        seg_head_module: PyTorch UniversalSegmentationHead module.
        device: ttnn device.

    Returns:
        dict with:
            'pred_masks'   – torch tensor (batch, num_queries, H, W)
            'semantic_seg' – torch tensor (batch, 1, H, W) or None
            'presence_logit' – torch tensor or None
    """
    seg_head_module.eval()

    with torch.no_grad():
        # decoder_output shape: (num_layers, batch, num_queries, d_model)
        # UniversalSegmentationHead.forward expects obj_queries and uses obj_queries[-1]
        # when not in aux_masks mode.  We pass the full tensor so indexing works.
        obj_queries = decoder_output

        # Build image_ids: each query maps to image 0 (single-image batch).
        # _embed_pixels uses image_ids to index pixel_embed when batch > 1.
        # For batch=1 it relies on broadcasting, but we still need a valid tensor.
        batch = fpn_features[0].shape[0]
        image_ids = torch.zeros(batch, dtype=torch.long)

        # encoder_hidden_states is required by UniversalSegmentationHead.
        # It is used to build per-query visual embeddings inside _embed_pixels.
        # Shape expected: (seq, batch, d_model) where seq covers the spatial
        # extent of the last FPN feature map.
        last_feat = fpn_features[-1]  # (batch, C, H, W)
        _, c, h, w = last_feat.shape
        # Flatten spatial dims and permute to (hw, batch, C)
        encoder_hidden_states = last_feat.flatten(2).permute(2, 0, 1)  # (hw, batch, C)

        outputs = seg_head_module(
            backbone_feats=fpn_features,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

    return outputs


def tt_dot_product_scoring(
    decoder_output: torch.Tensor,
    text_features: torch.Tensor,
    scoring_module,
    device,
) -> torch.Tensor:
    """Run dot product scoring on CPU.

    Args:
        decoder_output: torch tensor (num_layers, batch, num_queries, d_model) -
                        decoder output queries across all layers.
        text_features: torch tensor (seq, batch, d_model) - text encoder features.
        scoring_module: PyTorch DotProductScoring module.
        device: ttnn device.

    Returns:
        scores: torch tensor (num_layers, batch, num_queries, 1) - classification scores.
    """
    scoring_module.eval()

    with torch.no_grad():
        # DotProductScoring.forward expects:
        #   hs:          (num_layer, bs, num_query, d_model)
        #   prompt:      (seq, bs, d_model)
        #   prompt_mask: (bs, seq) – True where padding, False where valid
        seq_len = text_features.shape[0]
        batch = text_features.shape[1]
        # All tokens are valid (no padding)
        prompt_mask = torch.zeros(batch, seq_len, dtype=torch.bool)

        scores = scoring_module(
            hs=decoder_output,
            prompt=text_features,
            prompt_mask=prompt_mask,
        )

    return scores
