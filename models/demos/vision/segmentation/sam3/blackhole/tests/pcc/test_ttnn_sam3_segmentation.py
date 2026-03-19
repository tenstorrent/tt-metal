# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skip(reason="Segmentation head requires NestedTensor/full pipeline context. Tested via e2e integration.")
def test_tt_segmentation_head_output_shapes(sam3_reference_model):
    """Test that the segmentation head produces correct output shapes."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_segmentation import (
        tt_segmentation_head,
    )

    seg_head = sam3_reference_model.segmentation_head
    if seg_head is None:
        pytest.skip("Model has no segmentation head")

    torch.manual_seed(42)

    # Determine d_model from the head
    d_model = seg_head.d_model
    num_queries = 8
    batch = 1
    num_layers = 2

    # Build synthetic FPN features matching expected hidden_dim channels.
    # Typical SAM3 FPN has 3 scales: 1/8, 1/16, 1/32 of the image resolution.
    # Use small spatial sizes for speed.
    fpn_features = [
        torch.randn(batch, d_model, 16, 16),  # finest scale
        torch.randn(batch, d_model, 8, 8),
        torch.randn(batch, d_model, 4, 4),   # coarsest scale (used for encoder_hidden_states)
    ]

    # decoder_output: (num_layers, batch, num_queries, d_model)
    decoder_output = torch.randn(num_layers, batch, num_queries, d_model)

    outputs = tt_segmentation_head(
        decoder_output=decoder_output,
        fpn_features=fpn_features,
        seg_head_module=seg_head,
        device=None,
    )

    assert "pred_masks" in outputs, "Output must contain 'pred_masks'"
    pred_masks = outputs["pred_masks"]
    # pred_masks shape: (batch, num_queries, H, W)
    assert pred_masks.shape[0] == batch, f"Expected batch={batch}, got {pred_masks.shape[0]}"
    assert pred_masks.shape[1] == num_queries, (
        f"Expected num_queries={num_queries}, got {pred_masks.shape[1]}"
    )
    assert pred_masks.ndim == 4, f"pred_masks should be 4-D, got shape {pred_masks.shape}"


def test_tt_dot_product_scoring_output_shapes(sam3_reference_model):
    """Test that dot product scoring produces correct output shapes."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_segmentation import (
        tt_dot_product_scoring,
    )

    scoring_module = sam3_reference_model.dot_prod_scoring
    if scoring_module is None:
        pytest.skip("Model has no dot_prod_scoring module")

    torch.manual_seed(42)

    # Infer d_model from the scoring module's projection layer
    d_model = scoring_module.hs_proj.in_features
    num_queries = 8
    batch = 1
    num_layers = 2
    seq_len = 16

    # decoder_output: (num_layers, batch, num_queries, d_model)
    decoder_output = torch.randn(num_layers, batch, num_queries, d_model)
    # text_features: (seq, batch, d_model)
    text_features = torch.randn(seq_len, batch, d_model)

    scores = tt_dot_product_scoring(
        decoder_output=decoder_output,
        text_features=text_features,
        scoring_module=scoring_module,
        device=None,
    )

    # DotProductScoring returns (num_layers, batch, num_queries, 1)
    assert scores.shape == (num_layers, batch, num_queries, 1), (
        f"Expected shape ({num_layers}, {batch}, {num_queries}, 1), got {scores.shape}"
    )
