# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_text_encoder_shapes(device, reset_seeds, sam3_text_encoder):
    """Verify text encoder produces correct output shapes."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_text_encoder import (
        tt_text_encoder,
    )

    text_prompts = ["a cat sitting on a mat", "a dog running in the park"]
    batch = len(text_prompts)
    context_length = sam3_text_encoder.context_length  # 32
    d_model = sam3_text_encoder.resizer.out_features  # 256

    result = tt_text_encoder(text_prompts, sam3_text_encoder, device)

    assert "text_features" in result
    assert "text_mask" in result

    text_features_torch = ttnn.to_torch(result["text_features"]).float()
    text_mask_torch = ttnn.to_torch(result["text_mask"]).float()

    # text_memory_resized shape: [seq_len, batch, d_model]
    assert text_features_torch.shape == (context_length, batch, d_model), (
        f"Expected text_features shape ({context_length}, {batch}, {d_model}), "
        f"got {tuple(text_features_torch.shape)}"
    )

    # text_attention_mask shape: [batch, seq_len]
    assert text_mask_torch.shape == (batch, context_length), (
        f"Expected text_mask shape ({batch}, {context_length}), "
        f"got {tuple(text_mask_torch.shape)}"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_text_encoder_matches_reference(device, reset_seeds, sam3_text_encoder):
    """Verify tt_text_encoder output matches direct PyTorch reference call."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_text_encoder import (
        tt_text_encoder,
    )

    text_prompts = ["a photo of a person"]

    # Reference: run VETextEncoder directly on CPU
    sam3_text_encoder.eval()
    with torch.no_grad():
        ref_mask, ref_features, ref_embeds = sam3_text_encoder(
            text_prompts, input_boxes=None, device=None
        )

    result = tt_text_encoder(text_prompts, sam3_text_encoder, device)

    text_features_torch = ttnn.to_torch(result["text_features"]).float()
    text_mask_torch = ttnn.to_torch(result["text_mask"]).float()

    # text_features should match ref_features (text_memory_resized)
    assert_with_pcc(ref_features.float(), text_features_torch, 0.999)

    # text_mask should match ref_mask (text_attention_mask cast to float)
    assert_with_pcc(ref_mask.float(), text_mask_torch, 0.999)
