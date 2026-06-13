# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_tt_vit_backbone(device, reset_seeds, batch_size, sam3_vit_backbone):
    """Test the full ViT backbone (32 blocks) against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        move_backbone_params_to_device,
        preprocess_vit_backbone_weights,
        tt_vit_backbone,
    )

    torch.manual_seed(42)

    # Create input
    pixel_values = torch.randn(batch_size, 3, 1008, 1008)

    # Reference forward
    with torch.no_grad():
        ref_outputs = sam3_vit_backbone(pixel_values)
    ref_feats = ref_outputs[-1]  # (B, 1024, 72, 72)

    # Preprocess weights
    backbone_params = preprocess_vit_backbone_weights(sam3_vit_backbone)
    backbone_params = move_backbone_params_to_device(backbone_params, device)

    # Run ttnn backbone
    tt_outputs = tt_vit_backbone(pixel_values, backbone_params, device)
    tt_feats = tt_outputs[-1]  # (B, 1024, 72, 72)

    # Verify shape
    assert tt_feats.shape == ref_feats.shape, f"Shape mismatch: {tt_feats.shape} vs {ref_feats.shape}"

    # PCC check - 32 layers may accumulate some error
    assert_with_pcc(ref_feats.float(), tt_feats.float(), 0.90)
