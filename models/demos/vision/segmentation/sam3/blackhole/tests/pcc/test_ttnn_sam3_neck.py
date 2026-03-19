# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_fpn_neck(device, reset_seeds, sam3_vit_backbone, sam3_neck):
    """Test FPN neck against PyTorch reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_neck import (
        tt_fpn_neck,
        preprocess_neck_weights,
    )

    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 1008, 1008)

    # Run full reference pipeline (backbone + neck)
    with torch.no_grad():
        ref_output = sam3_neck(pixel_values)

    ref_fpn = ref_output[0]  # sam3_out is the first element of the 4-tuple

    # Get ViT backbone features
    with torch.no_grad():
        vit_features = sam3_vit_backbone(pixel_values)[-1]  # (1, 1024, 72, 72)

    # Run our neck
    neck_params = preprocess_neck_weights(sam3_neck)
    tt_output = tt_fpn_neck(vit_features, neck_params, device)

    # Compare each scale
    for i, (ref_feat, tt_feat) in enumerate(zip(ref_fpn, tt_output["backbone_fpn"])):
        tt_feat_torch = ttnn.to_torch(tt_feat).float()
        assert_with_pcc(ref_feat.float(), tt_feat_torch, 0.99)
