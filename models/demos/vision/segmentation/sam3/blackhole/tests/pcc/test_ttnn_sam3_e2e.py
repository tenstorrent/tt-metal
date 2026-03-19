# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for SAM3 on ttnn."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_sam3_pipeline_init(device, reset_seeds, sam3_reference_model):
    """Test that the TtSam3ImagePipeline initializes correctly."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_model import TtSam3ImagePipeline

    pipeline = TtSam3ImagePipeline(sam3_reference_model, device)

    assert pipeline.device is not None
    assert pipeline.backbone_params is not None
    assert len(pipeline.backbone_params["blocks"]) == 32


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_tt_sam3_vit_backbone_e2e(device, reset_seeds, sam3_reference_model, sam3_vit_backbone):
    """Test ViT backbone through the pipeline matches reference."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_model import TtSam3ImagePipeline

    torch.manual_seed(42)
    pipeline = TtSam3ImagePipeline(sam3_reference_model, device)

    pixel_values = torch.randn(1, 3, 1008, 1008)

    # Reference
    with torch.no_grad():
        ref_features = sam3_vit_backbone(pixel_values)
    ref_feat = ref_features[-1]

    # Pipeline
    tt_features = pipeline.run_vit_backbone(pixel_values)
    tt_feat = tt_features[-1]

    assert tt_feat.shape == ref_feat.shape
    assert_with_pcc(ref_feat.float(), tt_feat.float(), 0.90)


def test_tt_sam3_preprocess_image():
    """Test image preprocessing produces correct shapes and normalization."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_model import preprocess_image

    # Test with uint8-like input
    image = torch.randint(0, 256, (3, 480, 640), dtype=torch.float32)
    processed = preprocess_image(image)

    assert processed.shape == (1, 3, 1008, 1008)
    assert processed.min() >= -1.1  # approximately [-1, 1]
    assert processed.max() <= 1.1

    # Test with already normalized input
    image2 = torch.rand(1, 3, 1008, 1008)
    processed2 = preprocess_image(image2)
    assert processed2.shape == (1, 3, 1008, 1008)
