# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import os
from typing import Dict
from loguru import logger

from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from tests.ttnn.utils_for_testing import assert_with_pcc


PDL_L1_SMALL_SIZE = 37 * 1024  # Minimum L1 small size for Panoptic DeepLab


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_semseg(device, model_location_generator):
    """Test semantic segmentation head using the full model with real weights."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Determine weights path based on environment
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        # Use local path (old method)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        complete_weights_path = os.path.join(current_dir, "..", "..", "weights", "model_final_bd324a.pkl")
    else:
        # Check if weights already exist in CI v2 cache first
        cached_weights_path = (
            "/tmp/ttnn_model_cache/model_weights/vision-models/panoptic_deeplab/model_final_bd324a.pkl"
        )
        if os.path.exists(cached_weights_path):
            complete_weights_path = cached_weights_path
        else:
            # Use CI v2 model location generator to download
            complete_weights_path = (
                model_location_generator("vision-models/panoptic_deeplab", model_subdir="", download_if_ci_v2=True)
                / "model_final_bd324a.pkl"
            )

    # Model configuration
    batch_size = 1
    num_classes = 19
    project_channels = [32, 64]
    decoder_channels = [256, 256, 256]
    sem_seg_head_channels = 256
    ins_embed_head_channels = 32
    common_stride = 4
    train_size = (512, 1024)

    # Create test features matching ResNet output
    torch_features: Dict[str, torch.Tensor] = {
        "res2": torch.randn(1, 256, 128, 256, dtype=torch.bfloat16),
        "res3": torch.randn(1, 512, 64, 128, dtype=torch.bfloat16),
        "res5": torch.randn(1, 2048, 32, 64, dtype=torch.bfloat16),
    }

    ttnn_features: Dict[str, ttnn.Tensor] = {
        name: ttnn.from_torch(tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for name, tensor in torch_features.items()
    }

    try:
        # Load PyTorch model with real weights
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="SyncBN",
            train_size=train_size,
            weights_path=complete_weights_path,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters from the PyTorch model with loaded weights
        ttnn_parameters = create_panoptic_deeplab_parameters(pytorch_model, device)

        # Apply Conv+BatchNorm fusion to the parameters
        logger.info("Applying Conv+BatchNorm fusion to parameters...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)
        logger.info("Conv+BatchNorm fusion completed successfully")

        # Create TTNN model with fused parameters
        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="",
            train_size=train_size,
        )
    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    # Test semantic segmentation head
    logger.info("Running PyTorch semantic segmentation head test...")
    with torch.no_grad():
        torch_out, _ = pytorch_model.semantic_head(torch_features)

    logger.info("Running TTNN semantic segmentation head test...")
    ttnn_out_tt, _ = ttnn_model.semantic_head(ttnn_features)

    ttnn_out_torch = ttnn.to_torch(ttnn_out_tt).permute(0, 3, 1, 2)

    passed, msg = assert_with_pcc(torch_out, ttnn_out_torch, pcc=0.99)
    logger.info(f"Semantic segmentation PCC: {msg}")
    assert passed, f"Semantic segmentation PCC test failed: {msg}"
