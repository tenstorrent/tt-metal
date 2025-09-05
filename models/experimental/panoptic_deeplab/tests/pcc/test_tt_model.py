# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the complete TtPanopticDeepLab model.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_panoptic_deeplab(device):
    """Test PCC comparison between PyTorch and TTNN implementations."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    batch_size = 1
    num_classes = 19
    project_channels = [32, 64]
    decoder_channels = [256, 256, 256]
    sem_seg_head_channels = 256
    ins_embed_head_channels = 32
    common_stride = 4
    train_size = (512, 1024)

    shared_weight_tensor_kernel1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
    shared_weight_tensor_kernel3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
    shared_weight_tensor_kernel1_output5 = torch.randn(256, 1280, 1, 1, dtype=torch.bfloat16)

    sem_project_conv_weights = {
        "res2": torch.randn(project_channels[0], 256, 1, 1, dtype=torch.bfloat16),
        "res3": torch.randn(project_channels[1], 512, 1, 1, dtype=torch.bfloat16),
    }
    sem_fuse_conv_0_weights = {
        "res2": torch.randn(decoder_channels[0], project_channels[0] + decoder_channels[1], 3, 3, dtype=torch.bfloat16),
        "res3": torch.randn(decoder_channels[1], project_channels[1] + decoder_channels[2], 3, 3, dtype=torch.bfloat16),
    }
    sem_fuse_conv_1_weights = {
        "res2": torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16),
        "res3": torch.randn(decoder_channels[1], decoder_channels[1], 3, 3, dtype=torch.bfloat16),
    }
    sem_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    sem_head_1_weight = torch.randn(sem_seg_head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    sem_predictor_weight = torch.randn(num_classes, sem_seg_head_channels, 1, 1, dtype=torch.bfloat16)

    ins_project_conv_weights = sem_project_conv_weights
    ins_fuse_conv_0_weights = sem_fuse_conv_0_weights
    ins_fuse_conv_1_weights = sem_fuse_conv_1_weights
    center_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    center_head_1_weight = torch.randn(ins_embed_head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    center_predictor_weight = torch.randn(1, ins_embed_head_channels, 1, 1, dtype=torch.bfloat16)
    offset_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    offset_head_1_weight = torch.randn(ins_embed_head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    offset_predictor_weight = torch.randn(2, ins_embed_head_channels, 1, 1, dtype=torch.bfloat16)

    input_height, input_width = train_size[0], train_size[1]
    input_channels = 3

    pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)

    ttnn_input_torch = pytorch_input.permute(0, 2, 3, 1)

    ttnn_input = ttnn.from_torch(ttnn_input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    try:
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="SyncBN",
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            sem_project_conv_weights=sem_project_conv_weights,
            sem_fuse_conv_0_weights=sem_fuse_conv_0_weights,
            sem_fuse_conv_1_weights=sem_fuse_conv_1_weights,
            sem_head_0_weight=sem_head_0_weight,
            sem_head_1_weight=sem_head_1_weight,
            sem_predictor_weight=sem_predictor_weight,
            ins_project_conv_weights=ins_project_conv_weights,
            ins_fuse_conv_0_weights=ins_fuse_conv_0_weights,
            ins_fuse_conv_1_weights=ins_fuse_conv_1_weights,
            center_head_0_weight=center_head_0_weight,
            center_head_1_weight=center_head_1_weight,
            center_predictor_weight=center_predictor_weight,
            offset_head_0_weight=offset_head_0_weight,
            offset_head_1_weight=offset_head_1_weight,
            offset_predictor_weight=offset_predictor_weight,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # 6. Create TTNN model with same weights
        ttnn_model = TtPanopticDeepLab(
            device=device,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="SyncBN",
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            sem_project_conv_weights=sem_project_conv_weights,
            sem_fuse_conv_0_weights=sem_fuse_conv_0_weights,
            sem_fuse_conv_1_weights=sem_fuse_conv_1_weights,
            sem_head_0_weight=sem_head_0_weight,
            sem_head_1_weight=sem_head_1_weight,
            sem_predictor_weight=sem_predictor_weight,
            ins_project_conv_weights=ins_project_conv_weights,
            ins_fuse_conv_0_weights=ins_fuse_conv_0_weights,
            ins_fuse_conv_1_weights=ins_fuse_conv_1_weights,
            center_head_0_weight=center_head_0_weight,
            center_head_1_weight=center_head_1_weight,
            center_predictor_weight=center_predictor_weight,
            offset_head_0_weight=offset_head_0_weight,
            offset_head_1_weight=offset_head_1_weight,
            offset_predictor_weight=offset_predictor_weight,
        )
    except FileNotFoundError:
        pytest.fail("R-52.pkl file not found. Please place the weights file in the weights folder.")

    logger.info("Running PyTorch model...")
    with torch.no_grad():
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

    logger.info("Running TTNN model...")
    ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic).permute(0, 3, 1, 2)
    ttnn_center_torch = ttnn.to_torch(ttnn_center).permute(0, 3, 1, 2)
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset).permute(0, 3, 1, 2)

    sem_passed, sem_msg = assert_with_pcc(pytorch_semantic, ttnn_semantic_torch, pcc=0.99)
    logger.info(f"Semantic PCC: {sem_msg}")
    assert sem_passed, f"Semantic segmentation PCC failed: {sem_msg}"

    center_passed, center_msg = assert_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.91)
    logger.info(f"Center PCC: {center_msg}")
    assert center_passed, f"Center heatmap PCC failed: {center_msg}"

    offset_passed, offset_msg = assert_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.98)
    logger.info(f"Offset PCC: {offset_msg}")
    assert offset_passed, f"Offset map PCC failed: {offset_msg}"
