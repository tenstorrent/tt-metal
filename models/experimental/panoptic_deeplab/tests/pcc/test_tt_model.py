# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the complete TtPanopticDeepLab model.
"""

import pytest
import torch
import ttnn
from typing import Set

from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab, create_panoptic_deeplab_model
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65565}], indirect=True)
def test_panoptic_deeplab_model_creation(device):
    """Test that the complete model can be created successfully."""
    torch.manual_seed(0)

    # Create model with default configuration
    model = create_panoptic_deeplab_model(
        device=device, num_classes=19, use_real_weights=False, train_size=(512, 1024)  # Use random weights for testing
    )

    # Check model info
    info = model.get_model_info()
    assert info["model_type"] == "Panoptic-DeepLab"
    assert info["num_classes"] == 19
    assert info["common_stride"] == 4
    assert info["backbone"] == "ResNet-50"

    print("✅ Model creation test passed")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65565}], indirect=True)
def test_panoptic_deeplab_forward_pass(device):
    """Test the complete forward pass of the model."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Create model
    model = TtPanopticDeepLab(
        device=device,
        num_classes=19,
        use_real_weights=False,
        train_size=(512, 1024),
        common_stride=4,
        project_channels=[32, 64],
        decoder_channels=[256, 256, 256],
        sem_seg_head_channels=256,
        ins_embed_head_channels=32,
        norm="SyncBN",
    )

    # Create input tensor [B, H, W, C] in NHWC format
    batch_size = 1
    height, width = 512, 1024
    channels = 3

    input_tensor = ttnn.from_torch(
        torch.randn(batch_size, height, width, channels, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Forward pass
    semantic_logits, center_heatmap, offset_map, features = model.forward(input_tensor, return_features=True)

    # Check output shapes
    assert semantic_logits.shape[0] == batch_size
    assert semantic_logits.shape[1] == height
    assert semantic_logits.shape[2] == width
    assert semantic_logits.shape[3] == 19  # num_classes

    assert center_heatmap.shape[0] == batch_size
    assert center_heatmap.shape[1] == height
    assert center_heatmap.shape[2] == width
    assert center_heatmap.shape[3] == 1  # center prediction

    assert offset_map.shape[0] == batch_size
    assert offset_map.shape[1] == height
    assert offset_map.shape[2] == width
    assert offset_map.shape[3] == 2  # offset prediction (x, y)

    # Check that features are returned
    assert features is not None
    assert "res2" in features
    assert "res3" in features
    assert "res5" in features

    print("✅ Forward pass test passed")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_panoptic_deeplab_inference_pipeline(device):
    """Test the complete inference pipeline with post-processing."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # Create model
    model = TtPanopticDeepLab(
        device=device,
        num_classes=19,
        use_real_weights=False,
        train_size=(256, 512),  # Smaller size for faster testing
        common_stride=4,
        project_channels=[32, 64],
        decoder_channels=[256, 256, 256],
        sem_seg_head_channels=256,
        ins_embed_head_channels=32,
        norm="SyncBN",
    )

    # Create smaller input for faster testing
    batch_size = 1
    height, width = 256, 512
    channels = 3

    input_tensor = ttnn.from_torch(
        torch.randn(batch_size, height, width, channels, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Define thing classes (classes that are instances)
    thing_ids: Set[int] = {0, 1, 2, 3, 4, 5, 6, 7}  # Example thing classes

    # Run inference
    panoptic_seg, center_points = model.inference(
        x=input_tensor,
        thing_ids=thing_ids,
        label_divisor=1000,
        stuff_area=2048,
        void_label=255,
        threshold=0.1,
        nms_kernel=7,
        top_k=100,
    )

    # Check output shapes and types
    assert isinstance(panoptic_seg, torch.Tensor)
    assert isinstance(center_points, torch.Tensor)

    assert panoptic_seg.shape == (1, height, width)
    assert center_points.shape[0] == 1  # batch dimension
    assert center_points.shape[2] == 2  # (y, x) coordinates

    print(f"✅ Inference test passed. Found {center_points.shape[1]} center points")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_image_preprocessing(device):
    """Test the image preprocessing functionality."""
    torch.manual_seed(0)

    model = create_panoptic_deeplab_model(device=device, num_classes=19, use_real_weights=False)

    # Test with 3D tensor (single image)
    image_3d = torch.randn(3, 256, 512, dtype=torch.bfloat16)
    processed_3d = model.preprocess_image(image_3d)

    assert processed_3d.shape == (1, 256, 512, 3)  # [B, H, W, C]

    # Test with 4D tensor (batch of images)
    image_4d = torch.randn(2, 3, 256, 512, dtype=torch.bfloat16)
    processed_4d = model.preprocess_image(image_4d)

    assert processed_4d.shape == (2, 256, 512, 3)  # [B, H, W, C]

    print("✅ Image preprocessing test passed")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_panoptic_deeplab_pcc_comparison(device):
    """Test PCC comparison between PyTorch and TTNN implementations with real ResNet weights."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # 1. Define Model Configuration
    batch_size = 1
    num_classes = 19
    project_channels = [32, 64]
    decoder_channels = [256, 256, 256]
    sem_seg_head_channels = 256
    ins_embed_head_channels = 32
    common_stride = 4
    train_size = (512, 1024)

    # 2. Create shared weights for both models
    shared_weight_tensor_kernel1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
    shared_weight_tensor_kernel3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
    shared_weight_tensor_kernel1_output5 = torch.randn(256, 1280, 1, 1, dtype=torch.bfloat16)

    # Semantic head weights
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

    # Instance head weights (reuse same decoder weights for shared decoder)
    ins_project_conv_weights = sem_project_conv_weights
    ins_fuse_conv_0_weights = sem_fuse_conv_0_weights
    ins_fuse_conv_1_weights = sem_fuse_conv_1_weights
    center_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    center_head_1_weight = torch.randn(ins_embed_head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    center_predictor_weight = torch.randn(1, ins_embed_head_channels, 1, 1, dtype=torch.bfloat16)
    offset_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    offset_head_1_weight = torch.randn(ins_embed_head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
    offset_predictor_weight = torch.randn(2, ins_embed_head_channels, 1, 1, dtype=torch.bfloat16)

    # 3. Create input image tensor (same for both models)
    input_height, input_width = train_size[0], train_size[1]
    input_channels = 3

    # PyTorch input format: [B, C, H, W] (NCHW)
    pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)

    # TTNN input format: [B, H, W, C] (NHWC)
    ttnn_input_torch = pytorch_input.permute(0, 2, 3, 1)  # NCHW -> NHWC

    # 4. Create TTNN input tensor
    ttnn_input = ttnn.from_torch(ttnn_input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # 5. Create PyTorch model
    pytorch_model = PytorchPanopticDeepLab(
        num_classes=num_classes,
        use_real_weights=True,
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
        use_real_weights=True,
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

    # 7. Run forward passes
    print("Running PyTorch model...")
    with torch.no_grad():
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

    print("Running TTNN model...")
    ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

    # 8. Convert TTNN outputs to PyTorch format (NHWC -> NCHW)
    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic).permute(0, 3, 1, 2)
    ttnn_center_torch = ttnn.to_torch(ttnn_center).permute(0, 3, 1, 2)
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset).permute(0, 3, 1, 2)

    # 8. Compare outputs using PCC
    print("\n--- Comparing Semantic Segmentation ---")
    sem_passed, sem_msg = assert_with_pcc(pytorch_semantic, ttnn_semantic_torch, pcc=0.99)
    print(f"Semantic PCC: {sem_msg}")
    assert sem_passed, f"Semantic segmentation PCC failed: {sem_msg}"

    print("\n--- Comparing Center Heatmap ---")
    center_passed, center_msg = assert_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.91)
    print(f"Center PCC: {center_msg}")
    assert center_passed, f"Center heatmap PCC failed: {center_msg}"

    print("\n--- Comparing Offset Map ---")
    offset_passed, offset_msg = assert_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.95)
    print(f"Offset PCC: {offset_msg}")
    assert offset_passed, f"Offset map PCC failed: {offset_msg}"

    print("\n✅✅✅ FULL END-TO-END PCC COMPARISON TEST PASSED ✅✅✅")
