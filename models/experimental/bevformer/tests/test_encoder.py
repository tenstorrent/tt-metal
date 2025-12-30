# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import numpy as np
from typing import Dict, Any, List

from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerLayer, TTBEVFormerEncoder
from models.experimental.bevformer.reference.encoder import BEVFormerLayer, BEVFormerEncoder


from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,  # Legacy compatibility
    save_comparison_report,
    print_sparsity_analysis,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_bevformer_encoder_parameters,
    preprocess_bevformer_layer_parameters,
)

from loguru import logger


def create_sample_img_metas(batch_size: int, num_cams: int = 6) -> List[Dict[str, Any]]:
    """Create sample img_metas for testing with camera intrinsics and extrinsics."""
    img_metas = []

    for batch_idx in range(batch_size):
        # Sample camera intrinsics (3x3 matrix)
        camera_intrinsics = []
        for cam_idx in range(num_cams):
            intrinsic = torch.eye(3, dtype=torch.float32)
            intrinsic[0, 0] = 1000  # fx
            intrinsic[1, 1] = 1000  # fy
            intrinsic[0, 2] = 600  # cx
            intrinsic[1, 2] = 400  # cy
            camera_intrinsics.append(intrinsic)

        # Sample camera extrinsics (4x4 transformation matrices)
        camera_extrinsics = []
        for cam_idx in range(num_cams):
            extrinsic = torch.eye(4, dtype=torch.float32)
            # Add small rotation and translation for realism
            angle = cam_idx * np.pi / 3  # 60 degrees apart
            extrinsic[0, 0] = np.cos(angle)
            extrinsic[0, 1] = -np.sin(angle)
            extrinsic[1, 0] = np.sin(angle)
            extrinsic[1, 1] = np.cos(angle)
            extrinsic[0, 3] = cam_idx  # x translation
            extrinsic[1, 3] = 0  # y translation
            extrinsic[2, 3] = 1.5  # z translation (camera height)
            camera_extrinsics.append(extrinsic)

        # Combine intrinsics and extrinsics to create lidar2img matrices
        lidar2img = []
        for cam_idx in range(num_cams):
            # Convert 3x3 intrinsic to 4x4 by padding with [0,0,0,1]
            intrinsic_4x4 = torch.zeros(4, 4, dtype=torch.float32)
            intrinsic_4x4[:3, :3] = camera_intrinsics[cam_idx]
            intrinsic_4x4[3, 3] = 1.0

            # lidar2img = intrinsic @ extrinsic
            lidar2img_cam = intrinsic_4x4 @ camera_extrinsics[cam_idx]
            # Convert to nested list for reference encoder compatibility
            lidar2img.append(lidar2img_cam.tolist())

        meta = {
            "camera_intrinsics": [cam_int.tolist() for cam_int in camera_intrinsics],
            "camera_extrinsics": [cam_ext.tolist() for cam_ext in camera_extrinsics],
            "lidar2img": lidar2img,
            "img_shape": [(900, 1600) for _ in range(num_cams)],  # (height, width) for each camera
        }
        img_metas.append(meta)

    return img_metas


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_layers",
    [
        (1, 900, 256, 6),  # Single layer, small BEV grid (30x30)
        # (1, 2500, 256, 2),  # Two layers, medium BEV grid (50x50)
        # (1, 900, 256, 3),   # Three layers, small BEV grid
        # (1, 2500, 256, 6),  # Full 6 layers, medium BEV grid
        # (2, 900, 256, 1),   # Batch size 2, single layer
    ],
)
@pytest.mark.parametrize(
    "spatial_shapes",
    [
        [[200, 113], [100, 57], [50, 29], [25, 15]],  # nuScenes, input size 1600x900
        # [[160, 90], [80, 45], [40, 23], [20, 12]],   # nuScenes, input size 1280x720
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_bevformer_encoder_forward(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_layers,
    spatial_shapes,
    seed,
):
    """Test TTBEVFormerEncoder against PyTorch reference implementation."""
    torch.manual_seed(seed)
    print_detailed_comparison_flag = False

    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)
    num_cams = 6
    num_heads = 8
    num_levels = 4
    num_points = 4
    feedforward_channels = 1024

    # Create input tensors
    bev_query = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # Camera features: [num_cams, H*W, batch_size, embed_dims]
    encoder_total_key_length = [h * w for h, w in spatial_shapes.tolist()]
    camera_features = torch.randn(num_cams, sum(encoder_total_key_length), batch_size, embed_dims, dtype=torch.float32)

    # Level start index
    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Valid ratios (dummy for testing)
    valid_ratios = torch.ones(batch_size, num_levels, 2, dtype=torch.float32)

    # Previous BEV for temporal attention (optional)
    prev_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # Camera metadata for point sampling
    img_metas = create_sample_img_metas(batch_size, num_cams)

    # Create PyTorch reference model
    ref_model = BEVFormerEncoder(
        num_layers=num_layers,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_cams=num_cams,
        feedforward_channels=feedforward_channels,
        batch_first=True,
        return_intermediate=False,
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_bevformer_encoder_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_model = TTBEVFormerEncoder(
        device=device,
        params=tt_parameters,
        num_layers=num_layers,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_cams=num_cams,
        feedforward_channels=feedforward_channels,
        batch_first=True,
        return_intermediate=False,
    )

    # Forward pass with PyTorch reference model
    with torch.no_grad():
        ref_output = ref_model(
            bev_query=bev_query,
            key=camera_features,
            value=camera_features,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            prev_bev=None,  # No temporal attention
            img_metas=img_metas,
        )

    # Convert tensors to ttnn format
    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_prev_bev = ttnn.from_torch(prev_bev, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    tt_valid_ratios = ttnn.from_torch(valid_ratios, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    # Forward pass with ttnn model
    tt_output = tt_model(
        bev_query=tt_bev_query,
        key=tt_camera_features,
        value=tt_camera_features,
        bev_pos=tt_bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=tt_level_start_index,
        valid_ratios=tt_valid_ratios,
        prev_bev=None,  # No temporal attention
        img_metas=img_metas,
    )

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)

    # Comprehensive comparison using enhanced test utilities
    logger.info(f"Reference model output shape: {ref_output.shape}")
    logger.info(f"TT model output shape: {tt_output_torch.shape}")

    if print_detailed_comparison_flag:
        # Print detailed statistical comparison
        print_detailed_comparison(
            ref_output,
            tt_output_torch,
            tensor_name="bevformer_encoder_output",
            show_histograms=False,
        )

        # Individual sparsity analysis for each tensor
        print_sparsity_analysis(
            ref_output,
            tensor_name="bevformer_encoder_output_torch",
        )

        print_sparsity_analysis(
            tt_output_torch,
            tensor_name="bevformer_encoder_output_ttnn",
        )

    # Comprehensive tolerance checking with multiple criteria
    passed, results = check_with_tolerances(
        ref_output,
        tt_output_torch,
        pcc_threshold=0.997,  # Lower threshold for complex encoder
        abs_error_threshold=6e-2,
        rel_error_threshold=0.5,
        max_error_ratio=0.7,
        tensor_name="bevformer_encoder_output",
    )

    # Assert that the comprehensive check passes
    assert passed, f"Comprehensive tolerance check failed. Results: {results['individual_checks']}"

    logger.info("✅ All BEVFormer encoder tolerance checks passed successfully!")


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims",
    [
        (1, 900, 256),  # Small BEV grid (30x30)
        (1, 2500, 256),  # Medium BEV grid (50x50)
        (2, 900, 256),  # Batch size 2
    ],
)
@pytest.mark.parametrize(
    "spatial_shapes",
    [
        [[200, 113], [100, 57], [50, 29], [25, 15]],  # nuScenes
        [[160, 90], [80, 45], [40, 23], [20, 12]],  # nuScenes smaller
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_bevformer_layer_forward(
    device,
    batch_size,
    num_query,
    embed_dims,
    spatial_shapes,
    seed,
):
    """Test TTBEVFormerLayer against PyTorch reference implementation."""
    torch.manual_seed(seed)
    print_detailed_comparison_flag = False

    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)
    num_cams = 6
    num_heads = 8
    num_levels = 4
    num_points = 4
    feedforward_channels = 1024

    # Create input tensors
    bev_query = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # Camera features: [num_cams, H*W, batch_size, embed_dims]
    layer_total_key_length = [h * w for h, w in spatial_shapes.tolist()]
    camera_features = torch.randn(num_cams, sum(layer_total_key_length), batch_size, embed_dims, dtype=torch.float32)

    # Level start index
    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Valid ratios
    valid_ratios = torch.ones(batch_size, num_levels, 2, dtype=torch.float32)

    # Previous BEV for temporal attention
    prev_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # Generate 3D reference points for temporal self-attention
    D = 4  # Number of points per pillar
    reference_points_3d = torch.rand(batch_size, num_query, D, 3, dtype=torch.float32)

    # Determine BEV grid dimensions from num_query
    if num_query == 900:  # 30x30
        bev_h, bev_w = 30, 30
    elif num_query == 2500:  # 50x50
        bev_h, bev_w = 50, 50
    else:
        # Fallback: assume square grid
        bev_h = bev_w = int(num_query**0.5)

    bev_shape = torch.tensor([[bev_h, bev_w]], dtype=torch.long)

    # Pre-projected reference points and masks (simplified for single layer test)
    reference_points_cam = torch.rand(num_cams, batch_size, num_query, D, 2, dtype=torch.float32)
    bev_mask = torch.ones(num_cams, batch_size, num_query, D, dtype=torch.bool)
    # Randomly mask out 90% of points for realism
    total_points = bev_mask.numel()
    num_invalid = int(0.90 * total_points)
    invalid_indices = torch.randperm(total_points)[:num_invalid]
    bev_mask.view(-1)[invalid_indices] = False

    # Create PyTorch reference model
    ref_layer = BEVFormerLayer(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_cams=num_cams,
        feedforward_channels=feedforward_channels,
        batch_first=True,
    )
    ref_layer.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = preprocess_bevformer_layer_parameters(
        ref_layer,
        device=device,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_layer = TTBEVFormerLayer(
        device=device,
        params=tt_parameters,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_cams=num_cams,
        feedforward_channels=feedforward_channels,
        batch_first=True,
    )

    # Forward pass with PyTorch reference model
    with torch.no_grad():
        ref_output = ref_layer(
            bev_query=bev_query,
            key=camera_features,
            value=camera_features,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            bev_shape=bev_shape,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            prev_bev=prev_bev,
            reference_points_3d=reference_points_3d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
        )

    # Convert tensors to ttnn format
    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_prev_bev = ttnn.from_torch(prev_bev, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_reference_points_cam = ttnn.from_torch(
        reference_points_cam, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    tt_valid_ratios = ttnn.from_torch(valid_ratios, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_reference_points_3d = ttnn.from_torch(
        reference_points_3d, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    # Forward pass with ttnn model
    tt_output = tt_layer(
        bev_query=tt_bev_query,
        key=tt_camera_features,
        value=tt_camera_features,
        bev_pos=tt_bev_pos,
        spatial_shapes=spatial_shapes,
        bev_shape=bev_shape,
        level_start_index=tt_level_start_index,
        valid_ratios=tt_valid_ratios,
        prev_bev=tt_prev_bev,
        reference_points_3d=tt_reference_points_3d,
        reference_points_cam=tt_reference_points_cam,
        bev_mask=tt_bev_mask,
    )

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)

    # Comprehensive comparison
    logger.info(f"Reference layer output shape: {ref_output.shape}")
    logger.info(f"TT layer output shape: {tt_output_torch.shape}")

    if print_detailed_comparison_flag:
        print_detailed_comparison(
            ref_output,
            tt_output_torch,
            tensor_name="bevformer_layer_output",
            show_histograms=False,
        )

    # Tolerance checking
    passed, results = check_with_tolerances(
        ref_output,
        tt_output_torch,
        pcc_threshold=0.997,  # Lower threshold for complex layer
        abs_error_threshold=6e-2,
        rel_error_threshold=0.5,
        max_error_ratio=0.7,
        tensor_name="bevformer_layer_output",
    )

    assert passed, f"BEVFormerLayer tolerance check failed. Results: {results['individual_checks']}"

    logger.info("✅ All BEVFormer layer tolerance checks passed successfully!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
def test_bevformer_encoder_return_intermediate(device):
    """Test that TTBEVFormerEncoder correctly handles return_intermediate=True."""
    torch.manual_seed(42)

    batch_size = 1
    num_query = 900
    embed_dims = 256
    num_layers = 3
    num_cams = 6

    spatial_shapes = torch.tensor([[50, 29], [25, 15]], dtype=torch.long)

    # Create input tensors
    bev_query = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    encoder_total_key_length = [h * w for h, w in spatial_shapes.tolist()]
    camera_features = torch.randn(num_cams, sum(encoder_total_key_length), batch_size, embed_dims, dtype=torch.float32)

    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Create models with return_intermediate=True
    ref_model = BEVFormerEncoder(
        num_layers=num_layers,
        embed_dims=embed_dims,
        return_intermediate=True,
    )
    ref_model.eval()

    tt_parameters = create_bevformer_encoder_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.float32,
    )

    tt_model = TTBEVFormerEncoder(
        device=device,
        params=tt_parameters,
        num_layers=num_layers,
        embed_dims=embed_dims,
        return_intermediate=True,
    )

    # Forward pass
    with torch.no_grad():
        ref_outputs = ref_model(
            bev_query=bev_query,
            key=camera_features,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    tt_outputs = tt_model(
        bev_query=tt_bev_query,
        key=tt_camera_features,
        spatial_shapes=spatial_shapes,
        level_start_index=tt_level_start_index,
    )

    # Check that intermediate outputs are returned
    assert isinstance(ref_outputs, torch.Tensor), f"Expected torch.Tensor, got {type(ref_outputs)}"
    assert isinstance(tt_outputs, list), f"Expected list for TT intermediate outputs, got {type(tt_outputs)}"
    assert ref_outputs.shape[0] == num_layers, f"Expected {num_layers} intermediate outputs, got {ref_outputs.shape[0]}"
    assert len(tt_outputs) == num_layers, f"Expected {num_layers} TT intermediate outputs, got {len(tt_outputs)}"

    logger.info("✅ BEVFormer encoder intermediate outputs test passed!")
