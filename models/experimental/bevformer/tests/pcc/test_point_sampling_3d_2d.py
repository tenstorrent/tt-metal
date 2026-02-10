# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import numpy as np

from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)

from models.experimental.bevformer.tt.tt_point_sampling_3d_2d import (
    generate_reference_points_ttnn,
    point_sampling_3d_to_2d_ttnn,
)

from models.experimental.bevformer.config.encoder_config import (
    get_preset_config,
)

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,
)

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = True

# Default Test Configuration                                                  #
PRINT_DETAILED_COMPARISON_FLAG = False


# Helper functions for testing
def create_sample_camera_matrices(num_cams):
    """Create sample camera transformation matrices for testing."""
    torch.manual_seed(42)  # For reproducible tests

    # Create different camera matrices
    lidar2img_matrices = []

    for cam_idx in range(num_cams):
        # Camera intrinsic matrix
        fx, fy = 800.0, 450.0  # More reasonable focal lengths
        cx, cy = 800.0, 450.0  # Principal point at image center

        # Camera extrinsic parameters (rotation + translation)
        # Different pose for each camera around the vehicle
        angle = cam_idx * 60.0 * np.pi / 180.0  # 60 degrees apart

        # Create a viewing transformation that looks towards center from outside
        # Position camera at distance from origin
        cam_x = 3.0 * np.cos(angle)  # 3 meters from center
        cam_y = 3.0 * np.sin(angle)
        cam_z = 1.5  # 1.5m height

        # Camera looks towards the origin (simplified)
        # Create rotation to look towards center
        look_dir = np.array([-cam_x, -cam_y, -cam_z])
        look_dir /= np.linalg.norm(look_dir)

        # Simplified rotation matrix (just use identity for testing)
        R = np.eye(3, dtype=np.float32)

        # Translation vector
        t = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Create extrinsic matrix [R|t] in homogeneous coordinates
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        # Create intrinsic matrix in homogeneous coordinates
        intrinsic = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )

        # Combine intrinsic and extrinsic: K * [R|t]
        camera_matrix = intrinsic @ extrinsic

        # Scale down the result to get more reasonable coordinate ranges
        # This helps ensure projected coordinates are in a manageable range
        camera_matrix[:2, :] *= 0.001  # Scale down x,y projections

        lidar2img_matrices.append(torch.from_numpy(camera_matrix))

    return torch.stack(lidar2img_matrices)


# Test functions
@pytest.mark.parametrize(
    "config_name, bev_h, bev_w, batch_size, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_tiny", 100, 100, 1, 1.0, 0.0, 0, 0.0),  # Tiny model with perfect accuracy expected
        ("nuscenes_base", 200, 200, 1, 1.0, 0.0, 0, 0.0),  # Base model with perfect accuracy expected
        ("nuscenes_base", 200, 200, 2, 1.0, 0.0, 0, 0.0),  # Base model with batch size 2
        ("carla_base", 200, 200, 1, 1.0, 0.0, 0, 0.0),  # CARLA base model
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [42])
def test_generate_reference_points(
    device,
    config_name,
    bev_h,
    bev_w,
    batch_size,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
):
    """Test 3D reference point generation using configuration system."""
    torch.manual_seed(seed)

    # Get configuration from preset
    preset_config = get_preset_config(config_name)
    if preset_config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    dataset_config = preset_config.dataset_config
    z_cfg = dataset_config.z_cfg

    # --------------------------------------------------------------------------- #
    # Function Execution                                                          #
    # --------------------------------------------------------------------------- #

    # Test torch implementation
    torch_ref_points = generate_reference_points(bev_h, bev_w, z_cfg, batch_size=batch_size, device=torch.device("cpu"))

    # Test TTNN implementation
    ttnn_ref_points = generate_reference_points_ttnn(bev_h, bev_w, z_cfg, device, batch_size=batch_size)

    # Convert TTNN result to torch for comparison
    ttnn_ref_points_torch = ttnn.to_torch(ttnn_ref_points, dtype=torch.float32)

    # --------------------------------------------------------------------------- #
    # Output Comparison                                                           #
    # --------------------------------------------------------------------------- #

    # Check shapes - now includes batch dimension
    expected_shape = (batch_size, bev_h * bev_w, z_cfg["num_points"], 3)
    assert (
        torch_ref_points.shape == expected_shape
    ), f"Torch shape mismatch: {torch_ref_points.shape} vs {expected_shape}"
    assert (
        ttnn_ref_points_torch.shape == expected_shape
    ), f"TTNN shape mismatch: {ttnn_ref_points_torch.shape} vs {expected_shape}"

    # Check coordinate ranges (should be in [0, 1])
    assert torch.all(torch_ref_points >= 0.0), "Torch reference points should be >= 0"
    assert torch.all(torch_ref_points <= 1.0), "Torch reference points should be <= 1"
    assert torch.all(ttnn_ref_points_torch >= 0.0), "TTNN reference points should be >= 0"
    assert torch.all(ttnn_ref_points_torch <= 1.0), "TTNN reference points should be <= 1"

    if PRINT_DETAILED_COMPARISON_FLAG:
        # Detailed comparison
        print_detailed_comparison(
            torch_ref_points,
            ttnn_ref_points_torch,
            tensor_name="reference_points_generation",
            show_sparsity=False,  # Reference points shouldn't be sparse
        )

    # Check with expected tolerances from test parameters
    passed, results = check_with_tolerances(
        torch_ref_points,
        ttnn_ref_points_torch,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="reference_points_generation",
    )

    assert passed, f"Reference point generation comparison failed. Results: {results['individual_checks']}"

    if ENABLE_LOGGING:
        logger.info("✅ Reference point generation test passed!")


@pytest.mark.parametrize(
    "config_name, bev_h, bev_w, batch_size, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_tiny", 100, 100, 1, 0.999, 13.06, 0.02, 0.5),  # NuScenes tiny model
        ("nuscenes_base", 200, 200, 1, 0.999, 4.98, 0.03, 0.5),  # NuScenes base model
        ("carla_base", 200, 200, 1, 0.999, 5.21, 0.02, 0.5),  # CARLA base model
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [42])
def test_point_sampling_3d_to_2d(
    device,
    config_name,
    bev_h,
    bev_w,
    batch_size,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
):
    """Test 3D to 2D point sampling transformation using configuration system."""
    torch.manual_seed(seed)

    # Get configuration from preset
    preset_config = get_preset_config(config_name)
    if preset_config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    dataset_config = preset_config.dataset_config

    # Extract parameters from configs
    pc_range = dataset_config.pc_range
    z_cfg = dataset_config.z_cfg
    num_cams = dataset_config.num_cams
    img_shape = (dataset_config.input_size[1], dataset_config.input_size[0])  # (height, width)
    eps = 1e-5

    # --------------------------------------------------------------------------- #
    # Generate Inputs                                                             #
    # --------------------------------------------------------------------------- #

    # Create camera matrices for testing
    lidar2img = create_sample_camera_matrices(num_cams)
    lidar2img = lidar2img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Add batch dimension

    # Create img_metas for each batch item
    img_metas = []
    for batch_idx in range(batch_size):
        img_metas.append({"img_shape": [img_shape] * num_cams})

    # Generate reference points with batch dimension
    torch_ref_points = generate_reference_points(bev_h, bev_w, z_cfg, batch_size=batch_size, device=torch.device("cpu"))

    # --------------------------------------------------------------------------- #
    # Function Execution                                                          #
    # --------------------------------------------------------------------------- #

    # Test torch implementation
    torch_ref_points_cam, torch_bev_mask = point_sampling_3d_to_2d(
        torch_ref_points,
        pc_range,
        lidar2img,
        img_metas=img_metas,
        eps=eps,
    )

    # Test TTNN implementation
    ttnn_ref_points_cam, ttnn_bev_mask = point_sampling_3d_to_2d_ttnn(
        torch_ref_points,  # Use same input
        pc_range,
        lidar2img,
        img_metas=img_metas,
        eps=eps,
        device=device,
    )

    # Convert TTNN results to torch for comparison
    ttnn_ref_points_cam_torch = ttnn.to_torch(ttnn_ref_points_cam, dtype=torch.float32)
    ttnn_bev_mask_torch = ttnn.to_torch(ttnn_bev_mask, dtype=torch.bool)

    # --------------------------------------------------------------------------- #
    # Output Comparison                                                           #
    # --------------------------------------------------------------------------- #

    # Check shapes - now includes batch dimension in the expected shape
    expected_points_shape = (num_cams, batch_size, bev_h * bev_w, z_cfg["num_points"], 2)
    expected_mask_shape = (num_cams, batch_size, bev_h * bev_w, z_cfg["num_points"])

    assert (
        torch_ref_points_cam.shape == expected_points_shape
    ), f"Torch points shape mismatch: {torch_ref_points_cam.shape}"
    assert torch_bev_mask.shape == expected_mask_shape, f"Torch mask shape mismatch: {torch_bev_mask.shape}"
    assert (
        ttnn_ref_points_cam_torch.shape == expected_points_shape
    ), f"TTNN points shape mismatch: {ttnn_ref_points_cam_torch.shape}"
    assert ttnn_bev_mask_torch.shape == expected_mask_shape, f"TTNN mask shape mismatch: {ttnn_bev_mask_torch.shape}"

    # Check projected coordinates are in valid range [0, 1] only for valid points
    # Invalid points (where mask is False) can have any coordinate values

    # For torch implementation, check only valid points
    valid_torch_points = torch_ref_points_cam[torch_bev_mask]  # Use mask to filter valid points
    if valid_torch_points.numel() > 0:
        assert torch.all(valid_torch_points >= 0.0), "Torch valid projected points should be >= 0"
        assert torch.all(valid_torch_points <= 1.0), "Torch valid projected points should be <= 1"
    else:
        if ENABLE_LOGGING:
            logger.warning("No valid points found in torch implementation - this might indicate camera matrix issues")

    # For TTNN, check only valid points (where mask is True)
    valid_ttnn_points = ttnn_ref_points_cam_torch[ttnn_bev_mask_torch]
    if valid_ttnn_points.numel() > 0:
        assert torch.all(valid_ttnn_points >= 0.0), "TTNN valid projected points should be >= 0"
        assert torch.all(valid_ttnn_points <= 1.0), "TTNN valid projected points should be <= 1"
    else:
        if ENABLE_LOGGING:
            logger.warning("No valid points found in TTNN implementation - this might indicate camera matrix issues")

    # Check that we have some valid points (at least some points should be visible)
    torch_valid_ratio = torch_bev_mask.float().mean()
    ttnn_valid_ratio = ttnn_bev_mask_torch.float().mean()
    if ENABLE_LOGGING:
        logger.info(f"Valid point ratios - Torch: {torch_valid_ratio:.3f}, TTNN: {ttnn_valid_ratio:.3f}")

    # We should have at least some valid points for the test to be meaningful
    assert torch_valid_ratio > 0.001, f"Too few valid points in torch implementation: {torch_valid_ratio:.3f}"
    assert ttnn_valid_ratio > 0.001, f"Too few valid points in TTNN implementation: {ttnn_valid_ratio:.3f}"

    if PRINT_DETAILED_COMPARISON_FLAG:
        # Compare projected points
        print_detailed_comparison(
            torch_ref_points_cam,
            ttnn_ref_points_cam_torch,
            tensor_name="projected_reference_points",
            show_sparsity=True,  # Camera projections can be sparse due to out-of-view points
        )

        # Compare validity masks
        print_detailed_comparison(
            torch_bev_mask.float(),  # Convert to float for comparison
            ttnn_bev_mask_torch.float(),
            tensor_name="validity_mask",
            show_sparsity=True,
        )

    # Check with expected tolerances from test parameters
    check_with_tolerances(
        torch_ref_points_cam,
        ttnn_ref_points_cam_torch,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="projected_reference_points",
    )

    check_with_tolerances(
        torch_bev_mask.float(),
        ttnn_bev_mask_torch.float(),
        pcc_threshold=0.994,
        abs_error_threshold=0.1,
        rel_error_threshold=0.1,
        max_error_ratio=0.1,
        tensor_name="validity_mask",
    )

    points_passed, points_message = check_with_pcc(
        torch_ref_points_cam,
        ttnn_ref_points_cam_torch,
        expected_pcc,
    )

    assert points_passed, f"PCC check for projected_reference_points failed: {points_message}"

    mask_passed, mask_message = check_with_pcc(
        torch_bev_mask.float(),
        ttnn_bev_mask_torch.float(),
        0.994,
    )

    assert mask_passed, f"PCC check for validity_mask failed: {mask_message}"

    if ENABLE_LOGGING:
        logger.info("✅ Point sampling 3D to 2D test passed!")
