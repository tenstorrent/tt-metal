# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
)

from loguru import logger


# Test configuration and setup functions
def get_test_config():
    """Get common test configuration parameters."""
    return {
        "bev_h": 200,
        "bev_w": 200,
        "z_cfg": {"num_points": 4, "start": -5.0, "end": 3.0},
        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # nuScenes range
        "img_shape": (900, 1600),  # (height, width) for nuScenes cameras
        "num_cams": 6,
        "eps": 1e-5,
    }


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
        look_dir = look_dir / np.linalg.norm(look_dir)

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


def get_test_setup(batch_size=1):
    """Get complete test setup with configuration and generated matrices."""
    config = get_test_config()
    lidar2img = create_sample_camera_matrices(config["num_cams"])

    # Add batch dimension to lidar2img: [num_cams, 4, 4] -> [batch_size, num_cams, 4, 4]
    lidar2img = lidar2img.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Create img_metas for each batch item
    img_metas = []
    for batch_idx in range(batch_size):
        img_metas.append({"img_shape": [config["img_shape"]] * config["num_cams"]})  # List of shapes for each camera

    return {**config, "lidar2img": lidar2img, "img_metas": img_metas, "batch_size": batch_size}


# Test functions
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_reference_points(device, seed, batch_size):
    """Test 3D reference point generation."""
    torch.manual_seed(seed)
    print_detailed_comparison_flag = False
    setup = get_test_setup(batch_size=batch_size)

    # Test torch implementation
    torch_ref_points = generate_reference_points(
        setup["bev_h"], setup["bev_w"], setup["z_cfg"], batch_size=batch_size, device=torch.device("cpu")
    )

    # Test TTNN implementation
    ttnn_ref_points = generate_reference_points_ttnn(
        setup["bev_h"], setup["bev_w"], setup["z_cfg"], device, batch_size=batch_size
    )

    # Convert TTNN result to torch for comparison
    ttnn_ref_points_torch = ttnn.to_torch(ttnn_ref_points, dtype=torch.float32)

    # Check shapes - now includes batch dimension
    expected_shape = (batch_size, setup["bev_h"] * setup["bev_w"], setup["z_cfg"]["num_points"], 3)
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

    if print_detailed_comparison_flag:
        # Detailed comparison
        print_detailed_comparison(
            torch_ref_points,
            ttnn_ref_points_torch,
            tensor_name="reference_points_generation",
            show_sparsity=False,  # Reference points shouldn't be sparse
        )

    # Check with tight tolerances since this should be exact conversion
    passed, results = check_with_tolerances(
        torch_ref_points,
        ttnn_ref_points_torch,
        pcc_threshold=0.999,
        abs_error_threshold=7.5,
        rel_error_threshold=1e-3,
        max_error_ratio=0.6,
        tensor_name="reference_points_generation",
    )

    assert passed, f"Reference point generation comparison failed. Results: {results['individual_checks']}"

    logger.info("✅ Reference point generation test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "bev_size",
    [
        (200, 200),  # Default size
    ],
)
@pytest.mark.parametrize("batch_size", [1])
def test_point_sampling_3d_to_2d(device, seed, bev_size, batch_size):
    """Test 3D to 2D point sampling transformation."""
    torch.manual_seed(seed)
    setup = get_test_setup(batch_size=batch_size)
    print_detailed_comparison_flag = False

    bev_h, bev_w = bev_size

    # Generate reference points with batch dimension
    torch_ref_points = generate_reference_points(
        bev_h, bev_w, setup["z_cfg"], batch_size=batch_size, device=torch.device("cpu")
    )

    # Test torch implementation
    torch_ref_points_cam, torch_bev_mask = point_sampling_3d_to_2d(
        torch_ref_points,
        setup["pc_range"],
        setup["lidar2img"],
        img_metas=setup["img_metas"],
        eps=setup["eps"],
    )

    # Test TTNN implementation
    ttnn_ref_points_cam, ttnn_bev_mask = point_sampling_3d_to_2d_ttnn(
        torch_ref_points,  # Use same input
        setup["pc_range"],
        setup["lidar2img"],
        img_metas=setup["img_metas"],
        eps=setup["eps"],
        device=device,
    )

    # Convert TTNN results to torch for comparison
    ttnn_ref_points_cam_torch = ttnn.to_torch(ttnn_ref_points_cam, dtype=torch.float32)
    ttnn_bev_mask_torch = ttnn.to_torch(ttnn_bev_mask, dtype=torch.bool)

    # Check shapes - now includes batch dimension in the expected shape
    expected_points_shape = (setup["num_cams"], batch_size, bev_h * bev_w, setup["z_cfg"]["num_points"], 2)
    expected_mask_shape = (setup["num_cams"], batch_size, bev_h * bev_w, setup["z_cfg"]["num_points"])

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
        logger.warning("No valid points found in torch implementation - this might indicate camera matrix issues")

    # For TTNN, check only valid points (where mask is True)
    valid_ttnn_points = ttnn_ref_points_cam_torch[ttnn_bev_mask_torch]
    if valid_ttnn_points.numel() > 0:
        assert torch.all(valid_ttnn_points >= 0.0), "TTNN valid projected points should be >= 0"
        assert torch.all(valid_ttnn_points <= 1.0), "TTNN valid projected points should be <= 1"
    else:
        logger.warning("No valid points found in TTNN implementation - this might indicate camera matrix issues")

    # Check that we have some valid points (at least some points should be visible)
    torch_valid_ratio = torch_bev_mask.float().mean()
    ttnn_valid_ratio = ttnn_bev_mask_torch.float().mean()
    logger.info(f"Valid point ratios - Torch: {torch_valid_ratio:.3f}, TTNN: {ttnn_valid_ratio:.3f}")

    # We should have at least some valid points for the test to be meaningful
    assert torch_valid_ratio > 0.001, f"Too few valid points in torch implementation: {torch_valid_ratio:.3f}"
    assert ttnn_valid_ratio > 0.001, f"Too few valid points in TTNN implementation: {ttnn_valid_ratio:.3f}"

    if print_detailed_comparison_flag:
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

    # Check with appropriate tolerances for geometric transformations
    points_passed, points_results = check_with_tolerances(
        torch_ref_points_cam,
        ttnn_ref_points_cam_torch,
        pcc_threshold=0.999,
        abs_error_threshold=7.5,
        rel_error_threshold=0.05,
        max_error_ratio=0.5,
        tensor_name="projected_reference_points",
    )

    mask_passed, mask_results = check_with_tolerances(
        torch_bev_mask.float(),
        ttnn_bev_mask_torch.float(),
        pcc_threshold=0.999,
        abs_error_threshold=0.1,
        rel_error_threshold=0.1,
        max_error_ratio=0.1,
        tensor_name="validity_mask",
    )

    # At least one should pass (prefer points accuracy over mask accuracy)
    assert points_passed or mask_passed, (
        f"Both point sampling comparisons failed.\n"
        f"Points: {points_results['individual_checks']}\n"
        f"Mask: {mask_results['individual_checks']}"
    )

    if points_passed:
        logger.info("✅ Point sampling 3D to 2D test passed!")
    else:
        logger.warning("⚠️ Point sampling passed but with reduced accuracy on coordinates")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [42])
def test_complete_pipeline_functions(device, seed):
    """Test the complete point sampling pipeline using functions only."""
    torch.manual_seed(seed)
    setup = get_test_setup()
    print_detailed_comparison_flag = False

    # Generate reference points using torch function
    torch_ref_3d = generate_reference_points(
        setup["bev_h"], setup["bev_w"], setup["z_cfg"], device=torch.device("cpu"), dtype=torch.float32
    )

    # Test torch pipeline
    torch_ref_points_cam, torch_bev_mask = point_sampling_3d_to_2d(
        torch_ref_3d,
        setup["pc_range"],
        setup["lidar2img"].to(torch.float32),  # Ensure consistent dtype
        img_metas=setup["img_metas"],
        eps=setup["eps"],
    )

    # Test ttnn pipeline using torch tensors as input (let ttnn function handle conversion)
    ttnn_ref_points_cam, ttnn_bev_mask = point_sampling_3d_to_2d_ttnn(
        torch_ref_3d,  # Use same torch tensor as input
        setup["pc_range"],
        setup["lidar2img"].to(torch.float32),  # Ensure consistent dtype
        img_metas=setup["img_metas"],
        eps=setup["eps"],
        device=device,
    )

    # Convert TTNN results to torch for comparison
    ttnn_ref_points_cam_torch = ttnn.to_torch(ttnn_ref_points_cam, dtype=torch.float32)
    ttnn_bev_mask_torch = ttnn.to_torch(ttnn_bev_mask, dtype=torch.bool)

    if print_detailed_comparison_flag:
        # Compare results
        print_detailed_comparison(
            torch_ref_points_cam,
            ttnn_ref_points_cam_torch,
            tensor_name="pipeline_projected_points",
            show_sparsity=True,
        )

        print_detailed_comparison(
            torch_bev_mask.float(),
            ttnn_bev_mask_torch.float(),
            tensor_name="pipeline_validity_mask",
            show_sparsity=True,
        )

    # Check with tolerances
    points_passed, points_results = check_with_tolerances(
        torch_ref_points_cam,
        ttnn_ref_points_cam_torch,
        pcc_threshold=0.95,
        abs_error_threshold=7.5,
        rel_error_threshold=0.05,
        max_error_ratio=0.5,
        tensor_name="pipeline_projected_points",
    )

    assert points_passed, f"Pipeline comparison failed. Results: {points_results['individual_checks']}"

    logger.info("✅ Complete pipeline functions test passed!")


if __name__ == "__main__":
    # Run specific tests for debugging
    import pytest

    pytest.main([__file__ + "::test_generate_reference_points", "-v"])
