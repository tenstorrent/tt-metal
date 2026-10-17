# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)

from models.experimental.bevformer.tt.tt_point_sampling_3d_2d import (
    generate_reference_points_ttnn,
    point_sampling_3d_to_2d_ttnn,
)

from loguru import logger

from models.experimental.bevformer.tests.common import (
    PRESETS,
    assert_pcc,
    img_metas_for_dataset,
    lidar2img_for_dataset,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


@pytest.mark.parametrize(
    "config_name, bev_h, bev_w, batch_size",
    [
        ("nuscenes_tiny", 100, 100, 1),
        ("nuscenes_base", 200, 200, 1),
        ("nuscenes_base", 200, 200, 2),
        ("carla_base", 200, 200, 1),
    ],
)
@pytest.mark.parametrize("seed", [42])
def test_generate_reference_points(
    device,
    config_name,
    bev_h,
    bev_w,
    batch_size,
    seed,
):
    torch.manual_seed(seed)

    preset = PRESETS[config_name]
    z_cfg = preset.z_cfg

    torch_ref_points = generate_reference_points(bev_h, bev_w, z_cfg, batch_size=batch_size, device=torch.device("cpu"))

    ttnn_ref_points = generate_reference_points_ttnn(bev_h, bev_w, z_cfg, device, batch_size=batch_size)

    ttnn_ref_points_torch = ttnn.to_torch(ttnn_ref_points, dtype=torch.float32)

    expected_shape = (batch_size, bev_h * bev_w, z_cfg["num_points"], 3)
    assert (
        torch_ref_points.shape == expected_shape
    ), f"Torch shape mismatch: {torch_ref_points.shape} vs {expected_shape}"
    assert (
        ttnn_ref_points_torch.shape == expected_shape
    ), f"TTNN shape mismatch: {ttnn_ref_points_torch.shape} vs {expected_shape}"

    assert torch.all(torch_ref_points >= 0.0), "Torch reference points should be >= 0"
    assert torch.all(torch_ref_points <= 1.0), "Torch reference points should be <= 1"
    assert torch.all(ttnn_ref_points_torch >= 0.0), "TTNN reference points should be >= 0"
    assert torch.all(ttnn_ref_points_torch <= 1.0), "TTNN reference points should be <= 1"

    # Both build the same grid, element for element.
    torch.testing.assert_close(ttnn_ref_points_torch, torch_ref_points, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config_name, bev_h, bev_w, batch_size, expected_pcc",
    [
        ("nuscenes_tiny", 100, 100, 1, 0.999),
        ("nuscenes_base", 200, 200, 1, 0.999),
        ("carla_base", 200, 200, 1, 0.999),
    ],
)
@pytest.mark.parametrize("seed", [42])
def test_point_sampling_3d_to_2d(
    device,
    config_name,
    bev_h,
    bev_w,
    batch_size,
    expected_pcc,
    seed,
):
    torch.manual_seed(seed)

    preset = PRESETS[config_name]

    pc_range = preset.pc_range
    z_cfg = preset.z_cfg
    num_cams = preset.num_cams
    eps = 1e-5

    # The deterministic rig keeps bev_mask a property of the geometry rather than
    # of an RNG draw, so this test's projections match the encoder's.
    lidar2img = lidar2img_for_dataset(preset).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    img_metas = img_metas_for_dataset(preset, batch_size)

    torch_ref_points = generate_reference_points(bev_h, bev_w, z_cfg, batch_size=batch_size, device=torch.device("cpu"))

    torch_ref_points_cam, torch_bev_mask = point_sampling_3d_to_2d(
        torch_ref_points,
        pc_range,
        lidar2img,
        img_metas=img_metas,
        eps=eps,
    )

    ttnn_ref_points_cam, ttnn_bev_mask = point_sampling_3d_to_2d_ttnn(
        torch_ref_points,
        pc_range,
        lidar2img,
        img_metas=img_metas,
        eps=eps,
        device=device,
    )

    ttnn_ref_points_cam_torch = ttnn.to_torch(ttnn_ref_points_cam, dtype=torch.float32)
    ttnn_bev_mask_torch = ttnn.to_torch(ttnn_bev_mask, dtype=torch.bool)

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

    # Invalid points (where mask is False) can have any coordinate values

    valid_torch_points = torch_ref_points_cam[torch_bev_mask]
    if valid_torch_points.numel() > 0:
        assert torch.all(valid_torch_points >= 0.0), "Torch valid projected points should be >= 0"
        assert torch.all(valid_torch_points <= 1.0), "Torch valid projected points should be <= 1"
    else:
        logger.warning("No valid points found in torch implementation - this might indicate camera matrix issues")

    valid_ttnn_points = ttnn_ref_points_cam_torch[ttnn_bev_mask_torch]
    if valid_ttnn_points.numel() > 0:
        assert torch.all(valid_ttnn_points >= 0.0), "TTNN valid projected points should be >= 0"
        assert torch.all(valid_ttnn_points <= 1.0), "TTNN valid projected points should be <= 1"
    else:
        logger.warning("No valid points found in TTNN implementation - this might indicate camera matrix issues")

    torch_valid_ratio = torch_bev_mask.float().mean()
    ttnn_valid_ratio = ttnn_bev_mask_torch.float().mean()
    logger.info(f"Valid point ratios - Torch: {torch_valid_ratio:.3f}, TTNN: {ttnn_valid_ratio:.3f}")

    # We should have at least some valid points for the test to be meaningful
    assert torch_valid_ratio > 0.001, f"Too few valid points in torch implementation: {torch_valid_ratio:.3f}"
    assert ttnn_valid_ratio > 0.001, f"Too few valid points in TTNN implementation: {ttnn_valid_ratio:.3f}"

    # A point outside its camera's frustum carries whatever the projection divide
    # produced, so only points both implementations call valid are comparable.
    # Mask disagreement is a separate failure, caught by the mask check below.
    shared_mask = torch_bev_mask & ttnn_bev_mask_torch
    shared_torch_points = torch_ref_points_cam[shared_mask]
    shared_ttnn_points = ttnn_ref_points_cam_torch[shared_mask]

    assert_pcc(shared_torch_points, shared_ttnn_points, expected_pcc)
    assert_pcc(torch_bev_mask.float(), ttnn_bev_mask_torch.float(), 0.994)
