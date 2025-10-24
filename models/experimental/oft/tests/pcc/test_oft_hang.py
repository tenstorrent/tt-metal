# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.experimental.oft.tt.tt_oft import OFT
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft
from models.experimental.oft.reference.utils import make_grid
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from loguru import logger


def create_test_tensors(input_shapes, torch_model_dtype, seed=0):
    """Helper function to create test tensors."""
    torch.manual_seed(seed)

    features = []
    for shape in input_shapes:
        feature = torch.relu(torch.randn(*shape, dtype=torch.float32))
        features.append(feature.to(torch_model_dtype))

    calib = torch.tensor(
        [
            [
                [7.2154e02, 0.0000e00, 6.0956e02, 4.4857e01],
                [0.0000e00, 7.2154e02, 1.7285e02, 2.1638e-01],
                [0.0000e00, 0.0000e00, 1.0000e00, 2.7459e-03],
            ]
        ],
        dtype=torch_model_dtype,
    )

    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    grid = grid.unsqueeze(0).to(torch_model_dtype)

    return features, calib, grid


def create_oft_models(features_list, calib, grid, channels, cell_size, grid_height, scales, torch_model_dtype):
    """Helper function to create reference OFT models."""
    ref_ofts = []
    ref_outputs = []

    for i, (features, scale) in enumerate(zip(features_list, scales)):
        ref_oft = ReferenceOFT(channels, cell_size, grid_height, scale=scale, dtype=torch_model_dtype)
        ref_ofts.append(ref_oft)

        (
            ref_out,
            ref_integral_img,
            ref_bbox_top_left,
            ref_bbox_btm_right,
            ref_bbox_top_right,
            ref_bbox_btm_left,
        ) = ref_oft.forward(features, calib, grid)
        ref_outputs.append(ref_out)

    return ref_ofts, ref_outputs


def create_tt_tensors(features_list, calib, grid, device):
    """Helper function to convert tensors to TTNN format."""
    tt_features = []
    for features in features_list:
        features_nhwc = features.permute(0, 2, 3, 1)
        tt_feature = ttnn.from_torch(features_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_features.append(tt_feature)

    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    return tt_features, tt_calib, tt_grid


def create_tt_oft_models(
    device,
    ref_ofts,
    features_list,
    calib,
    grid,
    channels,
    cell_size,
    grid_height,
    scales,
    use_precomputed_grid,
    num_slices,
):
    """Helper function to create TT OFT models."""
    tt_ofts = []
    params_list = []

    for i, (ref_oft, features, scale, num_slice) in enumerate(zip(ref_ofts, features_list, scales, num_slices)):
        params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)
        params_list.append(params)

        tt_oft = OFT(
            device,
            params,
            channels,
            cell_size,
            grid_height,
            features.shape[2:],
            calib,
            grid,
            scale=scale,
            use_precomputed_grid=use_precomputed_grid,
            num_slices=num_slice,
        )
        tt_ofts.append(tt_oft)

    return tt_ofts, params_list


def run_tt_oft_forward(tt_ofts, device, tt_features, tt_calib, tt_grid):
    """Helper function to run TT OFT forward passes."""
    tt_outputs = []

    for tt_oft, tt_feature in zip(tt_ofts, tt_features):
        tt_out, tt_integral_img, bbox_top_left, bbox_btm_right, bbox_top_right, bbox_btm_left = tt_oft.forward(
            device, tt_feature, tt_calib, tt_grid
        )
        tt_outputs.append(tt_out)

    return tt_outputs


def check_pcc_results(ref_outputs, tt_outputs, pcc_output, scales):
    """Helper function to check PCC results."""
    assert_all = []

    for i, (ref_out, tt_out, pcc_threshold, scale) in enumerate(zip(ref_outputs, tt_outputs, pcc_output, scales)):
        tt_out_torch = ttnn.to_torch(tt_out).permute(0, 3, 1, 2)
        ref_out_reshaped = ref_out.reshape(tt_out_torch.shape)

        passed, pcc = check_with_pcc(ref_out_reshaped, tt_out_torch, pcc_threshold)
        assert_all.append(passed)
        logger.warning(f"OFT scale {scale}: {passed=}, {pcc=}")

    return assert_all


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height, scale, torch_model_dtype, use_precomputed_grid, pcc_integral_img, pcc_output, num_slices",
    [
        # fmt: off
        # feats8 {float32,bfloat16} x {use_precomputed_grid, no_use_precomputed_grid}
        ([(1, 256, 48, 160), (1,256,24, 80), (1, 256, 12, 40)], 256, 0.5, 4, [1 / 8, 1/16, 1/32 ] , torch.float32,  False, 0.999, (0.891, 0.533, 0.292), (18, 12, 12)), # pcc fails, no hang
        ([(1, 256, 48, 160), (1,256,24, 80), (1, 256, 12, 40)], 256, 0.5, 4, [1 / 8, 1/16, 1/32 ] , torch.float32,  False, 0.999, (0.891, 0.533, 0.292), (18, 12, 11)),
        # HANG!!!!
        # fmt: on
    ],
    ids=["feats8_fp32_no_precomputed_grid_no_hang", "feats8_fp32_no_precomputed_grid_hang"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_oft_forward_3(
    device,
    input_shape,
    channels,
    cell_size,
    grid_height,
    scale,
    torch_model_dtype,
    use_precomputed_grid,
    pcc_integral_img,
    pcc_output,
    num_slices,
    seed,
):
    skip_if_not_blackhole_20_cores(device)

    # Create test tensors
    features_list, calib, grid = create_test_tensors(input_shape, torch_model_dtype, seed)

    # Create reference OFT models and get outputs
    ref_ofts, ref_outputs = create_oft_models(
        features_list, calib, grid, channels, cell_size, grid_height, scale, torch_model_dtype
    )

    # Convert tensors to TTNN format
    tt_features, tt_calib, tt_grid = create_tt_tensors(features_list, calib, grid, device)

    # Create TT OFT models
    tt_ofts, params_list = create_tt_oft_models(
        device,
        ref_ofts,
        features_list,
        calib,
        grid,
        channels,
        cell_size,
        grid_height,
        scale,
        use_precomputed_grid,
        num_slices,
    )

    # Run TT OFT forward passes
    tt_outputs = run_tt_oft_forward(tt_ofts, device, tt_features, tt_calib, tt_grid)

    # Check PCC results
    assert_all = check_pcc_results(ref_outputs, tt_outputs, pcc_output, scale)
    assert all(assert_all)
