# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math
from models.experimental.oft.tt.tt_oft import OFT
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft
from models.experimental.oft.reference.utils import get_abs_and_relative_error, make_grid
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores

from loguru import logger


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height, scale, torch_model_dtype, use_precomputed_grid, pcc_integral_img, pcc_output, num_slices",
    [
        # fmt: off
        # feats8 {float32,bfloat16} x {use_precomputed_grid, no_use_precomputed_grid}
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.float32,  False, 0.999, 0.872, 18),
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.float32,   True, 0.999, 0.808, 18),
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.bfloat16, False, 0.999, 0.679, 18),
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.bfloat16,  True, 0.999, 0.642, 18),
        # feats16 {float32,bfloat16} x {use_precomputed_grid, no_use_precomputed_grid}
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.float32,  False, 0.999, 0.522, 12),
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.float32,   True, 0.999, 0.453, 12),
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.bfloat16, False, 0.999, 0.348, 12),
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.bfloat16,  True, 0.999, 0.334, 12),
        # feats32 {float32,bfloat16} x {use_precomputed_grid, no_use_precomputed_grid}
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.float32,  False, 0.999, 0.296, 11),
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.float32,   True, 0.999, 0.290, 11),
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.bfloat16, False, 0.999, 0.236, 11),
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.bfloat16,  True, 0.999, 0.229, 11),
        # fmt: on
    ],
    ids=[
        "feats8_fp32_no_precomputed_grid",
        "feats8_fp32_precomputed_grid",
        "feats8_bfp16_no_precomputed_grid",
        "feats8_bfp16_precomputed_grid",
        "feats16_fp32_no_precomputed_grid",
        "feats16_fp32_precomputed_grid",
        "feats16_bfp16_no_precomputed_grid",
        "feats16_bfp16_precomputed_grid",
        "feats32_fp32_no_precomputed_grid",
        "feats32_fp32_precomputed_grid",
        "feats32_bfp16_no_precomputed_grid",
        "feats32_bfp16_precomputed_grid",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_oft_forward(
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
    torch.manual_seed(seed)

    features = torch.relu(torch.randn(*input_shape, dtype=torch.float32))
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
    grid = grid.unsqueeze(0)

    ref_oft = ReferenceOFT(channels, cell_size, grid_height, scale=scale, dtype=torch_model_dtype)
    features = features.to(torch_model_dtype)
    calib = calib.to(torch_model_dtype)
    grid = grid.to(torch_model_dtype)

    (
        ref_out,
        ref_integral_img,
        ref_bbox_top_left,
        ref_bbox_btm_right,
        ref_bbox_top_right,
        ref_bbox_btm_left8,
    ) = ref_oft.forward(features, calib, grid)
    # Prepare TTNN input
    params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)

    features_nhwc = features.permute(0, 2, 3, 1)
    tt_features = ttnn.from_torch(features_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

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
        num_slices=num_slices,
    )
    tt_out, tt_integral_img, bbox_top_left, bbox_btm_right, bbox_top_right, bbox_btm_left8 = tt_oft.forward(
        device, tt_features, tt_calib, tt_grid
    )

    all_passed = []
    for i, (ref, tt, layer_name, exp_pcc) in enumerate(
        zip(
            [ref_integral_img, ref_out],
            [tt_integral_img, tt_out],
            ["integral_img", "out"],
            [pcc_integral_img, pcc_output],
        )
    ):
        if isinstance(tt, ttnn.Tensor):
            torch_tt = ttnn.to_torch(tt, dtype=torch.float32).permute(0, 3, 1, 2).reshape(ref.shape)
        else:
            torch_tt = tt.reshape(ref.shape)  # assume it's already a torch tensor in the right format

        passed, pcc = check_with_pcc(ref, torch_tt, exp_pcc)
        abs, rel = get_abs_and_relative_error(ref, torch_tt)

        all_passed.append(passed)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")
        if passed and float(pcc) - exp_pcc > 0.001:
            logger.warning(
                f"⚠️  Output {i} {layer_name} PCC is better than expected by {float(pcc)-exp_pcc:.3f}. Please update expected PCC value to {math.floor(float(pcc) * 1000) / 1000:.3f}."
            )

    assert all(all_passed), f"OFT module outputs did not pass the PCC check {all_passed=}"
