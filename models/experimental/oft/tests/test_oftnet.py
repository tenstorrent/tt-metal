import torch
import ttnn
import pytest
import os
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from models.experimental.oft.reference.utils import make_grid, load_calib, load_image

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 12 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "input_image_path, calib_path",
    [
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000022.jpg")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000022.txt")),
        )
    ],
)
@pytest.mark.parametrize(
    # fmt: off
    "use_host_oft, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    [
       (False, 0.074, 0.105, 0.124, 0.105),  # Using device OFT
    #  ( True, 0.86, 0.99, 0.99, 0.99, 0.99)
    ],
    # fmt: on
    # ids=["use_device_oft", "use_host_oft"],
)
def test_oftnet(
    device,
    input_image_path,
    calib_path,
    use_host_oft,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
):
    torch.manual_seed(42)

    input_tensor = load_image(input_image_path, pad_hw=(384, 1280))[None]
    calib = load_calib(calib_path)[None]
    # OFT configuration based on real model parameters
    grid_res = 0.5
    grid_size = (80.0, 80.0)
    grid_height = 4.0
    y_offset = 1.74
    grid = make_grid(grid_size, (-grid_size[0] / 2.0, y_offset, 0.0), grid_res)[None]

    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=grid_res,
        grid_height=grid_height,
    )

    parameters = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_calib = ttnn.from_torch(
        calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_grid = ttnn.from_torch(
        grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # with torch.inference_mode():
    tt_module = TTOftNet(
        device,
        parameters,
        parameters.conv_args,
        TTBasicBlock,
        [2, 2, 2, 2],
        ref_model.mean,
        ref_model.std,
        input_shape_hw=input_tensor.shape[2:],
        calib=calib,
        grid=grid,
        topdown_layers=topdown_layers,
    )

    tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_module.forward(
        device, ttnn_input, ttnn_calib, ttnn_grid
    )

    tt_scores = ttnn.to_torch(tt_scores)
    tt_pos_offsets = ttnn.to_torch(tt_pos_offsets)
    tt_dim_offsets = ttnn.to_torch(tt_dim_offsets)
    tt_ang_offsets = ttnn.to_torch(tt_ang_offsets)

    scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)

    scores_pcc_passed, scores_pcc = check_with_pcc(tt_scores, scores, pcc_scores_oft)
    logger.info(f"{scores_pcc_passed=}, {scores_pcc=}")
    positions_pcc_passed, positions_pcc = check_with_pcc(tt_pos_offsets, pos_offsets, pcc_positions_oft)
    logger.info(f"{positions_pcc_passed=}, {positions_pcc=}")
    dimensions_pcc_passed, dimensions_pcc = check_with_pcc(tt_dim_offsets, dim_offsets, pcc_dimensions_oft)
    logger.info(f"{dimensions_pcc_passed=}, {dimensions_pcc=}")
    angles_pcc_passed, angles_pcc = check_with_pcc(tt_ang_offsets, ang_offsets, pcc_angles_oft)
    logger.info(f"{angles_pcc_passed=}, {angles_pcc=}")

    assert (
        scores_pcc_passed and positions_pcc_passed and dimensions_pcc_passed and angles_pcc_passed
    ), f"Failed PCC OFT {scores_pcc_passed=}, {positions_pcc_passed=}, {dimensions_pcc_passed=}, {angles_pcc_passed=}"
