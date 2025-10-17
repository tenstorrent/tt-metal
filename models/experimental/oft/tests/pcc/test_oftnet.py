# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import os
import math
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.model_configs import ModelOptimizations
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from models.experimental.oft.tests.common import GRID_RES, GRID_SIZE, GRID_HEIGHT, Y_OFFSET, H_PADDED, W_PADDED
from models.experimental.oft.tests.common import load_checkpoint

from models.experimental.oft.reference.utils import make_grid, load_calib, load_image
from models.experimental.oft.reference.utils import get_abs_and_relative_error

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 14 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "input_image_path, calib_path",
    [
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources/000013.jpg")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources/000013.txt")),
        )
    ],
)
@pytest.mark.parametrize(
    "model_dtype, use_host_oft, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    # fmt: off
    [
       ( torch.bfloat16, False, 0.822, 0.892, 0.998, 0.904),
       ( torch.bfloat16,  True, 0.863, 0.966, 0.999, 0.896),
       ( torch.float32, False, 0.906, 0.978, 0.999, 0.929),
       ( torch.float32,  True, 0.916, 0.881, 0.998, 0.918)
    ],
    # fmt: on
    ids=[
        "bfp16_use_device_oft",
        "bfp16_use_host_oft",
        "fp32_use_device_oft",
        "fp32_use_host_oft",
    ],
)
def test_oftnet(
    device,
    input_image_path,
    calib_path,
    model_dtype,
    use_host_oft,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
    model_location_generator,
):
    skip_if_not_blackhole_20_cores(device)
    device.disable_and_clear_program_cache()  # test hangs without this line on P150
    torch.manual_seed(42)

    # OFT configuration based on real model parameters
    input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)[None].to(model_dtype)
    calib = load_calib(calib_path, dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=model_dtype,
    )

    ref_model = load_checkpoint(ref_model, model_location_generator)
    state_dict = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict)

    tt_input = input_tensor.permute((0, 2, 3, 1))
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # with torch.inference_mode():
    tt_module = TTOftNet(
        device,
        state_dict,
        state_dict.layer_args,
        TTBasicBlock,
        [2, 2, 2, 2],
        ref_model.mean,
        ref_model.std,
        input_shape_hw=input_tensor.shape[2:],
        calib=calib,
        grid=grid,
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        host_fallback_model=ref_model if use_host_oft else None,
        fallback_oft=use_host_oft,
        fallback_feedforward=False,
        fallback_lateral=False,
    )

    intermediates, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)
    (
        (tt_intermediates, intermediates_names),
        tt_scores,
        tt_pos_offsets,
        tt_dim_offsets,
        tt_ang_offsets,
    ) = tt_module.forward(device, tt_input, tt_calib, tt_grid)

    all_passed = True
    PCC_THRESHOLD = 0.990
    for i, (out, tt_out, layer_name) in enumerate(zip(intermediates, tt_intermediates, intermediates_names)):
        # conver tt output to torch, channel first, and correct shape
        if "bbox" in layer_name:
            # bbox layers have different shape in TTNN vs torch, so skip them for now
            logger.debug(f"Skipping PCC check for bbox layer {layer_name} due to different shape in TTNN vs torch")
            continue
        elif isinstance(tt_out, ttnn.Tensor):
            tt_out_torch = ttnn.to_torch(tt_out).permute(0, 3, 1, 2).reshape(out.shape)
        else:
            # logger.debug(f"Output {i} is not a ttnn.Tensor, skipping conversion")
            tt_out_torch = tt_out.reshape(out.shape)  # assume it's already a torch tensor in the right format
        passed, pcc = check_with_pcc(out, tt_out_torch, PCC_THRESHOLD)
        abs, rel = get_abs_and_relative_error(out, tt_out_torch)

        all_passed = all_passed and passed
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Intermediate {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

    tt_scores = ttnn.to_torch(tt_scores)
    tt_pos_offsets = ttnn.to_torch(tt_pos_offsets)
    tt_dim_offsets = ttnn.to_torch(tt_dim_offsets)
    tt_ang_offsets = ttnn.to_torch(tt_ang_offsets)

    all_passed = []
    ref_outs = [scores, pos_offsets, dim_offsets, ang_offsets]
    tt_outs = [tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets]
    names = ["scores", "pos_offsets", "dim_offsets", "ang_offsets"]
    expected_pcc = [pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft]
    for i, (out, tt_out, layer_name, exp_pcc) in enumerate(zip(ref_outs, tt_outs, names, expected_pcc)):
        tt_out_torch = tt_out.reshape(out.shape)  # assume it's already a torch tensor in the right format
        passed, pcc = check_with_pcc(out, tt_out_torch, exp_pcc)
        abs, rel = get_abs_and_relative_error(out, tt_out_torch)

        all_passed.append(passed)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")
        if passed and float(pcc) - exp_pcc > 0.001:
            logger.warning(
                f"⚠️  Output {i} {layer_name} PCC is better than expected by {float(pcc)-exp_pcc:.3f}. Please update expected PCC value to {math.floor(float(pcc) * 1000) / 1000:.3f}."
            )

    assert all(all_passed), f"OFTnet outputs did not pass the PCC check {all_passed=}"

    # Save outputs to files, useful when debugging encoder
    SAVE_OUTPUTS = False

    if SAVE_OUTPUTS:
        # Create directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "output_comparison")
        os.makedirs(output_dir, exist_ok=True)

        # Construct a unique filename based on test parameters
        test_config = f"{model_dtype}_host_oft_{use_host_oft}"
        output_file = os.path.join(output_dir, f"outputs_{test_config}.pt")

        # Package all outputs in a dictionary
        output_dict = {
            "ref_scores": ref_outs[0],
            "ref_pos_offsets": ref_outs[1],
            "ref_dim_offsets": ref_outs[2],
            "ref_ang_offsets": ref_outs[3],
            "tt_scores": tt_outs[0],
            "tt_pos_offsets": tt_outs[1],
            "tt_dim_offsets": tt_outs[2],
            "tt_ang_offsets": tt_outs[3],
        }

        # Save outputs to file using torch.save
        torch.save(output_dict, output_file)

        logger.info(f"Saved outputs to {output_file}")
