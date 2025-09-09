import torch
import ttnn
import pytest
import os
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from models.experimental.oft.reference.utils import make_grid, load_calib, load_image
from models.experimental.oft.reference.utils import get_abs_and_relative_error

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
    "model_dtype, use_host_oft, scale_features, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    # fmt: off
    [
       (torch.bfloat16, False, False, 0.210, 0.533, 0.986, 0.507),  # Using device OFT without scaling
       (torch.bfloat16, False,  True, 0.954, 0.991, 0.999, 0.850),  # Using device OFT with scaling
       (torch.bfloat16,  True, False, 0.793, 0.891, 0.998, 0.858),
       (torch.bfloat16,  True,  True, 0.886, 0.987, 0.999, 0.831),
       ( torch.float32, False, False, 0.211, 0.593, 0.989, 0.632),  # Using device OFT without scaling
       ( torch.float32, False,  True, 0.964, 0.994, 0.998, 0.806),  # Using device OFT with scaling
       ( torch.float32,  True, False, 0.923, 0.889, 0.997, 0.931),
       ( torch.float32,  True,  True, 0.921, 0.993, 0.998, 0.821)
    ],
    # fmt: on
    ids=[
        "bfp16_use_device_oft_no_scaling",
        "bfp16_use_device_oft_with_scaling",
        "bfp16_use_host_oft_no_scaling",
        "bfp16_use_host_oft_with_scaling",
        "fp32_use_device_oft_no_scaling",
        "fp32_use_device_oft_with_scaling",
        "fp32_use_host_oft_no_scaling",
        "fp32_use_host_oft_with_scaling",
    ],
)
@pytest.mark.parametrize("checkpoints_path", [r"/home/mbezulj/checkpoint-0600.pth"])
def test_oftnet(
    device,
    checkpoints_path,
    input_image_path,
    calib_path,
    model_dtype,
    scale_features,
    use_host_oft,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
):
    torch.manual_seed(42)

    input_tensor = load_image(input_image_path, pad_hw=(384, 1280), dtype=model_dtype)[None]
    calib = load_calib(calib_path, dtype=model_dtype)[None]
    # OFT configuration based on real model parameters
    grid_res = 0.5
    grid_size = (80.0, 80.0)
    grid_height = 4.0
    y_offset = 1.74
    grid = make_grid(grid_size, (-grid_size[0] / 2.0, y_offset, 0.0), grid_res, dtype=model_dtype)[None]

    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=grid_res,
        grid_height=grid_height,
        dtype=model_dtype,
        scale_features=scale_features,
    )

    if checkpoints_path is not None and os.path.isfile(checkpoints_path):
        logger.info(f"Loading model weights from {checkpoints_path}")
        checkpoint = torch.load(checkpoints_path, map_location="cpu")

        # Load state dict as is
        ref_model.load_state_dict(checkpoint["model"], strict=True)

        # Ensure all weights are converted to the specified dtype after loading
        ref_model.to(ref_model.dtype)
        logger.info(f"Converted all model weights to {ref_model.dtype}")
    else:
        assert False, f"Checkpoint path {checkpoints_path} is not a file"
        logger.warning(f"Checkpoint path {checkpoints_path} does not exist, using random weights")

    # Ensure all input tensors are of the right dtype before passing them to create_OFT_model_parameters
    model_dtype = ref_model.dtype
    input_tensor = input_tensor.to(model_dtype)
    calib = calib.to(model_dtype)
    grid = grid.to(model_dtype)
    logger.info(f"Converted all input tensors to {model_dtype}")

    parameters = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)

    tt_input = input_tensor.permute((0, 2, 3, 1))
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(
        calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_grid = ttnn.from_torch(
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
        grid_res=grid_res,
        grid_height=grid_height,
        host_fallback_model=ref_model if use_host_oft else None,
        OFT_fallback=use_host_oft,
        FeedForward_fallback=False,
        Lateral_fallback=False,
        scale_features=scale_features,
    )

    outputs, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)
    (tt_outputs, layer_names), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_module.forward(
        device, tt_input, tt_calib, tt_grid
    )

    all_passed = True
    PCC_THRESHOLD = 0.990
    for i, (out, tt_out, layer_name) in enumerate(zip(outputs, tt_outputs, layer_names)):
        # conver tt output to torch, channel first, and correct shape
        if isinstance(tt_out, ttnn.Tensor):
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
    assert all(all_passed), f"OFTnet outputs did not pass the PCC check {all_passed=}"
