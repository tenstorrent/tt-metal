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
       ( True, 0.997, 0.997, 0.997, 0.997)
    ],
    # fmt: on
    ids=["use_device_oft", "use_host_oft"],
)
@pytest.mark.parametrize("checkpoints_path", [r"/home/mbezulj/checkpoint-0600.pth"])
def test_oftnet(
    device,
    checkpoints_path,
    input_image_path,
    calib_path,
    use_host_oft,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
):
    torch.manual_seed(42)

    # Use bfloat16 dtype for consistency with model
    model_dtype = torch.bfloat16
    # model_dtype = torch.float32

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
        grid_res=grid_res,
        grid_height=grid_height,
        host_fallback_model=ref_model if use_host_oft else None,
        OFT_fallback=use_host_oft,
        FeedForward_fallback=True,
        Lateral_fallback=False,
    )

    outputs = ref_model(input_tensor, calib, grid)
    tt_outputs, layer_names = tt_module.forward(device, ttnn_input, ttnn_calib, ttnn_grid)

    all_passed = True
    PCC_THRESHOLD = 0.990
    for i, (out, tt_out, layer_name) in enumerate(zip(outputs, tt_outputs, layer_names)):
        # conver tt output to torch, channel first, and correct shape
        if isinstance(tt_out, ttnn.Tensor):
            tt_out_torch = ttnn.to_torch(tt_out).permute(0, 3, 1, 2).reshape(out.shape)
        else:
            logger.debug(f"Output {i} is not a ttnn.Tensor, skipping conversion")
            tt_out_torch = tt_out.reshape(out.shape)  # assume it's already a torch tensor in the right format
        passed, pcc = check_with_pcc(tt_out_torch, out, PCC_THRESHOLD)
        all_passed = all_passed and passed
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}")

        # if (i == 9):
        #     # this is one of the bbox_corener outputs
        #     # Print statistics for bbox_corner output
        #     out = out.to(dtype=torch.float32)
        #     tt_out_torch = tt_out_torch.to(dtype=torch.float32)
        #     logger.info(f"Output {i} statistics:")
        #     logger.info(f"Reference: min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}, std={out.std().item():.6f}")
        #     logger.info(f"TT model: min={tt_out_torch.min().item():.6f}, max={tt_out_torch.max().item():.6f}, mean={tt_out_torch.mean().item():.6f}, std={tt_out_torch.std().item():.6f}")

        #     # Create visualization of the bbox_corner outputs
        #     import matplotlib.pyplot as plt

        #     # Shape is [1,7,25281,2]
        #     fig, axs = plt.subplots(2, 7, figsize=(20, 6))
        #     fig.suptitle(f"Output {i}: {layer_name} Comparison")

        #     for col in range(out.shape[1]):  # 7 columns
        #         for row in range(out.shape[3]):  # 2 rows
        #             ref_data = out[0, col, :, row].detach().cpu().numpy()
        #             tt_data = tt_out_torch[0, col, :, row].detach().cpu().numpy()

        #             ax = axs[row, col]
        #             ax.scatter(np.arange(len(ref_data)), ref_data, s=1, alpha=0.5, label="Reference", color='blue')
        #             ax.scatter(np.arange(len(tt_data)), tt_data, s=1, alpha=0.5, label="TT model", color='red')
        #             ax.set_title(f"col={col}, row={row}")

        #             if col == 0:
        #                 ax.set_ylabel("Value")
        #             if row == 1:
        #                 ax.set_xlabel("Index")

        #     # Add a common legend
        #     handles, labels = axs[0, 0].get_legend_handles_labels()
        #     fig.legend(handles, labels, loc='upper right')

        #     plt.tight_layout()
        #     output_dir = os.path.dirname(os.path.abspath(__file__))
        #     output_path = os.path.join(output_dir, f"output_{i}_{layer_name}.png")
        #     plt.savefig(output_path)
        #     plt.close(fig)
        #     logger.info(f"Saved visualization to {output_path}")
    assert all_passed, "Failed PCC OFTNet"

    # scores, pos_offsets, dim_offsets, ang_offsets = outputs
    # tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_outputs
    # tt_scores = ttnn.to_torch(tt_scores)
    # tt_pos_offsets = ttnn.to_torch(tt_pos_offsets)
    # tt_dim_offsets = ttnn.to_torch(tt_dim_offsets)
    # tt_ang_offsets = ttnn.to_torch(tt_ang_offsets)

    # scores_pcc_passed, scores_pcc = check_with_pcc(tt_scores, scores, pcc_scores_oft)
    # logger.info(f"{scores_pcc_passed=}, {scores_pcc=}")
    # positions_pcc_passed, positions_pcc = check_with_pcc(tt_pos_offsets, pos_offsets, pcc_positions_oft)
    # logger.info(f"{positions_pcc_passed=}, {positions_pcc=}")
    # dimensions_pcc_passed, dimensions_pcc = check_with_pcc(tt_dim_offsets, dim_offsets, pcc_dimensions_oft)
    # logger.info(f"{dimensions_pcc_passed=}, {dimensions_pcc=}")
    # angles_pcc_passed, angles_pcc = check_with_pcc(tt_ang_offsets, ang_offsets, pcc_angles_oft)
    # logger.info(f"{angles_pcc_passed=}, {angles_pcc=}")

    # assert (
    #     scores_pcc_passed and positions_pcc_passed and dimensions_pcc_passed and angles_pcc_passed
    # ), f"Failed PCC OFT {scores_pcc_passed=}, {positions_pcc_passed=}, {dimensions_pcc_passed=}, {angles_pcc_passed=}"
