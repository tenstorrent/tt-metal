# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import pytest
import matplotlib.pyplot as plt
from loguru import logger

from models.experimental.oft.reference.bbox import visualize_objects
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.reference.utils import (
    get_abs_and_relative_error,
    load_calib,
    load_image,
    make_grid,
    visualize_score,
)
from models.experimental.oft.reference.utils import print_object_comparison
from models.experimental.oft.tests.common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
    visualize_tensor_distributions,
)
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters, create_decoder_model_parameters
from models.experimental.oft.tt.model_configs import ModelOptimizations
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from tests.ttnn.utils_for_testing import check_with_pcc
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "input_image_path, calib_path",
    [
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.jpg")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.txt")),
        ),
    ],
)
@pytest.mark.parametrize(
    "model_dtype, fallback_feedforward, fallback_lateral, fallback_oft, use_host_decoder, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    # fmt: off
    [
       ( torch.float32, False, False, False, False, 0.918, 0.977, 0.999, 0.927),
    ],
    # fmt: on
)
@torch.no_grad()
def test_demo_inference(
    device,
    input_image_path,
    calib_path,
    model_dtype,
    fallback_feedforward,
    fallback_lateral,
    fallback_oft,
    use_host_decoder,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
    model_location_generator,
):
    skip_if_not_blackhole_20_cores(device)
    device.disable_and_clear_program_cache()  # test hangs without this line on P150

    assert use_host_decoder == False, "Only use_host_decoder=False is supported for now"
    # Create output directory for saving visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    basename = os.path.basename(input_image_path).split(".")[0]

    torch.manual_seed(42)

    # ========================================================
    # OFT model configuration based on real model parameters

    # 1 Handle inputs
    input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)[None].to(model_dtype)
    calib = load_calib(calib_path, dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    # 2 Create reference OFTnet
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
    # Apply model optimizations
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict)

    # 3 Create reference encoder
    ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=model_dtype)

    # ========================================================
    # 4 Run torch oftnet inference pass
    intermediates, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)

    # ========================================================
    # 0 Load encoder parameters
    scores = scores.squeeze(0)
    pos_offsets = pos_offsets.squeeze(0)
    dim_offsets = dim_offsets.squeeze(0)
    ang_offsets = ang_offsets.squeeze(0)
    grid_ = grid.clone().squeeze(0)
    decoder_params = create_decoder_model_parameters(
        ref_encoder, [scores, pos_offsets, dim_offsets, ang_offsets, grid_], device
    )

    # ========================================================
    # 6 Run torch encoder inference pass
    ref_outs, ref_enc_intermediates = ref_encoder.decode(scores, pos_offsets, dim_offsets, ang_offsets, grid_)
    ref_objects = ref_encoder.create_objects(*ref_outs)

    # ========================================================
    # TT model configuration

    # 1 Handle inputs
    tt_input = input_tensor.permute((0, 2, 3, 1))
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grid_ = ttnn.from_torch(grid_, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # 2 Create tt OFTnet
    tt_model = TTOftNet(
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
        host_fallback_model=ref_model,
        fallback_feedforward=fallback_feedforward,
        fallback_lateral=fallback_lateral,
        fallback_oft=fallback_oft,  # True, #False, <----------------------------
    )

    # 3 Create tt encoder
    tt_encoder = TTObjectEncoder(device, decoder_params, nms_thresh=NMS_THRESH)

    # ========================================================
    # Run ttnn inference pass

    # 4 Run tt oftnet inference pass
    (tt_intermediates, layer_names), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_model.forward(
        device, tt_input, tt_calib, tt_grid
    )

    # 5 Run tt encoder inference pass
    tt_scores = ttnn.to_layout(ttnn.squeeze(tt_scores, 0), layout=ttnn.TILE_LAYOUT)
    tt_pos_offsets = ttnn.to_layout(ttnn.squeeze(tt_pos_offsets, 0), layout=ttnn.TILE_LAYOUT)
    tt_dim_offsets = ttnn.to_layout(ttnn.squeeze(tt_dim_offsets, 0), layout=ttnn.TILE_LAYOUT)
    tt_ang_offsets = ttnn.to_layout(ttnn.squeeze(tt_ang_offsets, 0), layout=ttnn.TILE_LAYOUT)

    tt_outs, tt_enc_intermediates, enc_names, enc_names_intermediates = tt_encoder.decode(
        device, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets, tt_grid_
    )
    tt_objects = tt_encoder.create_objects(*tt_outs)
    # ========================================================
    # Compare results

    test_id = f"{'fallback_ff_' if fallback_feedforward else ''}{'fallback_lat_' if fallback_lateral else ''}{'fallback_oft_' if fallback_oft else ''}host_decoder_{use_host_decoder}"

    # Check PCC on intermediates
    all_passed = True
    PCC_THRESHOLD = 0.990
    for i, (out, tt_out, layer_name) in enumerate(zip(intermediates, tt_intermediates, layer_names)):
        if "bbox" in layer_name:
            # bbox layers have different shape in TTNN vs torch, so skip them for now
            logger.debug(f"Skipping PCC check for bbox layer {layer_name} due to different shape in TTNN vs torch")
            continue
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

        # save latent and integral image distributions
        SAVE_TENSOR_DISTRIBUTION = False
        if SAVE_TENSOR_DISTRIBUTION and ("integral" in layer_name or "lat" in layer_name or "feat" in layer_name):
            # Visualize and save tensor distributions for integral layers
            tt_out_torch = ttnn.to_torch(tt_out).permute(0, 3, 1, 2).reshape(out.shape)
            fig = visualize_tensor_distributions(out, tt_out_torch, title1="Reference Integral", title2="TTNN Integral")

            # Create output filename with same naming pattern as other visualizations
            output_file = os.path.join(output_dir, f"oft_integral_{basename}_{layer_name}_{test_id}.png")
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved integral tensor distribution to {output_file}")
            plt.close(fig)

    # check PCC on the encoded outputs
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
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}, {exp_pcc=} {abs=:.3f}, {rel=:.3f}")

    # =======================================================
    # Visualization
    input_tensor = input_tensor.to(torch.float32)

    # Visualize scores/heatmaps
    visualize_score(scores[None], tt_scores[None], grid)
    plt.suptitle(f"scores {basename}", fontsize=16)
    plt.tight_layout()
    # Create an ID from the test parameters
    output_file = os.path.join(output_dir, f"oft_demo_scores_{basename}_{test_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved scores comparison visualization to {output_file}")

    # Visualize predictions
    _, (ax1, ax2) = plt.subplots(nrows=2)
    # Add a super title showing the basename of the image
    plt.suptitle(basename, fontsize=16)
    input_tensor = input_tensor.squeeze(
        0
    )  # TODO(mbezulj) align all shapes to get rid of squeezing/unsqueezing at random places
    visualize_objects(input_tensor, calib, ref_objects, ax=ax1)
    ax1.set_title("Ref detections")
    visualize_objects(input_tensor, calib, tt_objects, ax=ax2)
    ax2.set_title("TTNN detections")
    # Save the comparison plot to a file
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"oft_demo_detection_{basename}_{test_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved detection comparison visualization to {output_file}")

    # =======================================================
    # Compare encoder objects for debugging purposes
    logger.info("=== Comparing encoder objects ===")
    print_object_comparison(ref_objects, tt_objects)

    # =======================================================
    # Fail test based on PCC results
    assert all(all_passed), f"OFTnet outputs did not pass the PCC check {all_passed=}"
