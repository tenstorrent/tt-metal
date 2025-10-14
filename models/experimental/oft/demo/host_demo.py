# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
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
from models.experimental.oft.tests.common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
)
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize(
    "input_image_path, calib_path, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    [
        # fmt: off
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.jpg")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.txt")),
            0.879, 0.902, 0.998, 0.874,
        ),
        # fmt: on
    ],
)
@torch.no_grad()
def test_oftnet(
    input_image_path,
    calib_path,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
    model_location_generator,
):
    # Create output directory for saving visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    basename = os.path.basename(input_image_path).split(".")[0]

    torch.manual_seed(42)

    # ========================================================
    # OFT model configuration based on real model parameters
    input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=torch.float32)[None].to(
        torch.float32
    )
    calib = load_calib(calib_path, dtype=torch.float32)[None].to(torch.float32)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=torch.float32)[None].to(
        torch.float32
    )

    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=torch.float32,
    )

    ref_model = load_checkpoint(ref_model, model_location_generator)
    ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=torch.float32)

    # ========================================================
    # Create lower precision model
    test_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=torch.bfloat16,
    )
    test_model = load_checkpoint(test_model, model_location_generator)

    # ========================================================
    # Run torch fp32 and bfp16 inference pass

    intermediates, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)

    test_intermediates, test_scores, test_pos_offsets, test_dim_offsets, test_ang_offsets = test_model.forward(
        input_tensor.to(torch.bfloat16), calib, grid
    )

    grid = grid.squeeze(0)  # TODO(mbezulj) align all shapes to get rid of squeezing/unsqueezing at random places
    ref_pred_encoded = [t.squeeze(0) for t in (scores, pos_offsets, dim_offsets, ang_offsets)]
    ref_detections, _ = ref_encoder.decode(*ref_pred_encoded, grid)
    layer_names = (
        "image",
        "feats8",
        "feats16",
        "feats32",
        "lat8",
        "lat16",
        "lat32",
        "integral_img8",
        "integral_img16",
        "integral_img32",
        "bbox_top_left8",
        "bbox_btm_right8",
        "bbox_top_right8",
        "bbox_btm_left8",
        "bbox_top_left16",
        "bbox_btm_right16",
        "bbox_top_right16",
        "bbox_btm_left16",
        "bbox_top_left32",
        "bbox_btm_right32",
        "bbox_top_right32",
        "bbox_btm_left32",
        "ortho8",
        "ortho16",
        "ortho32",
        "ortho",
        "calib",
        "grid",
        "td",
    )
    test_pred_encoded = [
        t.squeeze(0)
        for t in (
            test_scores.to(torch.float32),
            test_pos_offsets.to(torch.float32),
            test_dim_offsets.to(torch.float32),
            test_ang_offsets.to(torch.float32),
        )
    ]
    test_detections, _ = ref_encoder.decode(*test_pred_encoded, grid)

    ref_objects = ref_encoder.create_objects(*ref_detections)
    test_objects = ref_encoder.create_objects(*test_detections)

    # ========================================================
    # Compare results

    # Check PCC on the intermediates
    all_passed = True
    INTERMEDIATES_PCC_THRESHOLD = 0.990
    for i, (ref_out, test_out, layer_name) in enumerate(zip(intermediates, test_intermediates, layer_names)):
        if "bbox" in layer_name:
            continue

        passed, pcc = check_with_pcc(ref_out, test_out, INTERMEDIATES_PCC_THRESHOLD)
        abs, rel = get_abs_and_relative_error(ref_out, test_out)

        all_passed = all_passed and passed
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Intermediate {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

    # Check PCC on the final outputs
    all_passed = []
    ref_outs = [scores, pos_offsets, dim_offsets, ang_offsets]
    tt_outs = [test_scores, test_pos_offsets, test_dim_offsets, test_ang_offsets]
    names = ["scores", "pos_offsets", "dim_offsets", "ang_offsets"]
    expected_pcc = [pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft]
    for i, (ref_out, tt_out, layer_name, exp_pcc) in enumerate(zip(ref_outs, tt_outs, names, expected_pcc)):
        test_out = tt_out.reshape(ref_out.shape)  # assume it's already a torch tensor in the right format
        passed, pcc = check_with_pcc(ref_out, test_out, exp_pcc)
        abs, rel = get_abs_and_relative_error(ref_out, test_out)

        all_passed.append(passed)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")
        if passed and float(pcc) - exp_pcc > 0.001:
            logger.warning(
                f"⚠️  Output {i} {layer_name} PCC is better than expected by {float(pcc)-exp_pcc:.3f}. Please update expected PCC value to {math.floor(float(pcc) * 1000) / 1000:.3f}."
            )

    # =======================================================
    # Visualization
    input_tensor = input_tensor.to(torch.float32)

    # Visualize scores/heatmaps
    visualize_score(scores, test_scores, grid.unsqueeze(0))
    plt.suptitle(basename, fontsize=16)
    plt.tight_layout()
    # Create an ID from the test parameters
    test_id = f"host_fp32_vs_bfp16"

    output_file = os.path.join(output_dir, f"oft_scores_{basename}_{test_id}.png")
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
    ax1.set_title("fp32 detections")
    visualize_objects(input_tensor, calib, test_objects, ax=ax2)
    ax2.set_title("bfp16 detections")

    # Save the comparison plot to a file
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"oft_detection_comparison_{basename}_{test_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved detection comparison visualization to {output_file}")

    # =======================================================
    # Fail test based on PCC results
    assert all(all_passed), f"OFTnet outputs did not pass the PCC check {all_passed=}"
