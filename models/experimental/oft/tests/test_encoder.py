# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from tests.ttnn.utils_for_testing import check_with_pcc
import pytest
from models.experimental.oft.reference.utils import (
    make_grid,
    get_abs_and_relative_error,
    visualize_score,
    print_object_comparison,
)
from models.experimental.oft.tt.model_preprocessing import create_decoder_model_parameters
from models.experimental.oft.tests.common import GRID_RES, GRID_SIZE, Y_OFFSET, NMS_THRESH
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
import matplotlib.pyplot as plt
from loguru import logger
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
@pytest.mark.parametrize(
    # fmt: off
    "pcc_peaks, pcc_scores, pcc_positions, pcc_dimensions, pcc_angles",
    [(0.999, 0.999, 0.999, 0.999, 0.999)],
    # fmt: on
)
@pytest.mark.parametrize("model_dtype", [torch.bfloat16], ids=["bfp16"])
def test_decode(
    device,
    model_dtype,
    pcc_peaks,
    pcc_scores,
    pcc_positions,
    pcc_dimensions,
    pcc_angles,
):
    skip_if_not_blackhole_20_cores(device)
    # Create output directory for saving visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    torch.manual_seed(1)
    encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=model_dtype)

    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None]

    # Prepare dummy inputs
    # Create a tensor with 3 Gaussian peaks
    scores = torch.zeros((1, 159, 159), dtype=model_dtype)
    peak_locations = [(10, 10), (120, 20), (80, 90)]
    spreads = (3, 2, 5)

    for s, (y, x) in enumerate(peak_locations):
        spread = spreads[s]
        for i in range(-spread, spread + 1):
            for j in range(-spread, spread + 1):
                if 0 <= y + i < 159 and 0 <= x + j < 159:
                    # Create Gaussian falloff based on distance from peak
                    dist = (i**2 + j**2) / (2 * (spread / 2) ** 2)
                    value = torch.exp(torch.tensor(-dist, dtype=model_dtype))
                    scores[0, y + i, x + j] = value
    scores = scores + torch.rand((1, 159, 159), dtype=model_dtype) * 0.05  # add slight noise
    pos_offsets = torch.rand((1, 3, 159, 159), dtype=model_dtype)
    dim_offsets = torch.rand((1, 3, 159, 159), dtype=model_dtype)
    ang_offsets = torch.rand((1, 2, 159, 159), dtype=model_dtype)

    # Convert to bfloat16 if needed based on model_dtype
    if model_dtype != scores.dtype:
        scores = scores.to(model_dtype)
    if model_dtype != pos_offsets.dtype:
        pos_offsets = pos_offsets.to(model_dtype)
    if model_dtype != dim_offsets.dtype:
        dim_offsets = dim_offsets.to(model_dtype)
    if model_dtype != ang_offsets.dtype:
        ang_offsets = ang_offsets.to(model_dtype)

    # Setup encoder and create TTNN encoder
    grid = grid.squeeze(0)
    decoder_params = create_decoder_model_parameters(
        encoder, [scores, pos_offsets, dim_offsets, ang_offsets, grid], device
    )
    tt_encoder = TTObjectEncoder(device, decoder_params, nms_thresh=NMS_THRESH)
    # Prepare TTNN inputs
    scores_ttnn = ttnn.from_torch(
        scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    pos_offsets_ttnn = ttnn.from_torch(
        pos_offsets, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    dim_offsets_ttnn = ttnn.from_torch(
        dim_offsets, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    ang_offsets_ttnn = ttnn.from_torch(
        ang_offsets, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    grid_ttnn = ttnn.from_torch(
        grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )

    # run reference and ttnn encoder
    ref_outs, ref_intermediates = encoder.decode(scores, pos_offsets, dim_offsets, ang_offsets, grid)
    ref_objects = encoder.create_objects(*ref_outs)

    tt_outs, tt_intermediates, names, names_intermediates = tt_encoder.decode(
        device, scores_ttnn, pos_offsets_ttnn, dim_offsets_ttnn, ang_offsets_ttnn, grid_ttnn
    )
    tt_objects = tt_encoder.create_objects(*tt_outs)

    # visualize smooth and mp
    for i, (ref, tt, name) in enumerate(zip(ref_intermediates, tt_intermediates, names_intermediates)):
        if name in ["peaks", "max_inds"]:
            continue
        logger.warning(f"Visualizing output {i} {name}")

        tt = tt.reshape(ref.shape)

        passed, pcc = check_with_pcc(ref, tt, 0.999)
        abs, rel = get_abs_and_relative_error(ref, tt)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

        visualize_score(ref, tt, grid.unsqueeze(0))
        plt.suptitle(name, fontsize=16)
        plt.tight_layout()
        # Create an ID from the test parameters
        output_file = os.path.join(output_dir, f"decoder_debug_{name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved scores comparison visualization to {output_file}")

        # Plot the absolute difference between reference and tt tensors
        plt.figure(figsize=(10, 6))
        diff = torch.abs(ref - tt)[0]
        plt.imshow(diff.detach().numpy().squeeze(), cmap="hot")
        plt.colorbar(label="Absolute Difference")
        plt.title(f"Absolute Difference for {name}")
        output_diff_file = os.path.join(output_dir, f"decoder_diff_{name}.png")
        plt.savefig(output_diff_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved absolute difference visualization to {output_diff_file}")

    # Use the shared function to print and compare objects
    print_object_comparison(ref_objects, tt_objects)

    all_passed = []
    for ref_out, tt_out, pcc, name in zip(
        ref_outs, tt_outs, (pcc_peaks, pcc_scores, pcc_positions, pcc_dimensions, pcc_angles), names
    ):
        tt_out = tt_out.to(torch.float32)
        ref_out = ref_out.to(torch.float32)
        tt_out = tt_out.reshape(ref_out.shape)
        passed, pcc = check_with_pcc(ref_out, tt_out, pcc)
        abs, rel = get_abs_and_relative_error(ref_out, tt_out)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")
        all_passed.append(passed)
    assert all(all_passed), f"Decoder outputs did not pass the PCC check {all_passed=}"
