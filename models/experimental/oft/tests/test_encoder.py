import torch
import ttnn
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from tests.ttnn.utils_for_testing import check_with_pcc
import pytest
from models.experimental.oft.reference.utils import make_grid, get_abs_and_relative_error, visualize_score
from models.experimental.oft.reference.utils_objects import print_object_comparison
from models.experimental.oft.tt.model_preprocessing import create_decoder_model_parameters
from models.experimental.oft.tests.common import GRID_RES, GRID_SIZE, Y_OFFSET

from loguru import logger
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    # fmt: off
    "use_host_peaks, pcc_peaks, pcc_scores_ttnn, pcc_positions_ttnn, pcc_dimensions_ttnn, pcc_angles_ttnn",
    [(False, 0.86, 0.99, 0.99, 0.99, 0.99)],
    # fmt: on
    ids=["use_ttnn_peaks"],
)
@pytest.mark.parametrize("model_dtype", [torch.bfloat16], ids=["bfp16"])
@pytest.mark.parametrize(
    "input_file_path",
    [
        # fmt: off
        (r"/localdev/mbezulj/tt-metal/models/experimental/oft/demo/outputs/encoded_outputs_000013_host_decoder_True.pt"),
        # fmt: on
    ],
)
def test_decode(
    device,
    model_dtype,
    input_file_path,
    use_host_peaks,
    pcc_peaks,
    pcc_scores_ttnn,
    pcc_positions_ttnn,
    pcc_dimensions_ttnn,
    pcc_angles_ttnn,
):
    torch.manual_seed(1)
    encoder = ObjectEncoder(nms_thresh=0.2, dtype=model_dtype)

    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None]

    # Load the precomputed outputs for testing
    if input_file_path is not None and os.path.isfile(os.path.join(os.path.dirname(__file__), input_file_path)):
        output_file = os.path.join(os.path.dirname(__file__), input_file_path)
        output_dict = torch.load(output_file)

        # Extract the outputs from the dictionary
        scores = output_dict["ref_scores"].squeeze(0)
        pos_offsets = output_dict["ref_pos_offsets"].squeeze(0)
        dim_offsets = output_dict["ref_dim_offsets"].squeeze(0)
        ang_offsets = output_dict["ref_ang_offsets"].squeeze(0)
    else:
        # Prepare dummy inputs
        scores = torch.rand((1, 159, 159), dtype=model_dtype)
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
    ttnn_encoder = TTObjectEncoder(device, decoder_params, nms_thresh=0.2)
    # Prepare TTNN inputs
    scores_ttnn = ttnn.from_torch(
        scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
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

    tt_outs, tt_intermediates, names, names_intermediates = ttnn_encoder.decode(
        device, scores_ttnn, pos_offsets_ttnn, dim_offsets_ttnn, ang_offsets_ttnn, grid_ttnn
    )
    tt_objects = ttnn_encoder.create_objects(*tt_outs)

    # visualize smooth and mp!
    import matplotlib.pyplot as plt

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
        output_file = f"decoder_debug_{name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved scores comparison visualization to {output_file}")

        # Plot the absolute difference between reference and tt tensors
        plt.figure(figsize=(10, 6))
        diff = torch.abs(ref - tt)[0]
        plt.imshow(diff.detach().numpy().squeeze(), cmap="hot")
        plt.colorbar(label="Absolute Difference")
        plt.title(f"Absolute Difference for {name}")
        output_diff_file = f"decoder_diff_{name}.png"
        plt.savefig(output_diff_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved absolute difference visualization to {output_diff_file}")

    # Save the max_pooled_with_indices tensor to file
    torch.save(ref_outs[-2], "mpwi.pt")
    logger.info(f"Saved max pooled with indices tensor to mpwi.pt")

    # Use the shared function to print and compare objects
    print_object_comparison(ref_objects, tt_objects)

    # visualize max_inds
    # for i, (ref, tt, name) in enumerate(zip(ref_outs, tt_outs, names)):
    #     if "max_inds" != name:
    #         continue

    #     logger.warning(f"Visualizing output {i} {name}")
    #     tt = tt.reshape(ref.shape)

    #     # Visualize max_inds (indices after max pooling)
    #     logger.warning(f"Visualizing max pooling indices comparison")

    #     # Extract data
    #     ref_indices = ref.squeeze().detach().cpu().numpy()
    #     tt_indices = tt.squeeze().detach().cpu().numpy()

    #     # Convert flattened indices to 2D coordinates
    #     height, width = 159, 159  # Based on the grid dimensions in your code
    #     ref_y = ref_indices // width
    #     ref_x = ref_indices % width

    #     tt_y = tt_indices // width
    #     tt_x = tt_indices % width

    #     # Get grid coordinates for plotting
    #     # Create coordinates from 1 to 159 for both dimensions
    #     x_coords = torch.arange(1, width + 1).reshape(1, width).repeat(height, 1)
    #     y_coords = torch.arange(1, height + 1).reshape(height, 1).repeat(1, width)

    #     # Map the indices to actual x-y coordinates in the grid, handling potential out-of-bounds indices
    #     max_idx = height * width - 1
    #     safe_ref_indices = torch.clamp(ref_indices, 0, max_idx)
    #     safe_tt_indices = torch.clamp(tt_indices, 0, max_idx)

    #     ref_points_x = x_coords.flatten()[safe_ref_indices]
    #     ref_points_y = y_coords.flatten()[safe_ref_indices]
    #     # Filter out invalid indices
    #     valid_ref_mask = ref_indices < (height * width)
    #     valid_tt_mask = tt_indices < (height * width)

    #     # Create a single plot with both sets of points
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(x_coords.flatten(), y_coords.flatten(), c='gray', s=1, alpha=0.2, label='Grid Points')
    #     plt.scatter(ref_points_x[valid_ref_mask], ref_points_y[valid_ref_mask], c='blue', marker='x', s=50, label='Reference')
    #     plt.scatter(tt_points_x[valid_tt_mask], tt_points_y[valid_tt_mask], c='red', marker='+', s=50, label='TTNN')
    #     plt.title('Max Pooling Indices Comparison')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.legend()

    #     # Save the figure
    #     output_indices_file = f"decoder_maxind_comparison.png"
    #     plt.savefig(output_indices_file, dpi=300, bbox_inches="tight")
    #     plt.close()
    #     logger.info(f"Saved max indices comparison visualization to {output_indices_file}")

    # # Preprocess print
    # for i, (out, tt_out, name) in enumerate(zip(ref_outs, tt_outs, names)):
    #     visualize_score(out, tt_out, grid.unsqueeze(0))

    return
    # pcc test each output
    exp_pcc = 0.990
    for i, (out, tt_out, name) in enumerate(zip(ref_outs, tt_outs, names)):
        if isinstance(tt_out, ttnn.Tensor):
            tt_out_torch = ttnn.to_torch(tt_out, dtype=torch.float32).permute(0, 3, 1, 2).reshape(out.shape)
        else:
            tt_out_torch = tt_out.to(dtype=torch.float32)

        if "peak" in name:
            indices_match = torch.equal(out, tt_out_torch)
            logger.warning(f"Indices match {indices_match=}")
            if indices_match == False:
                from models.experimental.oft.tt.utils import print_boolean_comparison

                print_boolean_comparison(out, tt_out_torch)
        else:
            if name in ["scores", "positions", "dimensions", "angles"]:
                tt_out_torch = tt_out_torch[ref_outs[0]]  # select only the peaks from ttnn output
            out = out.to(dtype=torch.float32)
            passed, pcc = check_with_pcc(out, tt_out_torch, exp_pcc)
            abs, rel = get_abs_and_relative_error(out, tt_out_torch)
            special_char = "✅" if passed else "❌"
            logger.warning(f"{special_char} Output {i} {name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

    peaks, max_inds, scores, classids, positions, dimensions, angles = ref_outs
    peaks_ttnn, max_inds_ttnn, scores_ttnn, classids_ttnn, positions_ttnn, dimensions_ttnn, angles_ttnn = tt_outs

    value = torch.allclose(max_inds, max_inds_ttnn, atol=0.01, rtol=0.01)
    logger.info(f"max indices allclose check: {value}")
    peaks_pcc_passed, peaks_pcc = check_with_pcc(peaks, peaks_ttnn, pcc_peaks)
    logger.info(f"{peaks_pcc_passed=}, {peaks_pcc=}")
    # plot_boolean_comparison(peaks, peaks_ttnn)

    if use_host_peaks:
        peaks_selected = peaks
    else:
        peaks_selected = peaks_ttnn

    scores_pcc_passed, scores_pcc = check_with_pcc(scores, scores_ttnn[peaks_selected], pcc_scores_ttnn)
    logger.info(f"{scores_pcc_passed=}, {scores_pcc=}")
    # classids are tensors with different lenght
    logger.info(f"{classids=}, {classids_ttnn=} | {classids.shape=}, {classids_ttnn.shape=}")

    # msg, pcc = check_with_pcc(classids.float(), classids_ttnn.float(), 0.99)
    # logger.info(f"PCC check classids {msg=}, {pcc=}")
    positions_pcc_passed, positions_pcc = check_with_pcc(positions, positions_ttnn[peaks_selected], pcc_positions_ttnn)
    logger.info(f"{positions_pcc_passed=}, {positions_pcc=}")
    dimensions_pcc_passed, dimensions_pcc = check_with_pcc(
        dimensions, dimensions_ttnn[peaks_selected], pcc_dimensions_ttnn
    )
    logger.info(f"{dimensions_pcc_passed=}, {dimensions_pcc=}")
    angles_pcc_passed, angles_pcc = check_with_pcc(angles, angles_ttnn[peaks_selected], pcc_angles_ttnn)
    logger.info(f"{angles_pcc_passed=}, {angles_pcc=}")

    assert (
        peaks_pcc_passed and scores_pcc_passed and positions_pcc_passed and dimensions_pcc_passed and angles_pcc_passed
    ), f"Failed test decode {peaks_pcc_passed=}, {scores_pcc_passed=}, {positions_pcc_passed=}, {dimensions_pcc_passed=}, {angles_pcc_passed=}"
