import torch
import ttnn
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from tests.ttnn.utils_for_testing import check_with_pcc
import pytest
from models.experimental.oft.reference.utils import make_grid
from models.experimental.oft.tt.model_preprocessing import create_decoder_model_parameters
from models.experimental.oft.tests.test_common import GRID_RES, GRID_SIZE, Y_OFFSET

from loguru import logger
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    # fmt: off
    "use_host_peaks, pcc_peaks, pcc_scores_ttnn, pcc_positions_ttnn, pcc_dimensions_ttnn, pcc_angles_ttnn",
    [(False, 0.86, 0.99, 0.99, 0.99, 0.99),
     ( True, 0.86, 0.99, 0.99, 0.99, 0.99)],
    # fmt: on
    ids=["use_ttnn_peaks", "use_host_peaks"],
)
@pytest.mark.parametrize("model_dtype", [torch.bfloat16], ids=["bfp16"])
@pytest.mark.parametrize(
    "input_file_path",
    [
        # fmt: off
    ("output_comparison/outputs_torch.float32_device_oft_False_host_oft_False.pt"),
        # ("output_comparison/outputs_torch.float32_device_oft_True_host_oft_False.pt")
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

    grid = grid.squeeze(0)
    decoder_params = create_decoder_model_parameters(
        encoder, [scores, pos_offsets, dim_offsets, ang_offsets, grid], device
    )
    print(f"{decoder_params=}")
    ttnn_encoder = TTObjectEncoder(device, decoder_params, nms_thresh=0.2)

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

    ret_val = encoder.decode(scores, pos_offsets, dim_offsets, ang_offsets, grid)
    ret_val_ttnn = ttnn_encoder.decode(
        device, scores_ttnn, pos_offsets_ttnn, dim_offsets_ttnn, ang_offsets_ttnn, grid_ttnn
    )

    peaks, max_inds, scores, classids, positions, dimensions, angles = ret_val
    peaks_ttnn, max_inds_ttnn, scores_ttnn, classids_ttnn, positions_ttnn, dimensions_ttnn, angles_ttnn = ret_val_ttnn

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
