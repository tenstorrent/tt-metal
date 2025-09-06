import torch
import ttnn
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from tests.ttnn.utils_for_testing import check_with_pcc
import pytest
from models.experimental.oft.reference.utils import make_grid
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "use_host_peaks, pcc_peaks, pcc_scores_ttnn, pcc_positions_ttnn, pcc_dimensions_ttnn, pcc_angles_ttnn",
    [(False, 0.86, 0.99, 0.99, 0.99, 0.99), (True, 0.86, 0.99, 0.99, 0.99, 0.99)],
    ids=["use_ttnn_peaks", "use_host_peaks"],
)
def test_decode(
    device, use_host_peaks, pcc_peaks, pcc_scores_ttnn, pcc_positions_ttnn, pcc_dimensions_ttnn, pcc_angles_ttnn
):
    torch.manual_seed(1)
    encoder = ObjectEncoder(nms_thresh=0.2)
    ttnn_encoder = TTObjectEncoder(device, nms_thresh=0.2)
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    # Prepare dummy inputs
    scores = torch.rand((1, 1, 159, 159), dtype=torch.bfloat16)
    pos_offsets = torch.rand((1, 1, 3, 159, 159), dtype=torch.bfloat16)
    dim_offsets = torch.rand((1, 1, 3, 159, 159), dtype=torch.bfloat16)
    ang_offsets = torch.rand((1, 1, 2, 159, 159), dtype=torch.bfloat16)

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
    # decoder_params = create_decoder_model_parameters(encoder.decode, scores, pos_offsets, dim_offsets, ang_offsets, grid, device)
    ret_val = encoder.decode(scores[0], pos_offsets[0], dim_offsets[0], ang_offsets[0], grid)
    ret_val_ttnn = ttnn_encoder.decode(
        device, scores_ttnn[0], pos_offsets_ttnn[0], dim_offsets_ttnn[0], ang_offsets_ttnn[0], grid_ttnn
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
