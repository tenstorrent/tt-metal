import torch
import ttnn
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.tt.tt_encoder import TTObjectEncoder
from tests.ttnn.utils_for_testing import check_with_pcc
import pytest
from models.experimental.oft.reference.utils import make_grid


@pytest.mark.parametrize("device_params", [{"l1_small_size": 100 * 1024}], indirect=True)
def test_decode(device):
    torch.manual_seed(1)
    encoder = ObjectEncoder(classnames=["Car"])
    ttnn_encoder = TTObjectEncoder(device, classnames=["Car"])
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
    print(f"max indices allclose check: {value}")
    msg, pcc = check_with_pcc(peaks, peaks_ttnn, 0.86)
    print(f"peaks check PCC {msg=}, {pcc=}")
    # for a testing putposes of angles, positions and dimensions we will take
    # only values at peaks (from torch)
    # we are comparing values at peaks from torch indexes
    msg, pcc = check_with_pcc(scores, scores_ttnn[peaks], 0.99)
    print(f"PCC check scores: {msg=}, {pcc=}")
    # classids are tensors with different lenght
    # msg, pcc = check_with_pcc(classids.float(), classids_ttnn.float(), 0.99)
    # print(f"PCC check classids {msg=}, {pcc=}")
    msg, pcc = check_with_pcc(positions, positions_ttnn[peaks], 0.99)
    print(f"PCC check positions {msg=}, {pcc=}")
    msg, pcc = check_with_pcc(dimensions, dimensions_ttnn[peaks], 0.99)
    print(f"PCC check dimensions {msg=}, {pcc=}")
    msg, pcc = check_with_pcc(angles, angles_ttnn[peaks], 0.99)
    print(f"PCC check angles {msg=}, {pcc=}")
