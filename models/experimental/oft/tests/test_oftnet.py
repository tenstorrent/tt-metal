import torch
import ttnn
import pytest
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from models.experimental.oft.reference.utils import make_grid

# from models.experimental.oft.tt.tt_oftnet import OftNet
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 12 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "cell_size",
    [0.5],
)
@pytest.mark.parametrize(
    "grid_height",
    [4],
)
def test_oftnet(device, grid_height, cell_size):
    torch.manual_seed(42)

    input_tensor = torch.rand((1, 3, 384, 1280))
    calib = torch.tensor(
        [
            [
                [7.2154e02, 0.0000e00, 6.0956e02, 4.4857e01],
                [0.0000e00, 7.2154e02, 1.7285e02, 2.1638e-01],
                [0.0000e00, 0.0000e00, 1.0000e00, 2.7459e-03],
            ]
        ],
        dtype=torch.float32,
    )
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    grid = grid.unsqueeze(0)

    topdown_layers = 8
    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=0.5,
        grid_height=4.0,
    )

    torch_module = model

    parameters = create_OFT_model_parameters(model, (input_tensor, calib, grid), device=device)

    mean = torch.tensor([0.485, 0.456, 0.406])  # .view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225])  # .view(1, 1, 3)
    ttnn_mean = ttnn.from_torch(
        mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_std = ttnn.from_torch(
        std, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

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
        ttnn_mean,
        ttnn_std,
        features=input_tensor,
        calib=calib,
        grid=grid,
        topdown_layers=topdown_layers,
    )

    tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_module.forward(
        device, ttnn_input, ttnn_calib, ttnn_grid
    )

    tt_scores = ttnn.to_torch(tt_scores)
    tt_pos_offsets = ttnn.to_torch(tt_pos_offsets)
    tt_dim_offsets = ttnn.to_torch(tt_dim_offsets)
    tt_ang_offsets = ttnn.to_torch(tt_ang_offsets)

    scores, pos_offsets, dim_offsets, ang_offsets = torch_module(input_tensor, calib, grid)

    passing, pcc = check_with_pcc(tt_scores, scores, 0.99)
    print(f"Scores: Passing: {passing}, PCC: {pcc}")
    passing, pcc = check_with_pcc(tt_pos_offsets, pos_offsets, 0.99)
    print(f"Pos Offsets: Passing: {passing}, PCC: {pcc}")
    passing, pcc = check_with_pcc(tt_dim_offsets, dim_offsets, 0.99)
    print(f"Dim Offsets: Passing: {passing}, PCC: {pcc}")
    passing, pcc = check_with_pcc(tt_ang_offsets, ang_offsets, 0.99)
    print(f"Ang Offsets: Passing: {passing}, PCC: {pcc}")
