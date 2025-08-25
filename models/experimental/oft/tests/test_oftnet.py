import torch
import ttnn
import pytest
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.tt_oftnet import TTOftNet
from models.experimental.oft.tt.tt_resnet import TTBasicBlock

# from models.experimental.oft.tt.tt_oftnet import OftNet
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "cell_size",
    [0.5],
)
@pytest.mark.parametrize(
    "grid_height",
    [4],
)
def test_oftnet(device, grid_height, cell_size):
    # disable_persistent_kernel_cache()

    torch.manual_seed(42)

    input_tensor = torch.rand((1, 3, 384, 1280))
    calib = torch.rand((1, 3, 4))
    grid = torch.rand((1, 160, 160, 3))

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )
    # state_dict = model.state_dict()

    # model.load_state_dict(state_dict)
    # torch_output = model(input_tensor, calib, grid)[0]
    torch_module = model
    parameters = create_OFT_model_parameters(model, (input_tensor, calib, grid), device=device)
    # print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    # print(f"Calib shape: {calib.shape}, dtype: {calib.dtype}")
    # print(f"Grid shape: {grid.shape}, dtype: {grid.dtype}")
    mean = torch.tensor([0.485, 0.456, 0.406])  # .view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225])  # .view(1, 1, 3)

    mean = ttnn.from_torch(
        mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    std = ttnn.from_torch(
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
        mean,
        std,
    )
    # ttnn_output = tt_module.forward(device, ttnn_input)#, ttnn_calib, ttnn_grid)
    # ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.permute((0, 3, 1, 2))
    ttnn_feats8 = tt_module.forward(device, ttnn_input, calib, grid)
    ttnn_feats8 = ttnn.to_torch(ttnn_feats8)
    # ttnn_feats16 = ttnn.to_torch(ttnn_feats16)
    # print(f"TTNN feats16 shape: {ttnn_feats16.shape}")
    # ttnn_feats32 = ttnn.to_torch(ttnn_feats32)
    # with torch.inference_mode():
    # torch_output = torch_module(input_tensor, calib, grid)
    feats8 = torch_module(input_tensor, calib, grid)
    # n, c, h, w = feats8.shape
    # feats8 = feats8.permute(0, 2, 3, 1)
    # feats8 = feats8.reshape(1, 1, n * h * w, c)
    # n, c, h, w = feats16.shape
    # feats16 = feats16.permute(0, 2, 3, 1)
    # feats16 = feats16.reshape(1, 1, n * h * w, c)

    # n, c, h, w = feats32.shape
    # feats32 = feats32.permute(0, 2, 3, 1)
    # feats32 = feats32.reshape(1, 1, n * h * w, c)

    passing, pcc = check_with_pcc(ttnn_feats8, feats8, 0.99)
    print(f"Passing: {passing}, PCC: {pcc}")
    # passing, pcc = check_with_pcc(ttnn_feats16, feats16, 0.99)
    # print(f"Passing: {passing}, PCC: {pcc}")
    # passing, pcc = check_with_pcc(ttnn_feats32, feats32, 0.99)
    # print(f"Passing: {passing}, PCC: {pcc}")
