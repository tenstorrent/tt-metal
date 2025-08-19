import torch
import ttnn
import torch.nn.functional as F
import pytest
from models.experimental.oft.tt.tt_oft import OFT, perspective_torch
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height",
    [
        ((1, 128, 48, 160), 128, 0.5, 4),  # feats8
        # ((1, 256, 24, 80), 256, 0.5, 4), #feats16
        # ((1, 512, 12, 40), 512, 0.5, 4), #feats32
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_oft_forward(device, input_shape, channels, cell_size, grid_height):
    torch.manual_seed(0)
    features = torch.randn(*input_shape)
    calib = torch.randn(1, 3, 4)
    grid = torch.randn(1, 160, 160, 3)

    ref_oft = ReferenceOFT(channels, cell_size, grid_height)
    ref_out = ref_oft.forward(features, calib, grid)

    # Prepare TTNN input
    params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)

    print(f"Reference OFT output shape: {ref_out.shape}")
    print(f"TTNN OFT parameters: {params}")
    # with open("params_oft.txt", "a") as f:
    #     f.write(str(params))
    tt_features = ttnn.from_torch(features, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # tt_oft = OFT(device, params, channels, cell_size, grid_height)
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    y_corners_tt = ttnn.from_torch(
        y_corners,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    img_corners = perspective_torch(calib.view(-1, 1, 1, 1, 3, 4), grid + y_corners)
    img_corners_tt = ttnn.from_torch(
        img_corners,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"TTNN: y_corners shape: {y_corners_tt.shape}, dtype: {y_corners_tt.dtype}")
    print(f"TTNN: img_corners shape: {img_corners_tt.shape}, dtype: {img_corners_tt.dtype}")
    tt_oft = OFT(device, params, y_corners_tt, img_corners_tt, scale=1 / 8.0)
    # tt_out = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = ttnn.to_torch(tt_out)

    print(f"Reference OFT output shape: {ref_out.shape}")
    print(f"TTNN OFT output shape: {tt_out.shape}")

    message, pcc = assert_with_pcc(tt_out, ref_out, 0.99)
    print(f"Passing: {message}, PCC: {pcc}")
