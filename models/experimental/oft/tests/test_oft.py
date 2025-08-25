import torch
import ttnn
import torch.nn.functional as F
import pytest
from models.experimental.oft.tt.tt_oft import OFT, perspective_torch
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft


def make_grid(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height",
    [
        ((1, 256, 48, 160), 256, 0.5, 4),  # feats8
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


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height, scale",
    [
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32),  # feats32
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_oft_forward_new(device, input_shape, channels, cell_size, grid_height, scale):
    torch.manual_seed(0)
    features = torch.randn(*input_shape)

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
    print(f"calib shape: {calib.shape}, dtype: {calib.dtype}")
    # calib = torch.randn(1, 3, 4)
    # grid = torch.randn(1, 160, 160, 3)
    grid = make_grid(grid_size=(80.0, 80.0), grid_offset=(-40.0, 1.74, 0.0), grid_res=0.5)
    print(f"grid shape: {grid.shape}, dtype: {grid.dtype}")
    # return grid
    grid = grid.unsqueeze(0)
    print(f"grid shape after unsqueeze: {grid.shape}, dtype: {grid.dtype}")

    ref_oft = ReferenceOFT(channels, cell_size, grid_height, scale=scale)
    ref_out = ref_oft.forward(features, calib, grid)
    # Prepare TTNN input
    params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)

    print(f"Reference OFT output shape: {ref_out.shape}")
    print(f"TTNN OFT parameters: {params}")
    # with open("params_oft.txt", "a") as f:
    #     f.write(str(params))
    features_nhwc = features.permute(0, 2, 3, 1)
    tt_features = ttnn.from_torch(features_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    print(f"TEST: features shape: {tt_features.shape}, dtype: {tt_features.dtype}")
    print(f"TEST: calib shape: {tt_calib.shape}, dtype: {tt_calib.dtype}")
    print(f"TEST: grid shape: {tt_grid.shape}, dtype: {tt_grid.dtype}")
    # tt_oft = OFT(device, params, channels, cell_size, grid_height)

    tt_oft = OFT(device, params, channels, cell_size, grid_height, features, calib, grid, scale=scale)
    # tt_out = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = ttnn.to_torch(tt_out)

    print(f"Reference OFT output shape: {ref_out.shape}")
    print(f"TTNN OFT output shape: {tt_out.shape}")

    n, c, h, w = ref_out.shape
    message, pcc = assert_with_pcc(tt_out, ref_out.permute(0, 2, 3, 1).view(n, 1, w * h, c), 0.99)
    # message, pcc = assert_with_pcc(tt_out, ref_out, 0.99)
    # message = "Tensors are equal" if torch.allclose(tt_out, ref_out, atol=1e-5) else "Tensors are NOT equal"
    # pcc = None

    print(f"Passing: {message}, PCC: {pcc}")


import ttnn
import torch


def test_div(device):
    vox_feats_torch = (218.0907 + 241.7774) * torch.rand((1, 256, 7, 25281), dtype=torch.bfloat16) - 241.7774
    area_torch = (7680 + 632) * torch.rand((1, 1, 7, 25281), dtype=torch.bfloat16) - 632
    torch_output = vox_feats_torch / (area_torch)
    vox_feats_ttnn = ttnn.from_torch(vox_feats_torch.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    area = ttnn.from_torch(
        area_torch.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    ttnn_output = ttnn.div(vox_feats_ttnn, area, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(ttnn_output, torch_output.permute(0, 2, 3, 1), 0.99)
