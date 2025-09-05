import torch
import ttnn
import pytest
from models.experimental.oft.tt.tt_oft import OFT
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft
from models.experimental.oft.reference.utils import make_grid


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height, scale, pcc_threshold",
    [
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, 0.88),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, 0.79),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, 0.60),  # feats32
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_oft_forward(device, input_shape, channels, cell_size, grid_height, scale, pcc_threshold, seed):
    torch.manual_seed(seed)
    features = torch.randn(*input_shape, dtype=torch.float32)
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

    ref_oft = ReferenceOFT(channels, cell_size, grid_height, scale=scale)
    ref_out = ref_oft.forward(features, calib, grid)
    # Prepare TTNN input
    params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)

    features_nhwc = features.permute(0, 2, 3, 1)
    tt_features = ttnn.from_torch(features_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_oft = OFT(device, params, channels, cell_size, grid_height, features, calib, grid, scale=scale)
    tt_out = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = ttnn.to_torch(tt_out)

    n, c, h, w = ref_out.shape
    ref_out = ref_out.permute(0, 2, 3, 1).view(1, 1, h * w, c)
    message, pcc = assert_with_pcc(tt_out, ref_out, pcc_threshold)

    print(f"Passing: {message}, PCC: {pcc} seed: {seed}")
