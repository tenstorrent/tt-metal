import torch
import ttnn
import pytest
from models.experimental.oft.tt.tt_oft import OFT
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft
from models.experimental.oft.reference.utils import make_grid
from loguru import logger


@pytest.mark.parametrize(
    "input_shape, channels, cell_size, grid_height, scale, torch_model_dtype, use_precomputed_grid, pcc_threshold",
    [
        # fmt: off
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.float32, False, 0.88),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.float32, False, 0.41),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.float32, False, 0.30),  # feats32
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.float32,  True, 0.80),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.float32,  True, 0.41),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.float32,  True, 0.27),  # feats32
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.bfloat16, False, 0.69),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.bfloat16, False, 0.35),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.bfloat16, False, 0.23),  # feats32
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, torch.bfloat16,  True, 0.64),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, torch.bfloat16,  True, 0.34),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, torch.bfloat16,  True, 0.22),
        # feats32
        # fmt: on
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_oft_forward(
    device,
    input_shape,
    channels,
    cell_size,
    grid_height,
    scale,
    torch_model_dtype,
    use_precomputed_grid,
    pcc_threshold,
    seed,
):
    torch.manual_seed(seed)
    features = (
        torch.randn(*input_shape, dtype=torch.float32) + 1.0
    )  # 0.1 to avoid negative values, features is output of ReLU
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

    ref_oft = ReferenceOFT(channels, cell_size, grid_height, scale=scale, dtype=torch_model_dtype)
    features = features.to(torch_model_dtype)
    calib = calib.to(torch_model_dtype)
    grid = grid.to(torch_model_dtype)

    (
        ref_out,
        ref_integral_img,
        ref_bbox_top_left,
        ref_bbox_btm_right,
        ref_bbox_top_right,
        ref_bbox_btm_left8,
    ) = ref_oft.forward(features, calib, grid)
    # Prepare TTNN input
    params = create_OFT_model_parameters_oft(ref_oft, (features, calib, grid), device)

    features_nhwc = features.permute(0, 2, 3, 1)
    tt_features = ttnn.from_torch(features_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_oft = OFT(
        device,
        params,
        channels,
        cell_size,
        grid_height,
        features.shape[2:],
        calib,
        grid,
        scale=scale,
        use_precomputed_grid=use_precomputed_grid,
    )
    tt_out, integral_img, bbox_top_left, bbox_btm_right, bbox_top_right, bbox_btm_left8 = tt_oft.forward(
        device, tt_features, tt_calib, tt_grid
    )
    tt_out = ttnn.to_torch(tt_out)

    n, c, h, w = ref_out.shape
    ref_out = ref_out.permute(0, 2, 3, 1).view(1, 1, h * w, c)
    message, pcc = assert_with_pcc(tt_out, ref_out, pcc_threshold)
    logger.info(f"Passing: {message}, PCC: {pcc} seed: {seed}")
