import torch
import ttnn
import pytest
from models.experimental.oft.tt.tt_oft import OFT
from models.experimental.oft.reference.oft import OFT as ReferenceOFT
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_oft
from loguru import logger
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores


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
    "input_shape, channels, cell_size, grid_height, scale, use_precomputed_grid, expected_pcc",
    [
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, True, 0.900),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, True, 0.771),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, True, 0.562),  # feats32
        ((1, 256, 48, 160), 256, 0.5, 4, 1 / 8, False, 0.880),  # feats8
        ((1, 256, 24, 80), 256, 0.5, 4, 1 / 16, False, 0.790),  # feats16
        ((1, 256, 12, 40), 256, 0.5, 4, 1 / 32, False, 0.603),  # feats32
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_oft_forward_new(
    device, input_shape, channels, cell_size, grid_height, scale, seed, use_precomputed_grid, expected_pcc
):
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(seed)
    # Generate random features with mean=1, std=1, limited between 0 and 5
    torch.manual_seed(seed)
    # real features have min 0, max 5, mean 1.0, std 1.0
    features = torch.randn(*input_shape)
    features = features * 1.0 + 1.0  # Adjust to mean=1, std=1
    features = torch.clamp(features, 0.0, 5.0)  # Limit values between 0 and 5
    print(f"features shape: {features.shape}, dtype: {features.dtype}")

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
    # ref_oft = ReferenceOFT_dtype(channels, cell_size, grid_height, scale=scale, dtype=torch.bfloat16)
    ref_out, intermediates = ref_oft.forward(features, calib, grid)
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
    tt_out, tt_intermediates = tt_oft.forward(device, tt_features, tt_calib, tt_grid)
    tt_out = ttnn.to_torch(tt_out)

    # Check intermediates
    print("Comparing intermediates:")
    for i, (ref_intermediate, tt_intermediate) in enumerate(zip(intermediates, tt_intermediates)):
        for j, (ref_tensor, tt_tensor) in enumerate(zip(ref_intermediate, tt_intermediate)):
            if not isinstance(tt_tensor, torch.Tensor):
                torch_tt_tensor = ttnn.to_torch(tt_tensor)
            else:
                torch_tt_tensor = tt_tensor
            torch_tt_tensor = torch_tt_tensor.permute(2, 0, 1)  # HWC to CHW
            assert (
                torch_tt_tensor.shape == ref_tensor.shape
            ), f"Intermediate {i},{j} shape mismatch: {ref_tensor.shape=} vs {torch_tt_tensor.shape=}"
            passing, pcc = check_with_pcc(ref_tensor, torch_tt_tensor, 0.990)
            logger.warning(f"Intermediate {i},{j} {passing} PCC {pcc}")

    n, c, h, w = ref_out.shape
    ref_out = ref_out.permute(0, 2, 3, 1).view(1, 1, h * w, c)
    pcc_passed, pcc_message = assert_with_pcc(tt_out, ref_out, expected_pcc)
    assert pcc_passed, pcc_message
    logger.info(f"Passing: {features.shape=} PCC: {pcc_message} seed: {seed}")
