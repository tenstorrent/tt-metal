import torch
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.vae.models import DepthToSpaceTime

import math


@pytest.mark.parametrize(
    "B,C,T,H,W,texp,sexp",
    [
        # From first block
        (1, 6144, 28, 60, 106, 3, 2),
        # From second block
        (1, 2048, 82, 120, 212, 2, 2),
        # From third block
        (1, 512, 163, 240, 424, 1, 2),
    ],
)
def test_depth_to_spacetime_torch(B, C, T, H, W, texp, sexp):
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    # Create input tensor
    input_shape = (B, C, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # Create DepthToSpaceTime module
    d2st = DepthToSpaceTime(texp, sexp)

    # Get output
    output = d2st(x)

    # Manual computation for verification
    out_channels = C // (texp * sexp * sexp)
    manual_output = x.reshape(B, out_channels, texp, sexp, sexp, T, H, W)
    manual_output = manual_output.permute(0, 1, 5, 2, 6, 3, 7, 4)
    manual_output = manual_output.reshape(B, out_channels, T * texp, H * sexp, W * sexp)

    # For first block with texp > 1, drop first texp-1 frames
    if texp > 1:
        manual_output = manual_output[:, :, texp - 1 :]

    # Compare outputs
    pcc, mse, mae = compute_metrics(output, manual_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Verify shapes
    expected_T = T * texp if texp == 1 else T * texp - (texp - 1)
    expected_shape = (B, out_channels, expected_T, H * sexp, W * sexp)
    assert output.shape == expected_shape
    assert torch.allclose(output, manual_output, rtol=1e-5, atol=1e-5)
    assert pcc > 0.99


@pytest.mark.parametrize(
    "B,C,T,H,W,texp,sexp",
    [
        # From first block
        (1, 6144, 28, 60, 106, 3, 2),
        # From second block
        (1, 2048, 82, 120, 212, 2, 2),
        # From third block
        (1, 512, 163, 240, 424, 1, 2),
    ],
)
@pytest.mark.parametrize("parallel_factor", [8])
def test_depth_to_spacetime_tt(device, B, C, T, H, W, texp, sexp, parallel_factor):
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    T = math.ceil(T / parallel_factor)
    # Create input tensor
    input_shape = (B, C, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # Create DepthToSpaceTime module for ground truth
    d2st = DepthToSpaceTime(texp, sexp)
    torch_output = d2st(x)

    # ttnn input will be of shape: B T (H W) (texp sexp sexp C)
    # it will also be tilized, coming out of the conv1x1 linear layer
    out_channels = C // (texp * sexp * sexp)
    x_perm = x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, out_channels, texp, sexp, sexp)
    x_perm = x_perm.permute(0, 1, 2, 4, 5, 6, 3).reshape(B, T, H * W, texp * sexp**2 * out_channels)

    tt_input = ttnn.from_torch(
        x_perm,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Implement DepthToSpaceTime using ttnn operations
    # First, go to row-major. This avoids OOM/segfaults
    tt_x = ttnn.to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)

    # 1. Reshape to separate expansion factors
    tt_x = ttnn.reshape(tt_x, [B, T, H, W, texp, sexp, sexp, out_channels])

    # 2. Permute to get: (B T texp H sexp W sexp out_channels)
    tt_x = ttnn.permute(tt_x, [0, 1, 4, 2, 5, 3, 6, 7])

    # 3. Reshape to get: B (T texp) (H sexp W sexp) out_channels
    tt_output = ttnn.reshape(tt_x, [B, T * texp, H * sexp * W * sexp, out_channels])

    # 4. If texp > 1, drop first texp-1 frames
    if texp > 1:
        tt_output = ttnn.slice(tt_output, [0, texp - 1, 0, 0], [B, T * texp, H * sexp * W * sexp, out_channels])

    # 5. Tilize for the rest of the compute graph
    tt_output = ttnn.to_layout(tt_output, ttnn.TILE_LAYOUT)

    # Convert back to torch
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # Now reshape and decombine HW to get channels first to compare to torch
    sliced_T = T * texp if texp == 1 else T * texp - (texp - 1)
    tt_output = tt_output.permute(0, 3, 1, 2).reshape(B, out_channels, sliced_T, H * sexp, W * sexp)

    assert tt_output.shape == torch_output.shape
    # Compare outputs
    pcc, mse, mae = compute_metrics(torch_output, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Verify shapes and values
    expected_T = T * texp if texp == 1 else T * texp - (texp - 1)
    expected_shape = (B, out_channels, expected_T, H * sexp, W * sexp)
    assert tt_output.shape == expected_shape
    assert pcc > 0.99
