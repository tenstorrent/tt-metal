import torch
import torch.nn as nn
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.vae.models import Conv1x1
import math


@pytest.mark.parametrize(
    "B,in_channels,T,H,W,out_channels, bias, texp, sexp",
    [
        (1, 768, 28, 60, 106, 6144, True, 3, 2),
        (1, 512, 82, 120, 212, 2048, True, 2, 2),
        (1, 256, 163, 240, 424, 512, True, 1, 2),
    ],
)
@pytest.mark.parametrize("parallel_factor", [8])
def test_conv1x1_tt_expand(device, B, in_channels, out_channels, T, H, W, bias, parallel_factor, texp, sexp):
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    T = math.ceil(T / parallel_factor)

    # Create input tensor
    input_shape = (B, in_channels, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # Create Conv1x1 module for ground truth
    conv1x1 = Conv1x1(in_channels, out_channels)
    torch_output = conv1x1(x)

    # Convert weights to ttnn
    weight = conv1x1.weight.data.transpose(0, 1)
    bias = conv1x1.bias.data
    out_chan_dim = out_channels // (texp * sexp * sexp)

    def swizzle_weight(w):
        # X (C texp sexp sexp) -> X (texp sexp sexp C)
        w = w.reshape(-1, out_chan_dim, texp, sexp, sexp)
        w = w.permute(0, 2, 3, 4, 1)
        w = w.reshape(-1, texp * sexp * sexp * out_chan_dim)
        return w.squeeze()

    weight = swizzle_weight(weight)
    bias = swizzle_weight(bias)

    tt_weight = ttnn.from_torch(
        weight,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_bias = ttnn.from_torch(
        bias,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get channels last
    x_perm = x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, in_channels)
    # Convert input to ttnn
    tt_input = ttnn.from_torch(
        x_perm,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.linear(tt_input, tt_weight, bias=tt_bias)

    # Convert back to torch
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    # Get channels first
    tt_output = tt_output.permute(0, 3, 1, 2).reshape(B, out_channels, T, H, W)

    # Undo output channel swizzling to compare to gt
    tt_output = tt_output.reshape(B, texp, sexp, sexp, out_chan_dim, T, H, W)
    tt_output = tt_output.permute(0, 4, 1, 2, 3, 5, 6, 7).reshape(B, out_channels, T, H, W)

    # Compare outputs
    pcc, mse, mae = compute_metrics(torch_output, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Verify shapes and values
    assert tt_output.shape == (B, out_channels, T, H, W)
    assert pcc > 0.99


@pytest.mark.parametrize(
    "B,in_channels,T,H,W,out_channels, bias",
    [
        (1, 128, 163, 480, 848, 3, True),
    ],
)
@pytest.mark.parametrize("parallel_factor", [8])
def test_conv1x1_tt_rgb(device, B, in_channels, out_channels, T, H, W, bias, parallel_factor):
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    T = math.ceil(T / parallel_factor)

    # Create input tensor
    input_shape = (B, in_channels, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # Create Conv1x1 module for ground truth
    conv1x1 = Conv1x1(in_channels, out_channels)
    torch_output = conv1x1(x)

    # Convert input to ttnn
    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert weights to ttnn
    weight = conv1x1.weight.data.transpose(0, 1)
    bias = conv1x1.bias.data

    tt_weight = ttnn.from_torch(
        weight,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_bias = ttnn.from_torch(
        bias,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Implement Conv1x1 using ttnn operations:
    # 1. Move channels to end
    tt_x = ttnn.permute(tt_input, [0, 2, 3, 4, 1])
    tt_x = ttnn.reshape(tt_x, [B * T, H, W, in_channels])

    # 2. Apply linear transformation
    tt_output = ttnn.linear(tt_x, tt_weight, bias=tt_bias)

    # 3. Move channels back to position 1
    tt_output = ttnn.reshape(tt_output, [B, T, H, W, out_channels])
    tt_output = ttnn.permute(tt_output, [0, 4, 1, 2, 3])

    # Convert back to torch
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # Compare outputs
    pcc, mse, mae = compute_metrics(torch_output, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Verify shapes and values
    assert tt_output.shape == (B, out_channels, T, H, W)
    assert pcc > 0.99
