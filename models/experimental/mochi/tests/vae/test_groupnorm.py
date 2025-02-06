from typing import Literal
import torch
import torch.nn as nn
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.vae.models import GroupNormSpatial

# from models.experimental.mochi.vae.modules import GroupNormSpatial as NewGroupNormSpatial


@pytest.mark.parametrize(
    "num_groups,B,C,T,H,W,affine",
    [
        # From blocks.0 and blocks.1 (first section)
        (32, 1, 768, 28, 60, 106, True),
        # From blocks.2 (second section)
        (32, 1, 512, 82, 120, 212, True),
        # From blocks.3 (third section)
        (32, 1, 256, 163, 240, 424, True),
        # From blocks.4 (fourth section)
        (32, 1, 128, 163, 480, 848, True),
    ],
)
def test_groupnorm_spatial_torch(num_groups, B, C, T, H, W, affine):
    # Set a manual seed for reproducibility
    torch.manual_seed(42)

    input_shape = (B, C, T, H, W)
    # Create random input tensor
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Create both GroupNorm implementations
    groupnorm_spatial = GroupNormSpatial(num_groups=num_groups, num_channels=input_shape[1], affine=affine)
    weight = torch.randn(C, dtype=torch.float32)
    bias = torch.randn(C, dtype=torch.float32)
    groupnorm_spatial.weight = nn.Parameter(weight)
    groupnorm_spatial.bias = nn.Parameter(bias)

    # Process each temporal frame independently with standard GroupNorm
    B, C, T, H, W = input_shape
    inp = input_tensor.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
    inp = inp.reshape(B * T * num_groups, -1)  # (B T num_groups) (C/num_groups H W)
    # input_reshaped = input_tensor.transpose(1, 2).reshape(B, T, num_groups, C//num_groups, H, W).reshape(B*T*num_groups, -1)
    norm_output = torch.nn.functional.layer_norm(inp, normalized_shape=(H * W * C // num_groups,))
    norm_output = (
        norm_output.reshape(B, T, num_groups, C // num_groups, H, W).reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)
    )  # B, T, H, W, C
    norm_output = norm_output * weight + bias
    test_output = norm_output.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
    # group_size = C // num_groups
    # for g in range(num_groups):
    #     gstart, gend = g * group_size, (g + 1) * group_size
    #     group_output = torch.nn.functional.layer_norm(
    #         input_reshaped[..., gstart:gend].reshape(B * T, H * W * group_size),
    #         normalized_shape=(H * W * group_size,),
    #     )
    #     group_output = group_output.reshape(B * T, H, W, group_size)
    #     group_output = group_output * weight[gstart:gend] + bias[gstart:gend]
    #     test_output[..., gstart:gend] = group_output
    # test_output = test_output.permute(0, 3, 1, 2).reshape(B, T, C, H, W)

    # Compute outputs
    output_spatial = groupnorm_spatial(input_tensor)

    # Compare outputs
    pcc, mse, mae = compute_metrics(test_output, output_spatial)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Assertions to verify correctness
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
    assert torch.allclose(
        output_spatial, test_output, rtol=1e-5, atol=1e-5
    ), "GroupNormSpatial output doesn't match standard GroupNorm"

    # Test that the shapes match
    assert (
        output_spatial.shape == input_tensor.shape
    ), f"Output shape {output_spatial.shape} doesn't match input shape {input_tensor.shape}"


def layernorm_decomp(x, epsilon=1e-5):
    # (x - mean(x)) / sqrt(var(x) + eps)
    mean = ttnn.mean(x, dim=3, keepdim=True)
    xmm = x - mean

    std = ttnn.sqrt(var + epsilon)
    recip_stc = ttnn.reciprocal(std)
    return xmm * recip_stc


@pytest.mark.parametrize(
    "num_groups,B,C,T,H,W,affine",
    [
        # From blocks.0 and blocks.1 (first section)
        (32, 1, 768, 28, 60, 106, True),
        # From blocks.2 (second section)
        (32, 1, 512, 82, 120, 212, True),
        # From blocks.3 (third section)
        (32, 1, 256, 163, 240, 424, True),
        # From blocks.4 (fourth section)
        (32, 1, 128, 163, 480, 848, True),
    ],
)
def test_groupnorm_spatial_tt(device, num_groups, B, C, T, H, W, affine):
    # Set a manual seed for reproducibility
    torch.manual_seed(42)

    input_shape = (B, C, T, H, W)
    # Create random input tensor
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Create both GroupNorm implementations
    groupnorm_spatial = GroupNormSpatial(num_groups=num_groups, num_channels=input_shape[1], affine=affine)
    weight = torch.randn(C, dtype=torch.float32)
    bias = torch.randn(C, dtype=torch.float32)
    groupnorm_spatial.weight = nn.Parameter(weight)
    groupnorm_spatial.bias = nn.Parameter(bias)

    # Process each temporal frame independently with standard GroupNorm
    B, C, T, H, W = input_shape
    inp = input_tensor.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
    inp = inp.reshape(1, 1, B * T * num_groups, -1)
    tt_input_tensor = ttnn.from_torch(
        inp,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_t = ttnn.from_torch(
        weight.reshape(1, 1, 1, -1),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        bias.reshape(1, 1, 1, -1),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    normed = ttnn.layer_norm(tt_input_tensor, epsilon=1e-5)  # (B T num_groups) (C/num_groups H W)
    # normed = layernorm_decomp(tt_input_tensor, epsilon=1e-5)
    normed = ttnn.reshape(normed, (B, T, num_groups, C // num_groups, H, W))
    normed = ttnn.reshape(normed, (B, T, C, H, W))
    normed = ttnn.permute(normed, (0, 1, 3, 4, 2))  # B, T, H, W, C
    normed = ttnn.mul(normed, gamma_t)
    normed = ttnn.add(normed, beta_t)
    normed = ttnn.permute(normed, (0, 4, 1, 2, 3))  # B, C, T, H, W
    tt_output = ttnn.from_device(normed)
    tt_output = ttnn.to_torch(tt_output)
    # tt_output = tt_output.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    output_gt = groupnorm_spatial(input_tensor)
    # Compare outputs
    pcc, mse, mae = compute_metrics(output_gt, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Assertions to verify correctness
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
