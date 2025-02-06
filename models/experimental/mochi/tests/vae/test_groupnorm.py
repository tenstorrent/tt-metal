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
    input_reshaped = input_tensor.transpose(1, 2).reshape(B * T, C, H, W).permute(0, 2, 3, 1)

    test_output = torch.zeros_like(input_reshaped)
    group_size = C // num_groups
    for g in range(num_groups):
        gstart, gend = g * group_size, (g + 1) * group_size
        group_output = torch.nn.functional.layer_norm(
            input_reshaped[..., gstart:gend].reshape(B * T, H * W * group_size),
            normalized_shape=(H * W * group_size,),
        )
        group_output = group_output.reshape(B * T, H, W, group_size)
        group_output = group_output * weight[gstart:gend] + bias[gstart:gend]
        test_output[..., gstart:gend] = group_output
    test_output = test_output.permute(0, 3, 1, 2).reshape(B, T, C, H, W)

    # Compute outputs
    output_spatial = groupnorm_spatial(input_tensor)
    # output_standard = groupnorm_standard(input_reshaped)
    test_output = test_output.reshape(B, T, C, H, W).transpose(1, 2)

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

    # Process each temporal frame independently with standard GroupNorm
    B, C, T, H, W = input_shape
    input_reshaped = input_tensor.transpose(1, 2).reshape(B * T, C, H, W)
    input_reshaped = input_reshaped.permute(0, 2, 3, 1)  # B*T, H, W, C
    input_reshaped = input_reshaped.reshape(B * T, 1, H * W, C)
    tt_input_tensor = ttnn.from_torch(
        input_reshaped,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_t = ttnn.from_torch(
        groupnorm_spatial.weight,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        groupnorm_spatial.bias,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=None,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    output_gt = groupnorm_spatial(input_tensor)
    # Compare outputs
    pcc, mse, mae = compute_metrics(output_gt, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Assertions to verify correctness
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
