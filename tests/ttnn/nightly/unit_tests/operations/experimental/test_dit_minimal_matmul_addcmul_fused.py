# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


def assert_quality(torch_output, tt_output):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()

    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_dit_minimal_matmul_addcmul_fused_test(
    device,
    M,
    K,
    N,
    scalar=1.0,
    dtype=ttnn.bfloat16,
    use_bias=False,
    M_block_size=8,
    K_block_size=8,
    N_block_size=8,
    subblock_h=2,
    subblock_w=2,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
):
    """
    Test dit_minimal_matmul_addcmul_fused: output = addcmul_input1 + (scalar * matmul_output * addcmul_input2).
    """
    torch.manual_seed(0)

    # Create torch inputs
    torch_matmul_input = torch.randn(M, K, dtype=torch.bfloat16)
    torch_matmul_weight = torch.randn(K, N, dtype=torch.bfloat16)
    torch_addcmul_a = torch.randn(M, N, dtype=torch.bfloat16)  # base value (full shape)
    torch_addcmul_b = torch.randn(1, N, dtype=torch.bfloat16)  # gate (broadcast like bias)
    torch_bias = torch.randn(1, N, dtype=torch.bfloat16) if use_bias else None

    # Compute expected torch output (full fused operation)
    with torch.no_grad():
        torch_matmul_output = torch_matmul_input @ torch_matmul_weight
        if torch_bias is not None:
            torch_matmul_output = torch_matmul_output + torch_bias
        # Full fused result (what we want in the future)
        torch_expected_fused = torch.addcmul(torch_addcmul_a, torch_matmul_output, torch_addcmul_b, value=scalar)

    # Convert to ttnn tensors
    tt_matmul_input = ttnn.from_torch(torch_matmul_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_matmul_weight = ttnn.from_torch(torch_matmul_weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_addcmul_a = ttnn.from_torch(torch_addcmul_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_addcmul_b = ttnn.from_torch(torch_addcmul_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bias = None
    if torch_bias is not None:
        tt_bias = ttnn.from_torch(torch_bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Configure compute
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )

    tt_output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
        tt_matmul_input,
        tt_matmul_weight,
        scalar,
        tt_addcmul_a,
        tt_addcmul_b,
        bias_tensor=tt_bias,
        config=matmul_config,
        compute_kernel_config=compute_config,
    )

    tt_output_torch = ttnn.to_torch(tt_output)

    check_result = assert_quality(torch_expected_fused, tt_output_torch)

    return check_result


@pytest.mark.parametrize("use_bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_minimal_matmul_addcmul_fused_basic(device, use_bias, dtype):
    """Basic functionality test with small shapes."""
    check_result = run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=256,
        K=512,
        N=1024,
        scalar=1.0,
        dtype=dtype,
        use_bias=use_bias,
    )
    assert check_result["pcc"] > 0.9995
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N, config_name",
    [
        (9472, 5120, 1280, "wan2.2_14b-720p-glx"),
        (2368, 5120, 1280, "wan2.2_14b-720p-single"),
    ],
)
def test_dit_minimal_matmul_addcmul_fused_wan2_shapes(device, M, K, N, config_name):
    """Test with actual Wan2.2 transformer shapes."""

    M_block = 8
    K_block = 8
    N_block = 8

    check_result = run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=M,
        K=K,
        N=N,
        scalar=1.0,
        dtype=ttnn.bfloat16,
        use_bias=True,
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=2,
        subblock_w=2,
    )
    assert check_result["pcc"] > 0.9995
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("scalar_value", [0.5, 1.0, 2.0], ids=["scalar_0.5", "scalar_1.0", "scalar_2.0"])
def test_dit_minimal_matmul_addcmul_fused_scalar_values(device, scalar_value):
    """Test with different scalar multiplier values."""
    check_result = run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=512,
        K=1024,
        N=2048,
        scalar=scalar_value,
        dtype=ttnn.bfloat16,
        use_bias=False,
    )
    assert check_result["pcc"] > 0.9995
    assert check_result["relative_rmse"] < 0.02
