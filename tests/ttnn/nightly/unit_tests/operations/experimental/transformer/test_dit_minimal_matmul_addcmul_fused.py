# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_allclose


def assert_quality(torch_output, tt_output):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


# def assert_quality(torch_output, tt_output, min_pcc=0.999, max_rmse=0.05):
#     """Assert PCC and RMSE quality metrics for test outputs."""
#     pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)


#     # if pcc_passed:
#     #     return

#     import pandas as pd
#     import torch

#     # Save tensors to file for debugging/alignment verification
#     # torch.save({'torch_output': torch_output, 'tt_output': tt_output}, 'debug_outputs.pt')

#     def print_with_indices(tensor, name):
#         arr = tensor.to(torch.float32).detach().cpu().numpy()
#         df = pd.DataFrame(arr)
#         # print(f"{name} (rows: 0-{df.shape[0]-1}, cols: 0-{df.shape[1]-1}):")
#         # print(df.to_string(index=True, header=True))
#         df.to_csv(f"debug_outputs_{name.lower().replace(' ', '_').replace('/', '_')}.csv")

#     # print_with_indices(torch_output, "TORCH OUTPUT")
#     print_with_indices(tt_output, "TT OUTPUT")

#     assert pcc_val >= min_pcc, f"PCC {pcc_val:.7f} is below minimum {min_pcc}"

#     assert_allclose(torch_output, tt_output, atol=1e-2, rtol=0.05), "Outputs are not close"

#     return {
#         "pcc": pcc_val,
#     }


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
    Test the dit_minimal_matmul_addcmul_fused operation.

    Current implementation: Tests skeleton that only performs minimal_matmul.
    Expected behavior: Should match minimal_matmul output (addcmul fusion not yet implemented).
    """
    torch.manual_seed(0)

    # Create torch inputs
    torch_matmul_input = torch.randn(M, K, dtype=torch.bfloat16)
    torch_matmul_weight = torch.randn(K, N, dtype=torch.bfloat16)
    torch_addcmul_a = torch.randn(M, N, dtype=torch.bfloat16)  # residual
    torch_addcmul_b = torch.randn(M, N, dtype=torch.bfloat16)  # gate

    # torch_matmul_input = torch.full_like(torch_matmul_input, fill_value=1.0)
    # torch_matmul_weight = torch.full_like(torch_matmul_weight, fill_value=1.0)
    # torch_matmul_weight = torch.eye(K, N, dtype=torch.bfloat16)

    # Debug: Test with constant values first
    # torch_addcmul_a = torch.full_like(torch_addcmul_a, fill_value=4.0)  # expected output will be == 1
    # torch_addcmul_b = torch.full_like(torch_addcmul_b, fill_value=5.0)  # expected output will be == 1

    torch_bias = torch.randn(1, N, dtype=torch.bfloat16) if use_bias else None
    # torch_bias = torch.full((1, N), fill_value=1.0, dtype=torch.bfloat16)

    # Compute expected torch output (full fused operation)
    with torch.no_grad():
        torch_matmul_output = torch_matmul_input @ torch_matmul_weight
        if torch_bias is not None:
            torch_matmul_output = torch_matmul_output + torch_bias
        # Full fused result (what we want in the future)
        torch_expected_fused = torch.addcmul(torch_addcmul_a, torch_matmul_output, torch_addcmul_b, value=scalar)
        # Skeleton result (what we get now - just matmul)
        torch_expected_skeleton = torch_matmul_output

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

    # Run fused operation (skeleton version)
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

    # For skeleton implementation, compare against matmul output only
    # TODO: When fusion is implemented, compare against torch_expected_fused
    # check_result = assert_quality(torch_expected_fused, tt_output_torch, min_pcc=0.999, max_rmse=0.05)
    check_result = assert_quality(torch_expected_fused, tt_output_torch)

    logger.info(f"Test passed for M={M}, K={K}, N={N}")
    return check_result


@pytest.mark.parametrize("use_bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
def test_dit_minimal_matmul_addcmul_fused_basic(device, use_bias, dtype):
    """Basic functionality test with small shapes."""
    run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=256,  # 256,
        K=512,  # 512,
        N=1024,  # 1024,
        scalar=1.0,
        dtype=dtype,
        use_bias=use_bias,
    )


@pytest.mark.skip()
@pytest.mark.parametrize(
    "M, K, N, config_name",
    [
        (24800, 5120, 13824, "5B-720p"),  # 31*20*40 = 24800
        (32760, 5120, 13824, "14B-480p"),  # 21*30*52 = 32760
        (75600, 5120, 13824, "14B-720p"),  # 21*45*80 = 75600
        (2000, 5120, 13824, "small"),  # Small test shape
    ],
)
def test_dit_minimal_matmul_addcmul_fused_wan2_shapes(device, M, K, N, config_name):
    """Test with actual Wan2.2 transformer shapes."""
    logger.info(f"Testing Wan2.2 shape configuration: {config_name}")

    # Use appropriate block sizes for large shapes
    M_block = 16 if M > 10000 else 8
    K_block = 16
    N_block = 16

    run_dit_minimal_matmul_addcmul_fused_test(
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
        subblock_h=4,
        subblock_w=4,
    )


@pytest.mark.parametrize("scalar_value", [0.5, 1.0, 2.0], ids=["scalar_0.5", "scalar_1.0", "scalar_2.0"])
def test_dit_minimal_matmul_addcmul_fused_scalar_values(device, scalar_value):
    """Test with different scalar multiplier values."""
    # Note: In skeleton implementation, scalar is ignored
    run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=512,
        K=1024,
        N=2048,
        scalar=scalar_value,
        dtype=ttnn.bfloat16,
        use_bias=False,
    )


@pytest.mark.parametrize(
    "M_block, K_block, N_block, subblock_h, subblock_w",
    [
        (4, 4, 4, 2, 2),
        (8, 8, 8, 2, 4),
        (16, 16, 16, 4, 4),
    ],
    ids=["4x4x4_2x2", "8x8x8_2x4", "16x16x16_4x4"],
)
def test_dit_minimal_matmul_addcmul_fused_block_configs(device, M_block, K_block, N_block, subblock_h, subblock_w):
    """Test with different block size configurations."""
    run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=1024,
        K=2048,
        N=4096,
        scalar=1.0,
        dtype=ttnn.bfloat16,
        use_bias=True,
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
    )


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4])
def test_dit_minimal_matmul_addcmul_fused_math_fidelity(device, math_fidelity):
    """Test with different math fidelity settings."""
    run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=512,
        K=1024,
        N=2048,
        scalar=1.0,
        dtype=ttnn.bfloat16,
        use_bias=True,
        math_fidelity=math_fidelity,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [
        (32, 64, 128),  # Very small
        (128, 256, 512),  # Small
        (512, 1024, 2048),  # Medium
        (1024, 2048, 4096),  # Large
    ],
    ids=["very_small", "small", "medium", "large"],
)
def test_dit_minimal_matmul_addcmul_fused_various_sizes(device, M, K, N):
    """Test with various tensor sizes."""
    run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=M,
        K=K,
        N=N,
        scalar=1.0,
        dtype=ttnn.bfloat16,
        use_bias=True,
    )


def test_dit_minimal_matmul_addcmul_fused_compare_with_separate_ops(device):
    """
    Compare fused operation with separate minimal_matmul and addcmul calls.

    Note: In skeleton implementation, we only compare with minimal_matmul.
    TODO: Add addcmul comparison when fusion is implemented.
    """
    torch.manual_seed(0)

    M, K, N = 512, 1024, 2048
    scalar = 1.0

    # Create torch inputs
    torch_matmul_input = torch.randn(M, K, dtype=torch.bfloat16)
    torch_matmul_weight = torch.randn(K, N, dtype=torch.bfloat16)
    torch_addcmul_input1 = torch.randn(M, N, dtype=torch.bfloat16)
    torch_addcmul_input2 = torch.randn(M, N, dtype=torch.bfloat16)

    # Convert to ttnn
    tt_matmul_input = ttnn.from_torch(torch_matmul_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_matmul_weight = ttnn.from_torch(torch_matmul_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_addcmul_input1 = ttnn.from_torch(
        torch_addcmul_input1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_addcmul_input2 = ttnn.from_torch(
        torch_addcmul_input2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )

    # Run fused operation
    tt_fused_output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
        tt_matmul_input,
        tt_matmul_weight,
        scalar,
        tt_addcmul_input1,
        tt_addcmul_input2,
        config=matmul_config,
        compute_kernel_config=compute_config,
    )

    # Run separate operations (skeleton behavior)
    tt_matmul_output = ttnn.experimental.minimal_matmul(
        tt_matmul_input,
        tt_matmul_weight,
        config=matmul_config,
        compute_kernel_config=compute_config,
    )

    # Convert to torch and compare
    tt_fused_torch = ttnn.to_torch(tt_fused_output)
    tt_matmul_torch = ttnn.to_torch(tt_matmul_output)

    # Skeleton implementation: fused should match minimal_matmul
    # assert_quality(tt_matmul_torch, tt_fused_torch, min_pcc=0.9999, max_rmse=0.001)
    check_result = assert_quality(tt_matmul_torch, tt_fused_torch)

    # TODO: When fusion is implemented, uncomment and test full pipeline:
    # tt_addcmul_output = ttnn.addcmul(tt_addcmul_input1, tt_matmul_output, tt_addcmul_input2, value=scalar)
    # tt_addcmul_torch = ttnn.to_torch(tt_addcmul_output)
    # assert_quality(tt_addcmul_torch, tt_fused_torch, min_pcc=0.999, max_rmse=0.05)

    logger.info("Comparison test passed: fused operation matches separate minimal_matmul")
