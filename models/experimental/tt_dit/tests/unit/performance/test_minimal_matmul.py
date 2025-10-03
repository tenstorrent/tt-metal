import pytest
import torch
import ttnn
from loguru import logger

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality


def run_test_linear(device, M, K, N):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    # torch_input = torch.full((M, K), 1.0, dtype=torch_dtype)
    # weight_input = torch.full((K, N), 3.0, dtype=torch_dtype)
    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(8, 8)

    # This is the optimal single-core config for 4096x4096x4096
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=32,
        N_block_size=4,
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=core_grid,
    )
    tt_output = ttnn.experimental.minimal_matmul(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=None,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)
    return check_result


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
def test_linear(device, M, K, N):
    check_result = run_test_linear(device, M, K, N)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


def test_linear_sweep_subblocks(device):
    M, K, N = 4096, 4096, 4096
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(1, 1)
    subblocks = [(1, 4), (4, 1), (2, 2)]
    block_sizes = (1, 2, 4, 8, 16, 32)

    """
    Skip all data movement since we're just testing subblocks here
    """
    import os

    os.environ["TT_MM_SKIP_IN0"] = "1"
    os.environ["TT_MM_SKIP_IN1"] = "1"
    os.environ["TT_MM_SKIP_OUT"] = "1"

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        block_sizes, block_sizes, block_sizes, subblocks
    ):
        if subblock_h > M_block_size or subblock_w > N_block_size:
            continue

        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}: {e}"
            )


def test_linear_sweep_blocks(device):
    M, K, N = 4096, 4096, 4096
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(8, 8)
    subblocks = [(2, 2)]
    mn_block_sizes = (2, 4, 8, 16)
    k_block_sizes = (2, 4, 8, 16, 32)

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        mn_block_sizes, k_block_sizes, mn_block_sizes, subblocks
    ):
        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}: {e}"
            )
