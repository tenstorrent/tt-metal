import pytest
import torch
import ttnn
import os
from loguru import logger

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality


def run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
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
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
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
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_ablate_datamovement(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    device.disable_and_clear_program_cache()  # since env vars aren't captured by hash
    for skip_in0 in [False, True]:
        for skip_in1 in [False, True]:
            for skip_out in [False, True]:
                if skip_in0:
                    os.environ["TT_MM_SKIP_IN0"] = "1"
                if skip_in1:
                    os.environ["TT_MM_SKIP_IN1"] = "1"
                if skip_out:
                    os.environ["TT_MM_SKIP_OUT"] = "1"

                print(f"skip_in0: {skip_in0}, skip_in1: {skip_in1}, skip_out: {skip_out}")
                check_result = run_test_linear(
                    device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w
                )
                os.environ.pop("TT_MM_SKIP_IN0", None)
                os.environ.pop("TT_MM_SKIP_IN1", None)
                os.environ.pop("TT_MM_SKIP_OUT", None)


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


@pytest.mark.parametrize(
    "M, K, N",
    [
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 2048),
        (4096, 4096, 1024),
        (4096, 4096, 512),
    ],
)
def test_linear_sweep_blocks(device, M, K, N):
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
    M_tiles_per_core = M // 32 // core_grid.y
    N_tiles_per_core = N // 32 // core_grid.x
    import math

    m_block_sizes = [2**i for i in range(1, min(int(math.log2(M_tiles_per_core)), 8) + 1)]
    n_block_sizes = [2**i for i in range(1, min(int(math.log2(N_tiles_per_core)), 8) + 1)]
    k_block_sizes = [4, 8]

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        if (M_block_size < subblock_h) or (N_block_size < subblock_w):
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
