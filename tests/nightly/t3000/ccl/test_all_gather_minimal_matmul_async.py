# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


def assert_quality(torch_output, tt_output):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_linear_impl(
    device,
    torch_input,
    weight_input,
    bias_input,
    tt_input,
    tt_weight,
    tt_bias,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    core_grid=None,
):
    core_grid = core_grid or device.compute_with_storage_grid_size()

    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)
    else:
        assert activation is None, f"Unsupported activation: {activation}"

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input

        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.AllGatherMinimalMatmulAsyncConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )
    tt_output = ttnn.experimental.all_gather_minimal_matmul_async(
        tt_input,
        tt_weight,
        bias_tensor=tt_bias,
        fused_activation=activation_fn,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)
    return check_result


def run_test_linear(
    device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    use_bias=False,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    dtype=ttnn.bfloat16,
    weight_dtype=None,
    bias_dtype=None,
    core_grid=None,
):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)
    bias_input = None
    if use_bias:
        bias_input = torch.randn((1, N), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=weight_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(bias_input, dtype=bias_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)

    return run_test_linear_impl(
        device=device,
        torch_input=torch_input,
        weight_input=weight_input,
        bias_input=bias_input,
        tt_input=tt_input,
        tt_weight=tt_weight,
        tt_bias=tt_bias,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        activation=activation,
        math_fidelity=math_fidelity,
        fp32_acc=fp32_acc,
        core_grid=core_grid,
    )


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(
        device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02
