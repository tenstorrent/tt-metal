# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# minimal_matmul on the LLK 2.0 compute kernel (TT_MINIMAL_MATMUL_LLK2=1, tt_metal/tt-llk/llk2/kernels/minimal_matmul.cpp),
# in the forms the Wan2.2 DiT uses: a bias, a bias with gelu_tanh, and a bias with a broadcast addcmul of scalar 1.0
# (dit_minimal_matmul_addcmul_fused). The reference math is the op tests' own.

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import (
    assert_quality,
    run_test_linear,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_dit_minimal_matmul_addcmul_fused import (
    run_dit_minimal_matmul_addcmul_fused_test,
)


@pytest.fixture(autouse=True)
def llk2_minimal_matmul(monkeypatch):
    monkeypatch.setenv("TT_MINIMAL_MATMUL_LLK2", "1")


BLOCKS = [(8, 8, 8, 2, 2)]


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1024, 1024, 1024),  # whole blocks
        (320, 96, 320),  # partial M, K and N blocks
        (9472, 5120, 1280),  # Wan2.2 shapes
        (9472, 3456, 5120),
    ],
)
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "fp16_acc"])
@pytest.mark.parametrize("M_block_size, K_block_size, N_block_size, subblock_h, subblock_w", BLOCKS)
def test_llk2_linear_bias(device, M, K, N, fp32_acc, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
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
        use_bias=True,
        fp32_acc=fp32_acc,
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("M, K, N", [(1024, 1024, 1024), (9472, 5120, 3456)])
def test_llk2_linear_bias_gelu_tanh(device, M, K, N):
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.float32)
    torch_weight = torch.randn((K, N), dtype=torch.float32) / K**0.5
    torch_bias = torch.randn((1, N), dtype=torch.float32)
    expected = torch.nn.functional.gelu(torch_input @ torch_weight + torch_bias, approximate="tanh")

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tt_output = ttnn.experimental.minimal_matmul(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        fused_activation=ttnn.UnaryOpType.GELU_TANH,
        compute_kernel_config=compute_config,
        config=ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=2,
            subblock_w=2,
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        ),
    )
    check_result = assert_quality(expected, ttnn.to_torch(tt_output).float())
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 512, 1024),
        (1024, 512, 256),
        (2368, 5120, 1280),  # Wan2.2 out-projection
    ],
)
def test_llk2_addcmul_fused(device, M, K, N):
    check_result = run_dit_minimal_matmul_addcmul_fused_test(
        device=device,
        M=M,
        K=K,
        N=N,
        scalar=1.0,
        dtype=ttnn.bfloat16,
        use_bias=True,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
    )
    assert check_result["pcc"] > 0.9995
    assert check_result["relative_rmse"] < 0.02
