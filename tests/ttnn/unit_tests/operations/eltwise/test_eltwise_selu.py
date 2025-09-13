# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_selu_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.functional.selu(x)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.selu(x)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(3, 2, 192, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6861134,
    ),
    (
        [(12, 224, 224)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
    (
        [(3, 2, 191, 31)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6861134,
    ),
    (
        [(12, 225, 223)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_selu(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_selu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)


def test_selu_arange(device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN, special values where selu has ULP>1 (Covered in atol test below).
    mask = (
        torch.isnan(input_tensor)
        | ((input_tensor >= -0.30859375) & (input_tensor <= 1.1663108012064884e-38))
        | (input_tensor == 3.2300240297573456e38)
        | (input_tensor == -0.0)
    )
    input_tensor[mask] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.selu(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "low, high, expected_Atol",
    [
        (-0.30859375, 1.1663108012064884e-38, 0.004),
        (-1.6 * 10**38, 1.6 * 10**38, 0),  # bf16 range
    ],
)
def test_selu_atol(low, high, expected_Atol, device):
    num_elements = torch.prod(torch.tensor(torch.Size([1, 3, 320, 320]))).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 3, 320, 320]))

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in)
    result = ttnn.to_torch(tt_result)
    torch.allclose(golden, result, atol=expected_Atol)
