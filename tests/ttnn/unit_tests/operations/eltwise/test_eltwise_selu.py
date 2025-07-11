# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_all_close
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


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([3, 2, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100.0, 100.0),
        (
            -1.6 * 10**38,
            1.6 * 10**38,
        ),
    ],
)
def test_unary_composite_selu_ttnn(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.selu(tt_in)
    result = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden_tensor = golden_function(torch_input)
    assert_with_ulp(golden_tensor, result)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 5, 5])),),
)
@pytest.mark.parametrize(
    "input_val, scale, alpha",
    [
        (0.3, 0.5, 1),
        (0.0, 1.0507, 1.6732),
        (1.0, 1.0507, 1.6732),
        (-1.0, 1.0507, 1.6732),
        (-5.0, 1.0507, 1.6732),
        (-0.5, 1.0, 1.5),
        (float("inf"), 1.0, float("inf")),
        (2.0, 1.2, 1.7),
        (float("-inf"), 1.0507, 1.6732),
        (float("inf"), 1.0507, 1.6732),
        (20.0, 1.0507, 1.6732),
        (-20.0, 1.0507, 1.6732),
        (-0.36719, 0.5, 1),
    ],
)
def test_selu_fill_val_bf16(input_shapes, input_val, scale, alpha, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(torch_input, scale, alpha, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    comp_pass = compare_all_close([tt_result], [golden], atol=0.5, rtol=0)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 1, 1])),),
)
@pytest.mark.parametrize(
    "input_val, scale, alpha",
    [
        (0.0, 1.0507, 1.6732),
        (1.0, 1.0507, 1.6732),
        (-5.0, 1.0507, 1.6732),
        (20.0, 1.0507, 1.6732),
        (-20.0, 1.0507, 1.6732),
    ],
)
def test_selu_fill_val_bf16_assert_with_ulp(input_shapes, input_val, scale, alpha, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(torch_input, scale, alpha, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert assert_with_ulp(golden, result)
