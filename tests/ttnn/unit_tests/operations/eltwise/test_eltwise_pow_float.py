# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_pow_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)
    random.seed(data_seed)

    # bfloat8_b input with negative base + fractional exponent yields NaN in torch.pow
    # but inf/zero on device; restrict to non-negative inputs for that path.
    in_low, in_high = (-100, 100)
    if dtype[0] == ttnn.bfloat8_b:
        in_low, in_high = (0, 100)

    x = torch.Tensor(size=input_shape[0]).uniform_(in_low, in_high)
    y = random.uniform(0, 10)

    try:
        # Quantize on host via from_torch (matches device dtype conversion; no device roundtrip).
        x_quantized = ttnn.to_torch(ttnn.from_torch(x, dtype=dtype[0], layout=dlayout[0], device=None)).to(
            torch.float32
        )
        ref_value = torch.pow(x_quantized, y)

        x = ttnn_ops.setup_ttnn_tensor(x_quantized, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.pow(x, y)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(160, 128), (160, 128)],
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
    ),
    (
        [(160, 128), (160, 128)],
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
    ),
    (
        [(160, 128), (160, 128)],
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed",
    (test_sweep_args),
)
def test_pow(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    run_pow_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)
