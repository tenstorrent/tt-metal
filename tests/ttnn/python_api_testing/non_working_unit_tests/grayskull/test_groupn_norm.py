# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_group_norm_tests(
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
    sh = input_shape[0]
    sh2 = (sh[1],)
    print(sh2)
    y = torch.Tensor(torch.Size(sh2)).uniform_(-100, 100).to(torch.bfloat16)
    z = torch.Tensor(torch.Size(sh2)).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.functional.group_norm(input=x, num_groups=1, weight=y, bias=z, eps=1e-05)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[0], dtype[0])
        z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.group_norm(x, num_groups=0, weight=y, bias=z)

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
        [(224, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        11079580,
    ),
    (
        [(64, 160)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
    (
        [(64, 160)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
    (
        [(5, 5, 192, 96)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
    (
        [(5, 5, 192, 96)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
    (
        [(2, 64, 32)],
        [ttnn.bfloat16],
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
def test_group_norm(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_group_norm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
