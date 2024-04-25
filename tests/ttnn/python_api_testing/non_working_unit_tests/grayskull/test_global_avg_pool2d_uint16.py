# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_global_avg_pool2d_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    # make random tensor with integers
    x = torch.randint(0, 20, input_shape)
    x_ref = x.to(torch.float32).detach().clone()

    output_size = (1, 1)

    try:
        # get ref result
        ref_value = torch.nn.functional.adaptive_avg_pool2d(x_ref, output_size)

        print(f"Pytorch: {ref_value[1:10, 1:10, 0, 0]}")

        input_tensor = torch.permute(x, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
        input_tensor = ttnn.from_torch(input_tensor, dtype=dtype[0], layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.global_avg_pool2d(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        tt_result = torch.permute(output_tensor, (0, 3, 1, 2))
        tt_result = tt_result.to(torch.float32)

        print(f"TT: {tt_result[1:10, 1:10, 0, 0]}")

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (6, 2, 224, 192),
        [ttnn.uint16],
        [ttnn.TILE_LAYOUT],
        (ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG),
        3378971,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_global_avg_pool2d(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_global_avg_pool2d_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
