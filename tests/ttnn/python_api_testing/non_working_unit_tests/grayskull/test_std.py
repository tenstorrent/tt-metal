# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

# from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_std_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    dim,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-10, 10).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.std(x, dim, keepdim=True)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        tt_result = ttnn.std(x, dim)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(224, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        (1, 0),
        19717156,
    ),
    (
        [(3, 10, 192, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        (2,),
        18539618,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed",
    (test_sweep_args),
)
def test_std(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device):
    run_std_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device)
