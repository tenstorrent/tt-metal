# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_eltwise_softmax_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-1, 1).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.softmax(x, dim=-1)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.softmax(x, dim=-1)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result, pcc=0.999)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(6, 160, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        3181992,
    ),
    (
        [(2, 4, 224, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        10739250,
    ),
    (
        [(6, 160, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
    (
        [(2, 4, 224, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        10739250,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_softmax(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_softmax_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
