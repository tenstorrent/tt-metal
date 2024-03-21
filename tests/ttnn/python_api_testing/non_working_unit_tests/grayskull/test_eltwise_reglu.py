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


def run_eltwise_reglu_tests(
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
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        a, b = torch.split(x, x.shape[dim] // 2, dim)
        ref_value = a * torch.nn.functional.gelu(b)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.reglu(x, dim=dim)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


# 'dim': -1}",11079580,(),error,"TT_FATAL @ tt_eager/tt_dnn/op_library/split/split_tiled.cpp:40: (chunk_size % TILE_WIDTH == 0)

test_sweep_args = [
    (
        [(3, 2, 192, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        -1,
        11079580,
    ),
    # (
    #     [(2, 4, 224, 64)],
    #     [ttnn.bfloat8_b],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.L1_MEMORY_CONFIG],
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     2699313,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed",
    (test_sweep_args),
)
def test_eltwise_reglu(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device):
    run_eltwise_reglu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device)
