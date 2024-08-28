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
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_topk_simmilarity


def run_topk_tests(
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
        ref_values, ref_indices = torch.topk(x, 32, dim=-1, largest=True, sorted=True)

        t0 = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_values, tt_indices = ttnn.topk(t0, 32, dim=-1, largest=True, sorted=True)

        tt_values = ttnn_ops.ttnn_tensor_to_torch(tt_values)
        tt_indices = ttnn_ops.ttnn_tensor_to_torch(tt_indices).to(torch.int64)

        tt_gather_values = torch.gather(x, -1, tt_indices)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    success, simmilarity_value = comp_topk_simmilarity([ref_values, ref_indices], [tt_values, tt_gather_values])
    logger.debug(simmilarity_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(4, 7, 32, 512)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        17155532,
    ),
    (
        [(1, 9, 32, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16561724,
    ),
    (
        [(6, 7, 224, 1024)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        9248746,
    ),
    (
        [(3, 7, 64, 64)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        12031119,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_topk(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_topk_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
