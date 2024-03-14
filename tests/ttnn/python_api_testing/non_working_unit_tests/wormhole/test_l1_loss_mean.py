# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_l1_loss_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.L1Loss(reduction="mean")(x, y)
        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[1], in_mem_config[1], dtype[1])

        tt_result = ttnn.l1_loss(x, y, loss_mode="mean")

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    tt_result = tt_result.squeeze(0)
    tt_result = tt_result.squeeze(0)

    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(224, 128), (224, 128)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        15991940,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_l1_loss(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_l1_loss_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
