# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_max_pool2d_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    batch_size = x.shape[0]
    input_height = x.shape[2]
    input_width = x.shape[3]

    try:
        # get ref result
        m = torch.nn.MaxPool2d(3, stride=2)
        ref_value = m(x)

        # get TT result
        m = ttnn.MaxPool2d(
            kernel_size=3,
            stride=2,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache={},
        )

        t0 = ttnn.from_torch(x, dtype=dtype[0], layout=ttnn.TILE_LAYOUT, device=device)
        t1 = m(t0)
        tt_result = ttnn.to_torch(t1)
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
def test_max_pool2d(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_max_pool2d_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
