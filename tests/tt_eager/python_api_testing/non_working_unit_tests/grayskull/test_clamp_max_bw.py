# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops, pytorch_ops
from models.utility_functions import tt2torch_tensor


def run_clamp_bw(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)

    ref_value = pytorch_ops.clamp_max_bw(x, y, scalar)

    tt_result = tt_lib_ops.clamp_max_bw(
        x=x,
        y=y,
        scalar=scalar,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 10, 192, 96),
        [
            ttnn.bfloat16,
            ttnn.bfloat16,
        ],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [
            None,
            None,
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        9234542,
        -99.0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_clamp_bw_test(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device):
    run_clamp_bw(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device)
