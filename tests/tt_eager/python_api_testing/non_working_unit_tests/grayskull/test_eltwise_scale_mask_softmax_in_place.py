# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import (
    eltwise_scale_mask_softmax_in_place as tt_eltwise_scale_mask_softmax_in_place,
)


def run_eltwise_scale_mask_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-1.0, 1.0)
    x_ref = x.detach().clone()

    y = torch.Tensor(size=input_shape).uniform_(-1.0, 1.0)
    y_ref = y.detach().clone()

    scale = random.uniform(1.0, 100.0)

    # get ref result
    ref_value = pytorch_ops.scale_mask_softmax_in_place(x_ref, y_ref, scale)

    tt_result = tt_eltwise_scale_mask_softmax_in_place(
        x=x,
        y=y,
        scale=scale,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=None,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 1, 32, 32),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        38346,
    ),
    (
        (1, 1, 32, 32),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        38346,
    ),
    (
        (1, 1, 32, 96),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        38346,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_scale_mask_softmax_in_place_test(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    run_eltwise_scale_mask_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device)
