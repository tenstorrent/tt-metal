# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import (
    untilize_with_unpadding as tt_untilize_with_unpadding,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_untilize_with_unpadding_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    output_tensor_end,
    device,
):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.untilize_with_unpadding(x_ref, output_tensor_end=output_tensor_end)

    tt_result = tt_untilize_with_unpadding(
        x=x,
        output_tensor_end=output_tensor_end,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (11, 17, 64, 448),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        5263366,
        [10, 9, 4, 1],
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, output_tensor_end",
    (test_sweep_args),
)
def test_untilize_with_unpadding_test(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    output_tensor_end,
    device,
):
    random.seed(0)
    run_untilize_with_unpadding_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        output_tensor_end,
        device,
    )
