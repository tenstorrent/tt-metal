# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.common import set_dispatch_mode


def run_untilize_with_unpadding(
    input_shape_1,
    output_tensor_start,
    output_tensor_end,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    dispatch_mode,
    device,
):
    torch.manual_seed(data_seed)
    set_dispatch_mode(dispatch_mode)

    x = gen_rand(size=input_shape_1, low=-100, high=100).to(torch.bfloat16)

    ref_value = pytorch_ops.untilize_with_unpadding(
        x=x,
        output_tensor_start=output_tensor_start,
        output_tensor_end=output_tensor_end,
    )

    tt_result = tt_lib_ops.untilize_with_unpadding(
        x=x,
        output_tensor_start=output_tensor_start,
        output_tensor_end=output_tensor_end,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        (10, 23, 352, 480),
        [0, 0, 0, 0],
        [8, 22, 253, 243],
        [ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE],
        [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        8799403,
        True,
    ),
]


@pytest.mark.parametrize(
    "input_shape_1, output_tensor_start, output_tensor_end, dtype, dlayout, in_mem_config, out_mem_config, data_seed, dispatch_mode",
    (test_sweep_args),
)
def test_untilize_with_unpadding(
    input_shape_1,
    output_tensor_start,
    output_tensor_end,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    data_seed,
    dispatch_mode,
    device,
):
    random.seed(0)
    run_untilize_with_unpadding(
        input_shape_1,
        output_tensor_start,
        output_tensor_end,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        data_seed,
        dispatch_mode,
        device,
    )
