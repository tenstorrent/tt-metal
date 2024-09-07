# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops, tt_lib_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand, gen_rand_infinite


def run_eltwise_isfinite_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = gen_rand_infinite(size=input_shape, low=-10, high=10)

    # compute ref value
    ref_value = pytorch_ops.isfinite(x=x)

    tt_result = tt_lib_ops.eltwise_isfinite(
        x=x,
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


test_sweep_args = []
random.seed(0)


def make_in_mem_config(buffer_type):
    if buffer_type is None:
        return None

    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)


for memorylayout in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
    for shape in [(4, 7, 32, 96), (6, 7, 192, 224)]:
        for bufertype in [ttnn.BufferType.DRAM, ttnn.BufferType.L1, None]:
            test_sweep_args.append(
                (
                    shape,
                    [ttnn.bfloat16],
                    [memorylayout],
                    [make_in_mem_config(bufertype)],
                    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                    random.randint(1000000, 10000000),
                )
            )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_isfinite(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_isfinite_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
