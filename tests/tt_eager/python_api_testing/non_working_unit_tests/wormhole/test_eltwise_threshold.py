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
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_threshold as tt_eltwise_threshold
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_eltwise_threshold_tests(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, threshold, value, device
):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = gen_rand(size=input_shape, low=-100, high=100).to(torch.bfloat16)
    # compute ref value
    x_ref = x.detach().clone()
    ref_value = pytorch_ops.threshold(x_ref, threshold=threshold, value=value)

    tt_result = tt_eltwise_threshold(
        x=x,
        threshold=threshold,
        value=value,
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


# 71.0	-16.375	4133507	(('TT_METAL_SLOW_DISPATCH_MODE', '1'),)	completed	Max ATOL Delta: 99.0, Max RTOL Delta: inf, PCC: 0.9968988009373057, Equal check failed	fail	NA	NA	NA	Details
# -56.25	-14.0625	6978585	(('TT_METAL_SLOW_DISPATCH_MODE', '1'),)	completed	Max ATOL Delta: 0.0, Max RTOL Delta: 0.0, PCC: 1.0	pass	NA	NA	NA	Details
# 71.0	1.5625	108451	(('TT_METAL_SLOW_DISPATCH_MODE', '1'),)	completed	Max ATOL Delta: 0.0, Max RTOL Delta: 0.0, PCC: 1.0	pass	NA	NA	NA	Details

test_sweep_args = [
    (
        (2, 20, 458, 74),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        6978585,
        -56.25,
        -14.0625,
    ),
    (
        (2, 20, 458, 74),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        108451,
        71.0,
        1.5625,
    ),
    (
        (4, 22, 346, 376),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        4133507,
        71.0,
        -16.375,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, threshold, value",
    (test_sweep_args),
)
def test_eltwise_threshold_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, threshold, value, device
):
    random.seed(0)
    run_eltwise_threshold_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, threshold, value, device
    )
