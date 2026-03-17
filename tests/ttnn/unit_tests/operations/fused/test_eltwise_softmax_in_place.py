# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.ttnn.unit_tests.operations.reduce.numeric_check import (
    collect_and_dump_numeric_metrics,
)


def run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = torch.softmax(x_ref, -1)

    tt_result = ttnn_ops.eltwise_softmax_in_place(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=None,
        numeric_stable=True,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_eltwise_softmax_in_place_test[input_shape={input_shape},dtype={dtype},dlayout={dlayout},in_mem_config={in_mem_config},data_seed={data_seed}]"
    collect_and_dump_numeric_metrics(
        ref_value,
        tt_result,
        test_name=test_name,
        csv_filename="test_eltwise_softmax_in_place_numeric_results.csv",
        test_params=None,
    )

    assert success


test_sweep_args = [
    (
        (1, 9, 32, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        38346,
    ),
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        17155532,
    ),
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        16305027,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_softmax_in_place_test(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device)
