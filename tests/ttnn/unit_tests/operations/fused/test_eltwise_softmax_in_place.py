# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Numeric thresholds from tests/ttnn/unit_tests/operations/fused/all_numeric_results_fused.csv
# (test_eltwise_softmax_in_place_numeric_results): PCC margin 1.5e-4 from min observed;
# atol / Frobenius = ceil(max * 1.1, 3 dp); rtol = max(max_rel where max_rel < 10) * 1.1, 3 dp;
# check_ulp when ceil(max_ulp * 1.1) < 12 (not met for current CSV).

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


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
    assert_numeric_metrics(
        ref_value,
        tt_result,
        pcc_threshold=0.998,
        rtol=0.741,
        atol=0.138,
        frobenius_threshold=0.068,
    )


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
