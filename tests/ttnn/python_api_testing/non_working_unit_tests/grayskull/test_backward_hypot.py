# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from itertools import product

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_backward_hypot_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    z = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = pytorch_ops.hypot_bw(x, y, z)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[0], dtype[1])
        z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[0], dtype[2])

        tt_result = ttnn.hypot_bw(x, y, z, memory_config=output_mem_config)
        tt_result = [
            ttnn_ops.ttnn_tensor_to_torch(tt_result[0], output_mem_config),
            ttnn_ops.ttnn_tensor_to_torch(tt_result[1], output_mem_config),
        ]

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    for i in range(2):
        assert len(tt_result[i].shape) == len(ref_value[i].shape)
        assert tt_result[i].shape == ref_value[i].shape
        assert_with_pcc(ref_value[i], tt_result[i], 0.99)


test_sweep_args = []

for dtype_1, dtype_2, dtype_3 in list(product([ttnn.bfloat16, ttnn.bfloat8_b], repeat=3)):
    if all([dtype_2 == ttnn.bfloat16, dtype_3 == ttnn.bfloat16]):
        continue
    for shape in [[224, 128], (2, 192, 64), (3, 5, 192, 64)]:
        for mem_cfg in [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]:
            test_sweep_args.append(
                (
                    [shape],
                    [dtype_1, dtype_2, dtype_3],
                    [ttnn.TILE_LAYOUT],
                    [mem_cfg],
                    mem_cfg,
                    13310914,
                )
            )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_hypot(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_hypot_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
