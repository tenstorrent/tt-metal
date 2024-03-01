# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_layernorm_residual_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100).to(torch.bfloat16)
    z = torch.Tensor(size=input_shape[2]).uniform_(-100, 100).to(torch.bfloat16)
    w = torch.Tensor(size=input_shape[3]).uniform_(-100, 100).to(torch.bfloat16)
    width = x.shape[1]

    try:
        # get ref result
        ref_value = torch.nn.functional.layer_norm(x + y, normalized_shape=[width], weight=z, bias=w)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[1], in_mem_config[1], dtype[1])
        z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[2], in_mem_config[2], dtype[2])
        w = ttnn_ops.setup_ttnn_tensor(w, device, dlayout[3], in_mem_config[3], dtype[3])

        tt_result = ttnn.layer_norm(x, residual_input_tensor=y, weight=z, bias=w)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, out_mem_config)

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(96, 64), (96, 64), (64,), (64,)],
        [
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ],
        [
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
        ttnn.DRAM_MEMORY_CONFIG,
        7164698,
    ),
    (
        [(224, 64), (224, 64), (64,), (64,)],
        [
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ],
        [
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ],
        ttnn.DRAM_MEMORY_CONFIG,
        16592185,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_layernorm_residual(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_layernorm_residual_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
