# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops, ttnn_pytorch_ops


def run_transformer_concat_heads_tests(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = ttnn_pytorch_ops.transformer_concatenate_heads(x)

        tt_result = ttnn_ops.transformer_concatenate_heads(
            x,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(4, 7, 21, 133)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4171614,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed",
    (test_sweep_args),
)
def test_transformer_concat_heads(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    run_transformer_concat_heads_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device)
