# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_softmax_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    scalar,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-1, 1).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[1]).uniform_(-1, 1).to(torch.bfloat16)
    y[y <= 0.50] = 0
    y[y > 0.50] = 1
    if scalar < 0:
        scalar = -scalar

    try:
        # get ref result
        ref_value = pytorch_ops.attention_softmax(x, y, scalar=scalar)
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config, dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config, dtype[0])

        tt_result = ttnn.transformer.attention_softmax(input_tensor=x, attention_mask=y, head_size=scalar)

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(1, 1, 224, 32), (1, 1, 32, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        (ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG),
        18139523,
        -51.5,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_softmax(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    run_softmax_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device)
