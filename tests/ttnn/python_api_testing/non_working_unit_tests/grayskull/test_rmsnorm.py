# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops, ttnn_pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_rmsnorm_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape[1]).uniform_(-200, 200)

    try:
        # get ref result
        ref_value = ttnn_pytorch_ops.rmsnorm(x, y, epsilon=1e-12)

        tt_result = ttnn_ops.rmsnorm(
            x,
            y,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(224, 128), (224, 128)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
    ),
    (
        [(8, 224, 160), (8, 224, 160)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
    ),
    (
        [(6, 224, 32), (6, 224, 32)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_rmsnorm(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_rmsnorm_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
