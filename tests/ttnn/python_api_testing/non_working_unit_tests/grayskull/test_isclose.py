# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


def run_isclose_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    rtol,
    atol,
    equal_nan,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.randn(input_shape[0])
    y = torch.Tensor(size=input_shape[1]).uniform_(-100, 100)

    try:
        # get ref result
        ref_value = torch.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

        t0 = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        t1 = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[1], in_mem_config[1], dtype[1])

        t2 = ttnn.isclose(t0, t1, rtol=rtol, atol=atol, equal_nan=equal_nan, memory_config=output_mem_config)

        tt_result = ttnn_ops.ttnn_tensor_to_torch(t2, output_mem_config)

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
        [(1, 1, 224, 128), (1, 1, 224, 128)],
        [ttnn.bfloat16, ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        1.0011717677116394e-07,
        9.968061931431293e-10,
        False,
        8687804,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, rtol, atol, equal_nan, data_seed",
    (test_sweep_args),
)
def test_isclose(input_shape, dtype, dlayout, in_mem_config, out_mem_config, rtol, atol, equal_nan, data_seed, device):
    run_isclose_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, rtol, atol, equal_nan, data_seed, device
    )
