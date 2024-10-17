# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal_print_error_value
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_singbit_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-1, 0)  # Torch might generate values in RM Layout

    try:
        # get ref result
        ref_value = torch.signbit(x)
        torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=128)
        input_torch = x
        print("\nTorch Input : ", x)

        x = ttnn.Tensor(x, dtype[0]).to(dlayout[0]).to(device)  # Convert it to required layout
        ttnn.set_printoptions(profile="full")
        y = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Convert it back to RM layout
        print("\nTT Input : ", y)
        tt_result = ttnn.signbit(x, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=128)
    print("Torch value : ", ref_value)
    print("TT Result   : ", tt_result)
    print("\n Details of the values causing assertion issue")
    success, pcc_value = comp_equal_print_error_value(ref_value, tt_result, input_torch, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        # [(32, 32)],
        [(224, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        17155532,
    ),
    # (
    #     [(6, 160, 64)],
    #     [ttnn.bfloat8_b],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.DRAM_MEMORY_CONFIG],
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     9456908,
    # ),
    # (
    #     [(6, 160, 64)],
    #     [ttnn.bfloat8_b],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.L1_MEMORY_CONFIG],
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     4689090,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_signbit(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_singbit_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
