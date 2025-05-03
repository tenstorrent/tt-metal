# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull


def run_eltwise_softplus_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    beta,
    threshold,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)

    try:
        # get ref result
        ref_value = torch.nn.functional.softplus(x, beta=beta, threshold=threshold)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        tt_result = ttnn.softplus(x, beta=beta, threshold=threshold, memory_config=output_mem_config)

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

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
        [(6, 6, 192, 224)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        0.0,
        28.125,
        19042500,
    ),
]


@skip_for_grayskull("Softplus is not available in Grayskull")
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, beta, threshold, data_seed",
    (test_sweep_args),
)
def test_eltwise_softplus(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, beta, threshold, data_seed, device
):
    run_eltwise_softplus_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, beta, threshold, data_seed, device
    )
