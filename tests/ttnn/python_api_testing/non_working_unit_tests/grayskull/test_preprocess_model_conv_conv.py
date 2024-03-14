# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

# from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops

from tests.tt_eager.python_api_testing.sweep_tests.model_tests import TorchConvConv, TTNNConvConv, run_conv_conv
from ttnn.model_preprocessing import preprocess_model


def run_preprocessing_model_conv_conv_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.float32)

    try:
        # get ref result
        torch_model = TorchConvConv()
        torch_model.eval()
        ref_value = torch_model(x)

        # get model parameters
        reader_patterns_cache = {}
        parameters = preprocess_model(
            initialize_model=lambda: torch_model,
            run_model=lambda model: model(x),
            reader_patterns_cache=reader_patterns_cache,
            device=device,
        )

        # create and run TTNN model
        ttnn_model = TTNNConvConv(parameters)
        output_tensor = run_conv_conv(ttnn_model, x)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


# ,16934480,(),error,"TT_THROW @ tt_metal/impl/program/program.cpp:492: tt::exception

test_sweep_args = [
    (
        [(4, 64, 160, 64)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
    # (
    #     [(3, 10, 192, 64)],
    #     [ttnn.bfloat16],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.DRAM_MEMORY_CONFIG],
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     (2,),
    #     18539618,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_preprocessing_model_conv_conv(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_preprocessing_model_conv_conv_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
    )
