# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_sinh_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-88, 88).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.sinh(x)

        tt_result = ttnn_ops.eltwise_sinh(
            x,
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
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = []

for shape in [(192, 224), (6, 6, 192, 224), (3, 2, 192, 32), (4, 231, 174), (176, 102)]:
    for dtype in [ttnn.bfloat16, ttnn.bfloat8_b]:
        for layout in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
            for mem_cfg in [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]:
                if layout == ttnn.ROW_MAJOR_LAYOUT or dtype == ttnn.bfloat8_b:
                    continue

                test_sweep_args.append(
                    (
                        [shape],
                        [dtype],
                        [layout],
                        [mem_cfg],
                        mem_cfg,
                        2474385,
                    )
                )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_sinh(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_sinh_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
