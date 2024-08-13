# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import ttnn.deprecated as ttl
from models.utility_functions import skip_for_grayskull


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

mem_configs = [
    ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    ),
    ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
    ),
]


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    (
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.FLOAT32),
        (torch.float32, ttnn.experimental.tensor.DataType.FLOAT32, ttnn.experimental.tensor.DataType.BFLOAT16),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.INT32),
        (torch.int, ttnn.experimental.tensor.DataType.INT32, ttnn.experimental.tensor.DataType.BFLOAT16),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.UINT16),
        (torch.int, ttnn.experimental.tensor.DataType.UINT16, ttnn.experimental.tensor.DataType.BFLOAT16),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.UINT32),
        (torch.int, ttnn.experimental.tensor.DataType.UINT32, ttnn.experimental.tensor.DataType.BFLOAT16),
        (torch.float32, ttnn.experimental.tensor.DataType.FLOAT32, ttnn.experimental.tensor.DataType.INT32),
        (torch.int, ttnn.experimental.tensor.DataType.INT32, ttnn.experimental.tensor.DataType.FLOAT32),
        (torch.float32, ttnn.experimental.tensor.DataType.FLOAT32, ttnn.experimental.tensor.DataType.UINT16),
        (torch.int, ttnn.experimental.tensor.DataType.UINT16, ttnn.experimental.tensor.DataType.FLOAT32),
        (torch.float32, ttnn.experimental.tensor.DataType.FLOAT32, ttnn.experimental.tensor.DataType.UINT32),
        (torch.int, ttnn.experimental.tensor.DataType.UINT32, ttnn.experimental.tensor.DataType.FLOAT32),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT8_B, ttnn.experimental.tensor.DataType.INT32),
        (torch.int, ttnn.experimental.tensor.DataType.INT32, ttnn.experimental.tensor.DataType.BFLOAT8_B),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT8_B, ttnn.experimental.tensor.DataType.UINT16),
        (torch.int, ttnn.experimental.tensor.DataType.UINT16, ttnn.experimental.tensor.DataType.BFLOAT8_B),
        (torch.bfloat16, ttnn.experimental.tensor.DataType.BFLOAT8_B, ttnn.experimental.tensor.DataType.UINT32),
        (torch.int, ttnn.experimental.tensor.DataType.UINT32, ttnn.experimental.tensor.DataType.BFLOAT8_B),
        (torch.int, ttnn.experimental.tensor.DataType.UINT16, ttnn.experimental.tensor.DataType.UINT32),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 320]],  # multi core
        [[1, 3, 320, 320]],  # multi core
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
class TestTypecast:
    def test_run_eltwise_typecast_op(
        self,
        tt_output_dtype,
        pt_input_dtype,
        tt_input_dtype,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        if tt_input_dtype == tt_output_dtype:
            pytest.skip("Same I/O data types. Skip.")
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=100), pt_input_dtype)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["tt_input_dtype"] = [tt_input_dtype]
        test_args["tt_output_dtype"] = [tt_output_dtype]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "eltwise-typecast",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
