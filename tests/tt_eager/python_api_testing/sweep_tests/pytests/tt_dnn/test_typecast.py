# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import ttnn


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype",
    (
        (torch.float16, ttnn.float32),
        (torch.float32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.int, ttnn.uint32),
    ),
)
@pytest.mark.parametrize(
    "pt_output_dtype, tt_output_dtype",
    (
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.bfloat8_b),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 320]],  # multi core
        [[1, 3, 320, 320]],  # multi core
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 384]],  # Multi core
        [[1, 3, 320, 384]],  # Multi core
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
class TestTypecast:
    def test_run_typecast_op(
        self,
        pt_output_dtype,
        tt_output_dtype,
        pt_input_dtype,
        tt_input_dtype,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        if tt_input_dtype in [ttnn.float32, ttnn.uint32]:
            pytest.skip(f"{tt_input_dtype} cannot be converted yet. Skip")
        if tt_input_dtype == tt_output_dtype:
            pytest.skip("Same I/O data types. Skip.")
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=10), pt_input_dtype)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["pt_input_dtype"] = [pt_input_dtype]
        test_args["tt_input_dtype"] = [tt_input_dtype]
        test_args["pt_output_dtype"] = [pt_output_dtype]
        test_args["tt_output_dtype"] = [tt_output_dtype]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "typecast",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
