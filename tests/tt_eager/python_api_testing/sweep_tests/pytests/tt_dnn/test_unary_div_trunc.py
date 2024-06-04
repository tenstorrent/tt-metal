# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import random
from functools import partial
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import skip_for_grayskull

mem_configs = [
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
]


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32], [1, 1, 32, 32]],
        [[1, 1, 320, 384], [1, 1, 320, 384]],
        [[1, 3, 320, 384], [1, 3, 320, 384]],
    ],
)
@pytest.mark.parametrize(
    "value",
    [-5.1, 0.0, 10.9],
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for floor")
class TestUnary_Div_Trunc:
    def test_run_unary_div_trunc(
        self,
        input_shapes,
        value,
        dst_mem_config,
        device,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"value": value})
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-unary_div_trunc",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
