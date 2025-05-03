# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
from functools import partial
import ttnn


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import skip_for_grayskull

mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],
        [[4, 3, 32, 32]],
        [[2, 2, 32, 32]],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) for _ in range(5)},
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
@skip_for_grayskull("#TODO: GS implementation needs to be done")
class TestBitwiseAnd:
    def test_run_bitwise_and_op(
        self,
        input_shapes,
        scalar,
        dst_mem_config,
        device,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=2147483647), torch.int)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "value": scalar,
                "dtype": [(ttnn.int32)],
            }
        )
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_equal

        run_single_pytorch_test(
            "bitwise-and", input_shapes, datagen_func, comparison_func, device, test_args, ttnn_op=True
        )
