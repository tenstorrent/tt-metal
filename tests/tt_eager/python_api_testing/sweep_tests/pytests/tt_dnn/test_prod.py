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

mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize(
    "dim",
    (3, 2, 1, 0, -1, -2, -3, -4),
)
@pytest.mark.parametrize("all_dimensions", [False, True])
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],
        [[4, 3, 32, 32]],
        [[2, 2, 32, 32]],
        # [[6, 4, 32, 32]], #Fails for all_dimensions = True ( expected result is inf but the result generated in nan )
        # [[1, 1, 320, 320]], #Fails for all_dimensions = True ( expected result is inf but the result generated in nan )
        # [[1, 3, 320, 64]], #Fails for all_dimensions = True ( expected result is inf but the result generated in nan )
    ],
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestProd:
    def test_run_prod_op(
        self,
        all_dimensions,
        dim,
        input_shapes,
        dst_mem_config,
        device,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=1, high=1.5), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "all_dimensions": all_dimensions,
                "dim": dim,
            }
        )
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "prod",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
