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
from models.common.utility_functions import is_grayskull

mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize("accurate_mode", [True])
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32], [1, 1, 32, 32]],
        [[1, 1, 320, 384], [1, 1, 320, 384]],
        [[1, 3, 320, 384], [1, 3, 320, 384]],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    {random.uniform(-100, 100) for _ in range(3)},
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestUnary_Div:
    def test_run_unary_div(
        self,
        accurate_mode,
        round_mode,
        input_shapes,
        scalar,
        dst_mem_config,
        device,
    ):
        if is_grayskull():
            if round_mode in ["trunc", "floor"]:
                pytest.skip("does not work for Grayskull -skipping")
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-1e6, high=1e6), torch.bfloat16)
        ] * 2
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "accurate_mode": accurate_mode,
                "round_mode": round_mode,
                "scalar": scalar,
            }
        )
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "eltwise-unary_div",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
