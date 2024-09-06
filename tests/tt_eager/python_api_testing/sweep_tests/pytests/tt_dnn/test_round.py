# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
from functools import partial
import ttnn
from models.utility_functions import skip_for_grayskull, skip_for_blackhole

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


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "decimals",
    [0],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],
        [[4, 3, 32, 32]],
        [[2, 2, 32, 32]],
        [[6, 4, 32, 32]],
        [[1, 1, 320, 320]],
        [[1, 3, 320, 64]],
    ],
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for Floor")
class TestRound:
    def test_run_round(
        self,
        decimals,
        input_shapes,
        dst_mem_config,
        device,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "decimals": decimals,
            }
        )
        test_args.update({"output_mem_config": dst_mem_config})
        if decimals == 0:
            comparison_func = comparison_funcs.comp_equal
        else:
            comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "eltwise-round",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
