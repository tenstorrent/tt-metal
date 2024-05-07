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
    "dst_mem_config",
    mem_configs,
)
class TestDiv_No_Nan:
    def test_run_div_no_nan(
        self,
        input_shapes,
        dst_mem_config,
        device,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] + [generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=-1), torch.bfloat16)]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc

        run_single_pytorch_test(
            "eltwise-div_no_nan",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
