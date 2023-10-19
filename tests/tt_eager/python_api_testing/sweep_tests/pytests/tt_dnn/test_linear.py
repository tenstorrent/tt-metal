# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import (
    is_wormhole_b0,
)

shapes  = [
    [[31, 4096] ,  [4096, 4096]],
    [[31, 4096] ,  [1024, 4096]],
    [[31, 4096] ,  [14336, 4096]],
    [[31, 14336],  [4096, 14336]],
    [[31, 4096] ,  [32000, 4096]],
    [[1, 4096] ,  [4096, 4096]],
    [[1, 4096] ,  [1024, 4096]],
    [[1, 4096] ,  [14336, 4096]],
    [[1, 14336],  [4096, 14336]],
    [[1, 4096] ,  [32000, 4096]],
]

@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT16", "BFLOAT16"],
)
def test_run_linear_test(input_shapes, device, dtype, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_with_pad, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "linear",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            "input_mem_config": [ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)] * 2,
            "output_mem_config": ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            ),
        },
    )
