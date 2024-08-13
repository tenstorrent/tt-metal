# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn.deprecated as ttl


@pytest.mark.parametrize(
    "input_shapes",
    (([[1, 1, 32, 32]], [[3, 1, 320, 384]], [[1, 1, 128, 1856]])),
)
@pytest.mark.parametrize(
    "untilize_args",
    (
        {
            "dtype": [ttnn.experimental.tensor.DataType.BFLOAT16],
            "layout": [ttnn.experimental.tensor.Layout.TILE],
            "input_mem_config": [
                ttnn.experimental.tensor.MemoryConfig(
                    ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
                )
            ],
            "output_mem_config": ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
            ),
        },
    ),
)
def test_untilize_test(input_shapes, untilize_args, device, function_level_defaults):
    datagen_func = [generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_arange), torch.bfloat16, True)]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "untilize",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        untilize_args,
    )
