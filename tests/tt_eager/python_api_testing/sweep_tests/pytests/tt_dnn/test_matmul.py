# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn.deprecated as ttl
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0

shapes_mm = [
    # Single core (won't be hit after padding is added for multicast)
    [[1, 1, 32, 32], [1, 1, 32, 32]],
    # Multi core (2% math util)
    [[1, 2, 320, 1024], [1, 1, 1024, 384]],
    # Multi core reuse (25% math util)
    [[1, 2, 512, 1024], [1, 1, 1024, 512]],
    # Multi core reuse multicast in0/in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 1, 1024, 6144]],
    # Multi core reuse multicast in0 (25% math util)
    [[1, 2, 512, 1024], [1, 1, 1024, 6144]],
    # Multi core reuse multicast in1 (25% math util)
    [[1, 2, 4608, 1024], [1, 1, 1024, 512]],
    # Multi core reuse with padding (?% math util)
    [[1, 2, 480, 1024], [1, 1, 1024, 480]],
    # Multi core reuse multicast in0/in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 1, 1024, 6112]],
    [[1, 2, 4416, 1024], [1, 1, 1024, 6048]],
    # Multi core reuse multicast in0 with padding (?% math util)
    [[1, 2, 480, 1024], [1, 1, 1024, 6112]],
    [[1, 2, 320, 1024], [1, 1, 1024, 6048]],
    # Multi core reuse multicast in1 with padding (?% math util)
    [[1, 2, 4576, 1024], [1, 1, 1024, 480]],
    [[1, 2, 4416, 1024], [1, 1, 1024, 320]],
]

if is_wormhole_b0():
    del shapes_mm[1:]


@pytest.mark.parametrize("input_shapes", shapes_mm)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.BFLOAT8_B),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
def test_run_matmul_test(input_shapes, device, dtype, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "matmul",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        {
            "dtype": [dtype, dtype],
            "layout": [ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.Layout.TILE],
            "input_mem_config": [
                ttnn.experimental.tensor.MemoryConfig(
                    ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
                )
            ]
            * 2,
            "output_mem_config": ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
            ),
        },
    )
