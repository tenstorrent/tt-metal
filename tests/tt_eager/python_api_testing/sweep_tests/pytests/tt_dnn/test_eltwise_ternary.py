# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from math import pi


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from models.utility_functions import is_wormhole_b0

shapes = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384]],  # Multi core
)
if is_wormhole_b0():
    shapes = (shapes[0],)


@pytest.mark.parametrize("input_shapes", shapes)
def test_run_eltwise_where_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_randint, low=-100, high=+100), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-5, high=+5), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=+10), torch.float32),
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-where",
        [input_shapes[0], input_shapes[0], input_shapes[0]],
        datagen_func,
        comparison_func,
        device,
    )
