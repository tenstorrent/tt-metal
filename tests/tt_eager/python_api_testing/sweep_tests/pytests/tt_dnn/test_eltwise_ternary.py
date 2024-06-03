# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
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


@pytest.mark.parametrize("input_shapes", shapes)
def test_run_eltwise_where_test_optional(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_randint, low=-100, high=+100), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-5, high=+5), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-10, high=+10), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-1, high=+1), torch.float32),
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-where-optional",
        [input_shapes[0], input_shapes[0], input_shapes[0], input_shapes[0]],
        datagen_func,
        comparison_func,
        device,
    )


shapes_scalar = (
    [[1, 1, 32, 32], [1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384], [1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384], [1, 3, 320, 384]],  # Multi core
)


@pytest.mark.parametrize("input_shapes", shapes_scalar)
def test_run_eltwise_where_scalar_optional(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_randint, low=-100, high=+100), torch.float32),
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-1, high=+1), torch.float32),
    ]
    test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
    test_args.update({"scalar_true": random.uniform(0.5, 75.5), "scalar_false": random.uniform(0.5, 95.5)})

    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-where-scalar-optional",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
    )
