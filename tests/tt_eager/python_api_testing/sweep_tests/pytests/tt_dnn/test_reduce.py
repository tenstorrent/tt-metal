# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

shapes = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 32, 3840]],  # Multi core h
    [[1, 3, 32, 3840]],  # Multi core h
)


@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize("minmax", ["min", "max"])
def test_run_reduce_max_h_test(input_shapes, minmax, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100, dim=-2),
            torch.bfloat16,
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        f"reduce-{minmax}-h",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )


shapes2 = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 3840, 32]],  # Multi core w
    [[1, 3, 3840, 32]],  # Multi core w
)


@pytest.mark.parametrize("input_shapes", shapes2)
@pytest.mark.parametrize("minmax", ["min", "max"])
def test_run_reduce_max_w_test(input_shapes, minmax, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100, dim=-1),
            torch.bfloat16,
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        f"reduce-{minmax}-w",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )


shapes3 = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 512, 512]],  # Multi core hw (== multi core w + multi core h)
    [[1, 3, 512, 512]],  # Multi core hw (== multi core w + multi core h)
)


@pytest.mark.parametrize(
    "input_shapes",
    shapes3,
)
@pytest.mark.parametrize("minmax", ["min", "max"])
def test_run_reduce_max_hw_test(input_shapes, minmax, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        f"reduce-{minmax}-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )


shapes5 = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 32, 3840]],  # Multi core h
    [[1, 1, 32, 3840]],  # Multi core h
)


@pytest.mark.parametrize("input_shapes", shapes5)
def test_run_reduce_sum_h_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32)
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-h",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )


shapes4 = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 3840, 32]],  # Multi core w
    [[1, 3, 3840, 32]],  # Multi core w
)


@pytest.mark.parametrize("input_shapes", shapes4)
def test_run_reduce_sum_w_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32)
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-w",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )


shapes5 = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 1, 512, 512]],  # Multi core hw (== multi core w + multi core h)
    [[1, 1, 512, 512]],  # Multi core hw (== multi core w + multi core h)
)


@pytest.mark.parametrize("input_shapes", shapes5)
def test_run_reduce_sum_hw_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32)
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
    )
