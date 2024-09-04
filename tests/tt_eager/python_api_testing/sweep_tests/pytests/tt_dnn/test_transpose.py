# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test

shape_wh = [
    [[1, 1, 32, 32]],  # Single core
    [[3, 1, 320, 384]],  # Multi core
]


@pytest.mark.parametrize("input_shapes", shape_wh)
def test_run_transpose_wh_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": -2,
            "dim1": -1,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[1, 32, 32, 32]],  # Single core
        [[3, 320, 384, 32]],  # Multi core
    ),
)
def test_run_transpose_hc_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": 1,
            "dim1": -2,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)


shape_cn = [
    [[1, 1, 32, 32]],  # Single core
    [[3, 5, 384, 96]],  # Single core
]


@pytest.mark.parametrize("input_shapes", shape_cn)
def test_run_transpose_cn_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": 0,
            "dim1": 1,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[32, 1, 32, 32]],  # Single core
        [[32, 3, 384, 96]],  # Single core
    ),
)
def test_run_transpose_nh_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": 0,
            "dim1": -2,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[32, 1, 32, 32]],  # Single core
        [[32, 3, 384, 96]],  # Single core
    ),
)
def test_run_transpose_nw_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": 0,
            "dim1": -1,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[1, 64, 32, 32]],  # Single core
        [[3, 64, 384, 96]],  # Single core
    ),
)
def test_run_transpose_cw_test(input_shapes, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "dim0": 1,
            "dim1": -1,
        }
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test("transpose", input_shapes, datagen_func, comparison_func, device, test_args)
