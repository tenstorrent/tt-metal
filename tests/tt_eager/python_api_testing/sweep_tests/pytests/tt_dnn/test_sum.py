# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[2, 2, 32, 64]],  # Single core
        [[4, 2, 320, 384]],  # Multi core
        [[8, 6, 320, 384]],  # Multi core
    ],
)
class TestSum:
    @pytest.mark.parametrize("fn_kind", ["sum-3", "sum-2", "sum-1", "sum-0"])
    def test_run_sum_ops(self, input_shapes, fn_kind, device, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"input_shapes": input_shapes})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
    ],
)
class TestSimpleSum:
    @pytest.mark.parametrize(
        "fn_kind",
        [
            "sum-3",
        ],
    )
    def test_run_sum_ops(self, input_shapes, fn_kind, device, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"input_shapes": input_shapes})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
