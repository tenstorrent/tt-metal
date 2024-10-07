# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import torch
from pathlib import Path
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_blackhole

shapes = (
    [[1, 1, 32, 32]],  # Single core
    [[1, 4, 320, 384]],  # Multi core h
    [[1, 3, 32, 2048]],  # Multi core h
    [[4, 3, 40, 640]],  # Multi core h
)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("input_shapes", shapes)
@pytest.mark.parametrize("dim", [0, 2, -4, -2, 1, 3])
@pytest.mark.parametrize("repeat", [2, 3, 4])
def test_run_repeat_interleave_test(input_shapes, dim, repeat, device):
    if is_grayskull and dim == 3:
        pytest.skip("Grayskull does not support dim=3 because we cannot tranpose WH reliably")
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100),
            torch.bfloat16,
        )
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"dim": dim, "repeat": repeat})
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        f"repeat_interleave",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
    )


@pytest.mark.parametrize("input_shapes", shapes[:-1])
@pytest.mark.parametrize("repeat", ([1, 2, 3, 4], [1, 1, 1, 2], [1, 1, 2, 1], [1, 2, 1, 1], [2, 1, 1, 1]))
def test_run_repeat_test(input_shapes, repeat, device):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100),
            torch.bfloat16,
        )
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"repeat": repeat})
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        f"repeat",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
    )
