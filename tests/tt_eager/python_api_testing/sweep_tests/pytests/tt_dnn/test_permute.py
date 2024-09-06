# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from models.utility_functions import skip_for_blackhole

params = [
    pytest.param([[1, 1, 32, 32]], permute_args) for permute_args in generation_funcs.gen_permute_args([[1, 1, 32, 32]])
]
params += [
    pytest.param([[32, 32, 32, 32]], permute_args)
    for permute_args in generation_funcs.gen_permute_args([[32, 32, 32, 32]])
]


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("input_shapes, permute_args", params)
def test_run_permute_test(input_shapes, permute_args, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "permute",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        permute_args,
    )
