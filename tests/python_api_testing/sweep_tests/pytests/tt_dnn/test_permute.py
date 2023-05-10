import pytest
import sys
import torch
from pathlib import Path
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test

params = [
    pytest.param([[1, 1, 32, 32]], permute_args, 0)
    for permute_args in generation_funcs.gen_permute_args([[1, 1, 32, 32]])
]
params += [
    pytest.param([[32, 32, 32, 32]], permute_args, 0)
    for permute_args in generation_funcs.gen_permute_args([[32, 32, 32, 32]])
]


@pytest.mark.parametrize("input_shapes, permute_args, pcie_slot", params)
def test_permute_test(input_shapes, permute_args, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "permute",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        permute_args,
    )
