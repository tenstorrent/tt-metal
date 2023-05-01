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
    pytest.param([[10, 10, 32, 32]], reshape_dims["reshape_dims"], 0)
    for reshape_dims in generation_funcs.gen_reshape_args([[10, 10, 32, 32]])
]
params += [
    pytest.param([[10, 10, 32, 32]], [-1, 5, 32, 32], 0),
    pytest.param([[10, 10, 32, 32]], [-1, 5, 32, 64], 0),
    pytest.param([[10, 10, 32, 32]], [5, -1, 32, 32], 0),
    pytest.param([[10, 10, 32, 32]], [5, -1, 32, 64], 0),
    pytest.param([[10, 10, 32, 32]], [10, 5, -1, 32], 0),
    pytest.param([[10, 10, 32, 32]], [10, 5, -1, 64], 0),
    pytest.param([[10, 10, 32, 32]], [10, 10, 32, -1], 0),
    pytest.param([[10, 10, 32, 32]], [10, 5, 32, -1], 0),
]


@pytest.mark.parametrize("input_shapes, reshape_dims, pcie_slot", params)
def test_reshape_test(input_shapes, reshape_dims, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "reshape",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        {"reshape_dims": reshape_dims},
    )
