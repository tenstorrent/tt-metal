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
import random

# Seed here for fixed params
# TODO @tt-aho: Move random param generation into the function so it's seeded by fixture
random.seed(213919)
torch.manual_seed(213919)

params = [
    pytest.param([[5, 5, 50, 50]], unpad_args, 0)
    for unpad_args in generation_funcs.gen_unpad_args([[5, 5, 50, 50]])
]
params += [
    pytest.param([[5, 5, 64, 96]], unpad_args, 0)
    for unpad_args in generation_funcs.gen_unpad_args([[5, 5, 64, 96]])
]


@pytest.mark.parametrize("input_shapes, unpad_args, pcie_slot", params)
def test_run_unpad_test(input_shapes, unpad_args, pcie_slot):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "unpad",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        unpad_args,
    )
