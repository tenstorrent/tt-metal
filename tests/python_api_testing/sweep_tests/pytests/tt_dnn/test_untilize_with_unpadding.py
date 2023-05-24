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
    pytest.param([[5, 5, 32, 32]], untilize_with_unpadding_args, 0)
    for untilize_with_unpadding_args in generation_funcs.gen_untilize_with_unpadding_args(
        [[5, 5, 32, 32]]
    )
]
params += [
    pytest.param([[5, 5, 64, 96]], untilize_with_unpadding_args, 0)
    for untilize_with_unpadding_args in generation_funcs.gen_untilize_with_unpadding_args(
        [[5, 5, 64, 96]]
    )
]


@pytest.mark.parametrize(
    "input_shapes, untilize_with_unpadding_args, pcie_slot", params
)
def test_run_untilize_with_unpadding_test(
    input_shapes, untilize_with_unpadding_args, pcie_slot
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16, True
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "untilize_with_unpadding",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        untilize_with_unpadding_args,
    )
