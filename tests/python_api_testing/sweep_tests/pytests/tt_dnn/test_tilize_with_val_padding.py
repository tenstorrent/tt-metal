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
    pytest.param([[5, 5, 50, 50]], tilize_with_val_padding_args, 0)
    for tilize_with_val_padding_args in generation_funcs.gen_tilize_with_val_padding_args(
        [[5, 5, 50, 50]]
    )
]
params += [
    pytest.param([[5, 5, 64, 96]], tilize_with_val_padding_args, 0)
    for tilize_with_val_padding_args in generation_funcs.gen_tilize_with_val_padding_args(
        [[5, 5, 64, 96]]
    )
]


@pytest.mark.parametrize(
    "input_shapes, tilize_with_val_padding_args, pcie_slot", params
)
def test_run_tilize_with_val_padding_test(
    input_shapes, tilize_with_val_padding_args, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "tilize_with_val_padding",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        tilize_with_val_padding_args,
    )
