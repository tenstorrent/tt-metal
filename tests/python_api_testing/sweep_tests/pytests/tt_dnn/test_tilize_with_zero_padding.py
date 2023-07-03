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
import tt_lib as ttl


@pytest.mark.parametrize(
    "input_shapes",
    (([[1, 1, 30, 32]], [[3, 1, 315, 384]], [[1, 1, 100, 7104]])),
)
@pytest.mark.parametrize(
    "tilize_with_zero_padding_args",
    (
        {
            "dtype": ttl.tensor.DataType.BFLOAT16,
            "on_device": True,
            "layout": ttl.tensor.Layout.ROW_MAJOR,
        },
    ),
)
@pytest.mark.parametrize("pcie_slot", ((0,)))
def test_tilize_with_zero_padding_test(
    input_shapes, tilize_with_zero_padding_args, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "tilize_with_zero_padding",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        tilize_with_zero_padding_args,
    )
