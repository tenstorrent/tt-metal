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
params = [
    pytest.param([[4, 4, 32, 32]], reshape_args, 0)
    for reshape_args in generation_funcs.gen_reshape_args([[4, 4, 32, 32]])
]
params += [
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [-1, 2, 32, 32]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [-1, 2, 32, 64]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [2, -1, 32, 32]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [2, -1, 32, 64]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [4, 2, -1, 32]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [4, 2, -1, 64]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [4, 4, 32, -1]}, 0),
    pytest.param([[4, 4, 32, 32]], {"dtype": ttl.tensor.DataType.BFLOAT16, "layout": ttl.tensor.Layout.TILE, "on_device": True, "reshape_dims": [4, 2, 32, -1]}, 0),
]


@pytest.mark.parametrize("input_shapes, reshape_args, pcie_slot", params)
def test_reshape_test(input_shapes, reshape_args, pcie_slot, function_level_defaults):
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
        reshape_args,
    )
