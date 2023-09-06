import pytest
import sys
import torch
import tt_lib as ttl
from pathlib import Path
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize(
    "input_shapes",
    ([[1, 1, 32, 64], [1, 1, 256, 64]], [[1, 1, 64, 128], [1, 1, 32, 128]]),
)
@pytest.mark.parametrize("pcie_slot", (0,))
def test_linear_no_bias(input_shapes, pcie_slot):
    comparison_func = partial(comparison_funcs.comp_pcc)

    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
        )
    ]
    weight = torch.randn(input_shapes[1])
    run_single_pytorch_test(
        "linear",
        [
            input_shapes[0],
        ],
        datagen_func,
        comparison_func,
        pcie_slot,
        {
            "weight": weight,
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
            "dtype": [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
            "buffer_type": [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM],
            "output_mem_config": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            "bias": None,
        },
    )


@pytest.mark.parametrize(
    "input_shapes",
    (
        [[1, 1, 32, 64], [1, 1, 256, 64], [1, 1, 1, 256]],
        [[1, 1, 64, 128], [1, 1, 32, 128], [1, 1, 1, 32]],
    ),
)
@pytest.mark.parametrize("pcie_slot", (0,))
def test_linear_with_bias(input_shapes, pcie_slot):
    comparison_func = partial(comparison_funcs.comp_pcc)

    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
        )
    ]
    weight = torch.randn(input_shapes[1])
    bias = torch.randn(input_shapes[2])
    run_single_pytorch_test(
        "linear",
        [
            input_shapes[0],
        ],
        datagen_func,
        comparison_func,
        pcie_slot,
        {
            "weight": weight,
            "bias": bias,
            "layout": [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.ROW_MAJOR],
            "dtype": [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
            "buffer_type": [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM],
            "output_mem_config": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        },
    )
