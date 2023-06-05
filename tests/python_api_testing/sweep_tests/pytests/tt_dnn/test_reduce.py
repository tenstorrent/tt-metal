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


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 32, 3840]], 0),  # Multi core h
        ([[1, 3, 32, 3840]], 0),  # Multi core h
    ),
)
def test_run_reduce_max_h_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100, dim=-2),
            torch.bfloat16,
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "reduce-max-h",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 3840, 32]], 0),  # Multi core w
        ([[1, 3, 3840, 32]], 0),  # Multi core w
    ),
)
def test_run_reduce_max_w_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_along_dim, low=-100, high=100, dim=-1),
            torch.bfloat16,
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "reduce-max-w",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 512, 512]], 0),  # Multi core hw (== multi core w + multi core h)
        ([[1, 3, 512, 512]], 0),  # Multi core hw (== multi core w + multi core h)
    ),
)
def test_run_reduce_max_hw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "reduce-max-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 32, 3840]], 0),  # Multi core h
        ([[1, 1, 32, 3840]], 0),  # Multi core h
    ),
)
def test_run_reduce_sum_h_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-h",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 3840, 32]], 0),  # Multi core w
        ([[1, 3, 3840, 32]], 0),  # Multi core w
    ),
)
def test_run_reduce_sum_w_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-w",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 512, 512]], 0),  # Multi core hw (== multi core w + multi core h)
        ([[1, 1, 512, 512]], 0),  # Multi core hw (== multi core w + multi core h)
    ),
)
def test_run_reduce_sum_hw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_checkerboard, low=0, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_allclose, atol=0.1, rtol=0)
    run_single_pytorch_test(
        "reduce-sum-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
