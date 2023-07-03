import pytest
import sys
import torch
from pathlib import Path
from functools import partial
from itertools import product

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
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384], [1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_add_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-add",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384], [1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_sub_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-sub",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384], [1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_min_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-min",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384], [1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_max_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-max",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384], [1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_mul_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-mul",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "cmp_kind, input_shapes, pcie_slot",
    (
        list(
            product(
                ("lt", "gt", "lte", "gte", "ne", "eq"),
                (
                    [[1, 1, 32, 32], [1, 1, 32, 32]],
                    [[1, 1, 64, 32], [1, 1, 64, 32]],
                    [[1, 1, 320, 384],[1, 1, 320, 384]],
                ),  # single, multi core sizes
                (0,),
            )
        )
    ),
)
def test_run_eltwise_cmp_test(
    cmp_kind, input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_randint, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    #if cmp_kind == 'eq':
    #    comparison_func = partial(comparison_funcs.comp_allclose_nortol)
    run_single_pytorch_test(
        f"eltwise-{cmp_kind}",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
