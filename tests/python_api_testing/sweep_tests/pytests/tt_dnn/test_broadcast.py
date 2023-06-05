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
        ([[1, 1, 32, 32], [1, 1, 1, 32]], 0),  # Single core
        ([[1, 1, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 3, 1, 32]], 0),  # Multi core h
    ),
)
def test_run_bcast_add_h_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-add-h",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 1]], 0),  # Single core
        ([[1, 1, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 3, 32, 1]], 0),  # Multi core w
    ),
)
def test_run_bcast_add_w_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-add-w",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 1, 1]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 3, 1, 1]], 0),  # Multi core hw
    ),
)
def test_run_bcast_add_hw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-add-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 1, 32]], 0),  # Single core
        ([[1, 1, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 3, 1, 32]], 0),  # Multi core h
    ),
)
def test_run_bcast_sub_h_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-sub-h",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 1]], 0),  # Single core
        ([[1, 1, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 3, 32, 1]], 0),  # Multi core w
    ),
)
def test_run_bcast_sub_w_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-sub-w",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 1, 1]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 3, 1, 1]], 0),  # Multi core hw
    ),
)
def test_run_bcast_sub_hw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-sub-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 1, 32]], 0),  # Single core
        ([[1, 1, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 1, 1, 32]], 0),  # Multi core h
        ([[1, 3, 3840, 32], [1, 3, 1, 32]], 0),  # Multi core h
    ),
)
def test_run_bcast_mul_h_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-mul-h",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 32, 1]], 0),  # Single core
        ([[1, 1, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 1, 32, 1]], 0),  # Multi core w
        ([[1, 3, 32, 3840], [1, 3, 32, 1]], 0),  # Multi core w
    ),
)
def test_run_bcast_mul_w_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-mul-w",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32], [1, 1, 1, 1]], 0),  # Single core
        ([[1, 1, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 1, 1, 1]], 0),  # Multi core hw
        ([[1, 3, 320, 384], [1, 3, 1, 1]], 0),  # Multi core hw
    ),
)
def test_run_bcast_mul_hw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bcast-mul-hw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
