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


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0, is_wormhole_b0

shape_wh =  [
    ([[1, 1, 32, 32]], 0),  # Single core
    ([[3, 1, 320, 384]], 0),  # Multi core
]
if is_wormhole_b0():
    del shape_wh[1:]


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    shape_wh
)
def test_run_transpose_wh_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-wh",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 32, 32, 32]], 0),  # Single core
        ([[3, 320, 384, 32]], 0),  # Multi core
    ),
)
def test_run_transpose_hc_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-hc",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )

shape_cn = [
    ([[1, 1, 32, 32]], 0),  # Single core
    ([[3, 5, 384, 96]], 0),  # Single core
]
if is_wormhole_b0():
    del shape_cn[1:]

@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    shape_cn
)
def test_run_transpose_cn_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-cn",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[32, 1, 32, 32]], 0),  # Single core
        ([[32, 3, 384, 96]], 0),  # Single core
    ),
)
def test_run_transpose_nh_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-nh",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[32, 1, 32, 32]], 0),  # Single core
        ([[32, 3, 384, 96]], 0),  # Single core
    ),
)
def test_run_transpose_nw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-nw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 64, 32, 32]], 0),  # Single core
        ([[3, 64, 384, 96]], 0),  # Single core
    ),
)
def test_run_transpose_cw_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "transpose-cw",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
