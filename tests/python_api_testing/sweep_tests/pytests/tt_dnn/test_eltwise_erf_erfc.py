import pytest
import sys
import torch
from pathlib import Path
from functools import partial
from math import pi

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 384]],  # Multi core
        [[1, 3, 320, 384]],  # Multi core
    ],
)
@pytest.mark.parametrize("pcie_slot", [0])
class TestEltwiseUnaryERF_ERFC:
    @pytest.mark.parametrize("fast_and_appx", [True, False])
    def test_run_eltwise_erf_op(
        self, input_shapes, fast_and_appx, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_randint, low=0, high=3), torch.bfloat16
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_appx"] = fast_and_appx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-erf",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_appx", [True, False])
    def test_run_eltwise_erfc_op(
        self, input_shapes, fast_and_appx, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_appx"] = fast_and_appx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-erfc",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )
