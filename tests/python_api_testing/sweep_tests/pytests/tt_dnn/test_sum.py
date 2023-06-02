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
    [
        [[2, 2, 32, 64]],  # Single core
        [[4, 2, 320, 384]],  # Multi core
        [[8, 6, 320, 384]],  # Multi core
    ],
)
@pytest.mark.parametrize("pcie_slot", [0])
class TestEltwiseUnary:
    @pytest.mark.parametrize("fn_kind", ["sum-3","sum-2","sum-1","sum-0"])
    def test_run_eltwise_unary_ops(
        self, input_shapes, fn_kind, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"fn_kind":fn_kind,"input_shapes":input_shapes})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )
