import pytest
import sys
import torch
from pathlib import Path
from functools import partial
from itertools import product
from math import pi
import random

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
class TestEltwiseUnary:
    @pytest.mark.parametrize(
        "input_range, comparison_func",
        (
            ({"low": -5, "high": 5}, comparison_funcs.comp_pcc),
            (
                {"low": -150, "high": -10},
                partial(comparison_funcs.comp_allclose, atol=5e-6, rtol=0),
            ),
            (
                {"low": -5e6, "high": -0.85e6},
                partial(comparison_funcs.comp_allclose, atol=5e-6, rtol=0),
            ),
        ),
    )
    def test_run_eltwise_exp_op(
        self,
        input_shapes,
        input_range,
        comparison_func,
        pcie_slot,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(
                    generation_funcs.gen_rand,
                    low=input_range["low"],
                    high=input_range["high"],
                ),
                torch.float32,
            )
        ]
        run_single_pytorch_test(
            "eltwise-exp",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_recip_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand_symmetric, low=1, high=100),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-recip",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_sqrt_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=0, high=100), torch.float32
            )
        ]
        comparison_func = partial(comparison_funcs.comp_pcc)
        run_single_pytorch_test(
            "eltwise-sqrt",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_gelu_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-gelu",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_relu_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-relu",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_sigmoid_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-sigmoid",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    @pytest.mark.parametrize("fn_kind",
                             ("relu_max",
                              "relu_min",
                              "inverted_relu_max",
                              "inverted_relu_min",
                             ))
    def test_run_eltwise_unary_test(
            self, input_shapes, fn_kind, pcie_slot, function_level_defaults
    ):
        v_low, v_high = 1, 100
        if fn_kind.find("_relu_"):
            v_low, v_high = -100, 100
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=v_low, high=v_high), torch.float32
            )
        ]
        comparison_func = partial(comparison_funcs.comp_allclose,atol=1e-2,rtol=1e-2)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "upper_limit": random.choice(range(20, 80)),
                "lower_limit": random.choice(range(-80, -20)),
            }
        )
        if fn_kind.startswith("inverted_relu"):
            # invert relu limits
            test_args["upper_limit"] *= -1.0
            test_args["lower_limit"] *= -1.0
            fn_kind = "_".join(fn_kind.split("_")[1:])

        run_single_pytorch_test(
            "eltwise-" + fn_kind,
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args
        )

    @pytest.mark.parametrize("input_value", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("lower_limit", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_run_eltwise_relu_min_op(
        self, input_shapes, input_value, lower_limit, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"lower_limit": lower_limit})
        run_single_pytorch_test(
            "eltwise-relu_min",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("input_value", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("upper_limit", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_run_eltwise_relu_max_op(
        self, input_shapes, input_value, upper_limit, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"upper_limit": upper_limit})
        run_single_pytorch_test(
            "eltwise-relu_max",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize(
        "fn_kind",
        [
            "ltz",
            "gtz",
            "lez",
            "gez",
            "eqz",
            "nez",
        ],
    )
    @pytest.mark.parametrize("fill_val", [-1, 0, 1])
    def test_run_eltwise_cmp_ops(
        self, fn_kind, input_shapes, fill_val, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=fill_val),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    @pytest.mark.parametrize("log_kind", ["log", "log2", "log10"])
    def test_run_eltwise_log_ops(
        self, log_kind, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=1, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{log_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_tanh_op(
        self, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-tanh",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_sin_op(self, input_shapes, pcie_slot, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-sin",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    def test_run_eltwise_cos_op(self, input_shapes, pcie_slot, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-cos",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    @pytest.mark.parametrize("input_value", [-1.0, 2.0])
    @pytest.mark.parametrize("exponent", [0, 1, 2, 3])
    def test_run_eltwise_power_op(
        self, input_shapes, input_value, exponent, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"exponent": exponent})
        run_single_pytorch_test(
            "eltwise-power",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("fn_kind", ["abs", "sign", "neg"])
    def test_run_eltwise_sign_ops(
        self, fn_kind, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-10, high=10), torch.bfloat16
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    @pytest.mark.parametrize(
        "unary_kind", ["add_unary", "sub_unary", "mul_unary", "div_unary"]
    )
    @pytest.mark.parametrize("scalar", [1.0, 2.0, 8.0])
    def test_run_eltwise_binop_to_unary_ops(
        self, unary_kind, input_shapes, pcie_slot, scalar, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = partial(comparison_funcs.comp_pcc)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"scalar": scalar})
        run_single_pytorch_test(
            f"eltwise-{unary_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("clip_kind", ["clip", "hardtanh"])
    def test_run_eltwise_clip_ops(
        self, clip_kind, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"low": -2.0, "high": +2.0})
        run_single_pytorch_test(
            f"eltwise-{clip_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

# explore ops for Single core and Multi core tensor sizes
@pytest.mark.parametrize(
    "fn_kind, input_shapes, pcie_slot",
    (
        list(
            product(
                (
                    "square",
                    "neg",
                    "abs",
                ),
                (
                    [[1, 1, 32, 32]],
                    [[1, 1, 320, 384]],
                    [[1, 3, 320, 384]],
                ),  # single, multi core sizes
                (0,),
            )
        )
    ))
def test_run_eltwise_multiple_op(
        fn_kind, input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ]
    comparison_func = comparison_funcs.comp_pcc
    run_single_pytorch_test(
        f"eltwise-{fn_kind}",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
