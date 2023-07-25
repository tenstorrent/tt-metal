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
class TestEltwiseUnary:
    @pytest.mark.parametrize(
        "fn_kind",
        ["relu", "sigmoid", "square", "tanh", "relu6", "add1", "deg2rad", "rad2deg"],
    )
    def test_run_eltwise_unary_ops(
        self, input_shapes, fn_kind, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_appx", [True, False])
    def test_run_eltwise_gelu_op(
        self, input_shapes, fast_and_appx, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_appx"] = fast_and_appx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-gelu",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("fast_and_appx", [True, False])
    def test_run_eltwise_rsqrt_op(
        self, input_shapes, fast_and_appx, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=1, high=1e8), torch.bfloat16
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["fast_and_appx"] = fast_and_appx
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-rsqrt",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

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
                partial(generation_funcs.gen_rand, **input_range),
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

    @pytest.mark.parametrize(
        "relu_type, limit_type", [["min", "lower"], ["max", "upper"]]
    )
    @pytest.mark.parametrize("input_value", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("limit", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_run_eltwise_relu_limit_ops(
        self,
        input_shapes,
        relu_type,
        limit_type,
        input_value,
        limit,
        pcie_slot,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=input_value),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_equal
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({f"{limit_type}_limit": limit})
        run_single_pytorch_test(
            f"eltwise-relu_{relu_type}",
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

    @pytest.mark.parametrize("trig_kind", ["sin", "cos"])
    def test_run_eltwise_periodic_trig_ops(
        self, input_shapes, trig_kind, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi),
                torch.float32,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{trig_kind}",
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

    @pytest.mark.parametrize("fn_kind", ["abs", "sign", "neg", "signbit"])
    @pytest.mark.parametrize("fill_val", [-1.0, 0.0, 1.0])
    def test_run_eltwise_sign_ops(
        self, fn_kind, input_shapes, fill_val, pcie_slot, function_level_defaults
    ):
        if fn_kind == "signbit" and fill_val == 0.0:
            pytest.skip("Signbit fails for 0 value")
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

    @pytest.mark.parametrize("value", [0.5])
    def test_run_eltwise_heaviside(
        self, input_shapes, value, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"value": value})
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            "eltwise-heaviside",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize(
        "unary_kind", ["add_unary", "sub_unary", "mul_unary", "div_unary"]
    )
    @pytest.mark.parametrize("scalar", [-2.0, 1.0, 2.0, 8.0])
    def test_run_eltwise_binop_to_unary_ops(
        self, unary_kind, input_shapes, pcie_slot, scalar, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
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
    @pytest.mark.parametrize(
        "clip_range", ({"low": -2.0, "high": 2.0}, {"low": -5.5, "high": 27.5})
    )
    def test_run_eltwise_clip_ops(
        self, clip_kind, clip_range, input_shapes, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(clip_range)
        run_single_pytorch_test(
            f"eltwise-{clip_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    @pytest.mark.parametrize("alpha", [-0.5, 0, 0.5])
    def test_run_eltwise_elu_op(
        self, input_shapes, alpha, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"alpha": alpha})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-elu",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
            test_args,
        )

    def test_run_eltwise_exp2(self, input_shapes, pcie_slot, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-10, high=10), torch.bfloat16
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-exp2",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    #@pytest.mark.skip("expm1 only returns -1.0")
    def test_run_eltwise_expm1(self, input_shapes, pcie_slot, function_level_defaults):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-10, high=10), torch.bfloat16
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-expm1",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )

    @pytest.mark.parametrize("arc_trig_kind", ["asin", "acos"])
    def test_run_eltwise_arc_trig_ops(
        self, input_shapes, arc_trig_kind, pcie_slot, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_rand, low=-1, high=1),
                torch.bfloat16,
            )
        ]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{arc_trig_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            pcie_slot,
        )
