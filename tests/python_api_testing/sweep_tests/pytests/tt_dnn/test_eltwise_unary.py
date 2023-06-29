import pytest
import sys
import torch
from pathlib import Path
from functools import partial
from itertools import product
from math import pi

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
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_exp_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-5, high=5), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-exp",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_exp_issue137_A_test(
    input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-150, high=-80), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_allclose_nortol)
    run_single_pytorch_test(
        "eltwise-exp",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_exp_issue137_B_test(
    input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-5e6, high=-0.85e6), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_allclose_nortol)
    run_single_pytorch_test(
        "eltwise-exp",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_recip_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand_symmetric, low=1, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-recip",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_sqrt_test(input_shapes, pcie_slot, function_level_defaults):
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
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_gelu_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-gelu",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_relu_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-300, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-relu",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_sigmoid_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-sigmoid",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
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
                    "ltz",
                    "gtz",
                    "lez",
                    "gez",
                    "eqz",
                    "nez",
                    "abs",
                    "relu_max",
                    "relu_min",
                ),
                (
                    [[1, 1, 32, 32]],
                    [[1, 1, 320, 384]],
                    [[1, 3, 320, 384]],
                ),  # single, multi core sizes
                (0,),
            )
        )
    ),
)
def test_run_eltwise_unary_test(
    fn_kind, input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=1, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-" + fn_kind,
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


# explore ops for Single core and Multi core tensor sizes
@pytest.mark.parametrize(
    "log_kind, input_shapes, pcie_slot",
    (
        list(
            product(
                ("log", "log2", "log10"),
                (
                    [[1, 1, 32, 32]],
                    [[1, 1, 320, 384]],
                    [[1, 3, 320, 384]],
                ),  # single, multi core sizes
                (0,),
            )
        )
    ),
)
def test_run_eltwise_log_with_base_test(
    log_kind, input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=1, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-" + log_kind,
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
        ([[1, 3, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_tanh_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-tanh",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_sin_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-sin",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_cos_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=0.0, high=2.0 * pi), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-cos",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_power_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_randint, low=0, high=10), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-power",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        ([[1, 1, 32, 32]], 0),  # Single core
        ([[1, 1, 320, 384]], 0),  # Multi core
    ),
)
def test_run_eltwise_sign_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "eltwise-sign",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


# explore ops for Single core and Multi core tensor sizes
@pytest.mark.parametrize(
    "unary_kind, input_shapes, pcie_slot,scalar",
    (
        list(
            product(
                ("add_unary", "sub_unary", "mul_unary", "div_unary"),
                (
                    [[1, 1, 32, 32]],
                    [[1, 1, 320, 384]],
                    [[1, 3, 320, 384]],
                ),  # single, multi core sizes
                (0,),
                (1.0, 2.0, 8.0),  # scalar
            )
        )
    ),
)
def test_run_eltwise_unary_kind_test(
    unary_kind, input_shapes, pcie_slot, scalar, function_level_defaults
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
        "eltwise-" + unary_kind,
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        test_args,
    )


# explore ops for Single core and Multi core tensor sizes
@pytest.mark.parametrize(
    "clip_kind, input_shapes, pcie_slot",
    (
        list(
            product(
                (
                    "clip",
                    "hardtanh",
                ),
                (
                    [[1, 1, 32, 32]],
                    [[1, 1, 320, 384]],
                    [[1, 3, 320, 384]],
                ),  # single, multi core sizes
                (0,),
            )
        )
    ),
)
def test_run_eltwise_clip_kind_test(
    clip_kind, input_shapes, pcie_slot, function_level_defaults
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-10, high=10), torch.float32
        )
    ]
    comparison_func = partial(comparison_funcs.comp_pcc)
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"low": -2.0, "high": +2.0})
    run_single_pytorch_test(
        "eltwise-" + clip_kind,
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        test_args,
    )
