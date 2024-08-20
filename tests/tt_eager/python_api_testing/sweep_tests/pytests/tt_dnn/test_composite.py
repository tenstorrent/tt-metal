# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from itertools import product
from collections import defaultdict
from math import pi
import random
import numpy as np


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0, is_grayskull


reference_pcc = defaultdict(lambda: 0.999)
reference_pcc["silu"] = 0.9714
reference_pcc["swish"] = reference_pcc["silu"]


def custom_compare(*args, **kwargs):
    function = kwargs.pop("function")
    if function in [
        "logical_xor",
        "logical_or",
        "logical_not",
        "is_close",
    ]:
        comparison_func = comparison_funcs.comp_equal
    elif function in ["empty"]:
        comparison_func = comparison_funcs.comp_shape
    else:
        comparison_func = partial(comparison_funcs.comp_pcc, pcc=reference_pcc[function])
    result = comparison_func(*args, **kwargs)
    return result


shapes = ([[1, 1, 32, 32]], [[1, 3, 320, 64]])
if is_wormhole_b0():
    shapes = (shapes[0],)


# TODO: This function should be split apart instead of having all these if cases
@pytest.mark.parametrize(
    "fn, input_shapes",
    list(
        product(
            (
                "lerp_binary",
                "lerp_ternary",
                "addcmul",
                "addcdiv",
                "swish",
                "log1p",
                "mish",
                "silu",
                "polyval",
                "mac",
                "cbrt",
                "threshold",
                "hypot",
                "hardswish",
                "hardsigmoid",
                "hardshrink",
                "softshrink",
                "sinh",
                "cosh",
                "tanhshrink",
                "xlogy",
                "asinh",
                "acosh",
                "atanh",
                "atan2",
                "subalpha",
                "bias_gelu_unary",
                "addalpha",
                "logit",
                "logical_xor",
                "isclose",
                "digamma",
                "lgamma",
                "multigammaln",
                "polygamma",
                "nextafter",
                "celu",
                # TO-DO:
                # "scatter",
            ),
            shapes,
        )
    ),  # Single core, and multi-core
)
def test_run_eltwise_composite_test(fn, input_shapes, device, function_level_defaults):
    options = defaultdict(lambda: (-1.0, 1.0))
    options["log1"] = (0.0, 1.0)
    options["polyval"] = (1, 100)
    options["logit"] = (0, 0.99)
    options["deg2rad"] = (-180, 180)
    options["bias_gelu_unary"] = (-1e10, 1e10)
    options["rad2deg"] = (0, 2 * pi)
    options["hypot"] = (1, 100)
    options["atan2"] = (-100, 100)
    options["celu"] = (-100, 100)
    options["cbrt"] = (-1000, 1000)
    options["hardsigmoid"] = (-100, 100)
    options["hardswish"] = (-100, 100)
    options["hardshrink"] = (-100, 100)
    options["softshrink"] = (-100, 100)
    options["leaky_shrink"] = (-100, 100)
    options["softsign"] = (1, 100)
    options["digamma"] = (1, 1000)
    options["lgamma"] = (0.1, 1e32)
    options["multigammaln"] = (1.6, 1e32)
    options["polygamma"] = (1, 10)

    options["sinh"] = (-9, 9)
    options["tanhshrink"] = (-100, 100)
    options["atanh"] = (-1, 1)
    options["cosh"] = options["sinh"]
    options["asinh"] = (-100, 100)
    options["isclose"] = (-100, 100)
    options["acosh"] = (1, 100)

    generator = generation_funcs.gen_rand

    if is_wormhole_b0():
        if fn in ["logit"]:
            pytest.skip("does not work for Wormhole -skipping")
    if is_grayskull():
        if fn in ["mish"]:
            pytest.skip("does not work for Grayskull -skipping")
    if fn in [
        "logical_xor",
    ]:
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generator, low=options[fn][0], high=options[fn][1]),
                torch.int32,
            )
        ]
    else:
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generator, low=options[fn][0], high=options[fn][1]),
                torch.bfloat16,
            )
        ]
    num_inputs = 1
    if fn in ["mac", "addcmul", "addcdiv", "lerp_ternary"]:
        num_inputs = 3
    elif fn in [
        "hypot",
        "scatter",
        "min",
        "max",
        "lerp_binary",
        "xlogy",
        "subalpha",
        "addalpha",
        "bias_gelu_unary",
        "atan2",
        "subalpha",
        "addalpha",
        "logical_xor",
        "isclose",
        "assign_binary",
        "nextafter",
    ]:
        num_inputs = 2

    input_shapes = input_shapes * num_inputs
    datagen_func = datagen_func * num_inputs
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"scalar": np.random.randint(-100, 100)})
    if fn == "arange":
        test_args.update({"start": -10, "end": 1024 - 10, "step": 1})
    elif fn == "polyval":
        test_args.update({"coeffs": [1.0, 2.0, 1.0, 2.0]})
    elif fn == "threshold":
        test_args.update({"threshold": 5.0, "value": 1.0})
    elif fn in ["softshrink", "hardshrink"]:
        test_args.update({"_lambda": np.random.randint(1, 100)})
    elif fn in ["addcmul", "addcdiv"]:
        test_args.update({"value": np.random.randint(1, 100)})
    elif fn in ["lerp_binary"]:
        test_args.update({"weight": np.random.randint(1, 100)})
    elif fn in ["subalpha", "celu"]:
        test_args.update({"alpha": np.random.randint(1, 100)})
    elif fn in ["addalpha"]:
        test_args.update({"alpha": np.random.randint(1, 100)})
    elif fn in ["bias_gelu_unary", "bias_gelu"]:
        test_args.update({"bias": np.random.randint(1, 100)})
    elif fn in ["logit"]:
        test_args.update({"eps": np.random.randint(-10, 0.99)})
    elif fn in ["polygamma"]:
        test_args.update({"k": np.random.randint(1, 10)})
    elif fn in ["isclose"]:
        test_args.update(
            {
                "rtol": random.choice([1e-3, 1e-5, 1e-7]),
                "atol": random.choice([1e-2, 1e-4, 1e-6]),
                "equal_nan": random.choice([False, True]),
            }
        )
    run_single_pytorch_test(
        "eltwise-%s" % (fn),
        input_shapes,
        datagen_func,
        partial(custom_compare, function=fn),
        device,
        test_args,
        ttnn_op=True,
    )


@pytest.mark.parametrize(
    "fn, input_shapes",
    list(
        product(
            (
                "min",
                "max",
            ),
            shapes,
        )
    ),  # Single core, and multi-core
)
def test_run_min_max_test(fn, input_shapes, device, function_level_defaults):
    generator = generation_funcs.gen_rand
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generator, low=-100, high=100),
            torch.bfloat16,
        )
    ]
    comparison_func = comparison_funcs.comp_equal
    num_inputs = 1
    input_shapes = input_shapes * num_inputs
    datagen_func = datagen_func * num_inputs
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]

    rank = len(input_shapes[0])
    choices = [(rank - 1,), (rank - 2,)]
    idx = np.random.choice(len(choices), 1)
    dims = choices[idx.item()]

    test_args.update({"dim": dims})

    run_single_pytorch_test(
        f"ttnn-{fn}", input_shapes, datagen_func, partial(custom_compare, function=fn), device, test_args, ttnn_op=True
    )
