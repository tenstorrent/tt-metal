import pytest
import sys
import torch
from pathlib import Path
from functools import partial
from itertools import product
from collections import defaultdict
from math import pi
import numpy as np

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


reference_pcc = defaultdict(lambda: 0.999)
reference_pcc["silu"] = 0.9714
reference_pcc["swish"] = reference_pcc["silu"]
reference_pcc["softplus"] = 0.9984
reference_pcc["relu_max"] = 0.9936789972261026


def custom_compare(*args, **kwargs):
    function = kwargs.pop("function")
    comparison_func = partial(comparison_funcs.comp_pcc, pcc=reference_pcc[function])
    result = comparison_func(*args, **kwargs)
    return result


@pytest.mark.parametrize(
    "fn, input_shapes, pcie_slot",
    list(
        product(
            (
                "swish",
                "log1p",
                "add1",
                "softplus",
                "mish",
                "silu",
                "polyval",
                "mac",
                "relu6",
                "cbrt",
                "deg2rad",
                "rad2deg",
                "threshold",
                "hypot",
                "hardswish",
                "hardsigmoid",
                "ones_like",
                "zeros_like",
                "full_like",
                "ones",
                "zeros",
                "full",
                "arange",
            ),
            ([[1, 1, 32, 32]], [[1, 3, 320, 64]]),
            (0,),
        )
    ),  # Single core, and multi-core
)
def test_run_eltwise_composite_test(
    fn, input_shapes, pcie_slot, function_level_defaults
):
    options = defaultdict(lambda: (-1.0, 1.0))
    options["log1"] = (0.0, 1.0)
    options["square"] = (1.0, 1e2)
    options["relu_max"] = (-100, +100)
    options["relu_min"] = (-100, +100)
    options["polyval"] = (1, 100)
    options["deg2rad"] = (-180, 180)
    options["rad2deg"] = (0, 2 * pi)
    options["hypot"] = (1, 100)
    options["cbrt"] = (-1000, 1000)
    options["relu6"] = (-100, 100)
    options["hardsigmoid"] = (-100, 100)
    options["hardswish"] = (-100, 100)

    generator = generation_funcs.gen_rand
    function = fn

    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generator, low=options[fn][0], high=options[fn][1]),
            torch.bfloat16,
        )
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update({"scalar": np.random.randint(-100, 100)})
    if fn == "arange":
        test_args.update({"start": -10, "end": 1024 - 10, "step": 1})
    run_single_pytorch_test(
        "eltwise-%s" % (function),
        input_shapes,
        datagen_func,
        partial(custom_compare, function=fn),
        pcie_slot,
        test_args,
    )
