# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import random
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.unary import TTL_UNARY_FUNCTIONS, TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER, REDUCE_UNARY_FUNCTIONS
from ttnn.createops import TTL_CREATE_FUNCTIONS, TTL_CREATE_FUNCTIONS_WITH_FLOAT_PARAMETER

TTL_FUNCTIONS = (
    TTL_UNARY_FUNCTIONS
    + TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER
    + TTL_CREATE_FUNCTIONS
    + TTL_CREATE_FUNCTIONS_WITH_FLOAT_PARAMETER
)


def get_reference_fn(fname):
    for name, ttl_hook, torch_hook in TTL_FUNCTIONS:
        if name == fname:
            return torch_hook
    raise ValueError(f"{fname} is not a valid entry in TTL_UNARY_FUNCTIONS")


SHAPE_OPS = [
    "ones",
    "zeros",
    "full",
    "empty",
]
SECOND_PARAM = [
    "elu",
    "pow",
    "prelu",
    "polygamma",
    "hardshrink",
    "leaky_relu",
    "logit",
    "tril",
    "triu",
    "full_like",
    "full",
    "softshrink",
]
custom_range = {"atanh": (-1.0, +1.0)}
# override pcc
pcc = {"exp2": 0.987, "atan": 0.978, "tanhshrink": 0.93, "digamma": 0.96}


@pytest.mark.parametrize(
    "unary_fn",
    [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "clone",
        "cos",
        "cosh",
        "deg2rad",
        "digamma",
        "elu",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "exp2",
        "expm1",
        # "geglu",
        "gelu",
        # "glu",
        "hardshrink",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "i0",
        "identity",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "leaky_relu",
        # "lerp",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "log_sigmoid",
        # "logaddexp",
        # "logaddexp2",
        "logit",
        # "max",
        # "min",
        "mish",
        "move",
        "multigammaln",
        "neg",
        # "permute",
        "polygamma",
        "pow",
        "prelu",
        "rad2deg",
        "recip",
        # "reglu",
        "relu",
        "relu6",
        "rsqrt",
        "sigmoid",
        "sign",
        # "signbit",
        "silu",
        "sin",
        "sinh",
        "sqrt",
        "square",
        # "swiglu",
        "swish",
        "tan",
        "tanh",
        "tanhshrink",
        "tril",
        "triu",
        "var_hw",
        "mean_hw",
        "std_hw",
        "normalize_hw",
        "normalize_global",
        "ones_like",
        "zeros_like",
        "full_like",
        "ones",
        "zeros",
        "full",
        # "empty",
        "softsign",
        "softplus",
        "softshrink",
    ],
)
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_unary(device, unary_fn, h, w):
    torch.manual_seed(0)

    low, high = custom_range.get(unary_fn, (0, 1))
    torch_input_tensor = low + torch.rand((h, w), dtype=torch.bfloat16) * (high - low)
    if unary_fn in ["lgamma", "digamma", "polygamma", "recip"]:
        torch_input_tensor += 10.0
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)

    if unary_fn in SHAPE_OPS:
        input_tensor = (2, 3, 4, 5)
        torch_input_tensor = input_tensor

    ttnn_unary_fn = getattr(ttnn, unary_fn)

    torch_ref_fn = get_reference_fn(unary_fn)
    if unary_fn == "polygamma":
        torch_ref_fn = lambda _a, _b: get_reference_fn(unary_fn)(_b, _a)

    if unary_fn in SECOND_PARAM:
        if unary_fn == "pow":
            arg = random.choice([1.5, 2.0])
        elif unary_fn == "prelu":
            arg = torch.Tensor((2.0,))
        else:
            arg = 1
        if torch_ref_fn:
            torch_output_tensor = torch_ref_fn(torch_input_tensor, arg)
        output_tensor = ttnn_unary_fn(input_tensor, arg)
    else:
        if torch_ref_fn:
            torch_output_tensor = torch_ref_fn(torch_input_tensor)
        output_tensor = ttnn_unary_fn(input_tensor)
    if not torch_ref_fn:
        pytest.skip("Cannot compare w/ reference function yet.")
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    if unary_fn in REDUCE_UNARY_FUNCTIONS:
        output_tensor = output_tensor[0, 0, 0, 0].reshape(1, 1)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc.get(unary_fn, 0.99))
