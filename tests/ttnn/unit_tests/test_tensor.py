# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

EXPECTED_TENSOR_METHODS = [
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
    "device",
    "digamma",
    "dtype",
    "elu",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "full_like",
    "geglu",
    "gelu",
    "glu",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "heaviside",
    "i0",
    "identity",
    "is_contiguous",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "layout",
    "leaky_relu",
    "lerp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_sigmoid",
    "logical_andi",
    "logical_not_unary",
    "logical_noti",
    "logical_ori",
    "logical_xori",
    "logit",
    "mish",
    "move",
    "multigammaln",
    "neg",
    "ones_like",
    "polygamma",
    "pow",
    "prelu",
    "rad2deg",
    "rdiv",
    "recip",
    "reglu",
    "relu",
    "relu6",
    "relu_max",
    "relu_min",
    "rpow",
    "rsqrt",
    "rsub",
    "shape",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "sinh",
    "softplus",
    "softshrink",
    "softsign",
    "sqrt",
    "square",
    "swiglu",
    "swish",
    "tan",
    "tanh",
    "tanhshrink",
    "threshold",
    "tril",
    "triu",
    "value",
    "zeros_like",
]


def test_check_symbols(device):
    for symbol in EXPECTED_TENSOR_METHODS:
        assert getattr(ttnn.Tensor, symbol)


@pytest.mark.parametrize(
    "input_shape",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 32, 64])),
        (torch.Size([1, 3, 64, 64])),
    ),
)
class TestErgonomicFluent:
    def test_mae_loss_none(self, input_shape, device):
        torch_a = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shape, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output_tensor = a.__add__(b).abs()

        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.to_torch(output_tensor)

        torch_output_tensor = torch_a + torch_b
        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
