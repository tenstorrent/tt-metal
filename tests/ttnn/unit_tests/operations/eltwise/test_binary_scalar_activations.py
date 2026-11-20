# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# The tensor-scalar overloads of maximum, minimum and bias_gelu must apply activations the same way the
# tensor-tensor overloads do. They used to accept and silently drop them (#66564).

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

SHAPE = (1, 1, 64, 64)
SCALAR = -1.0

ACTIVATIONS = [
    (ttnn.UnaryOpType.NEG, torch.neg),
    (ttnn.UnaryOpType.RELU, torch.relu),
    (ttnn.UnaryOpType.SQUARE, torch.square),
]


def _input(device):
    torch_input = torch.linspace(-4, 4, SHAPE[-2] * SHAPE[-1], dtype=torch.float32).reshape(SHAPE).bfloat16()
    return torch_input, ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("op_name", ["maximum", "minimum"])
@pytest.mark.parametrize("activation, torch_activation", ACTIVATIONS, ids=["neg", "relu", "square"])
@pytest.mark.parametrize("slot", ["post", "lhs"])
def test_compare_scalar_applies_activations(device, op_name, activation, torch_activation, slot):
    torch_op = torch.maximum if op_name == "maximum" else torch.minimum
    ttnn_op = getattr(ttnn, op_name)
    torch_input, tt_input = _input(device)
    torch_scalar = torch.full(SHAPE, SCALAR).bfloat16()
    tt_scalar_tensor = ttnn.from_torch(torch_scalar, layout=ttnn.TILE_LAYOUT, device=device)
    activations = [ttnn.UnaryWithParam(activation)]
    kwargs = {"activations": activations} if slot == "post" else {"input_tensor_a_activations": activations}

    if slot == "post":
        expected = torch_activation(torch_op(torch_input, torch_scalar))
    else:
        expected = torch_op(torch_activation(torch_input), torch_scalar)
    expected = expected.bfloat16()

    scalar_result = ttnn.to_torch(ttnn_op(tt_input, SCALAR, **kwargs))
    tensor_result = ttnn.to_torch(ttnn_op(tt_input, tt_scalar_tensor, **kwargs))

    # The scalar overload must match both torch and the tensor-tensor overload it mirrors.
    assert_with_ulp(expected_result=expected, actual_result=scalar_result, ulp_threshold=1)
    assert_with_ulp(expected_result=tensor_result, actual_result=scalar_result, ulp_threshold=1)


@pytest.mark.parametrize("op_name", ["maximum", "minimum"])
def test_compare_scalar_rejects_rhs_activations(device, op_name):
    _, tt_input = _input(device)
    with pytest.raises(RuntimeError, match="input_tensor_b_activations"):
        getattr(ttnn, op_name)(tt_input, SCALAR, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)])


@pytest.mark.parametrize("activation, torch_activation", ACTIVATIONS, ids=["neg", "relu", "square"])
@pytest.mark.parametrize("slot", ["post", "lhs"])
def test_bias_gelu_scalar_applies_activations(device, activation, torch_activation, slot):
    torch_input, tt_input = _input(device)
    bias = 0.5
    torch_bias = torch.full(SHAPE, bias).bfloat16()
    tt_bias_tensor = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    activations = [ttnn.UnaryWithParam(activation)]
    kwargs = {"activations": activations} if slot == "post" else {"input_tensor_a_activations": activations}

    scalar_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, bias, **kwargs))
    tensor_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, tt_bias_tensor, **kwargs))
    plain = ttnn.to_torch(ttnn.bias_gelu(tt_input, bias))

    # The activation must change the result, and the scalar overload must agree with the tensor-tensor one.
    assert not torch.equal(scalar_result, plain)
    assert_with_ulp(expected_result=tensor_result, actual_result=scalar_result, ulp_threshold=2)
