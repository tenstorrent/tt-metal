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


@pytest.mark.parametrize("op_name", ["maximum", "minimum"])
@pytest.mark.parametrize("slot", ["post", "lhs"])
def test_compare_scalar_float32_output_keeps_activation_precision(device, op_name, slot):
    # The square of a bfloat16 value is exact in float32 but not in bfloat16, so a float32 result that went
    # through bfloat16 on the way shows up as a mismatch, e.g. 1.0625 ** 2 = 1.12890625 becomes 1.125.
    torch_op = torch.maximum if op_name == "maximum" else torch.minimum
    scalar = -1.0 if op_name == "maximum" else 2.0
    torch_input, tt_input = _input(device)
    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
    kwargs = {"activations": activations} if slot == "post" else {"input_tensor_a_activations": activations}

    x = torch_input.float()
    expected = (
        torch.square(torch_op(x, torch.tensor(scalar))) if slot == "post" else torch_op(x * x, torch.tensor(scalar))
    )
    result = ttnn.to_torch(getattr(ttnn, op_name)(tt_input, scalar, dtype=ttnn.float32, **kwargs))

    assert result.dtype == torch.float32
    assert torch.equal(expected, result)


@pytest.mark.parametrize("activation, torch_activation", ACTIVATIONS, ids=["neg", "relu", "square"])
@pytest.mark.parametrize("slot", ["post", "lhs", "rhs"])
def test_bias_gelu_scalar_applies_activations(device, activation, torch_activation, slot):
    torch_input, tt_input = _input(device)
    bias = 0.5
    torch_bias = torch.full(SHAPE, bias).bfloat16()
    tt_bias_tensor = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    activations = [ttnn.UnaryWithParam(activation)]
    kwargs = {
        "post": {"activations": activations},
        "lhs": {"input_tensor_a_activations": activations},
        "rhs": {"input_tensor_b_activations": activations},
    }[slot]

    # The add packs its sum as bfloat16 before the gelu; post activations see the gelu before it is packed.
    x, b = torch_input.float(), torch_bias.float()
    if slot == "post":
        expected = torch_activation(torch.nn.functional.gelu((x + b).bfloat16().float()))
    elif slot == "lhs":
        expected = torch.nn.functional.gelu((torch_activation(x) + b).bfloat16().float())
    else:
        expected = torch.nn.functional.gelu((x + torch_activation(b)).bfloat16().float())
    expected = expected.bfloat16()

    scalar_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, bias, **kwargs))
    tensor_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, tt_bias_tensor, **kwargs))
    plain = ttnn.to_torch(ttnn.bias_gelu(tt_input, bias))

    # The activation must change the result (RELU leaves the positive bias alone), and the scalar overload
    # must agree with torch and with the tensor-tensor overload. The device gelu is a few bfloat16 ULP off
    # torch's (twice that after SQUARE), far closer than an activation applied in the wrong place.
    if not (slot == "rhs" and activation == ttnn.UnaryOpType.RELU):
        assert not torch.equal(scalar_result, plain)
    torch.testing.assert_close(scalar_result.float(), expected.float(), rtol=0.03, atol=1e-3)
    assert_with_ulp(expected_result=tensor_result, actual_result=scalar_result, ulp_threshold=2)


def test_bias_gelu_scalar_block_float_post_activation_before_pack(device):
    # Each 16-element bfloat8_b block mixes gelu(3.5) ~= 3.5 with gelu(-4.0) ~= -1.3e-4, which the shared
    # exponent rounds to zero. LTZ has to see the gelu before it is packed, as the tensor-tensor overload does.
    torch_input = torch.tensor([3.0, -4.5]).repeat(SHAPE[-2] * SHAPE[-1] // 2).reshape(SHAPE)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bias_tensor = ttnn.from_torch(
        torch.full(SHAPE, 0.5), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    activations = [ttnn.UnaryWithParam(ttnn.UnaryOpType.LTZ)]
    expected = (torch_input < 0).float()

    scalar_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, 0.5, activations=activations)).float()
    tensor_result = ttnn.to_torch(ttnn.bias_gelu(tt_input, tt_bias_tensor, activations=activations)).float()

    assert torch.equal(expected, tensor_result)
    assert torch.equal(expected, scalar_result)
