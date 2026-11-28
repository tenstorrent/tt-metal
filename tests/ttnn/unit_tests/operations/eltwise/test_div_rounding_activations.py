# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

P_NUMER = [3.5, -3.5, 4.0, -4.0, 0.5, -0.5, 2.5, -2.5, 7.0, -7.0, 1.5, -1.5, 5.0, -5.0, 6.5, -6.5]
P_DENOM = [1.5, 2.5, -2.0, 3.0, 0.5, 1.0, 2.0, -1.5, 0.75, 3.5, -3.0, 2.25, 1.25, 4.0, 0.25, 5.0]

ACTIVATIONS = {
    "NEG": (ttnn.UnaryOpType.NEG, torch.neg),
    "ABS": (ttnn.UnaryOpType.ABS, torch.abs),
    "SIGMOID": (ttnn.UnaryOpType.SIGMOID, torch.sigmoid),
}


def _patterns(dtype):
    numer = torch.tensor(P_NUMER * 64, dtype=dtype).reshape(1, 1, 32, 32)
    denom = torch.tensor(P_DENOM * 64, dtype=dtype).reshape(1, 1, 32, 32)
    return numer, denom


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("activation", list(ACTIVATIONS))
def test_div_tensor_tensor_round_then_activate(device, rounding_mode, activation):
    torch_numer, torch_denom = _patterns(torch.float32)
    op_type, torch_act = ACTIVATIONS[activation]

    tt_a = ttnn.from_torch(torch_numer.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_denom.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.div(tt_a, tt_b, rounding_mode=rounding_mode, activations=[ttnn.UnaryWithParam(op_type)])

    golden = torch_act(torch.div(torch_numer, torch_denom, rounding_mode=rounding_mode))
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.01)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_div_tensor_scalar_round_then_activate(device, rounding_mode):
    torch_numer, _ = _patterns(torch.float32)
    scalar = 2.0

    tt_a = ttnn.from_torch(torch_numer.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.div(tt_a, scalar, rounding_mode=rounding_mode, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)])

    golden = torch.neg(torch.div(torch_numer, scalar, rounding_mode=rounding_mode))
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.01)


def test_div_tensor_tensor_float32_round_then_activate(device):
    torch_numer, torch_denom = _patterns(torch.float32)

    tt_a = ttnn.from_torch(torch_numer, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_denom, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.div(tt_a, tt_b, rounding_mode="floor", activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)])

    golden = torch.neg(torch.div(torch_numer, torch_denom, rounding_mode="floor"))
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=1e-4)


# Controls: correct before and after the fix.


def test_div_activation_without_rounding(device):
    torch_numer, torch_denom = _patterns(torch.float32)

    tt_a = ttnn.from_torch(torch_numer.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_denom.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.div(tt_a, tt_b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])

    golden = torch.relu(torch_numer / torch_denom)
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.05)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_div_rounding_without_activation(device, rounding_mode):
    torch_numer, torch_denom = _patterns(torch.float32)

    tt_a = ttnn.from_torch(torch_numer.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_denom.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.div(tt_a, tt_b, rounding_mode=rounding_mode)

    golden = torch.div(torch_numer, torch_denom, rounding_mode=rounding_mode)
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.01)
