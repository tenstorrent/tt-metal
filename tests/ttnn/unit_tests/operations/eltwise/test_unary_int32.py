# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (-1e6, 1e6),
        (-2147483647, 0),  # max negative to zero
        (0, 2147483647),  # zero to max positive
        (2e9, 2147483647),  # large positive input
        (-2147483647, -2e9),  # large negative input
    ],
)
@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_not,
    ],
)
def test_unary_logical_int32(input_shapes, low_a, high_a, logical_op, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0  # every 5th element is zero
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = logical_op(input_tensor_a)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (-46340, 0),
        (0, 46340),
        (-46340, 46340),  # output overflows beyond this range
    ],
)
def test_unary_square_int32(input_shapes, low_a, high_a, device):
    if len(input_shapes) == 0:
        torch_input_tensor = torch.randint(low=-(2**15), high=2**15, size=(), dtype=torch.int32)
    else:
        num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
        torch_input_tensor = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
        torch_input_tensor = torch_input_tensor[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.square)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.square(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (-2147483647, 0),  # max negative to zero
        (0, 2147483647),  # zero to max positive
        (-2147483647, 2147483647),  # full representable range
    ],
)
def test_abs_int32(input_shapes, low_a, high_a, device):
    # Note: abs(-2147483648) is not representable in int32, so the range stops at -2147483647.
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0  # ensure the zero / sign-clear path is exercised
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)
    # Guard the documented INT_MIN avoidance: abs(-2147483648) overflows int32 and would
    # silently corrupt both golden and device results (a green-but-meaningless test).
    assert torch_input_tensor_a.min().item() != -2147483648, "INT_MIN leaked into abs test input"

    golden_function = ttnn.get_golden_function(ttnn.abs)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.abs(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(result)

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
@pytest.mark.parametrize("scalar", [0, 1, -0xFF, 2147483647, -2147483648])
def test_bitwise_scalar_int32(device, ttnn_function, scalar):
    x_torch = torch.tensor([[1, 2, 255, -4096, 2147483647, -2147483648, -2147483647, 0]], dtype=torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, scalar)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, scalar)
    tt_out = ttnn.to_torch(z_tt_out, dtype=torch.int32)

    assert torch.equal(z_torch, tt_out)
