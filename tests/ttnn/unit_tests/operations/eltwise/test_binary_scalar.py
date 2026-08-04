# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    (
        (ttnn.gt),
        (ttnn.lt),
        (ttnn.ne),
        (ttnn.ge),
        (ttnn.le),
        (ttnn.eq),
    ),
)
def test_binary_scalar_ops(input_shapes, device, ttnn_fn):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shapes, dtype=torch.bfloat16) * 100
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    scalar = random.randint(-80, 80)
    ttnn_fn(input_tensor, scalar, output_tensor=output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_fn(torch_input, scalar)

    out = ttnn.to_torch(output_tensor).to(torch.bool)

    assert torch.equal(out, golden_tensor)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize(
    "scalar",
    [
        7,
        -13,
        0,
        -1,
        2,
        10000,
    ],
)
def test_binary_scalar_int32_arithmetic(device, op_name, scalar):
    """Verify int32 tensor + int scalar passes the scalar as int32 (not float)."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            1,
            -1,
            0,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            1000,
            -1000,
            42,
            123456789,
            -123456789,
            500,
            -500,
            999,
            -999,
            77,
            -77,
            2,
            -2,
            10,
            -10,
            100,
            -100,
            7,
            9,
            11,
            15,
        ],
        dtype=torch.int32,
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result)


@pytest.mark.parametrize("op_name", ["add", "sub"])
@pytest.mark.parametrize("scalar", [0, 1, 100, 65535])
def test_binary_scalar_uint32_arithmetic(device, op_name, scalar):
    """Verify uint32 tensor + int scalar near and at uint32 max boundary."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            0,
            1,
            2,
            255,
            65535,
            100000,
            2147483647,
            2147483648,
            3000000000,
            4000000000,
            4294967290,
            4294967291,
            4294967294,
            4294967295,
            16777215,
            16777216,
            16777217,
            500,
            1000,
            10000,
            1000000,
            1000000000,
            2500000000,
            3500000000,
            3999999999,
            4294000000,
            4294900000,
            4294960000,
            4294967000,
            4294967200,
        ],
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_output = ttnn_fn(tt_input, scalar)

    expected_tt = ttnn.from_torch(
        expected,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    comparison = ttnn.eq(tt_output, expected_tt)
    comparison_torch = ttnn.to_torch(comparison)
    assert torch.all(comparison_torch), "Mismatch in uint32 scalar arithmetic"


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize("scalar", [1.5, -2.25, 0.0, 100.0])
def test_binary_scalar_float32_arithmetic(device, op_name, scalar):
    """Verify float32 tensor + float scalar still works correctly."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch.manual_seed(42)
    torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.allclose(expected, result, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "ttnn_fn",
    [ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le],
)
@pytest.mark.parametrize("scalar", [0, 1, -1, 42, -100])
def test_binary_scalar_int32_relational(device, ttnn_fn, scalar):
    """Verify relational ops with int32 tensor and int scalar."""
    torch_input = torch.tensor(
        [
            -100,
            42,
            -1,
            0,
            1,
            200,
            -200,
            -50,
            -10,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            300,
            -300,
            -150,
            -5,
        ],
        dtype=torch.int32,
    )

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    expected = golden_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output).to(torch.bool)

    assert torch.equal(expected, result)


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        -2147483647,
        -2147483540,
        -2147483648,
    ],
)
def test_binary_scalar_int32_large_values(scalar, device):
    """Verify that large int32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as int32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int32)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        4294967200,
        4294967294,
    ],
)
def test_binary_scalar_uint32_large_values(scalar, device):
    """Verify that large uint32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as uint32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int64)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output, dtype=torch.int64)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )
