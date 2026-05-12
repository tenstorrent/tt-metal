# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.max / ttnn.min vs torch.amax / torch.amin for int32 and float32 (SFPU reduce path)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("in_dtype", [ttnn.int32, ttnn.float32])
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
        (1, 1, 30, 96),
        (1, 1, 90, 32),
        (1, 3, 17, 19),
        (2, 4, 64, 60),
        (2, 1, 256, 2048),
        (2, 1, 2048, 256),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, -1, -2, (-1, -2), None])
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min(device, in_dtype, input_shape, dim, op):
    torch.manual_seed(0)
    if in_dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output_tensor = torch_op(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=in_dtype)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim))

    assert output_tensor.dtype == torch_input_tensor.dtype

    if in_dtype == ttnn.int32:
        assert_equal(output_tensor, torch_output_tensor)
    else:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999999)


@pytest.mark.parametrize("in_dtype", [ttnn.int32, ttnn.float32])
@pytest.mark.parametrize("scale", [2.0, 0.5, -3.0])
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 2, 64, 64), -1),
        ((1, 1, 96, 64), -2),
        ((1, 1, 64, 64), (-1, -2)),
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min_with_scaling(device, in_dtype, input_shape, dim, op, scale):
    torch.manual_seed(0)
    if in_dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    if in_dtype == ttnn.int32:
        torch_expected = torch_op(torch_input_tensor.float() * scale, dim=dim).to(torch.int32)
    else:
        torch_expected = torch_op(torch_input_tensor * scale, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=in_dtype)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim, scalar=scale))

    assert output_tensor.dtype == torch_input_tensor.dtype

    if in_dtype == ttnn.int32:
        assert_equal(output_tensor, torch_expected)
    else:
        assert_with_pcc(torch_expected, output_tensor, pcc=0.999999)
