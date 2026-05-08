# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 60),
        (1, 1, 100, 120),
        (1, 1, 30, 96),
        (1, 1, 90, 32),
        (2, 3, 64, 64),
        (1, 3, 17, 19),
        (2, 4, 64, 60),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, -1, -2, (-1, -2), None])
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min_int32(device, input_shape, dim, op):
    torch.manual_seed(0)

    torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output_tensor = torch_op(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn_op(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.dtype == torch.int32, f"Expected int32 output, got {output_tensor.dtype}"
    assert_equal(output_tensor, torch_output_tensor)


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
def test_max_min_int32_with_scaling(device, input_shape, dim, op, scale):
    torch.manual_seed(0)
    torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_expected = torch_op(torch_input_tensor.float() * scale, dim=dim).to(torch.int32)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn_op(input_tensor, dim=dim, scalar=scale)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.dtype == torch.int32, f"Expected int32 output, got {output_tensor.dtype}"
    assert_equal(output_tensor, torch_expected)
