# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.sum vs torch.sum for int32 (SFPU reduce path)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device

# Single tile, partial tiles, multi-batch, multi-chunk, and large multi-core shapes.
_SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 60),
    (1, 1, 100, 120),
    (1, 1, 30, 96),
    (1, 1, 90, 32),
    (2, 3, 64, 64),
    (1, 3, 17, 19),
    (2, 4, 64, 60),
    (1, 1, 64, 512),
    (1, 1, 512, 64),
    (2, 1, 256, 2048),
    (2, 1, 2048, 256),
]

_DIMS = [-1, -2, 0, 1, (-1, -2), None]

# Bound keeps the largest sum (2*2048*256 elems) well within int32 range.
_INT32_VALUE_BOUND = 1000


@pytest.mark.parametrize("input_shape", _SHAPES)
@pytest.mark.parametrize("dim", _DIMS)
def test_sum_int32(device, input_shape, dim):
    """ttnn.sum on int32 must match torch.sum exactly."""
    torch.manual_seed(0)
    torch_input = torch.randint(-_INT32_VALUE_BOUND, _INT32_VALUE_BOUND + 1, input_shape, dtype=torch.int32)
    torch_output = torch.sum(torch_input, dim=dim)  # torch promotes int32 -> int64

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input, dim=dim))

    assert ttnn_output.dtype == torch.int32
    assert_equal(ttnn_output.to(torch.int64).reshape(torch_output.shape), torch_output)


@pytest.mark.parametrize("dim", [-1, -2, (-1, -2), None])
def test_sum_int32_keepdim(device, dim):
    """keepdim=True keeps the reduced axis as size 1."""
    torch.manual_seed(0)
    shape = (2, 3, 64, 64)
    torch_input = torch.randint(-_INT32_VALUE_BOUND, _INT32_VALUE_BOUND + 1, shape, dtype=torch.int32)
    torch_output = torch.sum(torch_input, dim=dim, keepdim=True) if dim is not None else torch.sum(torch_input)

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input, dim=dim, keepdim=True))

    assert ttnn_output.dtype == torch.int32
    assert_equal(ttnn_output.to(torch.int64).reshape(torch_output.shape), torch_output)


def test_sum_int32_overflow_wraps(device):
    """int32 sum wraps in 2's-complement like torch (proves integer add, not float promotion)."""
    torch.manual_seed(0)
    torch_input = torch.full((1, 1, 64, 64), 2_000_000, dtype=torch.int32)  # sum > 2^31 -> wraps
    expected_wrapped = torch.sum(torch_input.to(torch.int64)).to(torch.int32)

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input))

    assert ttnn_output.dtype == torch.int32
    assert_equal(ttnn_output.reshape(expected_wrapped.shape), expected_wrapped)


def test_sum_int32_all_zeros_regression(device):
    """Guards the original silent all-zeros int32 sum bug."""
    torch.manual_seed(0)
    x = torch.randint(-_INT32_VALUE_BOUND, _INT32_VALUE_BOUND + 1, (1, 3, 17, 19), dtype=torch.int32)
    x_ttnn = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)

    torch_sum_all = torch.sum(x)
    ttnn_sum_all = ttnn.to_torch(ttnn.sum(x_ttnn))
    assert ttnn_sum_all.dtype == torch.int32
    assert_equal(ttnn_sum_all.to(torch.int64).reshape(torch_sum_all.shape), torch_sum_all)

    torch_sum_dim = torch.sum(x, dim=2)
    ttnn_sum_dim = ttnn.to_torch(ttnn.sum(x_ttnn, dim=2))
    assert ttnn_sum_dim.dtype == torch.int32
    assert_equal(ttnn_sum_dim.to(torch.int64).reshape(torch_sum_dim.shape), torch_sum_dim)


@pytest.mark.parametrize("scale", [2.0, 0.5, -3.0])
@pytest.mark.parametrize("input_shape, dim", [((1, 2, 64, 64), -1), ((1, 1, 96, 64), -2), ((1, 1, 64, 64), (-1, -2))])
def test_sum_int32_with_scaling(device, input_shape, dim, scale):
    """scalar= is a post-reduce fp32 multiply: result == trunc(scale * int_sum)."""
    torch.manual_seed(0)
    torch_input = torch.randint(-100, 100, input_shape, dtype=torch.int32)  # small so int_sum stays < 2^24
    torch_expected = (torch.sum(torch_input, dim=dim).to(torch.float32) * scale).to(torch.int32)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn.to_torch(ttnn.sum(input_tensor, dim=dim, scalar=scale))

    assert output_tensor.dtype == torch.int32
    assert_equal(output_tensor.reshape(torch_expected.shape), torch_expected)
