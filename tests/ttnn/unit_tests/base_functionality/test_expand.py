# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        [(4, 1), (4, 2)],
        [(1, 32), (32, -1)],
        [(1, 32), (64, 32)],
        [(8, 1), (8, 8)],
        [(8, 1), (-1, 32)],
    ],
)
@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_expand(input_shape, output_shape, tensor_layout, dtype, device):
    torch.manual_seed(2024)
    torch_input_tensor = random_torch_tensor(dtype, input_shape)
    torch_output_tensor = torch_input_tensor.expand(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=tensor_layout, device=device)
    with device.cache_entries_counter.measure():
        output_tensor = ttnn.expand(input_tensor, output_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    assert tuple(output_tensor.shape) == tuple(torch_output_tensor.shape)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        # rank 2 -> rank 3: right-align trailing dims
        [(3, 1), (2, 3, 1)],
        [(3, 1), (2, -1, 4)],
        [(1, 4), (3, 2, -1)],
        # rank 2 -> rank 4
        [(2, 1), (4, 3, 2, 5)],
        [(1, 1), (2, 3, 4, 5)],
        # rank 1 -> rank 3
        [(1,), (2, 3, 4)],
        [(4,), (2, 3, -1)],
        # rank 3 -> rank 4
        [(1, 3, 1), (2, -1, -1, 4)],
        [(1, 1, 1), (5, 4, 3, 2)],
        # rank 3 -> rank 5
        [(2, 1, 4), (3, 2, -1, 3, -1)],
        # zero-size dims (zero-volume tensors)
        [(3, 1), (0, 3, 1)],
        [(3, 1), (2, 3, 0)],
        # all-ones repetition (exercises repeat early-return + rank fixup)
        [(3, 1), (1, 3, 1)],
        [(3, 1), (1, 1, 3, 1)],
    ],
)
@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_expand_rank_increase(input_shape, output_shape, tensor_layout, dtype, device):
    """Expand with output rank > input rank (right-aligned dims)."""
    torch.manual_seed(2024)
    torch_input_tensor = random_torch_tensor(dtype, input_shape)
    torch_output_tensor = torch_input_tensor.expand(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=tensor_layout, device=device)
    output_tensor = ttnn.expand(input_tensor, output_shape)
    output_tensor = ttnn.to_torch(output_tensor)
    assert tuple(output_tensor.shape) == tuple(torch_output_tensor.shape)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "input_shape, output_shape, error_msg",
    [
        # negative on a leading dim that has no corresponding input dim
        [(3, 1), (-1, 3, 1), "Leading dimension must be non-negative"],
        # non-singleton trailing dim mismatch
        [(3, 1), (2, 4, 1), "Only size 1 dimensions can be expanded"],
        # negative size other than -1 on trailing dim
        [(3, 1), (2, 3, -2), "Expand dimension size must be -1"],
    ],
)
def test_expand_invalid(input_shape, output_shape, error_msg, device, expect_error):
    torch_input_tensor = torch.rand(input_shape).bfloat16().float()
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, error_msg):
        ttnn.expand(input_tensor, output_shape)


@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_expand_callback(tensor_layout, dtype, device):
    test_expand([32, 1], [32, 32], tensor_layout, dtype, device)
    num_cache_entries = device.cache_entries_counter.total
    assert num_cache_entries > 0

    test_expand([32, 1], [32, 32], tensor_layout, dtype, device)
    assert device.cache_entries_counter.total == num_cache_entries
