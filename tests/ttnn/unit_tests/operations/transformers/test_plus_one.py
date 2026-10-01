# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
def test_plus_one(device, w, dtype):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one_subdevice(device, w):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))])
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
def test_plus_one_subdevice_nd(device, input_shape):
    torch_input_tensor = torch.randint(32000, input_shape)
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor,
        sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))]),
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries(device, w, val):
    torch_input_tensor = torch.randint(32000, (w,))
    mask = torch.rand(w) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries_nd(device, input_shape, val):
    torch_input_tensor = torch.randint(32000, input_shape)
    mask = torch.rand(input_shape) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [4, 8, 32])
def test_plus_one_program_cache(device, w):
    """Two distinct same-spec DRAM tensors: the second call is a program-cache hit, so the
    cached program must be re-bound to the second tensor's buffer. A stale binding would
    increment the first tensor again and leave the second untouched — validating both
    tensors catches either failure."""
    torch_a = torch.randint(32000, (w,))
    torch_b = torch.randint(32000, (w,))

    tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.int32, device=device)
    tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.int32, device=device)

    ttnn.plus_one(tensor_a)
    ttnn.plus_one(tensor_b)
    assert device.num_program_cache_entries() == 1

    assert_with_pcc(torch_a + 1, ttnn.to_torch(tensor_a), 0.9999)
    assert_with_pcc(torch_b + 1, ttnn.to_torch(tensor_b), 0.9999)


@pytest.mark.parametrize("w", [4, 8, 32])
def test_plus_one_sharded_program_cache(device, w):
    """L1-sharded inputs: the program's working buffer is the input shard itself, so on a
    program-cache hit the framework must re-apply the second tensor's shard address.
    Validating both tensors catches a stale rebinding in either direction."""
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (1, w), ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_a = torch.randint(32000, (1, w))
    torch_b = torch.randint(32000, (1, w))

    tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.int32, device=device, memory_config=memory_config)
    tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.int32, device=device, memory_config=memory_config)

    ttnn.plus_one(tensor_a)
    ttnn.plus_one(tensor_b)
    assert device.num_program_cache_entries() == 1

    assert_with_pcc(torch_a + 1, ttnn.to_torch(tensor_a), 0.9999)
    assert_with_pcc(torch_b + 1, ttnn.to_torch(tensor_b), 0.9999)
