# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
def test_serialization(tmp_path, shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.tensorbin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttnn.load_tensor(str(file_name)).to_torch()

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing


core_ranges = ttnn.num_cores_to_corerangeset(56, [8, 7], True)


@pytest.mark.parametrize(
    "tensor_spec",
    [
        ttnn.TensorSpec((1, 2, 3, 4), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, tile=ttnn.Tile([16, 16])),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1),
        ttnn.TensorSpec(
            (2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims_except([0], core_ranges),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).block_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).height_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).width_sharded(
            core_ranges
        ),
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).sharded(
            (1, 37, 37), core_ranges, ttnn.ShardShapeAlignment.RECOMMENDED
        ),
    ],
)
def test_sharded_tensor_serialization(tmp_path, device, tensor_spec):
    torch.manual_seed(0)
    dtype = tt_dtype_to_torch_dtype[tensor_spec.dtype]
    py_tensor = torch.rand(list(tensor_spec.shape), dtype=dtype)
    tt_tensor = ttnn.from_torch(py_tensor, spec=tensor_spec, device=device)
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    ttnn_tensor_from_file = ttnn.load_tensor(str(file_name), device=device)
    assert ttnn_tensor_from_file.spec == tensor_spec
    torch_tensor_from_file = ttnn.to_torch(ttnn_tensor_from_file)
    assert torch.allclose(py_tensor, torch_tensor_from_file)
