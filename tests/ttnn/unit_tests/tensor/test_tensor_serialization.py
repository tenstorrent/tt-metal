# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib
import re

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype, TORCH_INTEGER_DTYPES

pytestmark = pytest.mark.use_module_device


def _kmd_supports_read_only_page_pinning():
    version_path = pathlib.Path("/sys/module/tenstorrent/version")
    if not version_path.exists():
        return False
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_path.read_text().strip())
    return match is not None and tuple(int(component) for component in match.groups()) >= (2, 9, 0)


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

    if dtype in TORCH_INTEGER_DTYPES:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.tensorbin")
    ttnn.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttnn.load_tensor(str(file_name)).to_torch()

    torch_tensor_from_file = torch_tensor_from_file.to(torch_tensor.dtype)

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing


def test_large_read_only_file_backed_tensor_upload(tmp_path, device):
    if not _kmd_supports_read_only_page_pinning():
        pytest.skip("Device-read-only page pinning requires KMD 2.9.0 or newer")

    # 1024 * 9216 * 4 bytes = 36 MiB, above Metal's 32 MiB pinned H2D threshold.
    shape = (1, 1, 1024, 9216)
    torch_tensor = torch.full(shape, 1.25, dtype=torch.float32)
    host_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.float32)
    file_name = tmp_path / "large_read_only.tensorbin"
    ttnn.dump_tensor(str(file_name), host_tensor)

    # load_tensor opens the file O_RDONLY and maps it PROT_READ | MAP_PRIVATE before uploading.
    device_tensor = ttnn.load_tensor(str(file_name), device=device)
    result = ttnn.to_torch(device_tensor)
    assert torch.equal(result, torch_tensor)


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
