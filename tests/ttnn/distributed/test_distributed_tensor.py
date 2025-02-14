# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from ttnn import (
    ShardTensorToMesh,
    ShardTensor2dMesh,
    ReplicateTensorToMesh,
    ConcatMeshToTensor,
    ConcatMesh2dToTensor,
    MeshToTensor,
    TensorToMesh,
)
from models.utility_functions import nearest_32


@pytest.mark.parametrize(
    "mesh_device",
    [
        32,
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_replicate_to_tensor_mesh(mesh_device, dtype):
    torch.manual_seed(1234)

    torch_tensor = torch.randn(1, 1, 32, 8192)
    to_repl = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    out_tensors = ttnn.ReplicateTensorToMesh(mesh_device).map(to_repl)

    test = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    out_pass, out_pcc = comp_pcc(out_tensors[0], test, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard_to_tensor_mesh(mesh_device, dtype):
    torch.manual_seed(1234)

    torch_tensor = torch.randn(1, 1, 8192, 32768)
    tensor_shards = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    test = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    out_tensors = ttnn.ShardTensorToMesh(mesh_device, dim=3).map(tensor_shards)

    out_tensor = ttnn.aggregate_as_tensor(out_tensors)

    out_pass, out_pcc = comp_pcc(out_tensor, test, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_to_tensor(mesh_device, dtype):
    torch.manual_seed(1234)

    torch_tensor = torch.randn(1, 1, 8192, 32768)
    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    test = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    sharded_tensors = ttnn.ShardTensorToMesh(mesh_device, dim=3).map(to_shard)

    out_tensor = ttnn.to_torch(ttnn.ConcatMeshToTensor(dim=3).compose(), dtype=ttnn.bfloat16, device=mesh_device)

    out_pass, out_pcc = comp_pcc(out_tensor, test, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 8192, 28 * 1024), pytest.param(32, 28 * 1024, 8192)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard2d_to_tensor_mesh(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    torch_tensor = torch.randn(1, 1, M, K)
    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (None, 3) if K < N else (3, None)  # None means to replicate along this dim

    K = K // mesh_shape[1] if K < N else K // mesh_shape[0]
    N = N // mesh_shape[0] if K < N else N // mesh_shape[1]

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M // core_grid.y, K // core_grid.x),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
    )

    out_tensors = ttnn.get_device_tensors(
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dim).map(to_shard)
    )

    for tensor in out_tensors:
        print(tensor)

    # out_pass, out_pcc = comp_pcc(tensor_shards, out, pcc=0.99)
    # logger.info(f"PCC value: {out_pcc}")
    # assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 8192, 28 * 1024), pytest.param(32, 28 * 1024, 8192)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat2d_to_tensor(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    torch_tensor = torch.randn(1, 1, M, K)
    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (None, 3) if K < N else (3, None)  # None means to replicate along this dim
    concat_dim = (3, 1) if K < N else (1, 3)  # dim 1 for reduce, dim 3 for concatenating fractures

    K = K // mesh_shape[1] if K < N else K // mesh_shape[0]
    N = N // mesh_shape[0] if K < N else N // mesh_shape[1]

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(M // core_grid.y, K // core_grid.x),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
    )

    sharded_tensors = ttnn.get_device_tensors(
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dim).map(to_shard)
    )

    out = ttnn.to_torch(
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dim, mesh_shape=mesh_shape).compose(
            sharded_tensors
        ),
    )

    out_pass, out_pcc = comp_pcc(out, torch_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass
