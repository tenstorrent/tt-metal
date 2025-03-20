# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import nearest_32


@pytest.mark.parametrize(
    "mesh_device",
    [
        32,
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_replicate_to_tensor_mesh_equality(mesh_device, dtype):
    torch.manual_seed(1234)

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)
    replicated_tensors = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        device=mesh_device,
    )

    orig_out_tensors = ttnn.to_torch(ttnn.get_device_tensors(replicated_tensors)[0])

    to_repl = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    replicated_tensors = ttnn.distribute_tensor(to_repl, mapper, mesh_device)
    out_tensors = ttnn.to_torch(ttnn.get_device_tensors(replicated_tensors)[0])

    out_pass, out_pcc = comp_pcc(orig_out_tensors, out_tensors, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard_to_tensor_mesh_equality(mesh_device, dtype):
    torch.manual_seed(1234)

    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, 32, 256))
    else:
        torch_tensor = torch.randn(2, 2, 32, 256)
    orig_sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        device=mesh_device,
    )

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    tensor_shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))
    orig_tensor_shards = ttnn.get_device_tensors(orig_sharded_tensor)

    out_pass1, out_pcc = comp_pcc(ttnn.to_torch(orig_tensor_shards[0]), ttnn.to_torch(tensor_shards[0]), pcc=0.99)
    logger.info(f"Shard 1 PCC value: {out_pcc}")
    out_pass2, out_pcc = comp_pcc(ttnn.to_torch(orig_tensor_shards[1]), ttnn.to_torch(tensor_shards[1]), pcc=0.99)
    logger.info(f"Shard 2 PCC value: {out_pcc}")

    assert out_pass1 and out_pass2


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard2d_to_tensor_mesh_equality(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, M, K))
    else:
        torch_tensor = torch.randn(1, 1, M, K)

    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (0, 3) if K < N else (3, 0)

    mapper = ttnn.ShardTensorTo2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    orig_sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        device=mesh_device,
    )

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tensor_shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))
    orig_tensor_shards = ttnn.get_device_tensors(orig_sharded_tensor)

    out_passes = []
    for i in range(len(orig_tensor_shards)):
        out_passes[i], out_pcc = comp_pcc(orig_tensor_shards[i], ttnn.to_torch(tensor_shards[i]), pcc=0.99)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_to_tensor_mesh_equality(mesh_device, dtype):
    torch.manual_seed(1234)

    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)
    orig_sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        device=mesh_device,
    )

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3)

    orig_concat_tensor = ttnn.to_torch(orig_sharded_tensor, mesh_composer=composer)

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

    concat_tensor = ttnn.to_torch(
        ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)
    )

    out_pass, out_pcc = comp_pcc(orig_concat_tensor, concat_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat2d_to_tensor_mesh_equality(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, M, K))
    else:
        torch_tensor = torch.randn(2, 2, M, K)

    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (0, 3) if K < N else (3, 0)
    concat_dim = (3, 1) if K < N else (1, 3)

    mapper = ttnn.ShardTensorTo2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        device=mesh_device,
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, dims=concat_dim)

    orig_concat_tensor = ttnn.to_torch(sharded_tensor, mesh_composer=composer)

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    mapper = ttnn.shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    composer = ttnn.concat_2d_mesh_to_tensor_composer(mesh_device, dims=concat_dim)

    concat_tensor = ttnn.to_torch(
        ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)
    )

    out_pass, out_pcc = comp_pcc(orig_concat_tensor, concat_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


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

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)

    to_repl = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    replicated_tensors = ttnn.distribute_tensor(to_repl, mapper, mesh_device)
    out_tensors = ttnn.get_device_tensors(replicated_tensors)

    out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensors[0]), torch_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard_to_tensor_mesh(mesh_device, dtype):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, 32, 256))
    else:
        torch_tensor = torch.randn(2, 2, 32, 256)
    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    tensor_shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))
    orig_tensor_shards = torch.chunk(torch_tensor, mesh_device.get_num_devices(), dim=3)

    out_pass1, out_pcc = comp_pcc(orig_tensor_shards[0], ttnn.to_torch(tensor_shards[0]), pcc=0.99)
    logger.info(f"Shard 1 PCC value: {out_pcc}")
    out_pass2, out_pcc = comp_pcc(orig_tensor_shards[1], ttnn.to_torch(tensor_shards[1]), pcc=0.99)
    logger.info(f"Shard 2 PCC value: {out_pcc}")

    assert out_pass1 and out_pass2


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_to_tensor(mesh_device, dtype):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)
    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

    out_tensor = ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)

    out_pass, out_pcc = comp_pcc(torch_tensor, ttnn.to_torch(out_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_slice_to_tensor(mesh_device, dtype):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)
    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

    sharded_tensor = ttnn.distribute_tensor(to_shard, mapper, mesh_device)

    shards = ttnn.get_device_tensors(sharded_tensor)

    out_tensor = ttnn.aggregate_tensor(shards, composer)

    out_pass, out_pcc = comp_pcc(torch_tensor, ttnn.to_torch(out_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard2d_to_tensor_mesh(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, M, K))
    else:
        torch_tensor = torch.randn(2, 2, M, K)
    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (0, 3) if K < N else (3, 0)

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    mapper = ttnn.shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))

    rows, cols = mesh_shape
    row_dim, col_dim = shard_dim

    # Shard along rows
    row_tensors = torch.chunk(torch_tensor, rows, dim=row_dim)

    # Shard along columns
    if col_dim == 0:
        orig_tensor_shards = [t.clone() for t in row_tensors for _ in range(cols)]
    else:
        orig_tensor_shards = [tt for t in row_tensors for tt in torch.chunk(t, cols, dim=col_dim)]

    out_passes = []
    for i in range(len(orig_tensor_shards)):
        out_passes[i], out_pcc = comp_pcc(orig_tensor_shards[i], ttnn.to_torch(shards[i]), pcc=0.99)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat2d_to_tensor(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, M, K))
    else:
        torch_tensor = torch.randn(1, 1, M, K)
    core_grid = ttnn.CoreGrid(y=1, x=8)

    # If K < N it's FF1-like test case, else FF2-like test case
    shard_dim = (0, 3) if K < N else (3, 0)
    concat_dim = (3, 1) if K < N else (1, 3)

    to_shard = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    mapper = ttnn.shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    composer = ttnn.concat_2d_mesh_to_tensor_composer(mesh_device, dims=concat_dim)

    out_tensor = ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)

    out_pass, out_pcc = comp_pcc(torch_tensor, ttnn.to_torch(out_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass
