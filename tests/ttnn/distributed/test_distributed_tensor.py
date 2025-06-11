# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import nearest_32


def generate_ttnn_tensor_of_shards(num_shards, dtype):
    torch.manual_seed(1234)

    unconcatenated_ttnn_tensor_shards = []

    for i in range(num_shards):
        if dtype == ttnn.uint16:
            unconcatenated_ttnn_tensor_shards.append(
                ttnn.from_torch(
                    torch.randint(0, 32767, (1, 1, 32, 64 // num_shards)), dtype=dtype, layout=ttnn.TILE_LAYOUT
                )
            )
        else:
            unconcatenated_ttnn_tensor_shards.append(
                ttnn.from_torch(torch.randn(1, 1, 32, 64 // num_shards), dtype=dtype, layout=ttnn.TILE_LAYOUT)
            )

    return ttnn.aggregate_as_tensor(unconcatenated_ttnn_tensor_shards)


def generate_2d_sharded_ttnn_tensor(num_shards, dtype, M, K, mesh_shape, shard_dim):
    rows, cols = mesh_shape
    row_dim, col_dim = shard_dim

    unsharded_torch_tensor_shards = []

    for i in range(num_shards):
        if dtype == ttnn.uint16:
            unsharded_torch_tensor_shards.append(torch.randint(0, 32767, (1, 1, M, K)))
        else:
            unsharded_torch_tensor_shards.append(torch.randn(1, 1, M, K))

    unsharded_torch_tensor = torch.cat(unsharded_torch_tensor_shards, dim=1)

    # Shard along rows
    row_tensors = torch.chunk(unsharded_torch_tensor, rows, dim=row_dim)

    # Shard along columns
    if col_dim == 0:
        torch_sharded_shards_as_ttnn_tensors = [
            ttnn.from_torch(t.clone(), dtype=dtype, layout=ttnn.TILE_LAYOUT) for t in row_tensors for _ in range(cols)
        ]
    else:
        torch_sharded_shards_as_ttnn_tensors = [
            ttnn.from_torch(tt, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            for t in row_tensors
            for tt in torch.chunk(t, cols, dim=col_dim)
        ]

    return ttnn.aggregate_as_tensor(torch_sharded_shards_as_ttnn_tensors)


@pytest.mark.parametrize(
    "mesh_device",
    [
        8,
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_replicate_to_tensor_mesh_torch_comparison(mesh_device, dtype):
    torch.manual_seed(1234)

    torch_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, 32, 256))
    else:
        torch_tensor = torch.randn(1, 1, 32, 256)
    torch_replicated_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=torch_mapper,
        device=mesh_device,
    )

    torch_replicated_shards = ttnn.get_device_tensors(torch_replicated_tensor)

    unreplicated_ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    xtensor_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    xtensor_replicated_tensor = ttnn.distribute_tensor(unreplicated_ttnn_tensor, xtensor_mapper, mesh_device)
    xtensor_replicated_shards = ttnn.get_device_tensors(xtensor_replicated_tensor)

    assert len(xtensor_replicated_shards) == len(torch_replicated_shards)

    out_passes = []
    for i in range(len(torch_replicated_shards)):
        out_pass, out_pcc = comp_pcc(
            ttnn.to_torch(xtensor_replicated_shards[i]), ttnn.to_torch(torch_replicated_shards[i]), pcc=0.99
        )
        out_passes.append(out_pass)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard_to_tensor_mesh_torch_comparison(mesh_device, dtype):
    torch.manual_seed(1234)

    torch_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, 32, 256))
    else:
        torch_tensor = torch.randn(2, 2, 32, 256)
    torch_sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=torch_mapper,
        device=mesh_device,
    )

    unsharded_ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    xtensor_mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

    xtensor_sharded_shards = ttnn.get_device_tensors(
        ttnn.distribute_tensor(unsharded_ttnn_tensor, xtensor_mapper, mesh_device)
    )
    torch_sharded_shards = ttnn.get_device_tensors(torch_sharded_tensor)

    out_pass1, out_pcc = comp_pcc(
        ttnn.to_torch(torch_sharded_shards[0]), ttnn.to_torch(xtensor_sharded_shards[0]), pcc=0.99
    )
    logger.info(f"Shard 1 PCC value: {out_pcc}")
    out_pass2, out_pcc = comp_pcc(
        ttnn.to_torch(torch_sharded_shards[1]), ttnn.to_torch(xtensor_sharded_shards[1]), pcc=0.99
    )
    logger.info(f"Shard 2 PCC value: {out_pcc}")

    assert out_pass1 and out_pass2


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 1), (2, 1), id="2x1_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard2d_to_tensor_mesh_torch_comparison(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (1, 1, M, K))
    else:
        torch_tensor = torch.randn(1, 1, M, K)

    shard_dim = (3, 0)

    torch_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

    torch_sharded_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=torch_mapper,
        device=mesh_device,
    )

    unsharded_ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    xtensor_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(shard_dim[0], shard_dim[1]),
    )

    xtensor_sharded_shards = ttnn.get_device_tensors(
        ttnn.distribute_tensor(unsharded_ttnn_tensor, xtensor_mapper, mesh_device)
    )
    torch_sharded_shards = ttnn.get_device_tensors(torch_sharded_tensor)

    assert len(torch_sharded_shards) == len(xtensor_sharded_shards)

    out_passes = []
    for i in range(len(torch_sharded_shards)):
        out_pass, out_pcc = comp_pcc(
            ttnn.to_torch(torch_sharded_shards[i]), ttnn.to_torch(xtensor_sharded_shards[i]), pcc=0.99
        )
        out_passes.append(out_pass)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_to_tensor_mesh_torch_comparison(mesh_device, dtype):
    torch_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3)

    num_shards = mesh_device.get_num_devices()

    unconcatenated_ttnn_tensor = generate_ttnn_tensor_of_shards(num_shards, dtype)

    torch_concat_tensor = ttnn.to_torch(unconcatenated_ttnn_tensor, mesh_composer=torch_composer)

    xtensor_composer = ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=3)
    xtensor_concat_tensor = ttnn.to_torch(ttnn.aggregate_tensor(unconcatenated_ttnn_tensor, xtensor_composer))

    out_pass, out_pcc = comp_pcc(torch_concat_tensor, xtensor_concat_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 1), (2, 1), id="2x1_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(16, 32, 64), pytest.param(16, 64, 32)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat2d_to_tensor_mesh_torch_comparison(M, K, N, dtype, mesh_shape, mesh_device):
    torch.manual_seed(1234)

    shard_dim = (3, 0)
    concat_dim = (1, 3)

    num_shards = 4

    unconcatenated_ttnn_tensor = generate_2d_sharded_ttnn_tensor(num_shards, dtype, M, K, mesh_shape, shard_dim)

    torch_composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, dims=concat_dim)
    torch_concat_tensor = ttnn.to_torch(unconcatenated_ttnn_tensor, mesh_composer=torch_composer)

    xtensor_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(concat_dim[0], concat_dim[1]),
    )
    xtensor_concat_tensor = ttnn.to_torch(ttnn.aggregate_tensor(unconcatenated_ttnn_tensor, xtensor_composer))

    out_pass, out_pcc = comp_pcc(torch_concat_tensor, xtensor_concat_tensor, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_replicate_to_tensor_mesh(mesh_device, dtype):
    pytest.skip(f"Covered through distributed.py paths for now")

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
    replicated_tensor = ttnn.distribute_tensor(to_repl, mapper, mesh_device)
    replicated_shards = ttnn.get_device_tensors(replicated_tensor)

    out_passes = []
    for i in range(len(replicated_shards)):
        out_pass, out_pcc = comp_pcc(ttnn.to_torch(replicated_shards[i]), torch_tensor, pcc=0.99)
        out_passes.append(out_pass)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard_to_tensor_mesh(mesh_device, dtype):
    pytest.skip(f"Covered through distributed.py paths for now")

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

    xtensor_sharded_shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))
    torch_sharded_shards = torch.chunk(torch_tensor, mesh_device.get_num_devices(), dim=3)

    assert len(xtensor_sharded_shards) == len(torch_sharded_shards) == 2

    out_pass1, out_pcc = comp_pcc(torch_sharded_shards[0], ttnn.to_torch(xtensor_sharded_shards[0]), pcc=0.99)
    logger.info(f"Shard 1 PCC value: {out_pcc}")
    out_pass2, out_pcc = comp_pcc(torch_sharded_shards[1], ttnn.to_torch(xtensor_sharded_shards[1]), pcc=0.99)
    logger.info(f"Shard 2 PCC value: {out_pcc}")

    assert out_pass1 and out_pass2


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_to_tensor(mesh_device, dtype):
    pytest.skip(f"Covered through distributed.py paths for now")

    torch.manual_seed(1234)

    num_shards = 4

    torch_shards = []

    for i in range(num_shards):
        if dtype == ttnn.uint16:
            torch_shards.append(torch.randint(0, 32767, (1, 1, 32, 64 // num_shards)))
        else:
            torch_shards.append(torch.randn(1, 1, 32, 64 // num_shards))

    # This will be the same as the generated unconcatenated_ttnn_tensor due to the shared seeding
    torch_concat_tensor = torch.cat(torch_shards, dim=3)

    unconcatenated_ttnn_tensor = generate_ttnn_tensor_of_shards(num_shards, dtype)

    composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

    xtensor_concat_tensor = ttnn.aggregate_tensor(unconcatenated_ttnn_tensor, composer)

    out_pass, out_pcc = comp_pcc(torch_concat_tensor, ttnn.to_torch(xtensor_concat_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat_slice_to_tensor(mesh_device, dtype):
    pytest.skip(f"Covered through distributed.py paths for now")

    torch.manual_seed(1234)

    num_shards = 4

    torch_shards = []

    for i in range(num_shards):
        if dtype == ttnn.uint16:
            torch_shards.append(torch.randint(0, 32767, (1, 1, 32, 64 // num_shards)))
        else:
            torch_shards.append(torch.randn(1, 1, 32, 64 // num_shards))

    # This will be the same as concatenating the generated unconcatenated_ttnn_shards due to the shared order and seeding
    torch_concat_tensor = torch.cat(torch_shards, dim=3)

    unconcatenated_ttnn_shards = generate_ttnn_tensor_of_shards(num_shards, dtype)

    composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

    xtensor_concat_tensor = ttnn.aggregate_tensor(unconcatenated_ttnn_shards, composer)

    out_pass, out_pcc = comp_pcc(torch_concat_tensor, ttnn.to_torch(xtensor_concat_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 1), (2, 1), id="2x1_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_shard2d_to_tensor_mesh(M, K, N, dtype, mesh_shape, mesh_device):
    pytest.skip(f"Covered through distributed.py paths for now")

    torch.manual_seed(1234)

    if dtype == ttnn.uint16:
        torch_tensor = torch.randint(0, 32767, (2, 2, M, K))
    else:
        torch_tensor = torch.randn(2, 2, M, K)

    shard_dim = (3, 0)

    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(shard_dim[0], shard_dim[1]))

    xtensor_sharded_shards = ttnn.get_device_tensors(ttnn.distribute_tensor(ttnn_tensor, mapper, mesh_device))

    rows, cols = mesh_shape
    row_dim, col_dim = shard_dim

    # Shard along rows
    row_tensors = torch.chunk(torch_tensor, rows, dim=row_dim)

    # Shard along columns
    if col_dim == 0:
        torch_sharded_shards = [t.clone() for t in row_tensors for _ in range(cols)]
    else:
        torch_sharded_shards = [tt for t in row_tensors for tt in torch.chunk(t, cols, dim=col_dim)]

    assert len(xtensor_sharded_shards) == len(torch_sharded_shards)

    out_passes = []
    for i in range(len(torch_sharded_shards)):
        out_pass, out_pcc = comp_pcc(torch_sharded_shards[i], ttnn.to_torch(xtensor_sharded_shards[i]), pcc=0.99)
        out_passes.append(out_pass)
        logger.info(f"Shard {i} PCC value: {out_pcc}")

    assert all(out_passes)


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 1), (2, 1), id="2x1_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N",
    [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
)
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
def test_concat2d_to_tensor(M, K, N, dtype, mesh_shape, mesh_device):
    pytest.skip(f"Covered through distributed.py paths for now")

    torch.manual_seed(1234)

    num_shards = 4

    shard_dim = (3, 0)
    concat_dim = (1, 3)

    unconcatenated_ttnn_tensor = generate_2d_sharded_ttnn_tensor(num_shards, dtype, M, K, mesh_shape, shard_dim)

    rows, cols = mesh_shape
    row_dim, col_dim = concat_dim

    torch_shards = [
        ttnn.to_torch(tt_input_tensor, mesh_composer=None)
        for tt_input_tensor in ttnn.get_device_tensors(unconcatenated_ttnn_tensor)
    ]

    # Reshape the list of shards into a 2D list representing the device mesh
    torch_shards_2d = [torch_shards[i : i + cols] for i in range(0, len(torch_shards), cols)]

    # Concatenate along columns first (within each row)
    row_concatenated = [torch.cat(rows, dim=col_dim) for rows in torch_shards_2d]

    # Then concatenate the resulting tensors along rows
    torch_concat_tensor = torch.cat(row_concatenated, dim=row_dim)

    composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(concat_dim[0], concat_dim[1]),
    )

    xtensor_concat_tensor = ttnn.aggregate_tensor(unconcatenated_ttnn_tensor, composer)

    out_pass, out_pcc = comp_pcc(torch_concat_tensor, ttnn.to_torch(xtensor_concat_tensor), pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")

    assert out_pass
