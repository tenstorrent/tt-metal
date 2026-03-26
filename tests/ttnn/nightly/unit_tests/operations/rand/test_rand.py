# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import math


def get_types_from_binding_framework():
    if hasattr(ttnn.DataType, "_member_map_"):
        # nanobind
        ALL_TYPES = [dtype for _, dtype in ttnn.DataType._member_map_.items() if dtype != ttnn.DataType.INVALID]
    else:
        raise Exception("test_rand.py: ttnn.DataType has unexpected way of holding values. Not matching nanobind.")

    return ALL_TYPES


DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]
ALL_TYPES = get_types_from_binding_framework()


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def check_uniform_distribution(data, value_range=(0, 1), is_discrete=False):
    n = data.numel()

    if n < 1000:
        print("[Warning] A meaningful analysis requires at least 1000 samples.")
        if n < 2:
            print("[Error] Cannot perform test with less than 2 data points.")
            return False

    start_value, end_value = value_range
    if is_discrete:
        min_val, max_val = start_value, end_value
    else:
        min_val, max_val = torch.aminmax(data)
        min_val = min_val.item()
        max_val = max_val.item()

    if min_val == max_val:
        return False

    # torch ops don't suport integer data types, convert to list
    data = data.detach().cpu().flatten().tolist()

    # Calculate sample statistics
    sample_mean = sum(data) / n
    sample_variance = sum([(x - sample_mean) ** 2 for x in data]) / n
    sample_std_dev = math.sqrt(sample_variance)

    # Calculate theoretical statistics
    if is_discrete:
        theoretical_mean = (start_value + end_value) / 2
        N = end_value - start_value + 1
        theoretical_std_dev = math.sqrt((N**2 - 1) / 12)
    else:
        theoretical_mean = (min_val + max_val) / 2
        theoretical_std_dev = (max_val - min_val) / math.sqrt(12)

    mean_diff = abs(sample_mean - theoretical_mean) / theoretical_mean * 100 if theoretical_mean != 0 else 0
    std_dev_diff = (
        abs(sample_std_dev - theoretical_std_dev) / theoretical_std_dev * 100 if theoretical_std_dev != 0 else 0
    )

    treshold_percentage = 4
    if mean_diff < treshold_percentage and std_dev_diff < treshold_percentage:
        return True

    return False


@pytest.mark.xfail(reason="BFLOAT4_B/UINT8 and `uint32/int32/BFLOAT8_B` for row major layout are not supported.")
@pytest.mark.parametrize("dtype", ALL_TYPES)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_tensor_dtype_and_value_range(device, dtype, layout):
    shape = (1024, 1024)
    if is_ttnn_float_type(dtype):
        tensor = ttnn.rand(shape, dtype=dtype, device=device, layout=layout)
        low = 0
        high = 1
    elif dtype == ttnn.int32:
        low = -100
        high = 100
        tensor = ttnn.rand(shape, low=low, high=high, dtype=dtype, device=device, layout=layout)
    else:
        low = 0
        high = 100
        tensor = ttnn.rand(shape, low=low, high=high, dtype=dtype, device=device, layout=layout)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)

    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_uniform_distribution(
        torch_tensor, value_range=(low, high), is_discrete=not is_ttnn_float_type(dtype)
    ), "The distribution of random values is not uniform!"


def test_rand_defaults(device):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_rand_shapes(device, shapes):
    tensor = ttnn.rand(shapes, device=device)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("dim", [i for i in range(32)])
def test_rand_dims(dim, device):
    shape = (dim, dim)
    tensor = ttnn.rand(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_rand_with_memory_config(device, mem_config):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


def test_rand_different_from_to_values(device):
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    dtype = ttnn.float32

    low_1, high_1 = 0.0, 1.0
    tensor_1 = ttnn.rand(shape, device=device, dtype=dtype, low=low_1, high=high_1)
    data_1 = ttnn.to_torch(tensor_1).float()
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after first rand, got {device.num_program_cache_entries()}"

    low_2, high_2 = 5.0, 10.0
    tensor_2 = ttnn.rand(shape, device=device, dtype=dtype, low=low_2, high=high_2)
    data_2 = ttnn.to_torch(tensor_2).float()
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after second rand (cache hit; from/to runtime-only), got {device.num_program_cache_entries()}"

    for torch_tensor, value_range in ((data_1, (low_1, high_1)), (data_2, (low_2, high_2))):
        assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
        assert check_uniform_distribution(
            torch_tensor, value_range=value_range, is_discrete=False
        ), "The distribution of random values is not uniform!"

    device.disable_and_clear_program_cache()


def test_rand_invalid_args(device):
    """
    Passing invalid args should raise TypeError.
    """

    with pytest.raises(TypeError):
        # expected list or tuple
        ttnn.rand(5, device=device)

    with pytest.raises(TypeError):
        # expected positive dim values
        ttnn.rand([2, -1], device=device)

    with pytest.raises(TypeError):
        # expected ttnn.LAYOUT type
        ttnn.rand([2, 2], device=device, layout="ROW_MAJOR")

    with pytest.raises(TypeError):
        # expected  ttnn.MemoryConfig type
        ttnn.rand([2, 2], device=device, memory_config="DRAM")

    with pytest.raises(TypeError):
        # expected  ttnn.Device type
        ttnn.rand([2, 2], device="WORMHOLE")

    with pytest.raises(TypeError):
        # expected  ttnn.DataType type
        ttnn.rand([2, 2], device=device, dtype="ttnn.bfloat16")


# ---------------------------------------------------------------------------
# Multi-device tests (mesh_mapper)
# ---------------------------------------------------------------------------


def _shard_placements(mesh_shape, shard_dim):
    """Build a placements list that shards `shard_dim` on the non-trivial mesh axis."""
    return [
        ttnn.PlacementShard(shard_dim) if mesh_shape[i] > 1 else ttnn.PlacementReplicate()
        for i in range(len(mesh_shape))
    ]


def _replicate_placements(mesh_shape):
    return [ttnn.PlacementReplicate() for _ in range(len(mesh_shape))]


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_shard(mesh_device):
    """
    Shard a random tensor across devices along dim 0, then verify:
      - mesh_mapper produces the right per-device shard shapes
      - unique_per_device seeding gives each device a distinct sequence
      - composed result is uniformly distributed
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 42
    shard_dim = 0
    per_device_rows = 256
    cols = 256
    full_shape = (per_device_rows * num_devices, cols)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_shard_placements(mesh_shape, shard_dim)),
    )

    composed = ttnn.to_torch(
        sharded_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
    ).float()

    assert tuple(composed.shape) == full_shape, f"Expected {full_shape}, got {tuple(composed.shape)}"
    assert not torch.isnan(composed).any(), "Composed tensor contains NaN values"
    assert check_uniform_distribution(composed), "Composed tensor is not uniformly distributed"

    shards = torch.chunk(composed, num_devices, dim=shard_dim)
    for i in range(1, len(shards)):
        assert not torch.equal(
            shards[0], shards[i]
        ), f"Shard 0 and shard {i} are identical — unique_per_device seeding did not work"


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_replicate(mesh_device):
    """
    Replicate a random tensor across devices with a fixed seed, then verify
    that every device holds the same data.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 42
    shape = (256, 256)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    replicated_tensor = ttnn.rand(
        shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_replicate_placements(mesh_shape)),
    )

    device_tensors = ttnn.get_device_tensors(replicated_tensor)
    shards = [ttnn.to_torch(t).float() for t in device_tensors]

    for i in range(1, len(shards)):
        assert torch.equal(
            shards[0], shards[i]
        ), f"Replicated shard 0 and shard {i} differ — replicate seeding is broken"

    assert not torch.isnan(shards[0]).any(), "Replicated tensor contains NaN values"
    assert check_uniform_distribution(shards[0]), "Replicated tensor is not uniformly distributed"


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(2, id="1x2_grid"), pytest.param((2, 1), id="2x1_grid")],
    indirect=True,
)
def test_rand_mesh_shard_matches_single_device(mesh_device):
    """
    Verify that each shard of a multi-device sharded ttnn.rand matches a
    replicated ttnn.rand run with the equivalent per-device seed.

    The kernel seeds core `i` on device at linear index `d` as:
        core_seed = user_seed + i + d * num_active_cores
    where num_active_cores = min(compute_grid_total, num_tiles).

    For each device d we run a replicated (no mesh_mapper) ttnn.rand with
    seed = user_seed + d * num_active_cores, then compare device d's copy
    against the corresponding shard from the sharded run.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("Need at least 2 devices")

    seed = 100
    shard_dim = 0
    per_device_rows = 256
    cols = 256
    full_shape = (per_device_rows * num_devices, cols)
    shard_shape = (per_device_rows, cols)
    dtype = ttnn.float32
    mesh_shape = tuple(mesh_device.shape)

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(_shard_placements(mesh_shape, shard_dim)),
    )
    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded_tensor)]

    # num_active_cores mirrors split_work_to_cores: min(grid_total, num_tiles)
    TILE_HW = 32 * 32
    grid = mesh_device.compute_with_storage_grid_size()
    num_tiles = (per_device_rows * cols) // TILE_HW
    num_active_cores = min(grid.x * grid.y, num_tiles)

    for d in range(num_devices):
        device_seed = seed + d * num_active_cores

        # Replicated rand — every device gets the same data; pick device d's copy
        reference_tensor = ttnn.rand(shard_shape, mesh_device, dtype=dtype, seed=device_seed)
        reference = ttnn.to_torch(ttnn.get_device_tensors(reference_tensor)[d]).float()

        assert tuple(shards[d].shape) == shard_shape, f"Shard {d}: expected {shard_shape}, got {tuple(shards[d].shape)}"
        assert torch.equal(shards[d], reference), (
            f"Shard {d} does not match replicated rand with seed={device_seed} " f"(offset {d * num_active_cores})"
        )


@pytest.mark.parametrize(
    "mesh_device, shard_mesh_dim",
    [
        pytest.param((2, 2), 0, id="2x2_shard_dim0"),
        pytest.param((2, 2), 1, id="2x2_shard_dim1"),
    ],
    indirect=["mesh_device"],
)
def test_rand_mesh_2d_shard_and_replicate(mesh_device, shard_mesh_dim):
    """
    On a 2D mesh, shard along one mesh dimension and replicate along the other.
    Verify:
      - Devices along the replicate axis hold identical data.
      - Devices along the shard axis hold distinct data.
    """
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape
    if rows * cols < 4:
        pytest.skip("Need at least 4 devices for a 2x2 mesh")

    seed = 77
    shard_dim = 0
    per_shard_rows = 256
    num_shards = mesh_shape[shard_mesh_dim]
    full_shape = (per_shard_rows * num_shards, 256)
    dtype = ttnn.float32

    placements = [
        ttnn.PlacementShard(shard_dim) if i == shard_mesh_dim else ttnn.PlacementReplicate()
        for i in range(len(mesh_shape))
    ]

    sharded_tensor = ttnn.rand(
        full_shape,
        mesh_device,
        dtype=dtype,
        seed=seed,
        mesh_mapper=ttnn.MeshMapperConfig(placements),
    )

    device_tensors = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sharded_tensor)]

    replicate_mesh_dim = 1 - shard_mesh_dim

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coord = (r, c)

            # Check replicas: devices differing only on the replicate axis must match.
            if coord[replicate_mesh_dim] > 0:
                replica_coord = list(coord)
                replica_coord[replicate_mesh_dim] = 0
                replica_idx = replica_coord[0] * cols + replica_coord[1]
                assert torch.equal(device_tensors[idx], device_tensors[replica_idx]), (
                    f"Device {coord} should be a replica of device {tuple(replica_coord)} " f"but data differs"
                )

            # Check shards: devices differing on the shard axis must differ.
            if coord[shard_mesh_dim] > 0:
                shard_neighbor = list(coord)
                shard_neighbor[shard_mesh_dim] = 0
                neighbor_idx = shard_neighbor[0] * cols + shard_neighbor[1]
                assert not torch.equal(device_tensors[idx], device_tensors[neighbor_idx]), (
                    f"Device {coord} and device {tuple(shard_neighbor)} are on different "
                    f"shards but hold identical data"
                )
