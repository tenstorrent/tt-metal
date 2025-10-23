#!/usr/bin/env python3
"""
Tests for automatic composition of multi-device sharded tensors using TensorTopology.

This test module validates that the auto-composition logic correctly infers
MeshToTensor composers from a sharded ttnn.Tensor's topology metadata.

It validates both host-sharded and device-sharded cases.
"""

import os

import pytest
import torch
import ttnn

from tt_transformers_v2.src.testing.auto_compose import to_torch_auto_compose

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def mesh_shape():
    """Get mesh shape from environment variable."""
    return {
        "N150": [1, 1],
        "N300": [1, 2],
        "N150x4": [1, 4],
        "T3K": [1, 8],
        "TG": [8, 4],
        "P150": [1, 1],
        "P300": [1, 2],
        "P150x4": [1, 4],
        "P150x8": [1, 8],
    }.get(os.environ.get("MESH_DEVICE"), [1, 1])


@pytest.fixture(scope="module")
def mesh_device(mesh_shape):
    """Create and yield a mesh device, cleanup on teardown."""
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape))
    except Exception as e:  # environment without TT hardware
        pytest.skip(f"Mesh device unavailable or no hardware detected: {e}")
        return
    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def num_devices(mesh_device):
    """Get number of devices in the mesh."""
    return mesh_device.get_num_devices()


# ======================================================================================
# Helper Functions
# ======================================================================================


def _make_known_pattern(num_chunks: int) -> torch.Tensor:
    """
    Produces shape [num_chunks, 1, 3, 1] with per-chunk distinct values.
    Chunk i contains [i*1, i*2, i*3].
    """
    rows = []
    for i in range(num_chunks):
        rows.append(torch.tensor([[[i * 1.0], [i * 2.0], [i * 3.0]]]).transpose(0, 1))  # [1,3,1]
    data = torch.stack(rows, dim=0)  # [num_chunks,1,3,1]
    return data.to(torch.bfloat16)


def _make_arange_bf16(shape: tuple[int, ...]) -> torch.Tensor:
    """Create a deterministic tensor with arange data and bfloat16 dtype."""
    numel = 1
    for s in shape:
        numel *= s
    data = torch.arange(numel, dtype=torch.float32).reshape(shape)
    return data.to(torch.bfloat16)


def _pos_dim(dim: int, rank: int) -> int:
    """Convert possibly-negative dim to positive index for given rank."""
    return dim % rank


def _get_hw_shard_unit() -> int:
    """
    Hardware-related shard unit threshold (default 32).
    Override via env var TT_TEST_SHARD_UNIT for future hardware.
    """
    try:
        return int(os.environ.get("TT_TEST_SHARD_UNIT", "32"))
    except Exception:
        return 32


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("min_devices", [2])
def test_host_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int, min_devices: int) -> None:
    """Test automatic composition of host-sharded 1D tensors."""
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Create a host-sharded tensor along dim=0
    tt_host_sharded = ttnn.from_torch(
        torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_host_sharded, device=mesh_device)

    # Reference using explicit composer
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on host-sharded tensor"


@pytest.mark.parametrize("min_devices", [2])
def test_device_sharded_1d(mesh_device: ttnn.MeshDevice, num_devices: int, min_devices: int) -> None:
    """Test automatic composition of device-sharded 1D tensors."""
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Distribute to mesh device directly (device storage)
    tt_dev_sharded = ttnn.from_torch(
        torch_in, device=mesh_device, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )
    # [INFO]{ equivalent to:
    # mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0)
    # tt_host = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16)
    # tt_dev_sharded = ttnn.distribute_tensor(tt_host, mapper, mesh_device)
    # }[INFO]

    # Validate that auto-composition returns original torch input
    torch_auto = to_torch_auto_compose(tt_dev_sharded, device=mesh_device)

    # Reference using explicit composer through high-level API
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch (device-sharded)"
    assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch on device-sharded tensor"


# --------------------------------------------------------------------------------------
# Additional coverage: shard various tensor dims on 1D meshes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("min_devices", [2])
def test_host_sharded_various_dims(mesh_device: ttnn.MeshDevice, num_devices: int, dim: int, min_devices: int) -> None:
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    torch_in = _make_arange_bf16(tuple(shape))

    tt_host_sharded = ttnn.from_torch(
        torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    )

    torch_auto = to_torch_auto_compose(tt_host_sharded, device=mesh_device)
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("min_devices", [2])
def test_device_sharded_various_dims(
    mesh_device: ttnn.MeshDevice, num_devices: int, dim: int, min_devices: int
) -> None:
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    torch_in = _make_arange_bf16(tuple(shape))

    tt_dev_sharded = ttnn.from_torch(
        torch_in, device=mesh_device, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    )

    torch_auto = to_torch_auto_compose(tt_dev_sharded, device=mesh_device)
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)


# --------------------------------------------------------------------------------------
# Coverage for 2D mesh sharding: shard-shard and replicate-shard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
def test_host_sharded_2d_shard_shard(mesh_device: ttnn.MeshDevice, dims_pair: tuple[int, int]) -> None:
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] <= 1 or mesh_shape[1] <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1, "Shard dims for 2D sharding must be distinct"

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    torch_in = _make_arange_bf16(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    tt_host_sharded = ttnn.from_torch(torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    torch_auto = to_torch_auto_compose(tt_host_sharded, device=mesh_device)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=composer)

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)


@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
def test_device_sharded_2d_shard_shard(mesh_device: ttnn.MeshDevice, dims_pair: tuple[int, int]) -> None:
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] <= 1 or mesh_shape[1] <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1, "Shard dims for 2D sharding must be distinct"

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    torch_in = _make_arange_bf16(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    tt_dev_sharded = ttnn.from_torch(torch_in, device=mesh_device, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    torch_auto = to_torch_auto_compose(tt_dev_sharded, device=mesh_device)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=composer)

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)


@pytest.mark.parametrize(
    "dims_pair,replicate_axis",
    [
        ((None, -1), 0),  # replicate along mesh dim 0, shard along last tensor dim
        ((1, None), 1),  # shard along tensor dim 1 on mesh dim 0, replicate mesh dim 1
    ],
)
def test_host_sharded_2d_with_replicate(
    mesh_device: ttnn.MeshDevice, dims_pair: tuple[object, object], replicate_axis: int
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[replicate_axis] <= 1 or mesh_shape[1 - replicate_axis] <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1 to observe replication")

    rank = 4
    # Determine which tensor axis is sharded (the non-None entry)
    shard_dim = [d for d in dims_pair if d is not None][0]
    shard_axis = _pos_dim(shard_dim, rank)
    shape = [2, 3, 4, 5]
    # Set size along sharded axis to corresponding mesh dim
    shape[shard_axis] = mesh_shape[1 - replicate_axis]
    torch_in = _make_arange_bf16(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims_pair)  # type: ignore[arg-type]
    tt_host = ttnn.from_torch(torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    torch_auto = to_torch_auto_compose(tt_host, device=mesh_device)

    # Expected: auto-composer concatenates replicas along axis 0
    repeat_factor = mesh_shape[replicate_axis]
    expected = torch_in.repeat((repeat_factor, 1, 1, 1))
    assert torch.equal(torch_auto, expected)


# --------------------------------------------------------------------------------------
# Tensor shape categories around hardware threshold (e.g., 32)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["lt", "eq", "gt"])  # per-shard length relative to threshold
@pytest.mark.parametrize("min_devices", [2])
def test_host_sharded_shape_thresholds(
    mesh_device: ttnn.MeshDevice, num_devices: int, category: str, min_devices: int
) -> None:
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    unit = _get_hw_shard_unit()
    if category == "lt":
        per_shard = max(1, unit - 1)
    elif category == "eq":
        per_shard = unit
    else:
        per_shard = unit + 1

    shard_dim = -1  # test last dimension as sharded axis (rank=4)
    rank = 4
    axis = _pos_dim(shard_dim, rank)
    # Global size across sharded dim = per_shard_len * num_devices
    shape = [2, 3, 4, 5]
    shape[axis] = per_shard * num_devices
    torch_in = _make_arange_bf16(tuple(shape))

    tt_host_sharded = ttnn.from_torch(
        torch_in, device=None, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    )

    torch_auto = to_torch_auto_compose(tt_host_sharded, device=mesh_device)
    torch_ref = ttnn.to_torch(tt_host_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim))

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)


@pytest.mark.parametrize("category", ["lt", "eq", "gt"])  # per-shard length relative to threshold
@pytest.mark.parametrize("min_devices", [2])
def test_device_sharded_shape_thresholds(
    mesh_device: ttnn.MeshDevice, num_devices: int, category: str, min_devices: int
) -> None:
    if num_devices < min_devices:
        pytest.skip(f"Test requires at least {min_devices} devices, found {num_devices}")

    unit = _get_hw_shard_unit()
    if category == "lt":
        per_shard = max(1, unit - 1)
    elif category == "eq":
        per_shard = unit
    else:
        per_shard = unit + 1

    shard_dim = -1  # test last dimension as sharded axis (rank=4)
    rank = 4
    axis = _pos_dim(shard_dim, rank)
    shape = [2, 3, 4, 5]
    shape[axis] = per_shard * num_devices
    torch_in = _make_arange_bf16(tuple(shape))

    tt_dev_sharded = ttnn.from_torch(
        torch_in,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
    )

    torch_auto = to_torch_auto_compose(tt_dev_sharded, device=mesh_device)
    torch_ref = ttnn.to_torch(tt_dev_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim))

    assert torch.equal(torch_ref, torch_in)
    assert torch.equal(torch_auto, torch_in)
