# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for automatic composition of multi-device sharded tensors using TensorTopology.

This test module validates that the auto-composition logic correctly infers
MeshToTensor composers from a sharded ttnn.Tensor's topology metadata.

It validates both host-sharded and device-sharded cases.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn
from models.common.auto_compose import _infer_mesh_composer_from_topology, to_torch_auto_compose

# ======================================================================================
# Test Parameters (for device-dependent tests)
# ======================================================================================


_DEVICE_TEST_MARKS = {
    "ttnn_mesh_device": pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            (1, 1),  # single device # [INFO] apply auto_compose on single device would incur error in c++ code
            (1, 2),  # 1D mesh, 2 devices
            (1, 4),  # 1D mesh, 4 devices
            (1, 8),  # 1D mesh, 8 devices
            (2, 4),  # 2D mesh, 8 devices
            (4, 8),  # 2D mesh, 32 devices
            (8, 4),  # 2D mesh, 32 devices
        ],
        ids=[
            "1x1",
            "1x2",
            "1x4",
            "1x8",
            "2x4",
            "4x8",
            "8x4",
        ],
        indirect=True,
    ),
    "layout_dtype": pytest.mark.parametrize(
        "layout,dtype",
        [
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),  # bfloat8_b only works with TILE_LAYOUT
            (ttnn.TILE_LAYOUT, ttnn.bfloat4_b),  # bfloat4_b only works with TILE_LAYOUT
        ],
        ids=["row_major_bf16", "tile_bf16", "tile_bf8b", "tile_bf4b"],
    ),
}


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


def _make_arange_dtype(
    shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16, min_value: float = 0, max_value: float = 100
) -> torch.Tensor:
    """Create a deterministic tensor with arange data and specified dtype."""
    numel = 1
    for s in shape:
        numel *= s
    # Generate values from min_value to max_value with step of 1
    values = torch.arange(min_value, max_value + 1, dtype=dtype)
    # Randomly sample indices (with replacement) to fill the tensor
    indices = torch.randint(0, len(values), size=(numel,))
    data = values[indices].reshape(shape)
    return data


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


def _build_and_compose_sharded(
    torch_in: torch.Tensor,
    device: ttnn.MeshDevice | None,
    layout,
    ttnn_mesh_device: ttnn.MeshDevice,
    shard_dim: int,
    dtype: torch.dtype = ttnn.bfloat16,
) -> tuple[ttnn.Tensor, torch.Tensor, torch.Tensor]:
    """Build sharded tensor and compose it back to torch."""
    tt_sharded = ttnn.from_torch(
        torch_in,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=shard_dim),
    )
    torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device if device is None else None)
    torch_ref = ttnn.to_torch(tt_sharded, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_mesh_device, dim=shard_dim))
    return tt_sharded, torch_auto, torch_ref


# ======================================================================================
# Device-Dependent Tests (require mesh device fixture)
# ======================================================================================


@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_sharded_1d_basic(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, storage: str) -> None:
    """Basic 1D sharding auto-composition for both host and device storage."""
    num_devices = ttnn_mesh_device.get_num_devices()

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Build sharded tensor on host or device along dim=0 and compose back
    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(
        torch_in, device, layout, ttnn_mesh_device, shard_dim=0, dtype=dtype
    )

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch"
        assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch"
    else:
        # For quantized dtypes, compare auto vs explicit composed results
        assert torch.equal(torch_auto, torch_ref), "Auto vs explicit composer mismatch for quantized dtype"


@pytest.mark.parametrize("storage", ["host", "device"])  # where the replicated tensor lives
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_replicate_1d_basic(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, storage: str) -> None:
    """Replicated 1D distribution should compose to identity for host and device storage."""
    # Any shape works; replication does not change global shape
    # ttnn.from_torch perform naive quantization to lower dtypes -- work on existing exponent and mantissa values
    # get range of values for bfloat4_b quantization which has 4 bits for the mantissa and shared 8-bit exponent
    min_value, max_value = -7, 7
    torch_in = _make_arange_dtype((2, 3, 4, 5), dtype=torch.float32, min_value=min_value, max_value=max_value)

    device = None if storage == "host" else ttnn_mesh_device
    tt_replicated = ttnn.from_torch(
        torch_in,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )

    # Auto-composition should detect full replication and yield identity
    if device is None:
        torch_auto = to_torch_auto_compose(tt_replicated, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_replicated)

    assert torch.equal(torch_auto, torch_in)


# --------------------------------------------------------------------------------------
# Shard various tensor dims on 1D meshes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_sharded_various_dims(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dim: int, storage: str) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    torch_in = _make_arange_dtype(tuple(shape))

    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(torch_in, device, layout, ttnn_mesh_device, dim, dtype=dtype)

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)


# --------------------------------------------------------------------------------------
# Coverage for 2D mesh sharding: shard-shard and replicate-shard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_sharded_2d_basic(
    ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dims_pair: tuple[int, int], storage: str
) -> None:
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 and torch.prod(torch.tensor(mesh_shape)).item() <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1, "Shard dims for 2D sharding must be distinct"

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    torch_in = _make_arange_dtype(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    device = None if storage == "host" else ttnn_mesh_device
    tt_sharded = ttnn.from_torch(torch_in, device=device, dtype=dtype, layout=layout, mesh_mapper=mapper)

    if device is None:
        torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_sharded)
    composer = ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    torch_ref = ttnn.to_torch(tt_sharded, mesh_composer=composer)

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)


@pytest.mark.parametrize(
    "dims_pair",
    [
        (None, -1),  # replicate along mesh dim 0, shard along last tensor dim
        (1, None),  # shard along tensor dim 1 on mesh dim 0, replicate mesh dim 1
    ],
)
@pytest.mark.parametrize("storage", ["host", "device"])  # host vs device sharded tensor
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_sharded_2d_with_replicate(
    ttnn_mesh_device: ttnn.MeshDevice,
    layout,
    dtype,
    dims_pair: tuple[object, object],
    storage: str,
) -> None:
    # None indicates replicate axis
    replicate_axis = [i for i, d in enumerate(dims_pair) if d is None][0]
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 and torch.prod(torch.tensor(mesh_shape)).item() <= 1:
        pytest.skip("Requires a 2D mesh with at least one dim > 1 to observe replication")

    rank = 4
    # Determine which tensor axis is sharded (the non-None entry)
    shard_dim = [d for d in dims_pair if d is not None][0]
    shard_axis = _pos_dim(shard_dim, rank)
    shape = [2, 3, 4, 5]
    # Set size along sharded axis rounded up to a multiple of the other mesh dim
    other_mesh_dim = mesh_shape[1 - replicate_axis]
    shape[shard_axis] = ((shape[shard_axis] + other_mesh_dim - 1) // other_mesh_dim) * other_mesh_dim

    # get range of values for bfloat4_b quantization which has 4 bits for the mantissa and shared 8-bit exponent
    torch_in = _make_arange_dtype(tuple(shape), dtype=torch.float32, min_value=-7, max_value=7)

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=dims_pair)  # type: ignore[arg-type]
    device = None if storage == "host" else ttnn_mesh_device
    tt_sharded = ttnn.from_torch(torch_in, device=device, dtype=dtype, layout=layout, mesh_mapper=mapper)

    if device is None:
        torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_sharded)

    assert torch.equal(torch_auto, torch_in)


# --------------------------------------------------------------------------------------
# Tensor shape categories around hardware threshold (e.g., 32)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["lt", "eq", "gt"])  # per-shard length relative to threshold
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
@_DEVICE_TEST_MARKS["ttnn_mesh_device"]
@_DEVICE_TEST_MARKS["layout_dtype"]
def test_sharded_shape_thresholds(
    ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, category: str, storage: str
) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

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
    torch_in = _make_arange_dtype(tuple(shape))

    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(
        torch_in, device, layout, ttnn_mesh_device, shard_dim, dtype=dtype
    )

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)


# --------------------------------------------------------------------------------------
# Test coverage for auto_compose
# --------------------------------------------------------------------------------------


def test_to_torch_auto_compose_exception_handler():
    """Test the exception handler in to_torch_auto_compose (lines 38-40)."""
    mock_tensor = MagicMock(spec=ttnn.Tensor)
    mock_topology = MagicMock()
    mock_tensor.tensor_topology.return_value = mock_topology
    mock_topology.placements.return_value = [ttnn.PlacementShard(0)]
    mock_topology.distribution_shape.return_value = [2]

    mock_device = MagicMock(spec=ttnn.MeshDevice)
    mock_tensor.device.return_value = mock_device

    with patch("ttnn.create_mesh_composer", return_value="fake_composer"):
        with patch("ttnn.to_torch", side_effect=RuntimeError("Mock failure")):
            with pytest.raises(RuntimeError, match="Mock failure"):
                to_torch_auto_compose(mock_tensor)


def test_to_torch_auto_compose_no_device_error():
    """Test RuntimeError when tensor is on host and no device is provided/available (lines 102-104)."""
    mock_tensor = MagicMock(spec=ttnn.Tensor)
    mock_topology = MagicMock()
    mock_tensor.tensor_topology.return_value = mock_topology
    mock_topology.placements.return_value = [ttnn.PlacementShard(0)]
    mock_topology.distribution_shape.return_value = [2]

    # Tensor on host
    mock_tensor.device.return_value = None

    with patch("ttnn.GetDefaultDevice", return_value=None):
        with pytest.raises(RuntimeError, match="Tensor is on host and no mesh_device provided"):
            to_torch_auto_compose(mock_tensor)


def test_infer_composer_1d_sharded_mock():
    """
    Use mocking to hit the 1D sharded paths (lines 113, 125-131)
    if real 1D meshes are hard to come by.
    """
    mock_tensor = MagicMock(spec=ttnn.Tensor)
    mock_topology = MagicMock()
    mock_tensor.tensor_topology.return_value = mock_topology

    # Case 1: 1D Sharded
    mock_topology.placements.return_value = [ttnn.PlacementShard(0)]
    mock_topology.distribution_shape.return_value = [2]

    mock_device = MagicMock(spec=ttnn.MeshDevice)
    mock_device.shape.dims.return_value = 1
    mock_tensor.device.return_value = mock_device

    with patch("ttnn.create_mesh_composer") as mock_create:
        mock_create.return_value = "fake_composer"
        composer = _infer_mesh_composer_from_topology(mock_tensor)
        assert composer == "fake_composer"
        mock_create.assert_called_once()

    # Case 2: 1D Replicated
    mock_topology.placements.return_value = [ttnn.PlacementReplicate()]
    mock_topology.distribution_shape.return_value = [2]
    composer = _infer_mesh_composer_from_topology(mock_tensor)
    assert composer is None
