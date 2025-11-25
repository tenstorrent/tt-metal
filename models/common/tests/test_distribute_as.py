# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for distributing a torch.Tensor using the topology of an existing TTNN tensor.

Follows the same style as test_auto_compose.py, but in the opposite
direction: given a reference distributed tensor, verify that
`from_torch_dist_as(torch_tensor, ref_tensor)` produces a TTNN tensor
with the same topology and data that composes back to the original
torch input.
"""

import pytest
import torch

import ttnn
from models.common.auto_compose import extract_tensor_topology_info, to_torch_auto_compose
from models.common.distribute_as import from_torch_dist_as

# ======================================================================================
# Test Parameters
# ======================================================================================


pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            (1, 1),  # single device
            (1, 2),  # 1D mesh, 2 devices
            (1, 8),  # 1D mesh, 8 devices
            (2, 4),  # 2D mesh, 8 devices
        ],
        ids=[
            "1x1",
            "1x2",
            "1x8",
            "2x4",
        ],
        indirect=True,
    ),
    pytest.mark.parametrize(
        "layout,dtype",
        [
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),  # bfloat8_b only works with TILE_LAYOUT
            (ttnn.TILE_LAYOUT, ttnn.bfloat4_b),  # bfloat4_b only works with TILE_LAYOUT
        ],
        ids=["row_major_bf16", "tile_bf16", "tile_bf8b", "tile_bf4b"],
    ),
]


# ======================================================================================
# Helpers
# ======================================================================================


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
    return dim % rank


def _topology_signature(tt: ttnn.Tensor) -> tuple[tuple[str, tuple[int, ...]], tuple[int, ...]]:
    placements, dist_shape = extract_tensor_topology_info(tt)
    sig_p = []
    for p in placements:
        if isinstance(p, ttnn.PlacementShard):
            sig_p.append(("shard", (p.dim,)))
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            sig_p.append(("replicate", ()))
    return tuple(sig_p), tuple(dist_shape)


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("storage", ["host", "device"])
def test_sharded_1d(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, storage: str) -> None:
    is_host_ref = storage == "host"
    num_devices = ttnn_mesh_device.get_num_devices()

    # Reference topology: host-sharded along dim 0
    shape = (num_devices, 1, 3, 1)
    # get range of values for bfloat4_b quantization which has 4 bits for the mantissa and shared 8-bit exponent
    max_value = 6  # allows for range of value to grow to 7 for new_input
    ref_input = _make_arange_dtype(shape, dtype=torch.float32, min_value=-7, max_value=max_value)
    ref_tt = ttnn.from_torch(
        ref_input,
        device=None if is_host_ref else ttnn_mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=0),
    )

    # Distribute a different tensor using the same topology
    new_input = ref_input + 1  # change values to avoid accidental equality
    if is_host_ref:
        tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)
    else:
        tt_as = from_torch_dist_as(new_input, ref_tt)

    # Verify topology matches reference
    expected_topology = _topology_signature(ref_tt)
    assert _topology_signature(tt_as) == expected_topology
    assert tt_as.dtype == dtype

    # Verify round-trip via auto-composition matches the source
    if is_host_ref:
        torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_as)
    assert torch.equal(torch_auto, new_input)


@pytest.mark.parametrize("storage", ["host", "device"])
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_sharded_various_dims(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dim: int, storage: str) -> None:
    is_host_ref = storage == "host"
    num_devices = ttnn_mesh_device.get_num_devices()

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    # get range of values for bfloat4_b quantization which has 4 bits for the mantissa and shared 8-bit exponent
    max_value = 6  # allows for range of value to grow to 7 for new_input
    ref_input = _make_arange_dtype(shape, dtype=torch.float32, min_value=-7, max_value=max_value)

    ref_tt = ttnn.from_torch(
        ref_input,
        device=None if is_host_ref else ttnn_mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=dim),
    )

    new_input = ref_input + 1  # change values to avoid accidental equality
    if is_host_ref:
        tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)
    else:
        tt_as = from_torch_dist_as(new_input, ref_tt)

    assert _topology_signature(tt_as) == _topology_signature(ref_tt)
    assert tt_as.dtype == dtype
    if is_host_ref:
        torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_as)
    assert torch.equal(torch_auto, new_input)


@pytest.mark.parametrize("storage", ["host", "device"])
@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
def test_sharded_2d(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dims_pair: tuple[int, int], storage: str) -> None:
    is_host_ref = storage == "host"
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 and torch.prod(torch.tensor(mesh_shape)).item() <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    # get range of values for bfloat4_b quantization which has 4 bits for the mantissa and shared 8-bit exponent
    max_value = 6  # allows for range of value to grow to 7 for new_input
    ref_input = _make_arange_dtype(shape, dtype=torch.float32, min_value=-7, max_value=max_value)

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    ref_tt = ttnn.from_torch(
        ref_input, device=None if is_host_ref else ttnn_mesh_device, dtype=dtype, layout=layout, mesh_mapper=mapper
    )

    new_input = ref_input + 1  # change values to avoid accidental equality
    if is_host_ref:
        tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)
    else:
        tt_as = from_torch_dist_as(new_input, ref_tt)

    assert _topology_signature(tt_as) == _topology_signature(ref_tt)
    assert tt_as.dtype == dtype
    if is_host_ref:
        torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_as)
    assert torch.equal(torch_auto, new_input)
