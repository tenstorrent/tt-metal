# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import EllipsisType

import pytest
import torch

import ttnn

from ...utils import tensor
from ...utils.tracing import Tracer


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_full(mesh_device: ttnn.MeshDevice) -> None:
    shape = (32, 32)
    dtype = ttnn.bfloat16
    value = 10

    full = Tracer(
        lambda: tensor.full(shape, value, dtype=dtype, device=mesh_device),
        device=mesh_device,
    )

    result = full()
    ref = torch.full(shape, value, dtype=torch.bfloat16)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "start", "end", "step"),
    [
        (ttnn.bfloat16, torch.bfloat16, 10, 20, 2),
        # start - step is negative: the offset has to be applied without a negative scalar,
        # which an unsigned tensor cannot encode.
        (ttnn.uint32, torch.int64, 0, 77, 1),
        (ttnn.int32, torch.int32, 0, 8, 1),
    ],
)
def test_arange(
    mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType, torch_dtype: torch.dtype, start: int, end: int, step: int
) -> None:
    arange = Tracer(
        lambda: tensor.arange(start, end, step, dtype=dtype, device=mesh_device),
        device=mesh_device,
    )

    result = arange()
    ref = torch.arange(start, end, step, dtype=torch_dtype)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result).to(torch_dtype)
    assert torch.equal(result_torch, ref)


@pytest.mark.parametrize(
    ("shape", "dim", "front", "back"),
    [
        # On device tile padding does not support front padding.
        # rank <= 4: direct ttnn.pad path
        ((2, 32, 64), 0, 0, 2),
        ((2, 32, 64), 1, 0, 3),
        # rank > 4, dim >= rank - 3: direct ttnn.pad path (last 3 dims)
        ((2, 2, 2, 32, 64), 3, 0, 2),
        ((2, 2, 2, 32, 64), -1, 0, 4),
        # rank > 4, dim < rank - 3: reshape workaround path (early dims)
        ((2, 2, 2, 32, 64), 0, 0, 2),
        ((2, 2, 2, 32, 64), 1, 0, 3),
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pad_single(mesh_device: ttnn.MeshDevice, shape: tuple, dim: int, front: int, back: int) -> None:
    dtype = ttnn.bfloat16

    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    pad = Tracer(
        lambda: tensor.pad_single(x_tt, dim=dim, front=front, back=back),
        device=mesh_device,
    )

    result = pad()

    pad_per_dim = [(front, back) if i == (dim % len(shape)) else (0, 0) for i in reversed(range(len(shape)))]
    ref = torch.nn.functional.pad(x_torch, [v for pair in pad_per_dim for v in pair])

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tril(mesh_device: ttnn.MeshDevice, diagonal: int) -> None:
    shape = (1, 1, 64, 64)
    dtype = ttnn.bfloat16

    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    tril = Tracer(
        lambda: tensor.tril(x_tt, diagonal=diagonal),
        device=mesh_device,
    )

    result = tril()
    ref = torch.tril(x_torch, diagonal=diagonal)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_triu(mesh_device: ttnn.MeshDevice, diagonal: int) -> None:
    shape = (1, 1, 64, 64)
    dtype = ttnn.bfloat16

    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    triu = Tracer(
        lambda: tensor.triu(x_tt, diagonal=diagonal),
        device=mesh_device,
    )

    result = triu()
    ref = torch.triu(x_torch, diagonal=diagonal)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 4)], indirect=True)
@pytest.mark.parametrize(
    "mesh_axes",
    [(None, None, None), (None, 0, None), (None, None, 1), (None, 1, 0), (0, None, 1)],
    ids=["replicated", "sharded_0", "sharded_1", "sharded_10", "sharded_01"],
)
def test_to_torch_round_trip(mesh_device: ttnn.MeshDevice, mesh_axes: tuple[int | None, ...]) -> None:
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x, device=mesh_device, mesh_axes=mesh_axes)
    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=mesh_axes), x)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", [(3, 96, 40)], ids=["tile_unaligned"])
def test_to_torch_uneven_shards(mesh_device: ttnn.MeshDevice, layout: ttnn.Layout, shape: tuple[int, ...]) -> None:
    """Shards that do not land on tile boundaries still reassemble.

    Every other case divides the mesh evenly into whole tiles, so this is the only place the
    per-shard padding matters. Shards of differing sizes cannot be tested: the mesh mapper rejects
    a dimension that does not divide evenly.
    """
    mesh_axes = (None, 1, 0)
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x, device=mesh_device, layout=layout, mesh_axes=mesh_axes)
    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=mesh_axes), x)


@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 4)], indirect=True)
@pytest.mark.parametrize(
    "mesh_axes",
    [(None, None, None), (None, 0, None), (None, None, 1), (None, 1, 0), (0, None, 1)],
    ids=["replicated", "sharded_0", "sharded_1", "sharded_10", "sharded_01"],
)
def test_to_torch_host_tensor(mesh_device: ttnn.MeshDevice, mesh_axes: tuple[int | None, ...]) -> None:
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x, device=mesh_device, mesh_axes=mesh_axes, on_host=True)
    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=mesh_axes, composer_device=mesh_device), x)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    ("mesh_axes", "spelled_out"),
    [
        ((...,), (None, None, None)),
        ((..., 0), (None, None, 0)),
        ((0, ...), (0, None, None)),
        ((None, ..., 1), (None, None, 1)),
    ],
    ids=["whole", "trailing", "leading", "middle"],
)
def test_to_torch_ellipsis(
    mesh_device: ttnn.MeshDevice,
    mesh_axes: tuple[int | None | EllipsisType, ...],
    spelled_out: tuple[int | None, ...],
) -> None:
    """An Ellipsis stands for the replicated axes the spelling leaves out."""
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x, device=mesh_device, mesh_axes=mesh_axes)
    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=mesh_axes), x)
    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=spelled_out), x)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_to_torch_ellipsis_repeated(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    with expect_error(ValueError, "at most one Ellipsis"):
        tensor.from_torch(x, device=mesh_device, mesh_axes=(..., 0, ...))


@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 4)], indirect=True)
def test_to_torch_1d_distribution(mesh_device: ttnn.MeshDevice) -> None:
    """A 1D mapper reports a flat distribution shape, which to_torch ignores in favour of the mesh.

    A dimension fractured over the whole mesh has no mesh_axes spelling, so only the replicated
    mapper is covered here.
    """
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper)
    assert tuple(x_tt.tensor_topology().distribution_shape()) == (mesh_device.get_num_devices(),)
    assert torch.equal(tensor.to_torch(x_tt), x)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_to_torch_ignores_stale_topology(mesh_device: ttnn.MeshDevice) -> None:
    """Ops leave the distribution shape and placements stale, so to_torch reads neither.

    Only the mesh coordinates are trusted. They are stamped back unchanged here, while the other two
    fields are replaced with a flat replicated layout that describes nothing about this tensor.
    """
    mesh_axes = (None, 1, 0)
    torch.manual_seed(0)  # every host must build the same tensor
    x = torch.randn(8, 128, 128, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x, device=mesh_device, mesh_axes=mesh_axes)

    coords = list(x_tt.tensor_topology().mesh_coords())
    stale = ttnn.TensorTopology(ttnn.MeshShape([len(coords)]), [ttnn.PlacementReplicate()], coords)
    x_tt.update_tensor_topology(stale)

    assert torch.equal(tensor.to_torch(x_tt, mesh_axes=mesh_axes), x)
