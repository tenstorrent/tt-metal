# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of ttnn.reshape and ttnn.experimental.view outputs.

A per-device reshape preserves the inter-device split when the outer prefix volume before the split dim is unchanged:
for input `old_shape` with `Shard(d)`, let `P = prod(old_shape[:d])`. When there is a unique outermost dim `d'` in
`new_shape` with `prod(new_shape[:d']) == P` and `new_shape[d'] > 1` (or `old_shape[d] == 1`), the split sits on `d'`
and preserves negative dim spelling. When the split dim is merged under an outer dim of size > 1, no single dim label
describes the result and the input label is kept.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)

REPLICATE = "PlacementReplicate()"


def _sharded(tensor, mesh_device, dim, layout=ttnn.TILE_LAYOUT, nd=True):
    if nd:
        config = ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)])
        mapper = ttnn.create_mesh_mapper(mesh_device, config)
    else:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=layout, device=mesh_device, mesh_mapper=mapper)


def _placements(tensor):
    return [repr(placement) for placement in tensor.tensor_topology().placements()]


def _gathered(tensor):
    pieces = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]
    dims = [p.dim for p in tensor.tensor_topology().placements() if isinstance(p, ttnn.PlacementShard)]
    if not dims:
        return pieces[0]
    (dim,) = dims
    return torch.cat(pieces, dim=dim)


@pytest.mark.parametrize("nd", [True, False], ids=["nd", "collapsed"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row-major"])
@pytest.mark.parametrize(
    "old_shape, split_dim, new_shape, expected_dim",
    [
        # Unsqueeze / squeeze of size-1 dims before split dim
        ((4, 64), 0, (1, 4, 64), 1),
        ((1, 4, 64), 1, (4, 64), 0),
        ((1, 4, 32, 64), 1, (4, 32, 64), 0),
        # Flatten / unflatten of dims strictly after split dim
        ((1, 4, 32, 64), 1, (1, 4, 2048), 1),
        ((1, 4, 32, 64), -3, (1, 4, 2048), -2),
        ((2, 64, 64), 1, (2, 128, 32), 1),
        # Flatten / unflatten of dims strictly before split dim
        ((2, 3, 4, 64), 2, (6, 4, 64), 1),
        ((6, 4, 64), 1, (2, 3, 4, 64), 2),
        # Negative spelling preserved when last dim is split across devices and outer dims flatten
        ((1, 2, 32, 128), -1, (2, 32, 128), -1),
    ],
)
def test_reshape_renumbers_shard_dim(mesh_device, nd, layout, old_shape, split_dim, new_shape, expected_dim):
    torch_input = torch.randn(old_shape, dtype=torch.bfloat16)

    out = ttnn.reshape(_sharded(torch_input, mesh_device, split_dim, layout=layout, nd=nd), new_shape)

    shard = f"PlacementShard({expected_dim})"
    assert _placements(out) == ([REPLICATE, shard] if nd else [shard])
    assert torch.equal(_gathered(out), torch_input.reshape(new_shape))


def test_reshape_merged_under_outer_dim_keeps_input_label(mesh_device):
    torch_input = torch.randn(2, 4, 64, dtype=torch.bfloat16)

    out = ttnn.reshape(_sharded(torch_input, mesh_device, 1), (8, 64))

    assert _placements(out) == [REPLICATE, "PlacementShard(1)"]
