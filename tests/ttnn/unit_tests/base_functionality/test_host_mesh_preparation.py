# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only mesh materialization: compare each shard with an independent conversion."""

import numpy as np
import pytest
import torch
import ttnn


def raw_shard(tensor, i=0, j=0):
    return np.from_dlpack(tensor.host_buffer().get_shard(ttnn.MeshCoordinate(i, j))).copy()


@pytest.mark.parametrize(
    "source_dtype,target_dtype,layout",
    [
        (torch.float32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        (torch.bfloat16, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (torch.int32, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        (torch.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.TILE_LAYOUT),
        (torch.float32, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("shape", [(64, 128), (2, 3, 64, 128), (1, 1, 66, 130)])
@pytest.mark.parametrize("dims", [(None, None), (None, -2), (None, -1), (-2, -1), (-1, -2)])
@pytest.mark.parametrize("strided", [False, True])
def test_host_mesh_shards(source_dtype, target_dtype, layout, shape, dims, strided):
    shape_before_transpose = (*shape[:-2], shape[-1], shape[-2]) if strided else shape
    source = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape_before_transpose).to(source_dtype)
    if strided:
        source = source.transpose(-2, -1)
    placements = [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims]
    mapper = ttnn.create_mesh_mapper(ttnn.MeshShape(2, 2), ttnn.MeshMapperConfig(placements))
    actual = ttnn.from_torch(source, dtype=target_dtype, layout=layout, mesh_mapper=mapper)
    for i in range(2):
        first = source if dims[0] is None else torch.chunk(source, 2, dim=dims[0])[i]
        for j in range(2):
            expected_source = first if dims[1] is None else torch.chunk(first, 2, dim=dims[1])[j]
            expected = ttnn.from_torch(expected_source.contiguous(), dtype=target_dtype, layout=layout)
            assert tuple(actual.shape) == tuple(expected.shape)
            assert tuple(actual.padded_shape) == tuple(expected.padded_shape)
            assert actual.dtype == expected.dtype
            assert actual.layout == expected.layout
            np.testing.assert_array_equal(raw_shard(actual, i, j), raw_shard(expected))


@pytest.mark.parametrize("dim", [None, 0, 1])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_host_mesh_preserves_borrowing_rules(dim, layout):
    source = torch.arange(64 * 128, dtype=torch.float32).reshape(64, 128)
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate() if dim is None else ttnn.PlacementShard(dim)]
    mapper = ttnn.create_mesh_mapper(ttnn.MeshShape(2, 2), ttnn.MeshMapperConfig(placements))
    tensor = ttnn.from_torch(source, dtype=ttnn.float32, layout=layout, mesh_mapper=mapper)
    before = [[raw_shard(tensor, i, j) for j in range(2)] for i in range(2)]
    source.fill_(7)
    borrows = dim in (None, 0) and layout == ttnn.ROW_MAJOR_LAYOUT
    for i in range(2):
        for j in range(2):
            actual = raw_shard(tensor, i, j)
            if borrows:
                expected_source = source if dim is None else source.chunk(2, dim=0)[j]
                expected = ttnn.from_torch(expected_source, dtype=ttnn.float32, layout=layout)
                np.testing.assert_array_equal(actual, raw_shard(expected))
            else:
                np.testing.assert_array_equal(actual, before[i][j])
