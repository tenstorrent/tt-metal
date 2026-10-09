# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode pool's row scatter (``tt/v41/pool.py scatter_rows``) on one chip: strided, based, sentinel-skipping."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.pool import scatter_rows


def _u32(mesh, host):
    return ttnn.from_torch(
        host.to(torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _column(t) -> list[int]:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0, :, 0].tolist()


@pytest.mark.parametrize("width", [64, 132])  # 256-byte rows, and the 528-byte rows of a SCALED_FP8 cache
def test_scatter_rows(width):
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        _check(mesh_device, width)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _check(mesh_device, width):
    rows = 8
    src = _u32(mesh_device, (torch.arange(rows).view(rows, 1) + 1).expand(rows, width).reshape(1, 1, rows, width))
    idx = _u32(mesh_device, torch.tensor([[[[7, 6, 5, 4, 3, 2, 1, -1]]]]))
    # row i -> 2 + 2 i + idx[i]; the sentinel row is skipped
    dst = _u32(mesh_device, torch.zeros(1, 1, 24, width))
    scatter_rows(src, dst, idx, dst_base=2, stride=2)
    expected = [0] * 24
    for i in range(rows - 1):
        expected[2 + 2 * i + (7 - i)] = i + 1
    assert _column(dst) == expected
    # the pure permutation (stride 0), from a source offset, over the first 4 rows only
    dst = _u32(mesh_device, torch.zeros(1, 1, 24, width))
    scatter_rows(src, dst, idx, src_base=2, rows=4)
    expected = [0] * 24
    for i in range(4):
        expected[7 - i] = 2 + i + 1
    assert _column(dst) == expected
    # a long index (one page, read whole) with the rows spread over the cores
    n = 4000
    big = _u32(mesh_device, torch.zeros(1, 1, n, width))
    big_src = _u32(mesh_device, (torch.arange(n).view(n, 1) + 1).expand(n, width).reshape(1, 1, n, width))
    scatter_rows(big_src, big, _u32(mesh_device, (n - 1 - torch.arange(n)).view(1, 1, 1, n)))
    assert _column(big) == list(range(n, 0, -1))
