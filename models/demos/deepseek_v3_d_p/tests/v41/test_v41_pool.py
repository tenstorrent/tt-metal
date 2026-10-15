# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode pool's row scatter (``tt/v41/pool.py scatter_rows``) on one chip: strided, based, sentinel-skipping,
into row-major rows, tiled bf16 rows and block pages; and per-rank predication through a sharded index tensor."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.pool import BLOCK_ROWS, scatter_rows


def _u32(mesh, host):
    return ttnn.from_torch(
        host.to(torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _bf16(mesh, host, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        host, device=mesh, dtype=ttnn.bfloat16, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _column(t) -> list[int]:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0, :, 0].tolist()


def _rows(t, chip: int = 0) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[chip]).to(torch.bfloat16)


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


def _expected_scatter(src: torch.Tensor, dst: torch.Tensor, idx: list[int], src_base, dst_base, stride, rows):
    """The torch reference of ``scatter_rows`` on logical rows ``[H, W]``."""
    out = dst.clone()
    for i in range(rows):
        if idx[i] == -1:
            continue
        out[dst_base + i * stride + idx[i]] = src[src_base + i]
    return out


# (index-K rows: 128 columns bf16, 256 bytes; and a 4-tile-wide check with a stride)
@pytest.mark.parametrize("mode", ["tiled", "blocks"])
def test_scatter_rows_layouts(mode):
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        _check_layout(mesh_device, mode)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _check_layout(mesh_device, mode):
    torch.manual_seed(0)
    width, height = 128, 32 * 6  # six tile rows / 24 block pages
    src_rows = 64
    src_host = torch.randn(1, 1, src_rows, width).to(torch.bfloat16)
    dst_host = torch.randn(1, 1, height, width).to(torch.bfloat16)
    # every in-tile row class (0, 15, 16, 31), both faces, page boundaries, the sentinel, a few repeats
    targets = [0, 15, 16, 31, 32, 33, 63, 64, 95, 96, 100, 127, 128, 159, 160, 191] + [-1] * 8
    # distinct destinations: the cores write in parallel, so two writes to one row have no defined order
    rest = [r for r in torch.randperm(height).tolist() if r not in targets]
    targets += rest[: src_rows - len(targets)]
    idx_host = torch.tensor(targets).view(1, 1, 1, src_rows)
    src = _bf16(mesh_device, src_host)
    idx = _u32(mesh_device, idx_host)
    if mode == "tiled":
        dst = _bf16(mesh_device, dst_host, ttnn.TILE_LAYOUT)
    else:
        dst = _bf16(mesh_device, dst_host.reshape(1, 1, height // BLOCK_ROWS, BLOCK_ROWS * width))

    def logical(t) -> torch.Tensor:
        return _rows(t).reshape(height, width)

    assert torch.equal(logical(dst), dst_host.reshape(height, width))  # the upload round-trips bytewise
    # a plain scatter of every source row
    scatter_rows(src, dst, idx)
    expected = _expected_scatter(
        src_host.reshape(src_rows, width), dst_host.reshape(height, width), targets, 0, 0, 0, src_rows
    )
    assert torch.equal(logical(dst), expected)
    # based and strided, over a subset of the rows, on top of the previous writes
    stride_targets = [3, 2, 1, 0, -1, 5, 6, 7]
    idx2 = _u32(mesh_device, torch.tensor(stride_targets).view(1, 1, 1, 8))
    scatter_rows(src, dst, idx2, src_base=5, dst_base=40, stride=16, rows=8)
    expected = _expected_scatter(src_host.reshape(src_rows, width), expected, stride_targets, 5, 40, 16, 8)
    assert torch.equal(logical(dst), expected)
    # the destination row index may address rows of several "users" laid end to end (the pool's u * (cap + 32))
    scatter_rows(src, dst, idx2, dst_base=height - 8, rows=8)
    expected = _expected_scatter(src_host.reshape(src_rows, width), expected, stride_targets, 0, height - 8, 0, 8)
    assert torch.equal(logical(dst), expected)


def test_scatter_rows_per_rank():
    """A sharded index tensor (sentinels on every rank but one) writes only that rank's destination."""
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 1))
    try:
        _check_per_rank(mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _check_per_rank(mesh_device):
    torch.manual_seed(1)
    width, height, n = 128, 64, 16
    src_host = torch.randn(1, 1, n, width).to(torch.bfloat16)
    dst_host = torch.randn(1, 1, height, width).to(torch.bfloat16)
    targets = torch.randperm(height)[:n].tolist()
    host_idx = torch.full((2, 1, 1, n), -1, dtype=torch.int32)
    host_idx[1, 0, 0] = torch.tensor(targets, dtype=torch.int32)
    idx = ttnn.from_torch(
        host_idx,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (2, 1), dims=(0, None)),
    )
    src = _bf16(mesh_device, src_host)
    for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        dst = _bf16(mesh_device, dst_host, layout)
        scatter_rows(src, dst, idx)
        assert torch.equal(_rows(dst, 0).reshape(height, width), dst_host.reshape(height, width))
        expected = _expected_scatter(src_host.reshape(n, width), dst_host.reshape(height, width), targets, 0, 0, 0, n)
        assert torch.equal(_rows(dst, 1).reshape(height, width), expected)


def test_scatter_rows_capacity():
    """One user's admission at 1M rows (256 MiB of index keys) into each layout, timed against the tilize the step
    does today; the whole destination compared bytewise."""
    import time

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        torch.manual_seed(2)
        width, n = 128, 1 << 20
        src_host = torch.randn(1, 1, n, width).to(torch.bfloat16)
        src = _bf16(mesh_device, src_host)
        idx = _u32(mesh_device, torch.arange(n).view(1, 1, 1, n))
        timings = {}
        for mode in ("rows", "tiled", "blocks"):
            if mode == "tiled":
                dst = _bf16(mesh_device, torch.zeros(1, 1, n, width, dtype=torch.bfloat16), ttnn.TILE_LAYOUT)
            elif mode == "blocks":
                dst = _bf16(mesh_device, torch.zeros(1, 1, n // BLOCK_ROWS, BLOCK_ROWS * width, dtype=torch.bfloat16))
            else:
                dst = _bf16(mesh_device, torch.zeros(1, 1, n, width, dtype=torch.bfloat16))
            scatter_rows(src, dst, idx)  # the kernel build
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(3):
                scatter_rows(src, dst, idx)
            ttnn.synchronize_device(mesh_device)
            timings[mode] = (time.perf_counter() - t0) / 3
            assert torch.equal(_rows(dst).reshape(n, width), src_host.reshape(n, width)), mode
            ttnn.deallocate(dst)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(3):
            k = ttnn.to_layout(src, ttnn.TILE_LAYOUT)
            ttnn.deallocate(k)
        ttnn.synchronize_device(mesh_device)
        timings["to_layout"] = (time.perf_counter() - t0) / 3
        print("scatter_rows 1M rows, seconds per call:", {k: round(v, 4) for k, v in timings.items()})
    finally:
        ttnn.close_mesh_device(mesh_device)


def test_scatter_rows_traced():
    """A captured scatter replays wherever the index tensor's current contents say (the step's contract)."""
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=1 << 20)
    try:
        torch.manual_seed(3)
        width, height, n = 128, 32 * 8, 32
        src_host = torch.randn(1, 1, n, width).to(torch.bfloat16)
        src = _bf16(mesh_device, src_host)
        dst_host = torch.zeros(1, 1, height, width, dtype=torch.bfloat16)
        dst = _bf16(mesh_device, dst_host, ttnn.TILE_LAYOUT)
        targets = list(range(0, height, 8))  # one row per block/tile quarter
        idx_host = torch.tensor(targets).view(1, 1, 1, n).to(torch.int32)
        idx = _u32(mesh_device, idx_host)
        scatter_rows(src, dst, idx)  # eager once: the kernel build
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        scatter_rows(src, dst, idx)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        expected = _expected_scatter(src_host.reshape(n, width), dst_host.reshape(height, width), targets, 0, 0, 0, n)
        for _ in range(100):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        assert torch.equal(_rows(dst).reshape(height, width), expected)
        # new indices in the same device tensor: the replay writes the new rows (and the sentinel skips one)
        targets2 = [t + 1 for t in targets]
        targets2[5] = -1
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor(targets2).view(1, 1, 1, n).to(torch.int32), dtype=ttnn.uint32), idx
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        expected = _expected_scatter(src_host.reshape(n, width), expected, targets2, 0, 0, 0, n)
        assert torch.equal(_rows(dst).reshape(height, width), expected)
        ttnn.release_trace(mesh_device, tid)
    finally:
        ttnn.close_mesh_device(mesh_device)
