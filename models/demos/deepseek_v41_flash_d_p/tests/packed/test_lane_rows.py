# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3K (mesh only, not gated): tt/lane_rows.LaneRows (PACKED_HOST_FAST=on: a packed chunk's rope rows of every lane,
each chip its own rows, one prepared program) is bit-exact against PB.3A's path (tt/rope.lane_slices, then
mesh_partition over SP) on every chip: TILE tables (tt/rope.RopeTable: fp32 [max_seq, R/2], bf16 [max_seq, R]) at
tile-row starts, ROW_MAJOR tables (tt/compressor.CompressRope: compressed rows at any start), lanes of different
starts (empty lanes at 0), repeated calls (the prepared program's values and output patched)."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True


def _old(mesh, table, starts, n):
    from models.demos.deepseek_v41_flash_d_p.tt.common import DRAM, SP_AXIS
    from models.demos.deepseek_v41_flash_d_p.tt.rope import lane_slices

    sl = lane_slices(table, starts, n // len(starts), int(table.shape[-1]))
    if int(mesh.shape[SP_AXIS]) > 1:
        part = ttnn.mesh_partition(sl, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM)
        ttnn.deallocate(sl)
        sl = part
    return sl


def _chips(t):
    return [ttnn.to_torch(d).contiguous().view(torch.uint8).reshape(-1) for d in ttnn.get_device_tensors(t)]


@mesh_parametrize(timeout_s=900)
def test_lane_rows(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import _upload
    from models.demos.deepseek_v41_flash_d_p.tt.lane_rows import lane_rows
    from models.demos.deepseek_v41_flash_d_p.tt.rope import LaneStarts

    g = torch.Generator().manual_seed(7)
    cases = [  # (layout, dtype, rows, width, n, starts)
        (ttnn.TILE_LAYOUT, ttnn.float32, 16384, 32, 8192, (4096, 0, 8192, 2048)),
        (ttnn.TILE_LAYOUT, ttnn.float32, 16384, 32, 8192, (8192, 6144, 0, 0)),
        (ttnn.TILE_LAYOUT, ttnn.bfloat16, 16384, 64, 8192, (2048, 10240, 4096, 6144)),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.float32, 4096, 32, 2048, (1032, 8, 2048, 512)),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.float32, 4096, 32, 2048, (16, 1544, 0, 0)),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 4096, 64, 2048, (264, 1000, 8, 3000)),
    ]
    bad = []
    for layout, dtype, rows, width, n, starts in cases:
        host = torch.randn(1, 1, rows, width, generator=g)
        table = _upload(mesh_device, host, dtype, layout)
        for rep in range(2):
            st = LaneStarts(starts)
            ref = _old(mesh_device, table, st, n)
            got = lane_rows(table, st, n)
            assert tuple(got.shape) == tuple(ref.shape), (got.shape, ref.shape)
            a, b = _chips(ref), _chips(got)
            same = [bool(torch.equal(x, y)) for x, y in zip(a, b)]
            print(
                f"PB.3K lane_rows {layout} {dtype} {width} n={n} starts={starts} rep {rep}: "
                f"{sum(same)} / {len(same)} chips bit-exact",
                flush=True,
            )
            if not all(same):
                bad.append((layout, dtype, starts, rep, same.index(False)))
            ttnn.deallocate(ref)
            ttnn.deallocate(got)
        ttnn.deallocate(table)
    assert not bad, bad
