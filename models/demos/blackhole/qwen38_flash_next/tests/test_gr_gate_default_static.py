# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The 32-row low_rank_gate program without a device: 4 low-rank cores x 3 tiles with the original compute, writer and
reader kernels (rows <= 16 take the gate v3 program instead)."""

from __future__ import annotations

import types

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

D = gr_read.TP_SIZE


class FakeTensor:
    def __init__(self, shape, dtype, mesh=None):
        self.shape = ttnn.Shape(list(shape))
        self.padded_shape = self.shape
        self.dtype = dtype
        self.layout = ttnn.TILE_LAYOUT
        self._mesh = mesh

    def device(self):
        return self._mesh

    def buffer_address(self):
        return 0x1000


class FakeMesh:
    def __init__(self, x, y):
        self.grid = types.SimpleNamespace(x=x, y=y)

    def compute_with_storage_grid_size(self):
        return self.grid


def build(monkeypatch, grid=(13, 10)):
    captured = {}
    mesh = FakeMesh(*grid)
    monkeypatch.setattr(fp, "allocate", lambda shape, dtype, layout, mesh_: FakeTensor(shape, dtype, mesh_))
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [0])
    monkeypatch.setattr(fp, "program_meta", lambda *a, **k: None)
    monkeypatch.setattr(fp, "run_program", lambda io, descriptor, meta=None: captured.setdefault("d", descriptor))
    # the logical grid doubles as the NoC grid for the rectangle check
    monkeypatch.setattr(
        gr_read, "noc_map", lambda mesh_: {(x, y): (x + 1, y + 1) for x in range(32) for y in range(32)}
    )
    rows = 32
    partials = FakeTensor((gr_read.TP_SIZE, 1, rows, gr_read.PARTIAL_WIDTH), gr_read.FP32, mesh)
    normalized = FakeTensor((1, 1, rows, gr_read.FLAT_WIDTH), gr_read.BF16, mesh)
    up = FakeTensor((1, 1, gr_read.PARTIAL_WIDTH, gr_read.FLAT_WIDTH), gr_read.BF16, mesh)
    gr_read.low_rank_gate(partials, normalized, up)
    return captured["d"]


def sources(descriptor):
    return [k.kernel_source for k in descriptor.kernels]


def core_count(core_range_set):
    return sum(r.grid_size().x * r.grid_size().y for r in core_range_set.ranges())


def test_default_selects_today_kernels(monkeypatch):
    d = build(monkeypatch)
    s = sources(d)
    assert gr_read.LOWRANK in s and gr_read.GATE in s and gr_read.MCAST_WRITER in s and gr_read.MCAST_READER in s
    low = [k for k in d.kernels if k.kernel_source == gr_read.LOWRANK]
    assert [list(k.compile_time_args)[:3] for k in low] == [[3, D, 3], [3, D, 1]]
    assert sum(core_count(k.core_ranges) for k in low) == 4
    reader = next(k for k in d.kernels if k.kernel_source == gr_read.MCAST_READER)
    assert list(reader.compile_time_args)[2] == 4  # senders
