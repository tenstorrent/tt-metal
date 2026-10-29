# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_GR_GATE_V2 without a device: the flag selects the v2 low_rank_gate program (12 one-tile low-rank cores, the
v2 compute / writer / reader kernels) and the default keeps today's (4 cores x 3 tiles, the original kernels); the
v2 kernels' invariants (K order, fp32 spill, per-sender semaphore bits)."""

from __future__ import annotations

import types

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

D = gr_read.TP_SIZE
KERNELS = fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_read" / "kernels"


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


def build(monkeypatch, grid=(13, 10), flag=None):
    if flag is None:
        monkeypatch.delenv(gr_read.GATE_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(gr_read.GATE_V2_ENV, flag)
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


def per_core_args(kernel):
    found = []
    for x in range(16):
        for y in range(12):
            try:
                found.append(list(kernel.runtime_args[x][y]))
            except IndexError:
                pass
    return found


def core_count(core_range_set):
    return sum(r.grid_size().x * r.grid_size().y for r in core_range_set.ranges())


def test_flag_parse():
    assert gr_read.gate_v2_enabled({}) is False
    assert gr_read.gate_v2_enabled({gr_read.GATE_V2_ENV: "0"}) is False
    assert gr_read.gate_v2_enabled({gr_read.GATE_V2_ENV: "1"}) is True
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        gr_read.gate_v2_enabled({gr_read.GATE_V2_ENV: "2"})


def test_default_selects_today_kernels(monkeypatch):
    d = build(monkeypatch)
    s = sources(d)
    assert gr_read.LOWRANK in s and gr_read.GATE in s and gr_read.MCAST_WRITER in s and gr_read.MCAST_READER in s
    assert not any(x in s for x in (gr_read.LOWRANK2, gr_read.GATE2, gr_read.MCAST_WRITER2, gr_read.MCAST_READER2))
    low = [k for k in d.kernels if k.kernel_source == gr_read.LOWRANK]
    assert [list(k.compile_time_args)[:3] for k in low] == [[3, D, 3], [3, D, 1]]
    assert sum(core_count(k.core_ranges) for k in low) == 4
    reader = next(k for k in d.kernels if k.kernel_source == gr_read.MCAST_READER)
    assert list(reader.compile_time_args)[2] == 4  # senders


@pytest.mark.parametrize("grid", [(13, 10), (11, 10)])
def test_flag_selects_v2_kernels(monkeypatch, grid):
    d = build(monkeypatch, grid, "1")
    s = sources(d)
    assert gr_read.LOWRANK2 in s and gr_read.GATE2 in s and gr_read.MCAST_WRITER2 in s and gr_read.MCAST_READER2 in s
    assert not any(x in s for x in (gr_read.LOWRANK, gr_read.GATE, gr_read.MCAST_WRITER, gr_read.MCAST_READER))
    low = [k for k in d.kernels if k.kernel_source == gr_read.LOWRANK2]
    assert [list(k.compile_time_args)[:3] for k in low] == [[1, D, 1], [1, D, 0]]
    assert sum(core_count(k.core_ranges) for k in low) == 12
    reader = next(k for k in d.kernels if k.kernel_source == gr_read.MCAST_READER2)
    assert list(reader.compile_time_args)[2] == 12  # senders = sem bits
    bits = []
    for k in d.kernels:
        if k.kernel_source == gr_read.MCAST_WRITER2:
            bits += [args[13] for args in per_core_args(k)]
    assert sorted(bits) == [1 << c for c in range(12)]
    offsets = {}
    for k in d.kernels:
        if k.kernel_source == gr_read.MCAST_WRITER2:
            for args in per_core_args(k):
                offsets[args[13]] = args[0]
    assert all(offsets[1 << c] == c for c in range(12))  # sender c lands at tile c: bit c = K tile c
    gate = next(k for k in d.kernels if k.kernel_source == gr_read.GATE2)
    assert list(gate.compile_time_args) == [12, 4, 8, 9, 10, 11, 12, 13, 14, 18, gr_read.UP_SPILL, 15]  # the same spill


def test_v2_kernel_invariants():
    gate = (KERNELS / "gate_compute.cpp").read_text()
    gate2 = (KERNELS / "gate_compute2.cpp").read_text()
    # the K order per dest tile and the spill machinery are the original's
    assert "matmul_block(c_lr, c_w, kt, b * Kt + kt, b, 0, 1, 1, 1);" in gate2
    for line in ("interm.reserve_back(B);", "pack_tile(b, c_interm);", "copy_tile(c_interm, b, b);"):
        assert line in gate and line in gate2
    assert "for (uint32_t kt = kb; kt < kb + block; ++kt) {\n                lr.wait_front(kt + 1);" in gate2
    assert gate2.count("copy_init(") == gate.count("copy_init(") - 1
    low = (KERNELS / "lowrank_compute.cpp").read_text()
    low2 = (KERNELS / "lowrank_compute2.cpp").read_text()
    assert "static_assert(T == 1" in low2 and "typecast_tile<fp32, bf16>(0);" in low2
    assert "add_tiles(c_pg, c_zero, d, 0, 0);" in low2
    reader2 = (KERNELS / "mcast_reader_v2.cpp").read_text()
    assert "(sem.value() & (1u << s)) == 0" in reader2 and "recv.push_back(PER_SENDER)" in reader2
    assert "sem.up(noc, x, y, sem_bits);" in (KERNELS / "mcast_writer_v2.cpp").read_text()
