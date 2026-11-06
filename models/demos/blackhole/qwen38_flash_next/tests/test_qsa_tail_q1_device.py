# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_OWNPOS and QWEN38_QSA_SELROW2 on the eight-die mesh: flag off against flag on, bit for bit.

 * test_index_tail_ownpos_bitwise / test_main_tail_ownpos_bitwise: rows 1, 8, 32 (main_tail unpaged and paged), random
   inputs and positions; the output tensors and the in-place states (ring, compressed cache, staging, KV cache / pool)
   of the two runs from the same initial state are bitwise equal.
 * test_selection_row_selrow2_bitwise: rows 1, 8, 32, unpaged, paged and live-rows, random keep / fill / offsets.
 * test_q1_timing: traced slope per call of each program with the flag off and on (Q1_TIMING lines), rows 8 and 32.
Run: R/go.sh "<-k expression>" <name> (device lock, QWEN38_FUSED_DEVICE_TEST=1).
"""

import os
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the held mesh")
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 20_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
BF16, U32, I32 = ttnn.bfloat16, ttnn.uint32, ttnn.int32
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
ROWS = (1, 8, 32)
TRACE_SHORT, TRACE_LONG, REPLAYS, REPEATS = 4, 12, 20, 3
INDEX_CACHE_ROWS = 256  # compressed cache rows per lane (positions < 4 * 256)
LANE_ROWS = 512
PAGE_SIZE, BLOCKS = 32, 16


def up(mesh, tensor, dtype, layout=RM):
    if dtype == U32:
        tensor = tensor.to(torch.int64).to(torch.uint32)
    # tilized on the host and then copied: a device-side tilize flushes denormals
    on_host = ttnn.from_torch(tensor, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    return ttnn.to_device(on_host, mesh, memory_config=DRAM)


def bits(tensor):
    """Every replica's raw bits (the replicas must agree); returns the first as int64."""

    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]

    def raw(t):
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).to(torch.int64) & 0xFFFF
        return t.to(torch.int64) & 0xFFFFFFFF

    first = raw(replicas[0])
    for other in replicas[1:]:
        assert torch.equal(first, raw(other)), "the dies disagree"
    return first


@contextmanager
def flags(ownpos="0", selrow2="0"):
    saved = {k: os.environ.get(k) for k in (qsa_block.OWNPOS_ENV, qsa_block.SELROW2_ENV)}
    os.environ[qsa_block.OWNPOS_ENV], os.environ[qsa_block.SELROW2_ENV] = ownpos, selrow2
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bf(generator, *shape):
    return torch.randn(*shape, generator=generator).to(torch.bfloat16)


def position_inputs(mesh, positions):
    rows = len(positions)
    hit = torch.zeros(1, rows, 32, 1, dtype=torch.bfloat16)
    for u in range(rows):
        hit[0, u, int(positions[u]) % 32, 0] = 1.0
    starts = (positions & ~31).reshape(1, 1, 1, rows)
    return SimpleNamespace(kv_block_start=up(mesh, starts, U32), kv_row_hit=up(mesh, hit, BF16, TILE))


def random_positions(generator, rows, limit):
    positions = torch.randint(0, limit, (rows,), generator=generator)
    if rows > 2:  # equal blocks, the block edges and the first position among the lanes
        positions[0], positions[1], positions[2] = 0, 31, 32
    return positions


# ------------------------------------------------------------------------------------------------ index_tail
class IndexCase:
    def __init__(self, mesh, rows, seed):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.rows = mesh, rows
        tile = lambda t: up(mesh, t, BF16, TILE)
        self.positions = random_positions(g, rows, 4 * INDEX_CACHE_ROWS)
        self.host = dict(
            q=bf(g, 1, 1, rows, 128),
            k=bf(g, 1, 1, rows, 128),
            qn=bf(g, 1, 1, 1, 128).abs() + 0.5,
            kn=bf(g, 1, 1, 1, 128).abs() + 0.5,
            tables=[torch.cos(bf(g, 1, 1, rows, 64).float()).to(torch.bfloat16) for _ in range(4)],
            ring=bf(g, 1, rows, 32, 128),
            cache=bf(g, rows, 1, INDEX_CACHE_ROWS, 128),
        )
        h = self.host
        self.static = dict(
            q=tile(h["q"]),
            k=tile(h["k"]),
            position=position_inputs(mesh, self.positions),
            qn=tile(h["qn"]),
            kn=tile(h["kn"]),
            tables=[tile(t) for t in h["tables"]],
        )

    def run(self):
        s, h = self.static, self.host
        ring, cache = up(self.mesh, h["ring"], BF16, TILE), up(self.mesh, h["cache"], BF16, TILE)
        out = qsa_block.index_tail(s["q"], s["k"], s["position"], s["qn"], s["kn"], *s["tables"], ring, cache)
        return {"query": bits(out), "ring": bits(ring), "cache": bits(cache)}

    def call(self):
        s = self.static
        return qsa_block.index_tail(
            s["q"], s["k"], s["position"], s["qn"], s["kn"], *s["tables"], self._ring, self._cache
        )

    def prepare_timing(self):
        h = self.host
        self._ring, self._cache = up(self.mesh, h["ring"], BF16, TILE), up(self.mesh, h["cache"], BF16, TILE)


@MESH
@PARAMS
@pytest.mark.parametrize("rows", ROWS)
def test_index_tail_ownpos_bitwise(mesh_device, rows):
    case = IndexCase(mesh_device, rows, 1000 + rows)
    with flags("0"):
        off = case.run()
    with flags("1"):
        on = case.run()
    for name in off:
        assert torch.equal(off[name], on[name]), f"index_tail {name} differs at rows={rows}"
    assert not torch.equal(off["ring"], bits(up(mesh_device, case.host["ring"], BF16, TILE))), "ring was not written"
    print(f"Q1_BITWISE index_tail rows={rows} OK")


# ------------------------------------------------------------------------------------------------ main_tail
class MainCase:
    def __init__(self, mesh, rows, paged, seed):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.rows, self.paged = mesh, rows, paged
        tile = lambda t: up(mesh, t, BF16, TILE)
        heads, hd = qsa_block.LOCAL_HEADS, qsa_block.HEAD_DIM
        limit = BLOCKS * PAGE_SIZE if paged else LANE_ROWS
        self.positions = random_positions(g, rows, limit)
        self.static = dict(
            qg_ws=tile(bf(g, 1, 1, rows, 2 * heads * hd)),
            k_ws=tile(bf(g, 1, 1, rows, hd)),
            v_ws=tile(bf(g, 1, 1, rows, hd)),
            position=position_inputs(mesh, self.positions),
            q_norm=tile(bf(g, 1, 1, 1, hd).abs() + 0.5),
            k_norm=tile(bf(g, 1, 1, 1, hd).abs() + 0.5),
            cos=tile(torch.cos(bf(g, 1, 1, rows, 64).float()).to(torch.bfloat16)),
            sin=tile(torch.sin(bf(g, 1, 1, rows, 64).float()).to(torch.bfloat16)),
        )
        self.staging_host = bf(g, 1, rows, 32, 512)
        if paged:
            table, pages = ref.random_page_table(rows, BLOCKS, g)
            self.cache_host = bf(g, 1, 1, pages * PAGE_SIZE, 512)
            self.kwargs = dict(page_table=up(mesh, table.reshape(1, 1, 32, BLOCKS), I32), page_size=PAGE_SIZE)
        else:
            self.cache_host = bf(g, 1, 1, rows * LANE_ROWS, 512)
            self.kwargs = dict(lane_rows=LANE_ROWS)

    def fresh(self):
        return up(self.mesh, self.staging_host, BF16, TILE), up(self.mesh, self.cache_host, BF16)

    def run(self):
        staging, cache = self.fresh()
        out = qsa_block.main_tail(**self.static, staging=staging, kv_cache=cache, **self.kwargs)
        return {"query": bits(out), "staging": bits(staging), "kv_cache": bits(cache)}

    def prepare_timing(self):
        self._staging, self._cache = self.fresh()

    def call(self):
        return qsa_block.main_tail(**self.static, staging=self._staging, kv_cache=self._cache, **self.kwargs)


@MESH
@PARAMS
@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@pytest.mark.parametrize("rows", ROWS)
def test_main_tail_ownpos_bitwise(mesh_device, rows, paged):
    case = MainCase(mesh_device, rows, paged, 2000 + rows + 100 * paged)
    with flags("0"):
        off = case.run()
    with flags("1"):
        on = case.run()
    for name in off:
        assert torch.equal(off[name], on[name]), f"main_tail {name} differs at rows={rows} paged={paged}"
    print(f"Q1_BITWISE main_tail rows={rows} paged={paged} OK")


# ------------------------------------------------------------------------------------------------ selection_row
class SelectionCase:
    def __init__(self, mesh, rows, paged, seed):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.rows, self.paged = mesh, rows, paged
        capacity = BLOCKS * PAGE_SIZE if paged else 4096
        ids = torch.randint(0, capacity // 4, (rows, 512), generator=g)
        if paged:
            offsets = (torch.arange(2048) % 4).unsqueeze(0).repeat(32, 1)
        else:
            offsets = (torch.arange(2048) % 4).unsqueeze(0) + (torch.arange(32) * capacity).unsqueeze(1)
        keep = torch.where(torch.rand(32, 2080, generator=g) < 0.8, 0xFFFFFFFF, 0).to(torch.int64)
        tail = torch.randint(0, capacity, (32, 2080), generator=g)
        tail[torch.rand(32, 2080, generator=g) < 0.5] = 0xFFFFFFFF
        fill = torch.where(keep == 0, tail, torch.zeros_like(tail))
        sentinel = torch.randint(0, capacity, (1, 32), generator=g)
        sentinel[0, ::5] = 0xFFFFFFFF
        args = (
            up(mesh, ids.reshape(1, 1, rows, 512), U32),
            up(mesh, sentinel.reshape(1, 1, 1, 32), U32),
            up(mesh, offsets.reshape(1, 1, 32, 2048), U32),
            up(mesh, keep.reshape(1, 1, 32, 2080), U32),
            up(mesh, fill.reshape(1, 1, 32, 2080), U32),
        )
        self.args = args
        if paged:
            table, _ = ref.random_page_table(rows, BLOCKS, g)
            self.kwargs = dict(page_table=up(mesh, table.reshape(1, 1, 32, BLOCKS), I32), page_size=PAGE_SIZE)
        else:
            self.kwargs = {}

    def run(self, live_rows=None):
        return {"row": bits(qsa_block.selection_row(*self.args, **self.kwargs, live_rows=live_rows))}

    def call(self):
        return qsa_block.selection_row(*self.args, **self.kwargs)

    def prepare_timing(self):
        pass


@MESH
@PARAMS
@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@pytest.mark.parametrize("rows", ROWS)
def test_selection_row_selrow2_bitwise(mesh_device, rows, paged):
    case = SelectionCase(mesh_device, rows, paged, 3000 + rows + 100 * paged)
    lives = [None] + ([] if paged or rows == 1 else [rows // 2, 1])  # the live-rows form is the unpaged one
    for live in lives:
        with flags(selrow2="0"):
            off = case.run(live)
        with flags(selrow2="1"):
            on = case.run(live)
        assert torch.equal(off["row"], on["row"]), f"selection_row differs at rows={rows} paged={paged} live={live}"
        assert (off["row"] != 0xFFFFFFFF).any() and (off["row"] == 0xFFFFFFFF).any()
    print(f"Q1_BITWISE selection_row rows={rows} paged={paged} OK")


# ------------------------------------------------------------------------------------------------ timing
def traced_us(mesh, call):
    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ops):
            ttnn.deallocate(call())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    ttnn.deallocate(call())  # warm: the program cache is filled before the capture
    short, long_ = capture(TRACE_SHORT), capture(TRACE_LONG)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return (tl - ts) / (TRACE_LONG - TRACE_SHORT)


@MESH
@PARAMS
def test_q1_timing(mesh_device):
    mesh = mesh_device
    for rows in (8, 32):
        cases = [
            ("index_tail", IndexCase(mesh, rows, 10 + rows), "ownpos"),
            ("main_tail", MainCase(mesh, rows, False, 20 + rows), "ownpos"),
            ("main_tail_paged", MainCase(mesh, rows, True, 30 + rows), "ownpos"),
            ("selection_row", SelectionCase(mesh, rows, False, 40 + rows), "selrow2"),
            ("selection_row_paged", SelectionCase(mesh, rows, True, 50 + rows), "selrow2"),
        ]
        for name, case, flag in cases:
            case.prepare_timing()
            best = {"0": 1e30, "1": 1e30}
            for mode in ("0", "1", "0", "1"):  # alternate; keep the best slope of each
                with flags(**{flag: mode}):
                    best[mode] = min(best[mode], traced_us(mesh, case.call))
            print(f"Q1_TIMING prog={name} rows={rows} off_us={best['0']:.2f} on_us={best['1']:.2f}")
