# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_ROUTER_SC_LANES (one fast top-k group per occupied sort pass) without a device: flag off gives today's program
parts (same kernels, defines, core ranges, arguments) for rows 1, 8 and 32; flag on gives one group per occupied pass on
disjoint cores of a 4 x 6 rectangle that fits the p150 grid beside the other moe_dense groups."""

from __future__ import annotations

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_dense as md
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import placement
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc as sc

READERS = frozenset({(0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4)})
RESERVED = frozenset(set(placement.STORAGE_CORES) | READERS | {(10, 9)})
PASS_ROWS = {2: 2, 8: 2, 9: 2, 16: 2, 17: 3, 18: 4, 32: 4}


class _T:
    dtype = ttnn.float32

    def device(self):
        return "mesh"

    def buffer_address(self):
        return 0x1000


@pytest.fixture
def stubs(monkeypatch):
    for name in ("QWEN38_ROUTER_SC_LANES", sc.SORTCORE_ENV, sc.PAR_ENV, sc.FAST_ENV, rt.LANES_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fp, "accessor_args", lambda tensor: [])
    monkeypatch.setattr(
        sc.gr_read, "noc_map", lambda mesh: {(x, y): (x + 18, y + 18) for x in range(12) for y in range(10)}
    )
    monkeypatch.setattr(sc, "float_index_template", lambda mesh: _T())
    monkeypatch.setattr(sc.fp, "tensor_bytes", lambda tensor: 1)
    return ttnn.CoreRange(ttnn.CoreCoord(7, 3), ttnn.CoreCoord(10, 8))


def _signature(parts):
    kernels, cbs, semaphores = parts
    return (
        [
            (
                k.kernel_source,
                [(r.start.x, r.start.y, r.end.x, r.end.y) for r in k.core_ranges.ranges()],
                list(k.compile_time_args),
                list(k.named_compile_time_args),
                list(k.defines),
                [(c.x, c.y, list(a)) for c, a in _runtime(k)],
            )
            for k in kernels
        ],
        [(c.total_size, [(r.start.x, r.start.y, r.end.x, r.end.y) for r in c.core_ranges.ranges()]) for c in cbs],
        [(s.id, [(r.start.x, r.start.y, r.end.x, r.end.y) for r in s.core_ranges.ranges()]) for s in semaphores],
    )


def _runtime(kernel):
    out = []
    for r in kernel.core_ranges.ranges():  # runtime_args is indexed [x][y]; iterating it does not end
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                args = kernel.runtime_args[x][y]
                if len(list(args)):
                    out.append((ttnn.CoreCoord(x, y), args))
    return out


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("flag", ["0"])  # unset is now on (test_flag_default_is_on)
def test_flag_off_is_todays_parts(monkeypatch, stubs, rows, flag):
    monkeypatch.setenv(sc.LANES_ENV, flag)
    call = lambda f, **k: f(_T(), _T(), _T(), _T(), rows=rows, top_k=10, rectangle=stubs, sem_base=0, **k)  # noqa: E731
    got = call(sc.program_parts)
    want = call(sc.two_core_parts) if rows == 1 else call(rt.program_parts)
    assert _signature(got) == _signature(want)
    assert not sc.lanes_enabled(rows)


def test_flag_default_is_on(monkeypatch, stubs):
    assert sc.lanes_flag() and sc.lanes_enabled(8) and not sc.lanes_enabled(1)
    monkeypatch.setenv(sc.LANES_ENV, "0")
    assert not sc.lanes_flag() and not sc.lanes_enabled(8)


def test_flag_off_keeps_the_top_k_rectangle(monkeypatch):
    seen = []
    monkeypatch.setenv(sc.LANES_ENV, "0")
    monkeypatch.setattr(md, "_PLANS", {})
    monkeypatch.setattr(
        md.placement,
        "free_rectangle",
        lambda mesh, w, h, **k: seen.append((w, h))
        or ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1)),
    )
    monkeypatch.setattr(md.gr_read, "noc_map", lambda mesh: {(x, y): (x, y) for x in range(12) for y in range(10)})
    mesh = object()
    md.plan_for(mesh)
    assert seen[0] == rt.RECTANGLE and set(md._PLANS) == {id(mesh)}
    monkeypatch.setenv(sc.LANES_ENV, "1")
    md.plan_for(mesh)
    assert seen[3] == sc.LANES_RECTANGLE and (id(mesh), "lanes") in md._PLANS


@pytest.mark.parametrize("rows", sorted(PASS_ROWS))
def test_groups_partition_the_live_rows(rows):
    groups = sc.lane_groups(rows)
    assert len(groups) == PASS_ROWS[rows]
    live = [t for g in groups for t in range(32) if (g["token_mask"] >> t) & 1]
    assert sorted(live) == list(range(rows))
    for g in groups:
        assert g["pass_mask"] == 1 << g["pass"] and g["face"] == g["pass"] >> 1
        tokens = rt.PASS_TOKENS[g["pass"]]
        assert all(t in tokens for t in range(32) if (g["token_mask"] >> t) & 1)
        assert g["iters"] == 8  # the full pass
    assert [(g["pass_mask"], g["token_mask"]) for g in groups] == rt.lanes_plan(rows)


@pytest.mark.parametrize("par", ["1", "0"])
@pytest.mark.parametrize("rows", [8, 9, 17, 32])
def test_flag_on_builds_one_group_per_pass_on_disjoint_cores(monkeypatch, stubs, rows, par):
    monkeypatch.setenv(sc.LANES_ENV, "1")
    monkeypatch.setenv(sc.PAR_ENV, par)
    assert sc.lanes_enabled(rows)
    monkeypatch.setattr(rt, "program_parts", lambda *a, **k: pytest.fail("fell back"))
    kernels, cbs, semaphores = sc.program_parts(
        _T(), _T(), _T(), _T(), rows=rows, top_k=10, rectangle=stubs, sem_base=0
    )
    groups = sc.lane_groups(rows)
    per_group = (8 if par == "1" else 6) - (0 if par == "1" else 0)
    mains = [k for k in kernels if k.kernel_source.endswith("compute_rtfl_main.cpp")]
    assert len(mains) == len(groups) == PASS_ROWS[rows]
    taken = set()
    rect = {(x, y) for x in range(7, 11) for y in range(3, 9)}
    for k in kernels:  # a kernel instance never shares a core with another group's kernel of the same RISC role
        pass
    by_core = {}
    for k in kernels:
        for r in k.core_ranges.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    by_core.setdefault((x, y), []).append(k.kernel_source.rsplit("/", 1)[-1])
    assert set(by_core) <= rect
    expect_cores = len(groups) * (6 if par == "1" else 2)
    assert len(by_core) == expect_cores
    for core, names in by_core.items():
        assert len(names) == len(set(names)) and len(names) <= (3 if par == "1" else 4), (core, names)
    # every kernel source is a real file; the lanes' own kernels live in kernels_lanes
    for k in kernels:
        assert (fp.REPO_ROOT / k.kernel_source).is_file(), k.kernel_source
    # core A_p: pass mask and token mask of its group; the same CB / semaphore ids everywhere
    for g, main in zip(groups, mains):
        ((_x, _y, args),) = [(c.x, c.y, list(a)) for c, a in _runtime(main)]
        assert args == [g["pass_mask"], g["token_mask"]]
        defines = dict(main.defines)
        assert defines["QWEN38_RTF_ROWTRIM_ITERS"] == str(g["iters"]) and ("QWEN38_RTFL_RC" in defines) == bool(
            g["face"]
        )
    names = {tuple(k.named_compile_time_args) for k in kernels}
    assert len(names) == 1
    assert semaphores[0].id == 0 and (len(semaphores) == 2) == (par == "1")
    cb_cores = {tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in c.core_ranges.ranges()) for c in cbs}
    assert len(cb_cores) == 1
    cores_cb = sum((r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1) for r in cbs[0].core_ranges.ranges())
    assert cores_cb == expect_cores
    assert sc.lanes_core_count(rows) == expect_cores


def test_group_core_map_rows_8_and_32():
    rect = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 5))
    for count in (2, 4):
        cmap = sc.lane_core_map(rect, count, True)
        cells = []
        for c in cmap:
            cells += [(c["a"].x, c["a"].y), (c["b"].x, c["b"].y)] + [(f.x, f.y) for f in c["folds"]]
        assert len(cells) == len(set(cells)) == 6 * count
        assert cmap[0]["a"].x == 0 and cmap[0]["a"].y == 0 and cmap[0]["b"].x == 1  # group 0 = the rows == 1 cores
        assert [(f.x, f.y) for f in cmap[0]["folds"]] == [(0, 1), (1, 1), (2, 1), (3, 1)]


@pytest.mark.parametrize("grid", [(11, 10), (12, 10)])
def test_rectangle_fits_beside_the_other_groups(grid):
    x, y = grid
    topk = placement.free_rectangle_in(x, y, RESERVED, *sc.LANES_RECTANGLE)
    taken = set(RESERVED) | set(placement.rectangle_cores(topk))
    workers = placement.free_rectangle_in(x, y, taken, *md.DOWN_SHAPES[0])
    taken |= set(placement.rectangle_cores(workers))
    untilize = placement.free_rectangle_in(x, y, taken, *md.UNTILIZE_SHAPE)
    assert not set(placement.rectangle_cores(untilize)) & taken
