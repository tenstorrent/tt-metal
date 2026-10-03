# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the shared-expert / dispatch overlap (tt/moe/shared_overlap.py, TtMiniMaxMoE.forward):
the sub-device split, the 2D matmul configs sized to it, the knobs (plus the routed-expert hybrid threshold), and
the op order of the overlap window (every ttnn op the forward calls is replaced by a recorder). The device tests are
test_ep_moe_vs_ref.py test_ep_moe_shared_schedule and test_ep_moe_overlap_release_recreate.
"""

import math
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.minimax_m3.tt.moe import shared_overlap
from models.demos.minimax_m3.tt.moe.shared_overlap import (
    CB_BUDGET_TILES,
    DEST_TILES,
    SharedExpertOverlap,
    cb_tiles,
    fuse_shared_rs_enabled,
    matmul_2d_config,
    overlap_shared_enabled,
    shared_expert_program_configs,
    split_grid,
)
from models.demos.minimax_m3.tt.moe.tt_minimax_moe import (
    DEFAULT_HYBRID_TOKEN_THRESHOLD,
    TtMiniMaxMoE,
    hybrid_token_threshold,
)

GRID = (11, 10)  # Blackhole galaxy compute grid
HIDDEN, SHARED_INTER, TP = 6144, 3072, 4


def _cores(crs):
    out = set()
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.add((x, y))
    return out


@pytest.mark.parametrize("rows", [1, 2])
def test_split_grid(rows, expect_error):
    dispatch, shared = split_grid(*GRID, rows)
    d, s = _cores(dispatch), _cores(shared)
    assert not d & s
    assert len(d) == GRID[0] * rows and len(s) == GRID[0] * (GRID[1] - rows)
    assert d | s == {(x, y) for x in range(GRID[0]) for y in range(GRID[1])}
    assert all(y < rows for _, y in d)
    with expect_error(AssertionError, "dispatch_rows"):
        split_grid(*GRID, GRID[1])


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("m_tiles", [4, 16, 64, 128, 256])
@pytest.mark.parametrize(
    "k_tiles, n_tiles",
    [(HIDDEN // 32, SHARED_INTER // TP // 32), (SHARED_INTER // TP // 32, HIDDEN // 32)],
    ids=["gate_up", "down"],
)
def test_matmul_2d_config_valid(rows, m_tiles, k_tiles, n_tiles):
    """The checks matmul validates for a 2D mcast config with interleaved in/out, the L1 budget, and that
    the launched grid stays inside the shared sub-device."""
    _, shared = split_grid(*GRID, rows)
    grid = shared.bounding_box().grid_size()
    c = matmul_2d_config(grid, m_tiles, k_tiles, n_tiles)
    assert k_tiles % c.in0_block_w == 0
    assert c.per_core_M % c.out_block_h == 0 and c.per_core_N % c.out_block_w == 0
    assert c.out_block_h % c.out_subblock_h == 0 and c.out_block_w % c.out_subblock_w == 0
    assert c.out_subblock_h * c.out_subblock_w <= DEST_TILES
    assert cb_tiles(c.in0_block_w, c.out_block_h, c.out_block_w, k_tiles) <= CB_BUDGET_TILES
    assert math.ceil(m_tiles / c.per_core_M) <= grid.y and math.ceil(n_tiles / c.per_core_N) <= grid.x
    assert c.per_core_M * grid.y >= m_tiles and c.per_core_N * grid.x >= n_tiles
    assert (c.compute_with_storage_grid_size.x, c.compute_with_storage_grid_size.y) == (grid.x, grid.y)


def test_shared_expert_program_configs_shapes():
    _, shared = split_grid(*GRID, 1)
    x = SimpleNamespace(padded_shape=(1, 1, 2048, HIDDEN))
    gate_w = SimpleNamespace(padded_shape=(1, 1, HIDDEN, SHARED_INTER // TP))
    down_w = SimpleNamespace(padded_shape=(1, 1, SHARED_INTER // TP, HIDDEN))
    gate, up, down = shared_expert_program_configs(shared, x, gate_w, down_w)
    assert gate is up and gate.fused_activation is None
    assert gate.per_core_M * 9 >= 64 and down.per_core_N * 11 >= HIDDEN // 32


class _FakeMesh:
    def __init__(self):
        self.created = []
        self.loaded = []
        self.removed = []

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(*GRID)

    def create_sub_device_manager(self, sub_devices, local_l1_size):
        self.created.append((len(sub_devices), local_l1_size))
        return f"mgr{len(self.created)}"

    def load_sub_device_manager(self, mgr):
        self.loaded.append(mgr)

    def clear_loaded_sub_device_manager(self):
        self.loaded.append("clear")

    def remove_sub_device_manager(self, mgr):
        self.removed.append(mgr)


def test_overlap_manager_shared_per_mesh(monkeypatch):
    monkeypatch.setattr(shared_overlap, "_MANAGERS", {})
    mesh, other = _FakeMesh(), _FakeMesh()
    a, b, c = SharedExpertOverlap(mesh), SharedExpertOverlap(mesh), SharedExpertOverlap(other)
    assert mesh.created == [(2, 0)] and other.created == [(2, 0)] and a.manager_id == b.manager_id
    assert a.shared_sub_device[0] == ttnn.SubDeviceId(1) and a.dispatch_sd_id == ttnn.SubDeviceId(0)
    assert a.shared_cores.num_cores() == 99 and a.dispatch_cores.num_cores() == 11
    a.load()
    a.clear()
    assert mesh.loaded == [a.manager_id, "clear"]


def test_overlap_release_and_recreate(monkeypatch):
    """release() clears a loaded manager, removes it and drops the entry; the next load re-creates it."""
    monkeypatch.setattr(shared_overlap, "_MANAGERS", {})
    mesh, other = _FakeMesh(), _FakeMesh()
    a, _ = SharedExpertOverlap(mesh), SharedExpertOverlap(other)
    first = a.manager_id
    a.load()
    shared_overlap.release(mesh)
    assert mesh.loaded == [first, "clear"] and mesh.removed == [first]
    assert id(mesh) not in shared_overlap._MANAGERS and id(other) in shared_overlap._MANAGERS
    shared_overlap.release(mesh)  # idempotent
    assert mesh.removed == [first]

    # not loaded: no clear; an instance built before the release re-creates the manager on load
    a.load()
    assert len(mesh.created) == 2 and a.manager_id != first and mesh.loaded[-1] == a.manager_id
    a.clear()
    shared_overlap.release(mesh)
    assert mesh.loaded[-1] == "clear" and mesh.loaded.count("clear") == 2 and mesh.removed[-1] == "mgr2"

    shared_overlap.release_all()
    assert other.removed == ["mgr1"] and other.loaded == []
    assert shared_overlap._MANAGERS == {}


def test_knobs(monkeypatch, expect_error):
    for var in ("M3_MOE_OVERLAP_SHARED", "M3_MOE_FUSE_SHARED_RS"):
        monkeypatch.delenv(var, raising=False)
    assert overlap_shared_enabled() and fuse_shared_rs_enabled()  # default on
    monkeypatch.setenv("M3_MOE_OVERLAP_SHARED", "0")
    monkeypatch.setenv("M3_MOE_FUSE_SHARED_RS", "0")
    assert not overlap_shared_enabled() and not fuse_shared_rs_enabled()
    monkeypatch.setenv("M3_MOE_OVERLAP_SHARED", "1")
    monkeypatch.setenv("M3_MOE_FUSE_SHARED_RS", "")
    assert overlap_shared_enabled() and fuse_shared_rs_enabled()
    monkeypatch.setenv("M3_MOE_OVERLAP_SHARED", "yes")
    with expect_error(ValueError, "must be 0 or 1"):
        overlap_shared_enabled()


def test_hybrid_threshold_knob(monkeypatch, expect_error):
    bh, wh = SimpleNamespace(arch=lambda: ttnn.Arch.BLACKHOLE), SimpleNamespace(arch=lambda: ttnn.Arch.WORMHOLE_B0)
    bf4, bf8 = ttnn.bfloat4_b, ttnn.bfloat8_b
    monkeypatch.delenv("M3_MOE_HYBRID_THRESHOLD", raising=False)
    assert DEFAULT_HYBRID_TOKEN_THRESHOLD == 128
    assert hybrid_token_threshold(bh, bf4) == 128
    assert hybrid_token_threshold(bh, bf8) is None and hybrid_token_threshold(wh, bf4) is None
    monkeypatch.setenv("M3_MOE_HYBRID_THRESHOLD", "")
    assert hybrid_token_threshold(bh, bf4) == 128
    monkeypatch.setenv("M3_MOE_HYBRID_THRESHOLD", "0")
    assert hybrid_token_threshold(bh, bf4) is None
    monkeypatch.setenv("M3_MOE_HYBRID_THRESHOLD", " 64 ")
    assert hybrid_token_threshold(bh, bf4) == 64 and hybrid_token_threshold(bh, bf8) == 64
    for bad in ("-1", "on", "1.5"):
        monkeypatch.setenv("M3_MOE_HYBRID_THRESHOLD", bad)
        with expect_error(ValueError, "M3_MOE_HYBRID_THRESHOLD"):
            hybrid_token_threshold(bh, bf4)


# ---- op order of TtMiniMaxMoE.forward, every ttnn call recorded ----


class _T:
    def __init__(self, name, shape=(1, 256, HIDDEN)):
        self.name, self.shape, self.dtype = name, shape, ttnn.bfloat16

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG

    def __repr__(self):
        return self.name


def _moe_with_recorder(monkeypatch, overlap):
    log = []

    def op(name):
        def f(t, *a, **k):
            log.append((name, getattr(t, "name", t)))
            return _T(f"{name}({getattr(t, 'name', t)})", getattr(t, "shape", (1, 256, HIDDEN)))

        return f

    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: log.append(("free", t.name)))
    for name in ("to_layout", "reshape", "to_memory_config", "typecast", "squeeze", "unsqueeze"):
        monkeypatch.setattr(ttnn, name, op(name))

    class _Dispatch:
        subdevice_id = None

        def __call__(self, x, *a, **k):
            log.append(("dispatch", x.name, self.subdevice_id))
            return _T("buf"), _T("meta")

    class _Overlap:
        dispatch_sd_id = "sd0"
        shared_sub_device = ("sd1", "cores1")

        def load(self):
            log.append(("load",))

        def clear(self):
            log.append(("clear",))

    moe = object.__new__(TtMiniMaxMoE)
    routing = (_T("offs"), _T("counts"), _T("regions"), _T("all_offs"))
    moe.__dict__.update(
        mesh_device=SimpleNamespace(shape=(2, 4)),
        num_routed_experts=128,
        num_experts_per_tok=4,
        emb_dim=HIDDEN,
        experts_per_chip=4,
        seq_len_per_chip=256,
        tt_expert_dispatch_table=_T("table"),
        routing_setup=lambda **k: routing,
        dispatch_module=_Dispatch(),
        routed_expert=lambda *a: (log.append(("experts",)), _T("exp_out"))[1],
        combine_module=lambda *a: (log.append(("combine",)), _T("combined"))[1],
        reduce_module=SimpleNamespace(num_links=2),
        overlap=_Overlap() if overlap else None,
    )

    class _Reduce:
        num_links = 2

        def __call__(self, combined, weights=None, indices=None, expert_dispatch_table=None, addend=None):
            log.append(("reduce", getattr(addend, "name", None)))
            return _T("routed")

    moe.reduce_module = _Reduce()
    return moe, log


def _shared_fn(log):
    def fn(sub_device, keep_alive):
        log.append(("shared", sub_device))
        keep_alive.append(_T("shared_act"))
        return _T("shared_partial")

    return fn


def _names(log, kind):
    return [i for i, e in enumerate(log) if e[0] == kind]


def test_forward_default_unchanged(monkeypatch):
    """No shared_fn: no load / clear, dispatch on no sub-device, x freed right after dispatch."""
    moe, log = _moe_with_recorder(monkeypatch, overlap=True)
    out = moe.forward(_T("x"), topk_indices=_T("idx"), topk_weights=_T("wts"))
    assert out.name.startswith("squeeze")
    assert not _names(log, "load") and not _names(log, "clear") and not _names(log, "shared")
    (d,) = _names(log, "dispatch")
    assert log[d][2] is None
    assert ("free", "x") in log[d + 1 : d + 3]
    assert log[_names(log, "reduce")[0]][1] is None


@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "no_overlap"])
@pytest.mark.parametrize("fuse", [True, False], ids=["fuse_rs", "own_rs"])
def test_forward_window_order(monkeypatch, overlap, fuse):
    moe, log = _moe_with_recorder(monkeypatch, overlap=overlap)
    routed, shared = moe.forward(
        _T("x"), topk_indices=_T("idx"), topk_weights=_T("wts"), shared_fn=_shared_fn(log), fuse_shared=fuse
    )
    (d,) = _names(log, "dispatch")
    (s,) = _names(log, "shared")
    assert d < s < _names(log, "experts")[0]
    if overlap:
        (lo,) = _names(log, "load")
        (cl,) = _names(log, "clear")
        assert lo < d and s < cl < _names(log, "experts")[0]
        assert log[d][2] == "sd0" and log[s][1] == ("sd1", "cores1")
        window = log[lo:cl]
        assert not any(e[0] == "free" for e in window), window
        # nothing but dispatch and the shared expert is enqueued inside the window
        assert {e[0] for e in window} <= {"load", "dispatch", "shared"}, window
    else:
        assert not _names(log, "load") and log[d][2] is None and log[s][1] is None
    frees = [e[1] for e in log if e[0] == "free"]
    assert "shared_act" in frees and "x" in frees
    # x (the shared expert's input) is freed only after the shared expert was enqueued
    assert log.index(("free", "x")) > s
    (r,) = _names(log, "reduce")
    if fuse:
        assert log[r][1] == "shared_partial" and shared is None and ("free", "shared_partial") in log[r:]
    else:
        assert log[r][1] is None and shared.name == "shared_partial"


@pytest.mark.parametrize("fail_in", ["dispatch", "shared", "load"])
def test_forward_window_clears_on_error(monkeypatch, fail_in, expect_error):
    """Once load() succeeded, clear() runs exactly once even when dispatch or the shared expert raises, and no
    keep-alive tensor is freed before it."""
    moe, log = _moe_with_recorder(monkeypatch, overlap=True)

    def boom(*a, **k):
        log.append((fail_in,))
        raise RuntimeError(fail_in)

    if fail_in == "dispatch":
        type(moe.dispatch_module).__call__ = lambda self, *a, **k: boom()
    elif fail_in == "load":
        moe.overlap.load = boom

    def shared_fn(sub_device, keep_alive):
        keep_alive.append(_T("shared_act"))
        if fail_in == "shared":
            boom()
        return _T("shared_partial")

    with expect_error(RuntimeError, fail_in):
        moe.forward(_T("x"), topk_indices=_T("idx"), topk_weights=_T("wts"), shared_fn=shared_fn, fuse_shared=True)
    if fail_in == "load":
        assert not _names(log, "clear")
    else:
        (lo,) = _names(log, "load")
        (cl,) = _names(log, "clear")
        assert lo < _names(log, fail_in)[0] < cl
        assert not any(e[0] == "free" for e in log[lo:cl])
    assert not _names(log, "experts")
