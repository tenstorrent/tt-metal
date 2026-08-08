# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP fracture correctness on a real mesh: a column-fractured matmul + all_gather must reproduce the
dense single-chip matmul (PCC ~ 1). Skips when ttnn / a multi-chip mesh is unavailable, so it is inert
in the offline venv and runs only on hardware. Proven on a QB2 (2,2) mesh: PCC 0.99997 across shapes.

The axis-selection tests below are NOT hardware tests and must not skip offline -- they assert which
cluster_axis all_gather_full picks for a given mesh shape, which is the logic that made this module
work on a (1,4) mesh and fail on a (2,2) one. They stub ttnn.all_gather with raising=False because
`pytest.importorskip("ttnn")` does not guard what it looks like it guards: offline, ttnn resolves to
the repo's source directory as a NAMESPACE PACKAGE, so the import succeeds and it is the first
attribute access that fails. The guard passed and the tests errored instead of skipping OR running."""
from pathlib import Path

import pytest


@pytest.mark.parametrize("m,k,n", [(128, 512, 1024), (256, 1024, 2048), (64, 2048, 4096)])
def test_column_fracture_matches_dense(m, k, n):
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize.tp_fracture import verify_fracture

    if not hasattr(ttnn, "open_mesh_device"):
        pytest.skip("ttnn mesh API unavailable")
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    except Exception as exc:
        pytest.skip(f"no multi-chip mesh available: {exc}")
    try:
        r = verify_fracture(mesh, m=m, k=k, n=n, tp=4)
        assert r["pcc"] > 0.99, r
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_sweep_degrees_returns_a_legal_fastest():
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize.tp_fracture import sweep_degrees

    if not hasattr(ttnn, "open_mesh_device"):
        pytest.skip("ttnn mesh API unavailable")
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    except Exception as exc:
        pytest.skip(f"no multi-chip mesh available: {exc}")
    try:
        r = sweep_degrees(mesh, m=32, k=8192, n=8192)
        assert r["best_tp"] in r["timings_ms"]
        assert r["timings_ms"][r["best_tp"]] == min(r["timings_ms"].values())
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _fake_mesh(rows, cols):
    class _M:
        shape = (rows, cols)

        def get_num_devices(self):
            return rows * cols

    return _M()


@pytest.mark.parametrize(
    "rows,cols,expect_axes",
    [(1, 4, [1]), (4, 1, [0]), (2, 2, [1, 0]), (1, 8, [1]), (2, 4, [1, 0])],
)
def test_all_gather_full_picks_axes_from_the_mesh_shape(rows, cols, expect_axes, monkeypatch):
    """The gather must cover EVERY device whatever the mesh shape.

    A bare all_gather works only on a 1-D mesh; with 1-D fabric on a 2-D mesh ttnn raises rather than
    guessing an axis, so this module passed on a (1,4) mesh and crashed on a (2,2) one -- in real runs
    too, since the mesh shape comes from the model. Runs offline: the mesh and the op are stubbed, so
    only the axis decision is under test.
    """
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize import tp_fracture

    seen = []

    def _ag(t, dim=None, cluster_axis=None, **kw):
        seen.append(cluster_axis)
        return t

    monkeypatch.setattr(ttnn, "all_gather", _ag, raising=False)
    out = tp_fracture.all_gather_full(_fake_mesh(rows, cols), object(), dim=-1)
    assert seen == expect_axes, "mesh (%d,%d) gathered along %s" % (rows, cols, seen)
    assert out is not None


def test_all_gather_full_on_a_single_device_mesh_is_a_no_op(monkeypatch):
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize import tp_fracture

    called = []
    monkeypatch.setattr(ttnn, "all_gather", lambda *a, **k: called.append(1), raising=False)
    sentinel = object()
    assert tp_fracture.all_gather_full(_fake_mesh(1, 1), sentinel, dim=-1) is sentinel
    assert not called, "a 1-device mesh has nothing to gather"


def test_all_gather_full_falls_back_when_the_shape_is_unreadable(monkeypatch):
    """An unreadable shape must still attempt the gather, not crash the lever."""
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize import tp_fracture

    seen = []
    monkeypatch.setattr(
        ttnn, "all_gather", lambda t, dim=None, cluster_axis=None, **k: seen.append(cluster_axis) or t, raising=False
    )

    class _Opaque:
        @property
        def shape(self):
            raise RuntimeError("no shape")

        def get_num_devices(self):
            raise RuntimeError("no count")

    tp_fracture.all_gather_full(_Opaque(), object(), dim=-1)
    assert seen == [None]


def _perf_mcp(tmp_path, monkeypatch):
    import importlib.util
    import sys as _sys

    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    for var in ("PERF_MCP_TP_MESH", "TT_PERF_MESH_ROWS", "TT_PERF_MESH_COLS"):
        monkeypatch.delenv(var, raising=False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pm_tpmesh_ut", root / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["pm_tpmesh_ut"] = mod
    spec.loader.exec_module(mod)
    # These cover the arithmetic fallback, so the box registry is neutralised: otherwise the result
    # depends on which board the suite happens to run on (this one answers p300).
    monkeypatch.setattr(mod, "_tp_box", lambda num=0: None)
    return mod


class _FakeTtnn:
    def __init__(self, num):
        self._num = num

    def get_num_devices(self):
        return self._num


@pytest.mark.parametrize(
    "num,expect",
    [
        (1, [(1, 1)]),
        (2, [(1, 2)]),
        (4, [(1, 4), (2, 2)]),
        (8, [(1, 8), (2, 4)]),
        (6, [(1, 6), (2, 3)]),
        (32, [(1, 32), (4, 8), (2, 16)]),
    ],
)
def test_tp_mesh_shapes_come_from_the_real_device_count(num, expect, tmp_path, monkeypatch):
    """A literal MeshShape(2, 2) was baked in from the QB2 bench machine, so a 1x8 board, a galaxy
    2x4 or a 2-chip p300c was asked for a topology it does not have."""
    monkeypatch.delenv("PERF_MCP_TP_MESH", raising=False)
    pm = _perf_mcp(tmp_path, monkeypatch)
    assert pm._tp_mesh_shapes(_FakeTtnn(num)) == expect


def test_tp_mesh_shape_can_be_overridden(tmp_path, monkeypatch):
    pm = _perf_mcp(tmp_path, monkeypatch)
    monkeypatch.setenv("PERF_MCP_TP_MESH", "2x4")
    assert pm._tp_mesh_shapes(_FakeTtnn(8)) == [(2, 4)]
    monkeypatch.setenv("PERF_MCP_TP_MESH", "1,8")
    assert pm._tp_mesh_shapes(_FakeTtnn(8)) == [(1, 8)]
    monkeypatch.setenv("PERF_MCP_TP_MESH", "garbage")
    assert pm._tp_mesh_shapes(_FakeTtnn(4)) == [(1, 4), (2, 2)]


def test_an_unreadable_device_count_does_not_crash_the_lever(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_TP_MESH", raising=False)
    pm = _perf_mcp(tmp_path, monkeypatch)

    class _Broken:
        def get_num_devices(self):
            raise RuntimeError("no cluster")

    assert pm._tp_mesh_shapes(_Broken()) == [(1, 1)]


def test_open_tp_mesh_falls_through_to_a_shape_the_board_accepts(tmp_path, monkeypatch):
    """A 1xN ring is preferred, but a board whose links cannot form one must still get a mesh."""
    monkeypatch.delenv("PERF_MCP_TP_MESH", raising=False)
    pm = _perf_mcp(tmp_path, monkeypatch)
    tried = []

    class _Fussy(_FakeTtnn):
        def MeshShape(self, r, c):
            return (r, c)

        def open_mesh_device(self, shape):
            tried.append(shape)
            if shape == (1, 4):
                raise RuntimeError("cannot form a 4-chip ring")
            return "mesh%s" % (shape,)

    assert pm._open_tp_mesh(_Fussy(4)) == "mesh(2, 2)"
    assert tried == [(1, 4), (2, 2)]


def test_open_tp_mesh_raises_the_real_error_when_nothing_opens(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_TP_MESH", raising=False)
    pm = _perf_mcp(tmp_path, monkeypatch)

    class _Dead(_FakeTtnn):
        def MeshShape(self, r, c):
            return (r, c)

        def open_mesh_device(self, shape):
            raise RuntimeError("board wedged")

    raised = None
    try:
        pm._open_tp_mesh(_Dead(4))
    except RuntimeError as exc:
        raised = exc
    assert raised is not None and "board wedged" in str(raised), raised
