# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP fracture correctness on a real mesh: a column-fractured matmul + all_gather must reproduce the
dense single-chip matmul (PCC ~ 1). Skips when ttnn / a multi-chip mesh is unavailable, so it is inert
in the offline venv and runs only on hardware. Proven on a QB2 (2,2) mesh: PCC 0.99997 across shapes."""
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

    monkeypatch.setattr(ttnn, "all_gather", _ag)
    out = tp_fracture.all_gather_full(_fake_mesh(rows, cols), object(), dim=-1)
    assert seen == expect_axes, "mesh (%d,%d) gathered along %s" % (rows, cols, seen)
    assert out is not None


def test_all_gather_full_on_a_single_device_mesh_is_a_no_op(monkeypatch):
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize import tp_fracture

    called = []
    monkeypatch.setattr(ttnn, "all_gather", lambda *a, **k: called.append(1))
    sentinel = object()
    assert tp_fracture.all_gather_full(_fake_mesh(1, 1), sentinel, dim=-1) is sentinel
    assert not called, "a 1-device mesh has nothing to gather"


def test_all_gather_full_falls_back_when_the_shape_is_unreadable(monkeypatch):
    """An unreadable shape must still attempt the gather, not crash the lever."""
    pytest.importorskip("torch")
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize import tp_fracture

    seen = []
    monkeypatch.setattr(ttnn, "all_gather", lambda t, dim=None, cluster_axis=None, **k: seen.append(cluster_axis) or t)

    class _Opaque:
        @property
        def shape(self):
            raise RuntimeError("no shape")

        def get_num_devices(self):
            raise RuntimeError("no count")

    tp_fracture.all_gather_full(_Opaque(), object(), dim=-1)
    assert seen == [None]
