# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""cards.py: which cards of a box form a mesh, found by probing, not hardcoded (no device here: the probe is faked)."""

from models.demos.common.bringup.testing import cards as C
from models.demos.common.bringup.testing import fork_source as FS
from models.demos.common.bringup.testing import fork_tests as FT


def test_candidates_try_consecutive_runs_first_then_every_subset():
    got = list(C.candidates(4, 2))
    assert got[:3] == [(0, 1), (1, 2), (2, 3)]
    assert sorted(got) == sorted({(a, b) for a in range(4) for b in range(a + 1, 4)})


def test_a_subset_fits_its_shape_or_the_transpose():
    assert C.fits((4, 1), (1, 4)) and C.fits((2, 2), (2, 2))
    assert not C.fits((4, 1), (2, 2)) and not C.fits(None, (1, 1))


def _box(tmp_path, monkeypatch, n):
    dev = tmp_path / "dev"
    dev.mkdir()
    for i in range(n):
        (dev / str(i)).touch()
    monkeypatch.setattr(C, "DEV_DIR", dev)
    monkeypatch.setattr(C, "CACHE", tmp_path / "cards.json")


def test_visible_devices_probes_once_per_shape_and_caches(tmp_path, monkeypatch):
    _box(tmp_path, monkeypatch, 8)
    # a 4x2 box cabled so that cards {0,1,4,5} form a 2x2 and every run of 4 consecutive cards a line
    shapes = {(0, 1, 4, 5): (2, 2)}
    calls = []

    def probe(c):
        calls.append(c)
        if len(c) == 1:
            return (1, 1)
        return shapes.get(c, (4, 1) if c == tuple(range(c[0], c[0] + 4)) else None)

    assert C.visible_devices((2, 2), probe) == "0,1,4,5"
    assert C.visible_devices((1, 4), probe) == "0,1,2,3"
    assert C.visible_devices((1, 1), probe) == "0"
    n = len(calls)
    assert C.visible_devices((2, 2), probe) == "0,1,4,5" and len(calls) == n  # cached
    assert C.visible_devices((4, 2), probe) is None  # every card: run as before


def test_a_shape_no_subset_forms_is_not_cached(tmp_path, monkeypatch):
    _box(tmp_path, monkeypatch, 4)
    assert C.visible_devices((1, 2), lambda c: None) is None
    assert C.visible_devices((1, 2), lambda c: (2, 1)) == "0,1"


def test_a_source_entry_names_its_mesh_or_gets_it_from_its_filter():
    assert FS.entry_mesh({"mesh": [2, 2]}) == (2, 2)
    assert FS.entry_mesh({"k": "4x1 and not fabric2d"}) == (4, 1)
    assert FS.entry_mesh({"k": "not perf"}) == (1, 1)


def test_a_model_case_node_maps_to_its_case_mesh():
    meshes = {"glm-2x2-s1": (2, 2), "glm-2x2-s1-w1": (4, 2)}
    assert FT.mesh_of("t.py::test_x[blackhole-glm-2x2-s1]", meshes) == (2, 2)
    assert FT.mesh_of("t.py::test_x[blackhole-glm-2x2-s1-w1]", meshes) == (4, 2)  # the longest id wins
    assert FT.mesh_of("t.py::test_reference_on_cpu", meshes) is None
