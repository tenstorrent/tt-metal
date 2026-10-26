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


def test_a_run_mesh_smaller_than_the_box_gets_its_cards_or_a_cached_reason(tmp_path, monkeypatch):
    """F192: at launch, a run's mesh smaller than the box shows only its cards; when none form it, the launch says so,
    and a resume does not probe again until the miss is deleted from the cache."""
    from models.demos.common.bringup import orchestrator as O

    _box(tmp_path, monkeypatch, 8)
    assert O.run_mesh_cards((2, 4), lambda c: None) == (None, None)  # every card
    assert O.run_mesh_cards((1, 2), lambda c: (2, 1)) == ("0,1", None)
    calls = []
    vis, why = O.run_mesh_cards((2, 2), lambda c: calls.append(c))
    assert vis is None and "no subset" in why and calls
    n = len(calls)
    assert O.run_mesh_cards((2, 2), lambda c: calls.append(c))[1] and len(calls) == n  # the miss is cached
    cache = C._load()
    cache[C._key()].pop("2x2:none")
    C._save(cache)
    assert O.run_mesh_cards((2, 2), lambda c: (2, 2)) == ("0,1,2,3", None)
    assert O.cached_cards((2, 2)) == "0,1,2,3" and O.cached_cards((2, 4)) is None


def test_no_selftest_sees_the_real_cards():
    """F192: a selftest of `resume` once probed a Galaxy's real cards next to a live run (conftest hides them)."""
    from models.demos.common.bringup import orchestrator as O

    assert C.num_cards() == 0 and O.run_mesh_cards((1, 1), lambda c: 1 / 0) == (None, None)
