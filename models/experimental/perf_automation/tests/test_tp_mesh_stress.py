# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress the TP mesh-shape derivation: it decides which topology the lever opens on a real board.

A literal MeshShape(2, 2) was baked in from the bench machine the lever was written on, so every
board that is not 4-chip-square was asked for a topology it does not have. Replacing a constant with
a derivation is only an improvement if the derivation holds for the whole range of boards -- a wrong
shape here does not fail cleanly, it opens the wrong number of chips and the sweep then reports
timings for a mesh nobody asked for.

Covers: every device count 1..64, primes, powers of two, the odd counts a partially-reset board
reports, malformed and hostile overrides, and the fall-through order when a board refuses a shape.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def pm(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    for var in ("PERF_MCP_TP_MESH", "TT_PERF_MESH_ROWS", "TT_PERF_MESH_COLS"):
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("pm_stress_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pm_stress_ut"] = mod
    spec.loader.exec_module(mod)
    # The arithmetic path is the LAST resort; these tests exercise it unless they say otherwise, so
    # the box registry is neutralised per-test rather than depending on the machine running them.
    monkeypatch.setattr(mod, "_tp_box", lambda num=0: None)
    return mod


class _Ttnn:
    def __init__(self, num):
        self._num = num

    def get_num_devices(self):
        return self._num

    def MeshShape(self, r, c):
        return (r, c)

    def open_mesh_device(self, shape):
        return "mesh%s" % (shape,)


@pytest.mark.parametrize("num", list(range(1, 65)))
def test_every_shape_covers_exactly_the_devices_present(num, pm):
    """THE INVARIANT: every candidate must account for all N chips, no more and no fewer. A shape
    that multiplies to less silently leaves chips idle and the TP degree it reports is a lie."""
    shapes = pm._tp_mesh_shapes(_Ttnn(num))
    assert shapes, num
    for r, c in shapes:
        assert r >= 1 and c >= 1, (num, r, c)
        assert r * c == num, "shape %sx%s does not cover %d devices" % (r, c, num)


@pytest.mark.parametrize("num", list(range(1, 65)))
def test_the_first_choice_is_always_the_1d_ring(num, pm):
    """1xN needs no cluster_axis under 1-D fabric, which is the whole reason the gather broke."""
    assert pm._tp_mesh_shapes(_Ttnn(num))[0] == (1, num)


@pytest.mark.parametrize("num", [2, 3, 5, 7, 11, 13, 17, 31, 61])
def test_a_prime_count_offers_only_the_ring(num, pm):
    assert pm._tp_mesh_shapes(_Ttnn(num)) == [(1, num)]


@pytest.mark.parametrize(
    "num,expect",
    [
        (4, [(1, 4), (2, 2)]),
        (8, [(1, 8), (2, 4)]),
        (16, [(1, 16), (4, 4), (2, 8)]),
        (32, [(1, 32), (4, 8), (2, 16)]),
        (64, [(1, 64), (8, 8), (4, 16), (2, 32)]),
    ],
)
def test_real_board_counts_get_the_expected_ladder(num, expect, pm):
    """p300c (2), QB2 (4), galaxy (32) and the shapes in between, most balanced first after the ring."""
    assert pm._tp_mesh_shapes(_Ttnn(num)) == expect


@pytest.mark.parametrize("num", [0, -1, -8])
def test_a_nonsense_device_count_degrades_to_one_chip(num, pm):
    """Never emit a 1x0 or a negative mesh: open_mesh_device would fail deep in UMD instead of here."""
    assert pm._tp_mesh_shapes(_Ttnn(num)) == [(1, 1)]


@pytest.mark.parametrize("raw", ["2x4", "2,4", " 2x4 ", "2X4", "2x4x9"])
def test_overrides_that_parse(raw, pm, monkeypatch):
    monkeypatch.setenv("PERF_MCP_TP_MESH", raw)
    assert pm._tp_mesh_shapes(_Ttnn(8)) == [(2, 4)]


@pytest.mark.parametrize("raw", ["", "  ", "garbage", "2", "x", "2x", "a,b", "1.5x4", "-", "None", "[1,8]"])
def test_a_malformed_override_falls_back_to_the_derivation(raw, pm, monkeypatch):
    """A typo in an env var must not silently pick a wrong topology -- it must be ignored."""
    monkeypatch.setenv("PERF_MCP_TP_MESH", raw)
    assert pm._tp_mesh_shapes(_Ttnn(4)) == [(1, 4), (2, 2)]


def test_an_override_is_honoured_even_when_it_disagrees_with_the_device_count(pm, monkeypatch):
    """An explicit override is an instruction, not a hint: a human debugging a fabric problem needs
    to pin a shape the auto-derivation would never choose."""
    monkeypatch.setenv("PERF_MCP_TP_MESH", "2x2")
    assert pm._tp_mesh_shapes(_Ttnn(32)) == [(2, 2)]


def test_a_broken_cluster_query_never_raises(pm):
    class _Broken:
        def get_num_devices(self):
            raise RuntimeError("cluster unreachable")

    assert pm._tp_mesh_shapes(_Broken()) == [(1, 1)]


def test_a_cluster_query_returning_junk_never_raises(pm):
    for junk in (None, "four", 3.7, [4]):

        class _Junk:
            def get_num_devices(self, v=junk):
                return v

        shapes = pm._tp_mesh_shapes(_Junk())
        assert shapes and all(isinstance(r, int) and isinstance(c, int) for r, c in shapes), junk


def test_open_falls_through_every_candidate_in_order(pm):
    tried = []

    class _RefusesAll(_Ttnn):
        def open_mesh_device(self, shape):
            tried.append(shape)
            raise RuntimeError("no")

    raised = None
    try:
        pm._open_tp_mesh(_RefusesAll(32))
    except RuntimeError as exc:
        raised = exc
    assert raised is not None, "a board that refuses every shape must not return a mesh"
    assert tried == [(1, 32), (4, 8), (2, 16)]


def test_open_stops_at_the_first_shape_that_works(pm):
    tried = []

    class _OnlySquare(_Ttnn):
        def open_mesh_device(self, shape):
            tried.append(shape)
            if shape != (4, 8):
                raise RuntimeError("cannot form that ring")
            return "mesh(4, 8)"

    assert pm._open_tp_mesh(_OnlySquare(32)) == "mesh(4, 8)"
    assert tried == [(1, 32), (4, 8)]


def test_open_surfaces_the_last_real_error_not_a_generic_one(pm):
    """The caller logs this string; a swallowed cause is how a wedged board looked like a bad shape."""

    class _Wedged(_Ttnn):
        def open_mesh_device(self, shape):
            raise RuntimeError("PCIe link down on chip 3")

    raised = None
    try:
        pm._open_tp_mesh(_Wedged(4))
    except RuntimeError as exc:
        raised = exc
    assert raised is not None and "PCIe link down on chip 3" in str(raised), raised


def test_open_on_a_single_device_board_still_returns_a_mesh(pm):
    assert pm._open_tp_mesh(_Ttnn(1)) == "mesh(1, 1)"


def test_no_hardcoded_mesh_literal_remains_in_the_tp_levers(pm):
    """The constant this replaced must not creep back into either entry point."""
    import ast

    src = (_ROOT / "cc_optimize" / "perf_mcp.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in ("tp_pick_degree", "verify_tp_fracture"):
            continue
        calls = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(getattr(n.func, "attr", None), "__str__", str)() == "MeshShape"
        ]
        assert not calls, "%s builds a MeshShape directly instead of calling _open_tp_mesh" % node.name
        assert "_open_tp_mesh" in ast.dump(node), "%s does not open the mesh via the shared helper" % node.name


# --- the sources that OUTRANK arithmetic -----------------------------------------------------------
# Chip count does not imply topology. QB2 has 4 chips but its fabric is 2x2 only, so the arithmetic
# "prefer a 1x4 ring" is a shape that cannot be formed there. These pin the priority order.


class _Box:
    def __init__(self, name, chips, mesh_shapes, default_mesh=None, board_types=()):
        self.name = name
        self.chips = chips
        self.mesh_shapes = mesh_shapes
        self.default_mesh = default_mesh
        self.board_types = board_types
        self.arch = "Blackhole"


_QB2 = _Box("QB2", 4, [(1, 1), (2, 2)], default_mesh=(2, 2), board_types=("p150c",))
_GALAXY = _Box("GalaxyBH", 32, [(1, 1), (1, 8), (4, 8), (2, 16), (1, 32)], default_mesh=(4, 8))


def test_the_run_planned_mesh_wins_over_everything_but_an_explicit_pin(pm, monkeypatch):
    """optimize._derive_topology_env exports the mesh plan_parallelism chose and the model opens.
    The TP lever must measure THAT topology, or it reports timings for a mesh the run never used."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2)
    monkeypatch.setenv("TT_PERF_MESH_ROWS", "1")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "4")
    assert pm._tp_mesh_shapes(_Ttnn(4))[0] == (1, 4)


def test_an_explicit_pin_outranks_the_planned_mesh(pm, monkeypatch):
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2)
    monkeypatch.setenv("TT_PERF_MESH_ROWS", "1")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "4")
    monkeypatch.setenv("PERF_MCP_TP_MESH", "2x2")
    assert pm._tp_mesh_shapes(_Ttnn(4)) == [(2, 2)]


def test_the_box_registry_is_used_when_no_mesh_was_planned(pm, monkeypatch):
    """THE CORRECTION: on QB2 the arithmetic path would lead with 1x4, which the fabric cannot form.
    The registry says [(1,1), (2,2)] with default (2,2), so (2,2) leads and 1x4 is never offered."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2)
    shapes = pm._tp_mesh_shapes(_Ttnn(4))
    assert shapes[0] == (2, 2), shapes
    assert (1, 4) not in shapes, shapes


def test_the_box_default_mesh_leads_on_galaxy(pm, monkeypatch):
    """Galaxy's canonical large-scale shape is (4,8), not the 1x32 ring arithmetic would pick."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _GALAXY)
    shapes = pm._tp_mesh_shapes(_Ttnn(32))
    assert shapes[0] == (4, 8), shapes
    assert all(r * c == 32 for r, c in shapes), shapes


def test_box_shapes_that_do_not_use_every_chip_are_dropped(pm, monkeypatch):
    """(1,1) is listed for single-chip bring-up; a TP sweep on it would measure no parallelism."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2)
    assert (1, 1) not in pm._tp_mesh_shapes(_Ttnn(4))


def test_a_half_set_planned_mesh_does_not_silently_become_1x1(pm, monkeypatch):
    """resolve_mesh_shape owns this: a half-set pair once opened 1x1 while the plan said TP=4."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: None)
    monkeypatch.setenv("TT_PERF_MESH_COLS", "4")
    monkeypatch.delenv("TT_PERF_MESH_ROWS", raising=False)
    assert pm._tp_mesh_shapes(_Ttnn(4))[0] == (1, 4)


def test_an_unparseable_planned_mesh_falls_through_to_the_box(pm, monkeypatch):
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2)
    monkeypatch.setenv("TT_PERF_MESH_ROWS", "junk")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "junk")
    assert pm._tp_mesh_shapes(_Ttnn(4))[0] == (2, 2)


def test_an_unrecognised_board_still_gets_candidates(pm, monkeypatch):
    """A board absent from the registry must not disable the lever -- arithmetic is the fallback."""
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: None)
    assert pm._tp_mesh_shapes(_Ttnn(8)) == [(1, 8), (2, 4)]


def test_box_lookup_never_raises_when_the_registry_is_absent(pm, monkeypatch):
    """Guarded import: perf_automation must run without scripts/tt_hw_planner importable."""
    import builtins

    real = builtins.__import__

    def _no_planner(name, *a, **k):
        if "tt_hw_planner" in name:
            raise ImportError("planner absent")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_planner)
    assert pm._tp_box(4) is None


def test_a_box_entry_that_describes_one_card_is_not_trusted_for_a_multi_card_system(pm, monkeypatch):
    """THE REAL CASE: this machine reports board_type p300 with 4 devices, and the registry's P300 is
    a 2-chip dual-ASIC CARD -- two are installed. Trusting its mesh_shapes would sweep 2 of 4 chips
    and call that the board's TP."""
    _p300_card = _Box("P300", 2, [(1, 1), (1, 2), (2, 1)], board_types=("p300",))
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _p300_card if num == 2 else None)
    shapes = pm._tp_mesh_shapes(_Ttnn(4))
    assert all(r * c == 4 for r, c in shapes), shapes
    assert shapes == [(1, 4), (2, 2)], shapes


def test_the_box_is_used_when_the_system_really_is_that_box(pm, monkeypatch):
    monkeypatch.setattr(pm, "_tp_box", lambda num=0: _QB2 if num == 4 else None)
    assert pm._tp_mesh_shapes(_Ttnn(4))[0] == (2, 2)
