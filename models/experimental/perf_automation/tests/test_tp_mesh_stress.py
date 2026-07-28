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
    monkeypatch.delenv("PERF_MCP_TP_MESH", raising=False)
    spec = importlib.util.spec_from_file_location("pm_stress_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pm_stress_ut"] = mod
    spec.loader.exec_module(mod)
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
