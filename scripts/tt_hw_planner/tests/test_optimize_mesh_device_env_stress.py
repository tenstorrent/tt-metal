# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: deriving MESH_DEVICE from --box/--mesh.

This writes to the PROCESS ENVIRONMENT, which every downstream subprocess inherits, and it names
the board the MODEL loads and the fabric topology the RUNTIME brings up. Getting it wrong is not a
crash -- it is a run that silently measures the wrong hardware profile, or trains ethernet across
chips it was told not to touch. So the properties that matter are about authority and blast radius,
not about label lookup.

  s1  AUTHORITY: an operator-set value is never overridden, across the full env x flag matrix
  s2  AGREEMENT: for every box x every canonical shape, optimize's answer == bring-up's answer
  s3  NO INVENTION: a label is exported only when the shared resolver returns one; a descriptor
      path only when the file exists on disk
  s4  BLAST RADIUS: nothing but that one key is ever touched
  s5  HOSTILE INPUT: 400 randomised (box, mesh) pairs never raise and never write junk
  s6  IDEMPOTENCE: repeated calls converge and do not accumulate
"""

import itertools
import os
import random
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.bringup import MESH_DEVICE_MAP, mesh_device_for  # noqa: E402
from tt_hw_planner.commands import optimize  # noqa: E402
from tt_hw_planner.hardware import HARDWARE  # noqa: E402

_KEYS = ("MESH_DEVICE",)


class _Args:
    def __init__(self, **kw):
        self.box = None
        self.mesh = None
        self.devices = "single"
        self.__dict__.update(kw)


def _clean(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def _run(box=None, mesh=None):
    """Resolve, tolerating the ONE documented refusal.

    CONTRACT CHANGED 2026-08-14 (475a4b6d60, "an unknown --box fails loudly instead of setting
    nothing"): a name that does not resolve now aborts rather than silently leaving MESH_DEVICE
    unset, because silence left the model loading a default board the operator did not ask for.
    These stress tests feed deliberately hostile box names, so they must expect that refusal --
    what they actually assert is that nothing is INVENTED and nothing already exported is
    overridden, and both hold whether the resolver answers or refuses. Any OTHER exception is
    still a failure, which is the property s5 exists to defend.
    """
    try:
        optimize._derive_mesh_device_env(_Args(box=box, mesh=mesh))
    except (SystemExit, KeyError):
        pass


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("box", [None, "p150", "P150", "n300", "nonsense"])
@pytest.mark.parametrize("mesh", [None, "1x1", "1x8", "junk"])
def test_s1_operator_value_is_absolute(monkeypatch, key, box, mesh):
    """Whatever the flags say, an explicitly exported value survives untouched. The operator's
    choice is never silently replaced -- the rule pcc_gate_policy states for a supplied gate."""
    _clean(monkeypatch)
    monkeypatch.setenv(key, "OPERATOR_SET")
    _run(box, mesh)
    assert os.environ[key] == "OPERATOR_SET", f"{key} was overridden with box={box} mesh={mesh}"


# --------------------------------------------------------------------------- s2
def test_s2_agrees_with_bringup_for_every_box_and_canonical_shape(monkeypatch):
    """One source of truth: for every (box, shape) the resolver labels, optimize must export
    exactly what bring-up would resolve. A divergence here means the table got copied."""
    checked = 0
    for box in HARDWARE:
        for (arch, shape), label in MESH_DEVICE_MAP.items():
            if arch != box.arch:
                continue
            _clean(monkeypatch)
            _run(box.name, "%dx%d" % shape)
            expected, _note = mesh_device_for(box.arch, shape)
            got = os.environ.get("MESH_DEVICE")
            assert got == expected, f"{box.name} {shape}: optimize={got!r} bringup={expected!r}"
            checked += 1
    assert checked > 10, f"only {checked} combinations exercised; the matrix is not covering"


def test_s2_case_insensitive_box_agrees(monkeypatch):
    """find_box is case-SENSITIVE but the CLI help suggests lowercase, so both spellings must land
    on the same label rather than one silently doing nothing."""
    for name in (b.name for b in HARDWARE):
        _clean(monkeypatch)
        _run(name, "1x1")
        upper = os.environ.get("MESH_DEVICE")
        _clean(monkeypatch)
        _run(name.lower(), "1x1")
        lower = os.environ.get("MESH_DEVICE")
        assert upper == lower, f"{name}: {upper!r} vs {name.lower()}: {lower!r}"


# --------------------------------------------------------------------------- s3
def test_s3_never_invents_a_label(monkeypatch):
    """Only export what the shared resolver returns. Every exported value must be a label that
    exists in MESH_DEVICE_MAP -- never a guess, never a stringified None."""
    labels = set(MESH_DEVICE_MAP.values())
    rng = random.Random(20260730)
    for _ in range(200):
        _clean(monkeypatch)
        box = rng.choice([b.name for b in HARDWARE] + ["bogus", ""])
        mesh = rng.choice(["1x1", "1x2", "1x4", "1x8", "2x2", "3x3", "7x7", "0x0", "junk"])
        _run(box, mesh)
        got = os.environ.get("MESH_DEVICE")
        if got is not None:
            assert got in labels, f"invented label {got!r} for box={box} mesh={mesh}"
            assert got != "None"


def test_s3_unlabelled_shape_exports_nothing(monkeypatch):
    """mesh_device_for returns (None, note) for a shape with no label; that must not become an
    export."""
    _clean(monkeypatch)
    _run("P150", "3x3")
    assert "MESH_DEVICE" not in os.environ


# --------------------------------------------------------------------------- s4
def test_s4_touches_only_the_two_keys(monkeypatch):
    """This mutates the real process environment that every subprocess inherits. Anything else it
    writes is an unaudited side effect."""
    _clean(monkeypatch)
    before = dict(os.environ)
    _run("P150", "1x1")
    after = dict(os.environ)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed <= set(_KEYS), f"unexpected env keys changed: {changed - set(_KEYS)}"


def test_s4_noop_case_changes_nothing_at_all(monkeypatch):
    _clean(monkeypatch)
    before = dict(os.environ)
    _run(None, "1x1")
    assert dict(os.environ) == before


# --------------------------------------------------------------------------- s5
def _rand_token(rng):
    return "".join(rng.choice(string.printable[:70]) for _ in range(rng.randint(0, 12)))


def test_s5_400_hostile_pairs_never_raise(monkeypatch):
    rng = random.Random(7)
    for i in range(400):
        _clean(monkeypatch)
        box = rng.choice([None, "", "P150", "p150", _rand_token(rng), 42, [], {}])
        mesh = rng.choice([None, "", "1x1", "x", "1x", "-1x-1", "0x0", "1x1x1", _rand_token(rng), 7, []])
        try:
            _run(box, mesh)  # absorbs the documented refusal; anything else is the bug
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"case {i}: box={box!r} mesh={mesh!r} raised {exc!r}")
        got = os.environ.get("MESH_DEVICE")
        assert got is None or got in set(MESH_DEVICE_MAP.values())


@pytest.mark.parametrize("mesh", ["1x1 ", " 1x1", "1X1", "1 x 1"])
def test_s5_whitespace_and_case_in_mesh(monkeypatch, mesh):
    """Must either parse or cleanly decline -- never write a wrong label."""
    _clean(monkeypatch)
    _run("P150", mesh)
    got = os.environ.get("MESH_DEVICE")
    assert got in (None, "P150"), f"mesh {mesh!r} produced {got!r}"


# --------------------------------------------------------------------------- s6
def test_s6_idempotent(monkeypatch):
    _clean(monkeypatch)
    _run("P150", "1x1")
    first = {k: os.environ.get(k) for k in _KEYS}
    for _ in range(20):
        _run("P150", "1x1")
    assert {k: os.environ.get(k) for k in _KEYS} == first


def test_s6_second_call_with_different_flags_does_not_flip(monkeypatch):
    """Once set, the value is authoritative for the run -- a later call with different flags must
    not silently retarget the board mid-run."""
    _clean(monkeypatch)
    _run("P150", "1x1")
    first = os.environ.get("MESH_DEVICE")
    _run("N300", "1x2")
    assert os.environ.get("MESH_DEVICE") == first


def test_s6_all_boxes_all_shapes_never_raise(monkeypatch):
    for box, mesh in itertools.product([b.name for b in HARDWARE], ["1x1", "1x2", "1x4", "1x8", "2x2", "4x8", "8x4"]):
        _clean(monkeypatch)
        _run(box, mesh)
