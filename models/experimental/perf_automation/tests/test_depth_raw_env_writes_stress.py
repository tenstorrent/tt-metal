# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue 5: the depth variable and PERF_MCP_FORCE_ALL_LAYERS must move together.

The invariant, stated once and then attacked from every direction:

    a POSITIVE depth  =>  var == str(depth)  AND  FORCE_ALL absent
    any OTHER depth   =>  var absent         AND  FORCE_ALL == "1"

There is no third state. A cap written while FORCE_ALL is armed is the worst cell, because the
depth guard strips the cap and the run profiles every layer while reporting the capped number --
silent, and it looks like a successful slice.

  s1  cross-product: {positive, zero, negative, None, junk} x {FORCE_ALL armed, absent} x keys
  s2  the invariant over 2000 random depth values
  s3  sequences: any ordering of set/unset must land in a consistent state (no accumulation)
  s4  companion variables in the knob are never disturbed
  s5  no call site anywhere in run.py writes a depth variable raw
  s6  aliasing: _knob_at copies, set_depth mutates in place (documented, relied upon)
"""

import importlib.util
import random
import re
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))

from agent.layer_depth import ENV, FORCE_ALL, set_depth  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("cc_run_raw_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _check(env, var, depth):
    """The one invariant."""
    try:
        d = int(depth)
    except (TypeError, ValueError):
        d = 0
    if d > 0:
        assert env.get(var) == str(d), f"cap not written: depth={depth!r} var={var} env={env}"
        assert FORCE_ALL not in env, f"FORCE_ALL still armed alongside a positive cap: {env}"
    else:
        assert var not in env, f"non-positive depth {depth!r} left a value behind: {env}"
        assert env.get(FORCE_ALL) == "1", f"all-layers did not arm FORCE_ALL: {env}"


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("depth", [1, 2, 16, 999, 0, -1, -999, None, "", "abc", 1.9, [], {}])
@pytest.mark.parametrize("pre_armed", [True, False])
@pytest.mark.parametrize("var", [ENV, "MY_MODEL_LAYERS", "n_layers", "X"])
def test_s1_invariant_in_every_cell(depth, pre_armed, var):
    env = {var: "7"}
    if pre_armed:
        env[FORCE_ALL] = "1"
    set_depth(env, depth, key=var)
    _check(env, var, depth)


@pytest.mark.parametrize("depth", [1, 4, 0, None])
@pytest.mark.parametrize("pre_armed", [True, False])
def test_s1_knob_at_invariant(depth, pre_armed):
    env = {"CAP": "7"}
    if pre_armed:
        env[FORCE_ALL] = "1"
    got = _mod()._knob_at(env, depth)
    _check(got, "CAP", depth)


# --------------------------------------------------------------------------- s2
def test_s2_two_thousand_random_depths():
    rng = random.Random(20260730)
    for _ in range(2000):
        depth = rng.choice([rng.randint(-50, 5000), rng.choice([None, "", "x", 0, -1]), str(rng.randint(0, 99))])
        env = {"CAP": "7"}
        if rng.random() < 0.5:
            env[FORCE_ALL] = "1"
        set_depth(env, depth, key="CAP")
        _check(env, "CAP", depth)


# --------------------------------------------------------------------------- s3
def test_s3_alternating_sequences_never_accumulate():
    env = {}
    for i in range(200):
        depth = (i % 5) if i % 2 else None  # 0/None/1/2/3/4 interleaved
        set_depth(env, depth, key="CAP")
        _check(env, "CAP", depth)
    # nothing but the two managed keys should ever have appeared
    assert set(env) <= {"CAP", FORCE_ALL}, f"set_depth accumulated junk: {env}"


def test_s3_all_layers_then_cap_disarms():
    env = {}
    set_depth(env, None, key="CAP")
    assert env[FORCE_ALL] == "1"
    set_depth(env, 4, key="CAP")
    assert FORCE_ALL not in env and env["CAP"] == "4"


def test_s3_cap_then_all_layers_removes():
    env = {}
    set_depth(env, 4, key="CAP")
    set_depth(env, None, key="CAP")
    assert "CAP" not in env and env[FORCE_ALL] == "1"


def test_s3_idempotent():
    a, b = {}, {}
    set_depth(a, 4, key="CAP")
    set_depth(b, 4, key="CAP")
    set_depth(b, 4, key="CAP")
    assert a == b


# --------------------------------------------------------------------------- s4
def test_s4_companion_variables_survive():
    env = {"CAP": "1", "ALLOW_PARTIAL": "yes", "UNRELATED": "keep"}
    set_depth(env, 8, key="CAP")
    assert env["ALLOW_PARTIAL"] == "yes" and env["UNRELATED"] == "keep"
    set_depth(env, None, key="CAP")
    assert env["ALLOW_PARTIAL"] == "yes" and env["UNRELATED"] == "keep"


def test_s4_default_key_does_not_touch_a_model_specific_var():
    env = {"MY_MODEL_LAYERS": "7"}
    set_depth(env, None)  # default key = ENV
    assert env["MY_MODEL_LAYERS"] == "7", "set_depth(default) clobbered an unrelated variable"


def test_s4_knob_at_leaves_non_numeric_entries_alone():
    got = _mod()._knob_at({"CAP": "1", "MODE": "fast"}, 4)
    assert got["MODE"] == "fast" and got["CAP"] == "4"


# --------------------------------------------------------------------------- s5
def test_s5_no_raw_depth_write_anywhere_in_run_py():
    """Whole-file sweep, not just the three known functions -- a fourth site must fail here."""
    src = (_CC / "run.py").read_text()
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    hits = re.findall(r"^\s*\w*env\[_?numkey\]\s*=\s*str\(", code, re.M)
    assert not hits, f"raw depth writes remain in run.py: {hits}"


def test_s5_set_depth_is_the_only_writer_of_force_all():
    """FORCE_ALL must be managed in one place; a call site setting it by hand can desync it."""
    lay = (_PA / "agent" / "layer_depth.py").read_text()
    assert lay.count("env[FORCE_ALL]") == 1, "FORCE_ALL is written in more than one branch"


# --------------------------------------------------------------------------- s6
def test_s6_set_depth_mutates_in_place_and_returns_the_same_object():
    env = {}
    out = set_depth(env, 4, key="CAP")
    assert out is env, "callers rely on in-place mutation"


def test_s6_knob_at_returns_a_copy():
    src = {"CAP": "1", FORCE_ALL: "1"}
    got = _mod()._knob_at(src, 4)
    assert got is not src
    assert src == {"CAP": "1", FORCE_ALL: "1"}, "_knob_at mutated its input"
