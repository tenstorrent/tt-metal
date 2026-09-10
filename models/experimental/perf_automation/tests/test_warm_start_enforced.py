# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The matmul-sweep warm start must be DELIVERED and VERIFIED, not merely suggested.

`--matmul-sweep` runs a device pre-pass that PCC-tests fidelity x dtype per matmul shape and writes
`matmul_sweep.json`. The optimize loop was told to use it only by prose in `run.py::_PROMPT`:

    "a matmul_sweep.json may exist in the model directory (Glob for it once) ... look up
     next_target's shape (m,k,n) in that table and APPLY its recommended fidelity/dtype FIRST"

Nothing made that happen and nothing recorded whether it did, so a table the agent never opened was
indistinguishable from a table with no matching shape -- you could pay for the pre-pass and get
nothing, silently.

Three changes, matching what the deterministic layer can actually know:

  1. DELIVER  termination_check resolves the shape itself and returns next_target["warm_start"],
              so the config is handed over as data instead of being a file to remember to read.
  2. VERIFY   the profile carries per-op MATH FIDELITY (tracy_tool.py:487), so whether the
              recommended fidelity actually ran is checkable against ground truth rather than
              self-reported. Per-op dtype is NOT captured (roofline.py:232 reads weight_dtype,
              which is never populated), so dtype is delivered but cannot be verified this way --
              stated here so the asymmetry is deliberate and visible.
  3. SCOPE    all of this applies ONLY when the file exists and holds the shape. No file, or no
              matching shape -> next_target is byte-for-byte what it is today.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _mod():
    spec = importlib.util.spec_from_file_location("pmcp_ws", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# THE SCHEMA matmul_sweep.py ACTUALLY WRITES -- summarize()'s dict plus summary["table"] = table
# (matmul_sweep.py:129-149 and :349). Shape is NESTED under row["shape"] in `seeds`, and FLAT on a
# `table` row whose winner sits under row["best"].
#
# This fixture used to invent an `entries` key with flat m/k/n. No producer has ever written that, so
# these tests asserted against a schema that does not exist: when _warm_start_for was corrected to
# read seeds/table, all 8 of them began failing and stayed that way. Building a fixture from a mental
# model instead of from the producer is exactly how the original defect shipped -- a table the loop
# could not read looked identical to a table with no matching shape.
SWEEP = {
    "shapes": 2,
    "seeded": 2,
    "improved": 2,
    "seeds": [
        {
            "shape": {"m": 32, "k": 4096, "n": 4096},
            "fidelity": "LoFi",
            "dtype": "bfloat8_b",
            "best_ms": 0.41,
            "baseline_ms": 0.87,
            "speedup": 2.122,
            "pcc": 0.998,
        },
        {
            "shape": {"m": 1, "k": 4096, "n": 14336},
            "fidelity": "HiFi2",
            "dtype": "bfloat16",
            "best_ms": 0.62,
            "baseline_ms": 0.70,
            "speedup": 1.129,
            "pcc": 0.999,
        },
    ],
    "table": [
        {"m": 32, "k": 4096, "n": 4096, "best": {"fidelity": "LoFi", "dtype": "bfloat8_b", "ms": 0.41, "pcc": 0.998}},
        {"m": 1, "k": 4096, "n": 14336, "best": {"fidelity": "HiFi2", "dtype": "bfloat16", "ms": 0.62, "pcc": 0.999}},
    ],
}


def _sweep_file(tmp_path, payload=None):
    p = tmp_path / "matmul_sweep.json"
    p.write_text(json.dumps(SWEEP if payload is None else payload))
    return tmp_path


def _lookup():
    m = _mod()
    fn = getattr(m, "_warm_start_for", None)
    if fn is None:
        pytest.fail(
            "perf_mcp has no _warm_start_for: the warm-start table is still consumed only by prose "
            "in run.py::_PROMPT, so nothing delivers it and nothing records whether it was used."
        )
    return m, fn


# --------------------------------------------------------------------------- 1. DELIVER
def test_shape_present_is_delivered(tmp_path):
    _m, fn = _lookup()
    got = fn(_sweep_file(tmp_path), "MatmulDeviceOperation 32 x 4096 x 4096")
    assert got == {"fidelity": "LoFi", "dtype": "bfloat8_b"}


def test_second_shape_resolves_independently(tmp_path):
    _m, fn = _lookup()
    got = fn(_sweep_file(tmp_path), "MatmulDeviceOperation 1 x 4096 x 14336")
    assert got == {"fidelity": "HiFi2", "dtype": "bfloat16"}


def test_shape_absent_returns_nothing(tmp_path):
    _m, fn = _lookup()
    assert fn(_sweep_file(tmp_path), "MatmulDeviceOperation 8 x 8 x 8") is None


def test_no_file_returns_nothing(tmp_path):
    _m, fn = _lookup()
    assert fn(tmp_path, "MatmulDeviceOperation 32 x 4096 x 4096") is None


def test_empty_sweep_returns_nothing(tmp_path):
    """The 0-shape file issue #14 produced must behave exactly like no file."""
    _m, fn = _lookup()
    root = _sweep_file(tmp_path, {"ok": True, "shapes": 0, "seeded": 0, "note": "no matmul ops enumerated"})
    assert fn(root, "MatmulDeviceOperation 32 x 4096 x 4096") is None


def test_non_matmul_op_returns_nothing(tmp_path):
    _m, fn = _lookup()
    assert fn(_sweep_file(tmp_path), "LayerNorm") is None


@pytest.mark.parametrize("junk", ["{not json", "", "[]", "null", '{"seeds": "nope"}', '{"table": "nope"}'])
def test_corrupt_table_degrades_to_nothing(tmp_path, junk):
    _m, fn = _lookup()
    (tmp_path / "matmul_sweep.json").write_text(junk)
    assert fn(tmp_path, "MatmulDeviceOperation 32 x 4096 x 4096") is None


# --------------------------------------------------------------------------- 2. VERIFY
def _verify():
    m = _mod()
    fn = getattr(m, "_warm_start_applied", None)
    if fn is None:
        pytest.fail(
            "perf_mcp has no _warm_start_applied: without a ground-truth check the only record of a "
            "warm start is the agent's own say-so, which is exactly what record_kernel_attempt "
            "already cannot verify."
        )
    return fn


def _prof(op, fidelity):
    return {"top_ops": [{"op_code": op, "fidelity": fidelity, "device_ms": 1.0}]}


def test_applied_fidelity_is_confirmed_from_the_profile():
    fn = _verify()
    op = "MatmulDeviceOperation 32 x 4096 x 4096"
    assert fn(_prof(op, "LoFi"), op, {"fidelity": "LoFi", "dtype": "bfloat8_b"}) is True


def test_wrong_fidelity_is_reported_as_not_applied():
    fn = _verify()
    op = "MatmulDeviceOperation 32 x 4096 x 4096"
    assert fn(_prof(op, "HiFi4"), op, {"fidelity": "LoFi", "dtype": "bfloat8_b"}) is False


def test_missing_op_in_profile_is_unknown_not_false():
    """Absent evidence is not evidence of absence -- a missing op must not read as 'ignored'."""
    fn = _verify()
    assert fn(_prof("SomethingElse", "LoFi"), "MatmulDeviceOperation 32 x 4096 x 4096", {"fidelity": "LoFi"}) is None


def test_no_fidelity_recommendation_is_unknown():
    """dtype-only entries cannot be verified: the profile carries no per-op dtype."""
    fn = _verify()
    op = "MatmulDeviceOperation 32 x 4096 x 4096"
    assert fn(_prof(op, "LoFi"), op, {"dtype": "bfloat8_b"}) is None


@pytest.mark.parametrize("bad", [None, {}, [], "x", 42])
def test_verify_hostile_inputs_never_raise(bad):
    fn = _verify()
    assert fn(bad, "op", {"fidelity": "LoFi"}) is None
    assert fn(_prof("op", "LoFi"), "op", bad) is None


# --------------------------------------------------------------------------- 3. SCOPE
def test_termination_check_wires_warm_start_into_next_target():
    """Wiring: the lookup must be reachable from termination_check, not just defined."""
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("def termination_check")
    # THE WHOLE FUNCTION, not a fixed byte window. This sliced `src[i : i + 8000]`, so any edit near
    # the top of termination_check pushed the wiring it checks out of view and failed a function that
    # had not changed -- a test measuring its own window size rather than the claim it states.
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_warm_start_for(" in body or "warm_start" in body, (
        "termination_check does not populate next_target['warm_start']; the table is still the " "agent's job to find"
    )
