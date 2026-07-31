# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS: warm-start delivery and verification must be exact, scoped, and never fabricate.

Two properties carry the whole feature:

  DELIVERY is EXACT      -- a recommendation is returned only for the shape it was measured on. A
                            near-miss must return nothing: applying another shape's PCC-gated dtype
                            is worse than a cold start, because it arrives wearing a verification
                            it does not have.
  VERIFICATION is HONEST -- True / False / None, where None means "unknown". Collapsing unknown to
                            False would report an agent as having ignored a table it obeyed.

  s1  600 random tables x op codes -- delivery matches an independent oracle in every cell
  s2  shape matching is exact: transposed, off-by-one, super/substring, whitespace, huge dims
  s3  scoping: absent file, empty table (the issue #14 shape), corrupt JSON, non-matmul ops
  s4  verification tri-state across the profile x recommendation cross-product
  s5  purity, determinism, and no writes to the model tree
  s6  the delivered dict is a fresh object a caller cannot corrupt for the next call
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _mod():
    spec = importlib.util.spec_from_file_location("pmcp_ws_stress", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _mod()
_LOOKUP = _M._warm_start_for
_VERIFY = _M._warm_start_applied

_FID = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
_DT = ["bfloat16", "bfloat8_b", "bfloat4_b"]


def _write(tmp_path, entries):
    """Write the schema matmul_sweep.py ACTUALLY produces, from convenient flat rows.

    Tests describe a shape as flat m/k/n because that reads well in a table; the producer nests it
    under row["shape"] inside `seeds` (matmul_sweep.py:140). Translating HERE keeps every test body
    unchanged while what lands on disk is what the sweep really writes.

    This helper used to emit an invented `entries` key. No producer has ever written that, so the
    whole file asserted against a schema that does not exist -- and when _warm_start_for was
    corrected to read seeds/table, these tests failed rather than the code. Only keys the caller
    supplied are carried through, so "partial row" cases stay partial.
    """
    seeds = []
    for r in entries:
        seed = {"shape": {"m": r["m"], "k": r["k"], "n": r["n"]}}
        for key in ("fidelity", "dtype", "pcc", "best_ms", "baseline_ms", "speedup"):
            if key in r:
                seed[key] = r[key]
        seeds.append(seed)
    payload = {"shapes": len(entries), "seeded": len(seeds), "seeds": seeds}
    (tmp_path / "matmul_sweep.json").write_text(json.dumps(payload))
    return tmp_path


def _op(m, k, n):
    return f"MatmulDeviceOperation {m} x {k} x {n}"


# --------------------------------------------------------------------------- s1
def test_s1_random_tables_match_the_oracle(tmp_path):
    rng = random.Random(20260730)
    for i in range(600):
        shapes = {(rng.choice([1, 32, 128]), rng.choice([512, 4096]), rng.choice([4096, 14336])) for _ in range(4)}
        entries = [
            {"m": m, "k": k, "n": n, "fidelity": rng.choice(_FID), "dtype": rng.choice(_DT)} for (m, k, n) in shapes
        ]
        d = tmp_path / f"c{i}"
        d.mkdir()
        _write(d, entries)
        probe = (rng.choice([1, 32, 128, 999]), rng.choice([512, 4096]), rng.choice([4096, 14336]))
        got = _LOOKUP(d, _op(*probe))
        want = next((e for e in entries if (e["m"], e["k"], e["n"]) == probe), None)
        if want is None:
            assert got is None, f"case {i}: invented a recommendation for unmeasured shape {probe}: {got}"
        else:
            assert got == {"fidelity": want["fidelity"], "dtype": want["dtype"]}, f"case {i}: {got} != {want}"


def test_s1_first_matching_entry_wins_deterministically(tmp_path):
    entries = [
        {"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"},
        {"m": 32, "k": 4096, "n": 4096, "fidelity": "HiFi4", "dtype": "bfloat16"},
    ]
    _write(tmp_path, entries)
    a = _LOOKUP(tmp_path, _op(32, 4096, 4096))
    b = _LOOKUP(tmp_path, _op(32, 4096, 4096))
    assert a == b == {"fidelity": "LoFi", "dtype": "bfloat8_b"}


# --------------------------------------------------------------------------- s2
def test_s2_transposed_shape_is_not_a_match(tmp_path):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 14336, "fidelity": "LoFi", "dtype": "bfloat8_b"}])
    assert _LOOKUP(tmp_path, _op(32, 14336, 4096)) is None, "a transposed shape is a different matmul"


@pytest.mark.parametrize("probe", [(31, 4096, 4096), (32, 4095, 4096), (32, 4096, 4097), (320, 4096, 4096)])
def test_s2_near_misses_are_not_matches(tmp_path, probe):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}])
    assert _LOOKUP(tmp_path, _op(*probe)) is None, f"{probe} matched a table entry for (32,4096,4096)"


def test_s2_whitespace_variants_still_match(tmp_path):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}])
    for op in [
        "MatmulDeviceOperation 32 x 4096 x 4096",
        "MatmulDeviceOperation 32x4096x4096",
        "MatmulDeviceOperation  32  x  4096  x  4096",
    ]:
        assert _LOOKUP(tmp_path, op) is not None, f"failed to match {op!r}"


def test_s2_huge_dims_do_not_overflow(tmp_path):
    big = 10**9
    _write(tmp_path, [{"m": big, "k": big, "n": big, "fidelity": "LoFi", "dtype": "bfloat16"}])
    assert _LOOKUP(tmp_path, _op(big, big, big)) == {"fidelity": "LoFi", "dtype": "bfloat16"}


# --------------------------------------------------------------------------- s3
def test_s3_absent_file(tmp_path):
    assert _LOOKUP(tmp_path, _op(32, 4096, 4096)) is None


def test_s3_the_issue_14_empty_table(tmp_path):
    """The exact artifact the --case bug produced must behave as if absent."""
    (tmp_path / "matmul_sweep.json").write_text(
        json.dumps({"ok": True, "shapes": 0, "seeded": 0, "note": "no matmul ops enumerated"})
    )
    assert _LOOKUP(tmp_path, _op(32, 4096, 4096)) is None


@pytest.mark.parametrize(
    "junk",
    [
        "{not json",
        "",
        "[]",
        "null",
        "42",
        # Malformed on the REAL key. Probing a key the reader ignores would pass against any
        # garbage and prove nothing -- which is what these cases did while they named `entries`.
        '{"seeds": null}',
        '{"seeds": [1,2,3]}',
        '{"seeds": [{}]}',
        '{"seeds": [{"shape": {"m": "x", "k": 1, "n": 1}}]}',
        '{"seeds": [{"shape": null, "fidelity": "LoFi"}]}',
        '{"table": [{"m": 1, "k": 1, "n": 1, "best": null}]}',
    ],
)
def test_s3_corrupt_tables_degrade_silently(tmp_path, junk):
    (tmp_path / "matmul_sweep.json").write_text(junk)
    assert _LOOKUP(tmp_path, _op(32, 4096, 4096)) is None


@pytest.mark.parametrize("op", ["LayerNorm", "Softmax", "", None, "AllGather 32 x 4096 x 4096"])
def test_s3_non_matmul_ops_get_nothing(tmp_path, op):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}])
    assert _LOOKUP(tmp_path, op) is None


@pytest.mark.parametrize("root", [None, Path("/nonexistent_xyz"), Path("/dev/null")])
def test_s3_hostile_roots_never_raise(root):
    assert _LOOKUP(root, _op(32, 4096, 4096)) is None


def test_s3_entry_without_config_yields_nothing(tmp_path):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096}])
    assert _LOOKUP(tmp_path, _op(32, 4096, 4096)) is None, "an entry recommending nothing is not a recommendation"


def test_s3_partial_entry_returns_only_what_it_has(tmp_path):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi"}])
    assert _LOOKUP(tmp_path, _op(32, 4096, 4096)) == {"fidelity": "LoFi"}


# --------------------------------------------------------------------------- s4
def _prof(op, fid):
    return {"top_ops": [{"op_code": op, "fidelity": fid, "device_ms": 1.0}]}


@pytest.mark.parametrize(
    "ran,want,expect",
    [
        ("LoFi", "LoFi", True),
        ("lofi", "LoFi", True),
        ("  LoFi  ", "LoFi", True),
        ("HiFi4", "LoFi", False),
        ("HiFi2", "HiFi4", False),
        (None, "LoFi", None),
        ("", "LoFi", None),
    ],
)
def test_s4_verification_tristate(ran, want, expect):
    op = _op(32, 4096, 4096)
    assert _VERIFY(_prof(op, ran), op, {"fidelity": want}) is expect


def test_s4_unknown_is_never_false():
    """The distinction the whole telemetry rests on: 'we could not tell' != 'they ignored it'."""
    op = _op(32, 4096, 4096)
    assert _VERIFY({"top_ops": []}, op, {"fidelity": "LoFi"}) is None
    assert _VERIFY(_prof("Other", "LoFi"), op, {"fidelity": "LoFi"}) is None
    assert _VERIFY(_prof(op, "LoFi"), op, {"dtype": "bfloat8_b"}) is None


def test_s4_dtype_only_is_unknown_not_verified():
    """Per-op dtype is not captured, so a dtype recommendation can never be confirmed here. If this
    ever starts returning True/False, the profile gained a dtype column and this module should use
    it rather than keep pretending it cannot."""
    op = _op(32, 4096, 4096)
    assert _VERIFY(_prof(op, "LoFi"), op, {"dtype": "bfloat4_b"}) is None


@pytest.mark.parametrize("bad", [None, {}, [], "x", 42, {"top_ops": "nope"}, {"top_ops": [None]}])
def test_s4_hostile_profiles_never_raise(bad):
    assert _VERIFY(bad, "op", {"fidelity": "LoFi"}) is None


# --------------------------------------------------------------------------- s5
def test_s5_deterministic_and_read_only(tmp_path):
    entries = [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}]
    _write(tmp_path, entries)
    before = (tmp_path / "matmul_sweep.json").read_text()
    results = [_LOOKUP(tmp_path, _op(32, 4096, 4096)) for _ in range(50)]
    assert all(r == results[0] for r in results)
    assert (tmp_path / "matmul_sweep.json").read_text() == before, "the lookup wrote to the model tree"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["matmul_sweep.json"]


# --------------------------------------------------------------------------- s6
def test_s6_result_is_not_shared_state(tmp_path):
    _write(tmp_path, [{"m": 32, "k": 4096, "n": 4096, "fidelity": "LoFi", "dtype": "bfloat8_b"}])
    a = _LOOKUP(tmp_path, _op(32, 4096, 4096))
    a["fidelity"] = "CORRUPTED"
    b = _LOOKUP(tmp_path, _op(32, 4096, 4096))
    assert b["fidelity"] == "LoFi", "a caller mutating the result poisoned the next lookup"
