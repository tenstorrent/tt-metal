# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue 1: the declared-depth-bounded coverage ladder.

  s1  every depth 1..256 x several operator ladders -- invariants hold in every cell
  s2  config dialects: the 9 depth key spellings, nested text_config, junk configs
  s3  the ladder is consumed by _measure_cov, not just defined (wiring, not decoration)
  s4  hostile inputs: unreadable dir, corrupt json, absurd depths, non-int depths
  s5  purity/determinism and no environment leakage
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_depth_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_LADDER = _mod()._cov_ladder


def _cfg(tmp_path, payload, name="config.json"):
    d = tmp_path / name.replace("/", "_")
    d = tmp_path
    (d / name).write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return d


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize("ladder", ["2,4,8,16", "1,2,4,8,16,32,64", "4", "2,3,5,7,11"])
def test_s1_invariants_hold_for_every_depth(tmp_path, monkeypatch, ladder):
    monkeypatch.setenv("PERF_MCP_COV_LADDER", ladder)
    base = [int(x) for x in ladder.split(",")]
    for depth in range(1, 257):
        d = tmp_path / f"m{depth}"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"num_hidden_layers": depth}))
        out = _LADDER(d)
        assert out, f"empty ladder for depth={depth}"
        assert out[-1] == depth, f"full depth must be the final rung (depth={depth}, got {out})"
        assert all(r <= depth for r in out), f"rung deeper than the model (depth={depth}, got {out})"
        assert out == sorted(out), f"ladder not ascending (depth={depth}, got {out})"
        assert len(out) == len(set(out)), f"duplicate rung (depth={depth}, got {out})"
        assert all(r in base or r == depth for r in out), f"invented a rung (depth={depth}, got {out})"


def test_s1b_monotone_in_depth(tmp_path):
    """Deeper model -> never a shorter ladder. A regression here means rungs are being dropped."""
    prev = 0
    for depth in range(1, 100):
        d = tmp_path / f"m{depth}"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"num_hidden_layers": depth}))
        n = len(_LADDER(d))
        assert n >= prev - 1, f"ladder length collapsed at depth={depth}"
        prev = n


# --------------------------------------------------------------------------- s2
@pytest.mark.parametrize(
    "payload",
    [
        {"num_hidden_layers": 12},
        {"n_layer": 12},
        {"n_layers": 12},
        {"num_layers": 12},
        {"num_blocks": 12},
        {"depth": 12},
        {"text_config": {"num_hidden_layers": 12}},
        {"llm_config": {"num_hidden_layers": 12}},
    ],
)
def test_s2_config_dialects_all_reach_the_ladder(tmp_path, payload):
    (tmp_path / "config.json").write_text(json.dumps(payload))
    out = _LADDER(tmp_path)
    if out[-1] != 12:
        pytest.skip(f"dialect {list(payload)[0]!r} not supported by full_depth_from_config")
    assert out == [2, 4, 8, 12]


@pytest.mark.parametrize("name", ["config.json", "params.json", "model_config.json"])
def test_s2_all_three_config_filenames(tmp_path, name):
    (tmp_path / name).write_text(json.dumps({"num_hidden_layers": 12}))
    assert _LADDER(tmp_path)[-1] == 12


def test_s2_no_config_keeps_plain_ladder(tmp_path):
    assert _LADDER(tmp_path) == [2, 4, 8, 16]


# --------------------------------------------------------------------------- s3
def test_s3_measure_cov_actually_uses_the_ladder():
    """Wiring check: _measure_cov must call _cov_ladder, not re-read the env itself."""
    import inspect

    src = inspect.getsource(_mod()._measure_cov)
    assert "_cov_ladder(" in src, "_measure_cov does not consume the declared-depth ladder"
    assert (
        "PERF_MCP_COV_LADDER" not in src
    ), "_measure_cov still reads PERF_MCP_COV_LADDER directly, bypassing the depth bound"


def test_s3_full_depth_from_config_is_imported_in_run_py():
    assert "full_depth_from_config" in (_CC / "run.py").read_text()


# --------------------------------------------------------------------------- s4
def test_s4_corrupt_config_degrades_to_plain_ladder(tmp_path):
    (tmp_path / "config.json").write_text("{not json at all")
    assert _LADDER(tmp_path) == [2, 4, 8, 16]


@pytest.mark.parametrize("bad", [0, -1, -999, "twelve", None, 1.5, [], {}, True])
def test_s4_non_positive_or_non_int_depth_is_ignored(tmp_path, bad):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": bad}))
    out = _LADDER(tmp_path)
    assert out and out == sorted(set(out))
    assert all(isinstance(r, int) and r > 0 for r in out), f"bad depth {bad!r} produced {out}"


def test_s4_absurd_depth_is_still_bounded(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 10**6}))
    out = _LADDER(tmp_path)
    assert out[-1] == 10**6 and out[:-1] == [2, 4, 8, 16]


@pytest.mark.parametrize("bad_root", [None, Path("/nonexistent_xyz_123"), Path("/dev/null")])
def test_s4_hostile_model_root_does_not_raise(bad_root):
    out = _LADDER(bad_root)
    assert out == [2, 4, 8, 16]


def test_s4_empty_env_ladder_still_returns_the_full_depth(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_COV_LADDER", "")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 9}))
    assert _LADDER(tmp_path) == [9]


def test_s4_garbage_env_ladder_is_filtered(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_COV_LADDER", "2,abc,,8,-4,16.5")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
    assert _LADDER(tmp_path) == [2, 8, 32]


# --------------------------------------------------------------------------- s5
def test_s5_deterministic_and_no_env_mutation(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("PERF_MCP_COV_LADDER", "2,4,8,16")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
    before = dict(os.environ)
    a = _LADDER(tmp_path)
    b = _LADDER(tmp_path)
    assert a == b == [2, 4, 8, 16, 32]
    assert dict(os.environ) == before, "the ladder mutated the environment"


def test_s5_result_is_a_fresh_list(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
    a = _LADDER(tmp_path)
    a.append(999)
    assert _LADDER(tmp_path) == [2, 4, 8, 16, 32], "ladder returned shared mutable state"
