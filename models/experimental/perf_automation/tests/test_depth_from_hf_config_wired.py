# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 1: the model's DECLARED depth must actually reach the depth path.

`agent/layer_depth.py::full_depth_from_config()` reads the block count straight out of the HF
config (32 for meta-llama/Llama-3.1-8B-Instruct) without building or running anything -- but it had
ZERO production callers. Every consumer instead used the hardcoded ladder `2,4,8,16`, so on
llama3_1_8b_p150 the coverage search topped out at 16 with 2 op types still uncovered, reported
that as "measured", and the run carried a false `(16 layers)` label for the rest of its life.

The declared depth is the authority the ladder was missing:
  * the ladder must never probe DEEPER than the model actually is (wasted device time, and the
    probe silently clamps so two rungs measure the same thing), and
  * the FULL depth must be the last rung, because at full depth coverage is total by construction
    -- which is exactly the rung that would have covered the 2 missing op types.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _run_module():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_depth", str(_CC / "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ladder_fn():
    mod = _run_module()
    fn = getattr(mod, "_cov_ladder", None)
    if fn is None:
        pytest.fail(
            "run.py has no _cov_ladder helper: the coverage ladder is still the hardcoded "
            "PERF_MCP_COV_LADDER default '2,4,8,16' and full_depth_from_config() still has zero "
            "production callers, so the model's declared depth never reaches the depth path. "
            "That is how llama3_1_8b_p150 stopped at 16 with 2 op types uncovered and still "
            "labelled the profile '(16 layers)'."
        )
    return fn


def _model_dir(tmp_path, n_layers):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": n_layers}))
    return tmp_path


def test_declared_depth_becomes_the_final_rung(tmp_path):
    # Llama-3.1-8B: 32 layers. The default ladder stops at 16, so 32 must be appended -- at full
    # depth every op type is present by construction.
    assert _ladder_fn()(_model_dir(tmp_path, 32)) == [2, 4, 8, 16, 32]


def test_ladder_never_probes_deeper_than_the_model(tmp_path):
    # A 6-layer model: rungs 8 and 16 do not exist. Probing them wastes device time and produces
    # two rungs that measure the identical thing.
    assert _ladder_fn()(_model_dir(tmp_path, 6)) == [2, 4, 6]


def test_shallow_model_collapses_to_full_depth(tmp_path):
    assert _ladder_fn()(_model_dir(tmp_path, 2)) == [2]
    assert _ladder_fn()(_model_dir(tmp_path, 1)) == [1]


def test_exact_power_of_two_depth_is_not_duplicated(tmp_path):
    assert _ladder_fn()(_model_dir(tmp_path, 16)) == [2, 4, 8, 16]


def test_undeclared_depth_falls_back_to_the_plain_ladder(tmp_path):
    # No config anywhere -> we must NOT invent a depth; keep the old behaviour so the builder can
    # still reveal its own depth.
    assert _ladder_fn()(tmp_path) == [2, 4, 8, 16]


def test_explicit_env_override_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_COV_LADDER", "3,9")
    # the operator asked for specific rungs; still bounded by the real depth
    assert _ladder_fn()(_model_dir(tmp_path, 32)) == [3, 9, 32]
    assert _ladder_fn()(_model_dir(tmp_path, 5)) == [3, 5]


def test_full_depth_from_config_now_has_a_production_caller():
    """The point of the issue: the function must be reachable from run.py, not just from tests."""
    src = (_CC / "run.py").read_text()
    assert (
        "full_depth_from_config" in src
    ), "full_depth_from_config() still has no caller in run.py -- the HF config is not wired in"
