# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): the device is judged against the full-precision reference; the reference at its own precision only
sets the margin. These guard the rule across the framework, not one model."""

import ast
import json
import re
from pathlib import Path

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.reference import drift as D
from models.demos.common.bringup.reference import golden as G
from models.demos.common.bringup.selftest.test_plan import Reference  # noqa: F401 (the plan tests' fixture model)
from models.demos.common.bringup.testing import e2e_limits as E

ROOT = Path(E.__file__).resolve().parents[1]


def test_the_limit_is_calib_times_the_reference_drift_and_never_below_resolution(monkeypatch):
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    d = {"pcc_layer_L00": {"drift": 0.99, "res": 0.99999}, "pcc_layer_L01": {"drift": 1.0, "res": 0.99999}}
    assert E.limit(d, "pcc_layer_L00") == pytest.approx(1 - E.CALIB * 0.01)
    assert E.limit(d, "pcc_layer_L01") == pytest.approx(1 - E.CALIB * 1e-5)  # an exact layer: the bf16 floor
    failed = []
    assert E.check(d, "pcc_layer_L00", 0.985, failed) and not failed
    assert not E.check(d, "pcc_layer_L00", 0.97, failed) and "limit" in failed[0]


def test_a_rung_without_its_drift_fails_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="F128"):
        E.load(tmp_path)


class _G:
    """A golden in memory: layer outs, states and the model outputs of the last chunk."""

    def __init__(self, out, state, n_chunks=2):
        self.n_chunks, self._out, self._state = n_chunks, out, state

    def layer(self, c, i):
        return {"out": self._out[i]}

    def state(self, i):
        return self._state[i]


def test_drift_measures_the_official_reference_against_full_precision():
    g = torch.Generator().manual_seed(0)
    full = torch.randn(64, 32, generator=g)
    rounded = full + 0.05 * torch.randn(64, 32, generator=g)
    idx = torch.stack([torch.randperm(100, generator=g)[:10] for _ in range(8)])
    flipped = idx.clone()
    flipped[:4, 0] = 500
    target = _G({0: full}, {0: {"kv": full, "topk": idx}})
    official = _G({0: rounded}, {0: {"kv": rounded, "topk": flipped}})
    d = D.drift(target, official, [0], full_stack=False)
    assert 0.99 < d["pcc_layer_L00"]["drift"] < 0.9999 and d["pcc_layer_L00"]["res"] > 0.9999
    assert d["overlap_state_topk_L00"]["drift"] == pytest.approx(1 - 4 / 80)
    assert set(d) == {"pcc_layer_L00", "pcc_state_kv_L00", "overlap_state_topk_L00"}


def test_no_hook_without_a_numerics_switch_falls_back_to_rounded_arithmetic():
    class Hooks:
        @staticmethod
        def reference(spec, layers=None, dtype=None):
            return "official only"

    class S:
        def __init__(self, data):
            self.data = data

        def hooks(self):
            return Hooks

        def get(self, key, default=None):
            return self.data.get(key, default)

    with pytest.raises(SystemExit, match="numerics"):
        G.reference_at(S({}), "native", layers=None)
    assert G.reference_at(S({"reference.no_rounding": True}), "native", layers=None) == "official only"
    assert G.reference_at(S({}), "official", layers=None) == "official only"


ACCURACY = re.compile(r"^(pcc_|overlap_|top1|top5|err_)")


def test_no_generated_gate_holds_a_fixed_accuracy_number(fx):
    """Every device accuracy condition the ledger writes compares with the limit its test records (F76 / F80 per step, F128
    end to end); a fixed number here is the rule broken again."""
    s = Spec.load(fx())
    fixed = []
    tasks = generate(s, Reference())["tasks"]
    assert any(t["id"].startswith("L.") for t in tasks)
    for t in tasks:
        if not t.get(
            "device"
        ):  # the reference's own checks (R.*: CPU vs the official code, exact) are not device accuracy
            continue
        for k, cond in (t.get("gate", {}).get("metrics") or {}).items():
            if ACCURACY.match(k) and not str(cond).endswith("limit"):
                fixed.append(f"{t['id']}: {k} {cond}")
    assert not fixed, fixed


def test_the_end_to_end_tests_use_no_fixed_threshold():
    for f in ("testing/ladder.py", "testing/contract.py"):
        tree = ast.parse((ROOT / f).read_text())
        calls = [
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        ]
        assert "threshold" not in calls, f"{f} calls threshold(): end-to-end limits come from the drift (F128)"


def test_goldens_and_per_step_references_default_to_full_precision():
    from models.demos.common.bringup.core import defaults

    assert defaults.get("golden.numerics") == "native" and defaults.get("tests.step_numerics") == "native"
    assert json.dumps(defaults.get("thresholds")).find("state_overlap") < 0
