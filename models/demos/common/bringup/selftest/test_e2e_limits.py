# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): the device is judged against the full-precision reference; the reference at its own precision only
sets the margin. These guard the rule across the framework, not one model."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

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
        E.load(tmp_path, "s256")


class _G:
    """A golden in memory: layer outs and states of the last chunk."""

    n_chunks, seq, chunk, state_fixed = 2, 64, 32, []

    def __init__(self, out, state):
        self._out, self._state = out, state

    def layer(self, c, i):
        return {"out": self._out[i]}

    def state(self, i, at=None):
        return self._state[i]


class _Ref:
    def state_tensors(self, state, i, length):
        return state[i]


def _entries(target, official_out, official_state, keep=()):
    s = SimpleNamespace(get=lambda k, d=None: {"golden.keep_fp32": list(keep)}.get(k, d), num_layers=99)
    rep = {"trail": {0: official_out}, "state": {0: official_state}}
    return D.entries(s, _Ref(), target, [0], rep, 0)


def test_drift_measures_the_official_reference_against_full_precision():
    g = torch.Generator().manual_seed(0)
    full = torch.randn(64, 32, generator=g)
    rounded = full + 0.05 * torch.randn(64, 32, generator=g)
    idx = torch.stack([torch.randperm(100, generator=g)[:10] for _ in range(8)])
    flipped = idx.clone()
    flipped[:4, 0] = 500
    d = _entries(_G({0: full}, {0: {"kv": full, "topk": idx}}), rounded, {"kv": rounded, "topk": flipped})
    assert 0.99 < d["pcc_layer_L00"]["drift"] < 0.9999 and d["pcc_layer_L00"]["res"] > 0.9999
    assert d["overlap_state_topk_L00"]["drift"] == pytest.approx(1 - 4 / 80)
    assert set(d) == {"pcc_layer_L00", "pcc_state_kv_L00", "overlap_state_topk_L00"}


def test_the_drift_floor_is_one_rounding_in_the_stored_dtype_and_one_miss_per_selection(monkeypatch):
    """F133: a state the official golden keeps fp32 gets an fp32 floor (not bf16); an exact index state with 10^5
    selected entries a limit above 0.9999 at calib 2 (not the fixed 0.98 of one miss per hundred)."""
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    g = torch.Generator().manual_seed(0)
    full = torch.randn(256, 64, generator=g)
    idx = torch.stack([torch.randperm(4096, generator=g)[:1000] for _ in range(100)])
    target = _G({0: full}, {0: {"rec": full, "topk": idx}})
    d = _entries(target, full.clone(), {"rec": full.clone(), "topk": idx.clone()}, keep=("rec",))
    assert 1 - d["pcc_layer_L00"]["res"] > 1e-7  # stored bf16: a bf16 floor
    assert 1 - d["pcc_state_rec_L00"]["res"] < 1e-12  # kept fp32
    assert E.limit(d, "overlap_state_topk_L00") > 0.9999


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

    for numerics in ("native", "official"):  # F132: either way the arithmetic is unknown
        with pytest.raises(SystemExit, match="numerics"):
            G.reference_at(S({}), numerics, layers=None)
    assert G.reference_at(S({"reference.no_rounding": True}), "native", layers=None) == "official only"


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

    assert G.TARGET_NUMERICS == "native"  # F132: a constant, not a switch
    assert defaults.get("golden.numerics", None) is None and defaults.get("tests.step_numerics", None) is None
    assert json.dumps(defaults.get("thresholds")).find("state_overlap") < 0
