# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): the device is judged against the full-precision reference; the reference at its own precision only
sets the margin. These guard the rule across the framework, not one model."""

import ast
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
    """F135: the per-step convention end to end: calib x rel for an error, calib^2 on a PCC's miss, calib on an
    overlap's miss; never below one rounding (one miss)."""
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    d = {
        "layer_L00": {"rel": 0.01, "rel_res": 0.002, "pcc": 0.9999, "pcc_res": 0.999998},
        "layer_L01": {"rel": 0.0, "rel_res": 0.002, "pcc": 1.0, "pcc_res": 0.999998},
        "top5": {"overlap": 0.99, "overlap_res": 0.999},
    }
    assert E.limit(d, "layer_L00", "err_layer_L00") == pytest.approx(E.CALIB * 0.01)
    assert E.limit(d, "layer_L01", "err_layer_L01") == pytest.approx(E.CALIB * 0.002)  # an exact layer: one rounding
    assert E.limit(d, "layer_L00", "pcc_layer_L00") == pytest.approx(1 - E.CALIB**2 * 1e-4)
    assert E.limit(d, "top5", "top5_overlap") == pytest.approx(1 - E.CALIB * 0.01)
    failed = []
    assert E.check(d, "layer_L00", "err_layer_L00", 0.015, failed) and not failed
    assert not E.check(d, "layer_L00", "err_layer_L00", 0.03, failed) and "limit" in failed[0]


def test_a_two_percent_scale_fails_end_to_end_though_its_pcc_is_one(monkeypatch):
    """F81's reason end to end: PCC cannot see a scale; the gated rel error does."""
    from models.demos.common.bringup.testing.ladder import rel

    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    g = torch.Generator().manual_seed(0)
    want = torch.randn(512, 64, generator=g)
    official = want + 0.002 * torch.randn(512, 64, generator=g)
    d = {"layer_L00": D._float_entry(official.bfloat16(), want)}
    dev = 1.02 * want
    assert metrics.pcc(dev, want) > 0.99999
    assert not E.check(d, "layer_L00", "err_layer_L00", rel(dev, want), [])
    assert E.check(d, "layer_L00", "err_layer_L00", rel(official, want), [])


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
    assert 0.01 < d["layer_L00"]["rel"] < 0.1 and d["layer_L00"]["rel_res"] < 0.01
    assert 0.99 < d["layer_L00"]["pcc"] < 0.9999 and d["layer_L00"]["pcc_res"] > 0.9999
    assert d["state_topk_L00"]["overlap"] == pytest.approx(1 - 4 / 80)
    assert set(d) == {"layer_L00", "state_kv_L00", "state_topk_L00"}


def test_the_drift_floor_is_one_rounding_in_the_stored_dtype_and_one_miss_per_selection(monkeypatch):
    """F133: a state the official golden keeps fp32 gets an fp32 floor (not bf16); an exact index state with 10^5
    selected entries a limit above 0.9999 at calib 2 (not the fixed 0.98 of one miss per hundred)."""
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)
    g = torch.Generator().manual_seed(0)
    full = torch.randn(256, 64, generator=g)
    idx = torch.stack([torch.randperm(4096, generator=g)[:1000] for _ in range(100)])
    target = _G({0: full}, {0: {"rec": full, "topk": idx}})
    d = _entries(target, full.clone(), {"rec": full.clone(), "topk": idx.clone()}, keep=("rec",))
    assert d["layer_L00"]["rel_res"] > 1e-3  # stored bf16: a bf16 floor
    assert 1e-9 < d["state_rec_L00"]["rel_res"] < 1e-6  # kept fp32: one fp32 rounding
    assert E.limit(d, "state_topk_L00", "overlap_state_topk_L00") > 0.9999


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
    assert defaults.get("thresholds", None) is None  # F138: no fixed accuracy number at all


def test_an_all_zero_state_gets_a_zero_limit_not_an_absurd_one():
    """F159: DeepSeek-V4.1 err_state_compress_partial: the target state was exactly zero, one rounding's rel came out
    9e255 (a 1e-300 clamp) and the limit 1.8e256 let anything pass. Now an all-zero target needs an exact zero."""
    import torch

    from models.demos.common.bringup.reference import drift as D

    z = torch.zeros(4, 64)
    e = D._float_entry(z, z)
    assert e["zero_target"] and e["rel"] == 0.0 and E.err_limit({"k": e}, "k") == 0.0
    assert D._float_entry(z + 1e-3, z)["rel"] == 1.0
    old = {"rel": 0.0, "rel_res": 9.14e255, "pcc": 1.0, "pcc_res": 0.0, "region": None}  # a drift file from before
    assert E.err_limit({"k": old}, "k") == 0.0
    t = torch.randn(4, 64)
    assert 0 < E.err_limit({"k": D._float_entry(t.to(torch.bfloat16).float(), t)}, "k") < 0.02


def test_a_state_over_its_limit_passes_only_with_its_owner_waiver_and_is_reported(monkeypatch, capsys):
    """F226: an owner waiver state.<name> passes that state over its limit, printed WAIVED and recorded
    <metric>_waived = 1 (the gate reports it WAIVED); other states, and the state without a waiver, still fail."""
    from models.demos.common.bringup.testing import ladder as L

    rec = {}
    monkeypatch.setattr(metrics, "record", lambda k, v, **kw: rec.__setitem__(k, v))
    d = {"state_topk_idx_L14": {"overlap": 0.92, "overlap_res": 0.999}, "state_kv_L14": {"rel": 0.01, "rel_res": 1e-3}}
    lim = E.limit(d, "state_topk_idx_L14", "err_state_topk_idx_L14")
    failed = []
    assert not L._state_check(d, "state_topk_idx_L14", "topk_idx", lim + 0.02, failed, None) and len(failed) == 1
    failed = []
    w = {"state.topk_idx": "owner: 4-bit index"}
    assert L._state_check(d, "state_topk_idx_L14", "topk_idx", lim + 0.02, failed, w) and not failed
    assert rec["err_state_topk_idx_L14_waived"] == 1 and "WAIVED" in capsys.readouterr().out
    failed = []
    assert not L._state_check(d, "state_kv_L14", "kv", 0.5, failed, w) and len(failed) == 1  # another state: fails
    assert "err_state_kv_L14_waived" not in rec
