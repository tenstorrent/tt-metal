# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F138 (owner): no fixed accuracy number anywhere. Every accuracy limit is calib x a noise floor (per step) or a
drift (end to end) the test records; a gate holds `limit`, never a number. CPU only."""

import ast
import fnmatch
from pathlib import Path

import pytest

from models.demos.common.bringup.core import defaults
from models.demos.common.bringup.core import freeze as F
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.runs import IMPL_ENV
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_device_side import gspec, noise  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parents[1]


def test_the_ledger_refuses_a_fixed_accuracy_number_on_a_device_check(tmp_path):
    led = Ledger(tmp_path)
    led.write_tasks(
        {
            "tasks": [
                {
                    "id": "R.2",
                    "title": "r",
                    "step": "reference",
                    "gate": {"cmd": "x", "metrics": {"pcc_logits": ">= 0.9999"}},
                },
                {
                    "id": "P.1",
                    "title": "p",
                    "step": "perf",
                    "gate": {"cmd": "x", "metrics": {"pcc_layer_L*": ">= 0.97"}},
                },
                {
                    "id": "L.x",
                    "title": "l",
                    "device": True,
                    "gate": {"cmd": "x", "metrics": {"err_layer_L*": "<= limit"}},
                },
            ]
        }
    )
    errs = led.validate()
    assert len(errs) == 1 and errs[0].startswith("P.1: pcc_layer_L*") and "F138" in errs[0]


def test_no_spec_key_holds_a_fixed_accuracy_number(fx):
    errs = Spec.load(fx(thresholds={"component": 0.99})).validate()
    assert any("'thresholds'" in e and "F138" in e for e in errs)
    assert defaults.get("thresholds", None) is None and defaults.get("tests.low_precision_kinds", None) is None


def test_the_freeze_refuses_a_numeric_threshold(tmp_path):
    f = tmp_path / "test_c.py"
    f.write_text("THRESHOLD = 0.99\nCOMPARE = None\n")
    assert F.literal_threshold(f) == 0.99
    f.write_text("THRESHOLD = None  # the noise floor\n")
    assert F.literal_threshold(f) is None


def test_the_per_step_tests_take_no_fixed_threshold():
    for f in ("testing/component.py", "testing/component_checks.py", "testing/noise_floor.py", "testing/ladder.py"):
        calls = {
            getattr(n.func, "attr", getattr(n.func, "id", ""))
            for n in ast.walk(ast.parse((ROOT / f).read_text()))
            if isinstance(n, ast.Call)
        }
        assert not calls & {"threshold"}, f
    from models.demos.common.bringup.testing.component_checks import COMPONENT_DEFAULTS

    assert set(COMPONENT_DEFAULTS) == {"component_calib", "component_bias_share"} | {
        k for k in COMPONENT_DEFAULTS if k.startswith("struct_")
    }, "a component_* number other than calib is an accuracy limit (F138); structural probes are struct_*"


def test_every_limit_condition_is_recorded_by_its_gates_test(gspec, noise, monkeypatch):  # noqa: F811
    """The generated ledger's per-step and ladder gates on the fixture model (CPU): every `limit` condition finds its
    `<metric>_limit` recorded by the test the gate runs (the contract, the profile and the positions need a device)."""
    from models.demos.common.bringup.testing.component import run_component_test, run_swap_test
    from models.demos.common.bringup.testing.ladder import run_ladder

    s = gspec()
    noise(1e-3)
    tasks = generate(s, fixture_model.Reference())["tasks"]
    missing, n = [], 0
    for t in tasks:
        b, tid = t.get("brief") or {}, t["id"]
        conds = {g for g, c in (t["gate"].get("metrics") or {}).items() if str(c).endswith("limit")}
        if not conds:
            continue
        if tid.startswith("C."):
            monkeypatch.setenv(IMPL_ENV, "reference")
            run_component_test(s, b["step"], b["layer"])
        elif tid.startswith("S."):
            monkeypatch.setenv(IMPL_ENV, "reference")
            run_swap_test(s, b["block_type"], b["swapped"], checks="steps")
        elif tid.startswith(("L.", "M.1")) and "BRINGUP_RUNG=" in t["gate"]["cmd"]:
            monkeypatch.setenv(IMPL_ENV, "device")
            run_ladder(s, t["gate"]["cmd"].split("BRINGUP_RUNG=")[1].split()[0], None)
        else:
            continue
        m, n = got(), n + 1
        for g in conds:
            if not any(fnmatch.fnmatchcase(k[: -len("_limit")], g) for k in m if k.endswith("_limit")):
                missing.append(f"{tid}: {g}")
    assert n >= 5 and not missing, missing


@pytest.mark.xfail(strict=True, reason="F140: the existing bring-ups' ledgers still hold fixed e2e numbers")
def test_no_committed_ledger_holds_a_fixed_accuracy_number():
    bad = []
    for p in sorted(MODELS.glob("*/bringup/tasks.yaml")):
        bad += [f"{p.parent.parent.name}: {e}" for e in Ledger(p.parent).validate() if "F138" in e]
    assert not bad, f"{len(bad)} fixed accuracy gates, e.g. {bad[:5]}"


def test_a_gate_with_a_fixed_accuracy_number_fails_before_it_runs(sandbox):
    """F138: an overseer-added task (a perf pick) is checked at its gate too, not only by the ledger generator."""
    from models.demos.common.bringup.core.gate import run_gate

    led = sandbox.tasks(
        {"id": "P.1", "title": "p", "step": "perf", "gate": {"cmd": "true", "metrics": {"pcc_chunk_out": ">= 0.97"}}}
    )
    res = run_gate(sandbox.spec, led, "P.1", commit=False)
    assert res.verdict == "FAIL" and res.rc is None and any("F138" in ln for ln in res.lines)
