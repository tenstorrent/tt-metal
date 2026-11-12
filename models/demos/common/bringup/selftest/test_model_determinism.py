# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.25 (owner 2026-11-12: determinism is the highest of all priorities): the model-level determinism check
(testing/model_determinism.py) and its place in every model-level gate (core/gate.py with_determinism, gate_limits;
the A/B's determinism item). A deliberately non-deterministic fake model fails the check and the gate."""

import os
import stat

import pytest
import torch

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.gate import gate_command, gate_limits, gate_units, run_gate
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.selftest import fixture_model as F
from models.demos.common.bringup.selftest.conftest import PY, got
from models.demos.common.bringup.selftest.test_core import task
from models.demos.common.bringup.testing import model_determinism as MD

LADDER = "BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py"
PROFILE = "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py"
CONTRACT = "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py"
OWN = "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/x/tests/test_determinism.py"


def _tokens(n=256):
    g = torch.Generator().manual_seed(1)
    return {"novel": torch.randint(0, F.V, (n,), generator=g), "code": torch.randint(0, F.V, (n,), generator=g)}


@pytest.fixture
def fspec(fx):
    return Spec.load(fx())


def test_a_deterministic_model_passes_and_records_zero(fspec):
    m = F.FakeDeviceModel(fspec, fspec.layers())
    err, worst = MD.check(fspec, m, fspec.layers(), tokens=_tokens())
    g = got()
    assert err is None and worst == 0.0
    assert g["determinism_max_abs_diff"] == 0.0 and g["determinism_elements_differ"] == 0
    assert g["determinism_chunks"] == 4 and g["determinism_runs"] == 3  # defaults.yaml tests.model_determinism


def test_a_nondeterministic_model_fails_naming_the_chunk_and_the_layer(fspec):
    m = F.NondeterministicFakeModel(fspec, fspec.layers(), layer=1)
    err, worst = MD.check(fspec, m, fspec.layers(), tokens=_tokens())
    assert err is not None and worst > 0
    assert "prompt novel run 1 vs run 0 (run 0: the cold sequence" in err
    assert "first differing chunk 0" in err
    assert "first differing layer state L02" in err  # layer 1's output feeds layer 2's KV state
    assert got()["determinism_max_abs_diff"] > 0


def test_a_bits_only_difference_fails_too(fspec):
    m = F.NondeterministicFakeModel(fspec, fspec.layers(), layer=2, bits_only=True)
    err, worst = MD.check(fspec, m, fspec.layers(), tokens=_tokens())
    assert err is not None and worst == 0.0  # -0 vs +0: equal values, different bits
    assert got()["determinism_max_abs_diff"] > 0  # the gated metric still fails (2^-149)
    assert got()["determinism_elements_differ"] > 0


def test_coverage_fits_the_target_and_needs_two_runs(fspec, monkeypatch):
    assert MD.coverage(fspec) == (4, 3)
    monkeypatch.setenv("DET_CHUNKS", "99")
    assert MD.coverage(fspec)[0] == 256 // 64
    monkeypatch.setenv("DET_RUNS", "1")
    with pytest.raises(AssertionError, match="at least two"):
        MD.coverage(fspec)


def _t(cmd, step="perf", **kw):
    return task("A.1", cmd, step=step, device=True, **kw)


def test_every_model_level_gate_runs_it_first_in_its_first_shared_process_unit():
    t = _t(f"{LADDER} && {CONTRACT}")
    units = gate_units(t)
    assert [u for u, _ in units] == ["A.1#test_ladder", "A.1#test_contract"]  # unit names unchanged
    assert units[0][1].startswith(B.DETERMINISM_CMD + " && ")  # first in the ladder's process
    assert B.unit_command(units[0][1]) is not None  # one process with the ladder: no model load of its own
    assert B.DETERMINISM_TEST not in units[1][1]
    assert gate_limits(t)["determinism_max_abs_diff"] == "== 0"
    assert B.DETERMINISM_TEST in gate_command(t)


def test_a_gate_without_a_shared_unit_gets_its_own_first_unit():
    fix = _t("scripts/run_safe_pytest.sh --run-all --no-precompile models/x/tests/test_prefill_end.py", "overseer_fix")
    assert [u for u, _ in gate_units(fix)] == ["A.1#test_determinism", "A.1"]  # a fix step's device gate
    t = _t(CONTRACT, step="contract")
    assert [u for u, _ in gate_units(t)] == ["A.1#test_determinism", "A.1"]
    assert gate_units(t)[0][1] == B.DETERMINISM_CMD
    # a unit with a model switch (another configuration) is not where the shipped model's check runs
    sw = _t(f"DS41_X=on {LADDER}")
    assert [u for u, _ in gate_units(sw)] == ["A.1#test_determinism", "A.1"]


def test_a_gate_naming_a_determinism_test_and_a_non_model_gate_are_left_alone():
    own = _t(f"{LADDER} && {OWN}")
    assert B.DETERMINISM_CMD not in gate_command(own)
    assert gate_limits(own)["determinism_max_abs_diff"] == "== 0"  # still gated
    comp = task("C.1", "scripts/run_safe_pytest.sh models/demos/x/tests/test_c_attn.py", step="component", device=True)
    assert gate_units(comp) == [("C.1", comp["gate"]["cmd"])] and "determinism_max_abs_diff" not in gate_limits(comp)
    cpu = task("FIX.1", "scripts/run_safe_pytest.sh models/demos/x/reference/test_ref.py", step="overseer_fix")
    assert gate_units(cpu) == [("FIX.1", cpu["gate"]["cmd"])]  # not a device gate (device: false)


def test_a_tasks_yaml_limit_is_never_loosened():
    t = _t(LADDER, metrics={"determinism_max_abs_diff": "== 0", "pcc_*": ">= 0.99"})
    assert gate_limits(t) == {"determinism_max_abs_diff": "== 0", "pcc_*": ">= 0.99"}


def test_it_runs_first_in_a_batch():
    runs = [B.parse(x) for x in (LADDER, PROFILE, B.DETERMINISM_CMD)]
    assert all(runs)
    assert sorted(runs, key=B.phase)[0]["args"][0] == B.DETERMINISM_TEST


FAKE_TEST = """import os, sys
import torch
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.spec import Spec
test = [a for a in sys.argv[1:] if a.endswith(".py")][-1]
if test.endswith("test_determinism.py"):
    from models.demos.common.bringup.selftest import fixture_model as F
    from models.demos.common.bringup.testing.model_determinism import check
    s = Spec.load(os.environ["FIXTURE_SPEC"])
    g = torch.Generator().manual_seed(1)
    toks = {w: torch.randint(0, F.V, (256,), generator=g) for w in ("novel", "code")}
    err, _ = check(s, F.device_model(None, s, s.layers()), s.layers(), tokens=toks)
    if err:
        print("E   AssertionError: " + err)
        sys.exit(1)
else:
    M.record("pcc_ladder", 0.999)
"""


def test_a_nondeterministic_fake_model_fails_the_gate(sandbox, fx, monkeypatch):
    """End to end: a ladder gate (tasks.yaml names no determinism test) through run_gate, with a stand-in safe runner
    in the sandbox repo; the deterministic fixture passes, the non-deterministic one fails the gate."""
    root = sandbox.repo.parent
    (root / "fake_test.py").write_text(FAKE_TEST)
    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir(exist_ok=True)
    runner.write_text(f'#!/bin/bash\nexec {PY} {root / "fake_test.py"} "$@"\n')
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FIXTURE_SPEC", fx())
    sandbox.write_spec(tests={"one_process": False})
    led = sandbox.tasks(_t(LADDER, paths=["src"]))
    res = run_gate(sandbox.spec, led, "A.1")
    assert res.verdict == "PASS", res.summary()
    m = M.load("A.1", led.results_dir)
    assert m["determinism_max_abs_diff"]["value"] == 0.0 and "pcc_ladder" in m
    led.reset(["A.1"])
    monkeypatch.setenv("FIXTURE_NONDETERMINISTIC", "1")
    res = run_gate(sandbox.spec, led, "A.1", fresh=True)
    assert res.verdict == "FAIL"
    log = res.log.read_text()
    assert "identical inputs gave different outputs" in log and "first differing chunk 0" in log
    assert "first differing layer state L02" in log
    assert "pcc_ladder" not in (M.load("A.1", led.results_dir))  # && : the ladder never ran after it
    assert os.environ.get("FIXTURE_NONDETERMINISTIC") == "1"
