# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for quasar/open_risks.py (the open-risks gate) and build_report.pr_body."""

import importlib.util
import json
import sys
from pathlib import Path

_Q = Path(__file__).parent / "quasar"
sys.path.insert(0, str(_Q))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _Q / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


open_risks = _load("open_risks")
build_report = _load("build_report")

CLEAN = """# Agent: llk-tester
## Assumptions made
- Int32 Dest is two's complement — siblings default to it — wrong if a caller opts in.
## Open risks
- R1 CLOSED: -0.0 sign through datacopy — evidence: test_x[fpu_route-Float32]
- R2 DEFERRED: settle delay copied from a test kernel —
  PR: PRNG settle of 1600 cycles needs HW-owner confirmation
"""


NONE_LOG = "# x\n## Open risks\n- None.\n"


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text)


def _run(tmp_path, cycle=1, **logs):
    """Write state.json + a clean log for every expected self-log, overridden by `logs`."""
    (tmp_path / "state.json").write_text(json.dumps({"CYCLE": cycle}))
    for name in open_risks.expected_logs(str(tmp_path)):
        _write(tmp_path, name, logs.get(name, NONE_LOG))


def test_clean_logs_pass(tmp_path):
    _run(tmp_path, **{"agent_tester_cycle1.md": CLEAN})
    findings, closed, deferred = open_risks.check(str(tmp_path))
    assert findings == [] and (closed, deferred) == (1, 1)
    assert open_risks.deferred_lines(str(tmp_path)) == [
        "PRNG settle of 1600 cycles needs HW-owner confirmation"
    ]


def test_none_bullet_and_indented_evidence(tmp_path):
    log = "# t\n## Open risks\n- R1 CLOSED: NaN sign\n  - evidence: test_x[nan]\n"
    _run(
        tmp_path,
        **{
            "agent_tester_cycle1.md": log,
            "agent_analyzer.md": "# a\n## Open risks\n- none\n",
        },
    )
    assert open_risks.check(str(tmp_path)) == ([], 1, 0)


def test_open_missing_and_malformed_block(tmp_path):
    _run(
        tmp_path,
        **{
            "agent_analyzer.md": "# a\n## Open risks\n- R1 OPEN: NaN untested (R1)\n- R2 CLOSED: no evidence\n",
            "agent_optimizer.md": "# o\n## Reasoning summary\nok\n",
        },
    )
    findings, _, _ = open_risks.check(str(tmp_path))
    kinds = sorted(f.split()[0] for f in findings)
    assert kinds == ["MALFORMED", "MISSING", "OPEN"]


def test_absent_expected_log_is_missing(tmp_path):
    _run(tmp_path, cycle=2)
    (tmp_path / "agent_tester_cycle2.md").unlink()
    _write(tmp_path, "agent_tester_cycle1.md", CLEAN)  # superseded cycle does not count
    _write(
        tmp_path, "agent_arch_lookup.md", "# lookup\nassumes LREG8 is read-only\n"
    )  # not a pipeline log
    findings, _, _ = open_risks.check(str(tmp_path))
    assert findings == ["MISSING agent_tester_cycle2.md: self-log not written"]
    assert "agent_analysis_refiner_v1.md" in open_risks.expected_logs(str(tmp_path))


def test_waiver_word_needs_an_entry_reference(tmp_path):
    log = (
        CLEAN
        + "## Notes\n- NaN path untested\n- seed path untested (R2)\n- SFPNOP delay not verified on Quasar\n"
    )
    _run(tmp_path, **{"agent_tester_cycle1.md": log})
    findings, _, _ = open_risks.check(str(tmp_path))
    assert [f.split(":")[0] for f in findings] == ["WAIVER agent_tester_cycle1.md"] * 2
    assert "NaN path untested" in findings[0] and "not verified" in findings[1]


def test_waiver_in_code_fence_is_ignored(tmp_path):
    _run(
        tmp_path,
        **{
            "agent_writer_cycle1.md": "# w\n```\nconservative\n```\n## Open risks\nnone\n"
        },
    )
    assert open_risks.check(str(tmp_path))[0] == []


def _pr_run(tmp_path, risks, drift):
    log = "# t\n## Open risks\n" + "".join(
        f"- R{i} DEFERRED: r{i} — PR: risk {i % 6}\n" for i in range(risks)
    )
    _run(tmp_path, **{"agent_tester_cycle1.md": log})
    (tmp_path / "pre_pr_gates.txt").write_text(drift)
    run = {
        "kernel": "rand",
        "reference_file": "ref.h",
        "generated_file": "gen.h",
        "tests_total": 20,
        "tests_passed": 20,
        "test_file": "python_tests/quasar/test_rand_quasar.py",
        "coverage": {"covered": ["seed", "lockup"], "not_covered": {}},
    }
    return build_report.pr_body(run, str(tmp_path)).splitlines()


def test_pr_body_caps_risks_and_points_at_the_rest(tmp_path):
    drift = "".join(
        f"DRIFT: reference x changed on main: c{i}\n"
        for i in range(build_report.PR_BODY_MAX_DRIFT)
    )
    lines = _pr_run(tmp_path, risks=8, drift=drift)  # 6 distinct after dedupe
    assert len(lines) == build_report.PR_BODY_MAX_LINES
    assert "- (+2 more: see the run report's Open risks)" in lines
    assert (
        sum(ln.startswith("- DRIFT") for ln in lines) == build_report.PR_BODY_MAX_DRIFT
    )
    assert "Out of scope: none." in lines  # COMPUTE_API_GATED not set


def test_pr_body_drops_drift_none(tmp_path):
    lines = _pr_run(tmp_path, risks=1, drift="PERF_HEADER_GATE: PASS\nDRIFT: none\n")
    assert lines[-2:] == ["### Follow-ups", "None."]
    assert "- risk 0" in lines


def test_pr_body_flags_unwired_gated_compute_api(tmp_path):
    _pr_run(tmp_path, risks=0, drift="")
    (tmp_path / "state.json").write_text(
        json.dumps({"CYCLE": 1, "COMPUTE_API_GATED": True})
    )
    run = {"kernel": "rand", "coverage": {}}
    assert any(
        "compute-API wiring" in ln
        for ln in build_report.pr_body(run, str(tmp_path)).splitlines()
    )
