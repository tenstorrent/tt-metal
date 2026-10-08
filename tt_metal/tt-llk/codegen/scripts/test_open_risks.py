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


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text)


def test_clean_logs_pass(tmp_path):
    _write(tmp_path, "agent_tester_cycle1.md", CLEAN)
    _write(tmp_path, "agent_prettifier.md", "# p\n## Open risks\nnone\n")
    findings, closed, deferred = open_risks.check(str(tmp_path))
    assert findings == [] and (closed, deferred) == (1, 1)
    assert open_risks.deferred_lines(str(tmp_path)) == [
        "PRNG settle of 1600 cycles needs HW-owner confirmation"
    ]


def test_open_missing_and_malformed_block(tmp_path):
    _write(tmp_path, "agent_analyzer.md", "# a\n## Open risks\n- R1 OPEN: NaN untested (R1)\n- R2 CLOSED: no evidence\n")
    _write(tmp_path, "agent_optimizer.md", "# o\n## Reasoning summary\nok\n")
    findings, _, _ = open_risks.check(str(tmp_path))
    kinds = sorted(f.split()[0] for f in findings)
    assert kinds == ["MALFORMED", "MISSING", "OPEN"]


def test_waiver_word_needs_an_entry_reference(tmp_path):
    log = CLEAN + "## Open questions / handoffs\n- NaN path untested\n- seed path untested (R2)\n"
    _write(tmp_path, "agent_tester_cycle1.md", log)
    findings, _, _ = open_risks.check(str(tmp_path))
    assert len(findings) == 1 and findings[0].startswith("WAIVER agent_tester_cycle1.md:")
    assert "NaN path untested" in findings[0]


def test_waiver_in_code_fence_is_ignored(tmp_path):
    _write(tmp_path, "agent_writer_cycle1.md", "# w\n```\nconservative\n```\n## Open risks\nnone\n")
    assert open_risks.check(str(tmp_path))[0] == []


def test_only_latest_cycle_is_read(tmp_path):
    _write(tmp_path, "agent_tester_cycle1.md", "# t\n## Open risks\n- R1 OPEN: stale\n")
    _write(tmp_path, "agent_tester_cycle2.md", CLEAN)
    assert open_risks.check(str(tmp_path))[0] == []


def test_pr_body_is_short_and_carries_deferred_risks(tmp_path):
    _write(tmp_path, "agent_tester_cycle1.md", CLEAN)
    (tmp_path / "pre_pr_gates.txt").write_text("DRIFT: reference x changed on main: abc fix\n")
    run = {
        "kernel": "rand",
        "reference_file": "ref.h",
        "generated_file": "gen.h",
        "tests_total": 20,
        "tests_passed": 20,
        "test_file": "python_tests/quasar/test_rand_quasar.py",
        "coverage": {"covered": ["seed", "lockup"], "not_covered": {}},
    }
    body = build_report.pr_body(json.loads(json.dumps(run)), str(tmp_path))
    lines = body.splitlines()
    assert len(lines) <= build_report.PR_BODY_MAX_LINES
    assert "- PRNG settle of 1600 cycles needs HW-owner confirmation" in lines
    assert any("compute-API wiring" in ln for ln in lines)
    assert any(ln.startswith("- DRIFT") for ln in lines)
