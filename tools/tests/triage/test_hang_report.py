# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Run the hang app (which dispatches tt-triage on timeout). If JUnit or triage text reports a hang,
the test fails on purpose so CI stays red with real hang logs for triage. If no hang is reported,
the test passes (run completed without a detected hang).
"""

import os
import subprocess
from pathlib import Path

import pytest
from xml.etree.ElementTree import parse as XMLParse

if not os.environ.get("TT_METAL_HOME"):
    pytest.skip("TT_METAL_HOME not set; skipping hang report tests", allow_module_level=True)

METAL_HOME = Path(os.environ.get("TT_METAL_HOME"))
HANG_APP = METAL_HOME / "build/tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang"
TRIAGE_SCRIPT = METAL_HOME / "tools/tt-triage.py"
HANG_REPORT_SCRIPT = METAL_HOME / ".github/scripts/utils/hang_report.py"
REPORT_DIR = METAL_HOME / "generated/test_reports"

FAKE_TEST_ID = "fake/test_hang.py::test_hung_operation[device0]"


def _hang_reported_in_junit(xml_path: Path) -> bool:
    try:
        testcase = XMLParse(str(xml_path)).getroot().find(".//testcase")
        if testcase is None:
            return False
        if testcase.get("classname") != "fake.test_hang":
            return False
        if testcase.get("name") != "test_hung_operation[device0]":
            return False
        props = {p.get("name"): p.get("value") for p in testcase.findall("properties/property")}
        if props.get("failure_type") != "hang":
            return False
        failure = testcase.find("failure")
        if failure is None:
            return False
        msg = failure.get("message") or ""
        body = failure.text or ""
        return "[HANG DETECTED]" in msg and "Triage summary" in body
    except Exception:
        return False


@pytest.mark.skipif(not HANG_APP.exists(), reason="Hang app binary not built")
@pytest.mark.skipif(not TRIAGE_SCRIPT.exists(), reason="tt-triage.py not found")
@pytest.mark.skipif(not HANG_REPORT_SCRIPT.exists(), reason="hang_report.py not found")
def test_hang_triggers_and_fails_when_hang_reported():
    existing = set(REPORT_DIR.glob("hang_report_*.xml")) if REPORT_DIR.exists() else set()

    triage_output_path = METAL_HOME / "generated/triage_output.txt"
    if triage_output_path.exists():
        triage_output_path.unlink()

    hang_report_cmd = f"python3 {HANG_REPORT_SCRIPT}"
    triage_cmd = f"python3 {TRIAGE_SCRIPT} --disable-progress --skip-version-check --triage-summary-path=generated/triage_summary.txt"
    dispatch_cmd = f"{hang_report_cmd}; mkdir -p generated; {triage_cmd} 2>&1 | tee {triage_output_path} 1>&2; {hang_report_cmd} --update"

    result = subprocess.run(
        [str(HANG_APP)],
        cwd=str(METAL_HOME),
        capture_output=True,
        timeout=90,
        env={
            **os.environ,
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
            "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": dispatch_cmd,
            "PYTEST_CURRENT_TEST": f"{FAKE_TEST_ID} (call)",
        },
    )

    print(result.stdout.decode(errors="replace"))
    print(result.stderr.decode(errors="replace"))

    assert result.returncode == 0, f"Hang app failed (rc={result.returncode})"

    new_reports = sorted(set(REPORT_DIR.glob("hang_report_*.xml")) - existing)
    triage_text = ""
    if triage_output_path.exists():
        triage_text = triage_output_path.read_text(errors="replace")

    hang_in_junit = bool(new_reports) and _hang_reported_in_junit(new_reports[0])
    hang_in_triage_log = "[HANG DETECTED]" in triage_text or "Card hang detected" in triage_text

    if hang_in_junit or hang_in_triage_log:
        pytest.fail(
            "Hang was reported (JUnit and/or triage output). "
            "Treating as test failure so the workflow job fails for triage."
        )
