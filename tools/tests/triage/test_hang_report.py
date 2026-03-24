# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test: trigger a card hang, verify tt-triage writes a JUnit XML report.

Requires a built hang app binary and a Tenstorrent device.
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


@pytest.mark.skipif(not HANG_APP.exists(), reason="Hang app binary not built")
@pytest.mark.skipif(not TRIAGE_SCRIPT.exists(), reason="tt-triage.py not found")
@pytest.mark.skipif(not HANG_REPORT_SCRIPT.exists(), reason="hang_report.py not found")
def test_hang_generates_junit_xml():
    existing = set(REPORT_DIR.glob("hang_report_*.xml")) if REPORT_DIR.exists() else set()

    triage_output_path = METAL_HOME / "generated/triage_output.txt"
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

    # It is useful to see the tt-triage output to confirm correct test execution.
    print(result.stdout.decode(errors="replace"))
    print(result.stderr.decode(errors="replace"))
    assert result.returncode == 0, f"Hang app failed (rc={result.returncode})"

    new_reports = sorted(set(REPORT_DIR.glob("hang_report_*.xml")) - existing)
    assert new_reports, f"No hang report XML generated in {REPORT_DIR}"
    xml_path = new_reports[0]

    testcase = XMLParse(str(xml_path)).getroot().find(".//testcase")
    assert testcase is not None

    assert testcase.get("classname") == "fake.test_hang"
    assert testcase.get("name") == "test_hung_operation[device0]"

    props = {p.get("name"): p.get("value") for p in testcase.findall("properties/property")}
    assert props["failure_type"] == "hang"

    failure = testcase.find("failure")
    assert "[HANG DETECTED]" in (failure.get("message") or "")
    assert "Triage summary" in (failure.text or "")

    assert triage_output_path.exists(), f"Triage output file not created at {triage_output_path}"
    triage_output_content = triage_output_path.read_text()
    assert len(triage_output_content) > 0, "Triage output file is empty"
