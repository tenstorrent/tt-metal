# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Generate a JUnit XML test report for a hung test so it appears in CI artifacts.

When a dispatch timeout occurs, this script writes a JUnit XML file that marks
the currently-running pytest test as failed with failure_type=hang.  This
guarantees the hung test appears in test reports even if the process is later
killed and pytest never finalises its own XML.

Invoked from the CI dispatch-timeout command in two phases:

  Phase 1 (before triage):
      python3 hang_report.py
    Writes an initial report with no summary so a record exists even if
    tt-triage itself crashes or is interrupted.

  Phase 2 (after triage):
      python3 hang_report.py --update
    Reads the triage summary from ``generated/triage_summary.txt`` (written
    by tt-triage) and overwrites the report with the full summary.

Both phases derive the report filename from PYTEST_CURRENT_TEST so they
target the same file regardless of PID.

Callers with no pytest process pass --test-id for the identity and --report-dir for the
location. Concurrent writers of one identity collapse onto a single report.
"""

import argparse
import hashlib
import os
import tempfile
import sys
from datetime import datetime, timezone
from html import escape
from textwrap import dedent

REPORT_DIR = "generated/test_reports"
TRIAGE_SUMMARY_PATH = "generated/triage_summary.txt"


def _report_path_for_test(test_id: str, report_dir: str = REPORT_DIR) -> str:
    name_hash = hashlib.sha256(test_id.encode()).hexdigest()[:16]
    return os.path.join(report_dir, f"hang_report_{name_hash}.xml")


def resolve_test_id(fallback: str | None = None) -> str | None:
    """PYTEST_CURRENT_TEST when running under pytest, else the fallback identity."""
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if current_test:
        return current_test.rsplit(" (", 1)[0]
    return fallback or None


def write_hang_junit_xml(
    triage_summary: str = "", test_id: str | None = None, report_dir: str = REPORT_DIR
) -> str | None:
    """Write (or overwrite) a JUnit XML report for the hung test.

    Returns the report path, or ``None`` if no test identity is available.
    """
    test_id = resolve_test_id(test_id)
    if not test_id:
        return None

    filepath, sep, test_name = test_id.partition("::")
    if not sep:
        test_name = "unknown"

    classname = filepath.replace("/", ".").removesuffix(".py")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hostname = os.environ.get("HOSTNAME", "unknown")

    failure_message = "[HANG DETECTED] Card hang detected during test execution. tt-triage was invoked."

    xml_content = dedent(
        """\
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites name="pytest tests">
          <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="1" time="0" timestamp="{timestamp}" hostname="{hostname}">
            <testcase classname="{classname}" name="{test_name}" time="0">
              <properties>
                <property name="failure_type" value="hang"/>
                <property name="start_timestamp" value="{timestamp}"/>
                <property name="end_timestamp" value="{timestamp}"/>
              </properties>
              <failure message="{failure_message}">{failure_message}

        Triage summary:
        {triage_summary}
              </failure>
            </testcase>
          </testsuite>
        </testsuites>"""
    ).format(
        timestamp=timestamp,
        hostname=escape(hostname),
        classname=escape(classname),
        test_name=escape(test_name),
        failure_message=escape(failure_message),
        triage_summary=escape(triage_summary),
    )

    report_path = _report_path_for_test(test_id, report_dir)
    try:
        os.makedirs(report_dir, exist_ok=True)
        # Writers can race on this path; rename so it is never half-written.
        fd, tmp_path = tempfile.mkstemp(dir=report_dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(xml_content)
        # mkstemp is 0600; on shared storage the reader can be another user.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, report_path)
        print(f"[INFO] Hang report written to {report_path}", file=sys.stderr)
        return report_path
    except Exception as e:
        print(f"[WARN] Failed to write hang JUnit XML report: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a JUnit XML hang report for CI.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Read the triage summary and overwrite the report.",
    )
    parser.add_argument(
        "--summary-path",
        default=TRIAGE_SUMMARY_PATH,
        help=f"Triage summary --update reads (default: {TRIAGE_SUMMARY_PATH}).",
    )
    parser.add_argument(
        "--test-id",
        default=None,
        help="Identity as 'file::name', used when PYTEST_CURRENT_TEST is unset.",
    )
    parser.add_argument(
        "--report-dir",
        default=REPORT_DIR,
        help=f"Directory to write the report into (default: {REPORT_DIR}).",
    )
    args = parser.parse_args()

    summary = ""
    if args.update and os.path.isfile(args.summary_path):
        with open(args.summary_path) as f:
            summary = f.read()

    write_hang_junit_xml(summary, test_id=args.test_id, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
