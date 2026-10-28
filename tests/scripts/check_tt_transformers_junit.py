#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail unless the JUnit report of a tt-transformers run has a passed test.

usage: tests/scripts/check_tt_transformers_junit.py <report prefix>

run_tt_transformers_tests.sh passes pytest --junitxml <report prefix>.xml. Under CI=true,
tt-transformers' conftest renames that report to <report prefix>_<YYYYmmdd_HHMMSS>.xml, so
the report is found by its prefix. The prefix is unique per run: exactly one report must match.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET


def main(prefix):
    directory, stem = os.path.split(prefix)
    name = re.compile(re.escape(stem) + r"(_\d{8}_\d{6})?\.xml")
    reports = sorted(os.path.join(directory, f) for f in os.listdir(directory or ".") if name.fullmatch(f))
    if len(reports) != 1:
        return f"Expected exactly one JUnit report {prefix}[_<timestamp>].xml, found {len(reports)}: {reports}"
    report = reports[0]
    cases = ET.parse(report).iter("testcase")
    if not any(all(case.find(tag) is None for tag in ("skipped", "failure", "error")) for case in cases):
        return f"No tt-transformers test passed (report: {report})"
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
