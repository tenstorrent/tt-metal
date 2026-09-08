# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expose existing C++ reduction tests to pytest and run_safe_pytest's device guard."""

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_generate_tests(metafunc):
    if "cpp_case" not in metafunc.fixturenames:
        return
    manifest = json.loads(Path(os.environ["TT_REDUCE_SUITE_MANIFEST"]).read_text())
    group = next(group for group in manifest["groups"] if group["id"] == os.environ["TT_REDUCE_SUITE_GROUP"])
    enabled = os.environ.get("TT_REDUCE_SUITE_DISABLED_GTESTS") == "1"
    cases = []
    for name in group["tests"]:
        marks = (
            [] if enabled or "DISABLED_" not in name else [pytest.mark.skip(reason="Disabled in upstream C++ source")]
        )
        cases.append(pytest.param((group["binary"], name), id=name, marks=marks))
    metafunc.parametrize("cpp_case", cases)


def test_cpp_factory(cpp_case, tmp_path):
    binary, name = cpp_case
    executable = ROOT / binary
    assert executable.is_file(), f"Missing {executable}; build the TTNN test target before running C++ cases"
    xml = tmp_path / "gtest.xml"
    command = [str(executable), f"--gtest_filter={name}", f"--gtest_output=xml:{xml}"]
    if "DISABLED_" in name:
        command.append("--gtest_also_run_disabled_tests")
    # Do not leave a device-using child behind if pytest's timeout or Ctrl-C interrupts the adapter.
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        returncode = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    assert returncode == 0, f"{name} exited with status {returncode}; see captured gtest output"
    assert xml.is_file(), f"{name} produced no gtest XML result"
    cases = ET.parse(xml).getroot().findall(".//testcase")
    assert len(cases) == 1, f"Expected exactly {name}; found {len(cases)} cases (binary may be stale)"
    case = cases[0]
    assert f"{case.attrib['classname']}.{case.attrib['name']}" == name
    assert case.find("failure") is None, ET.tostring(case, encoding="unicode")
    skipped = case.find("skipped")
    if skipped is not None:
        pytest.skip(skipped.attrib.get("message", "Skipped by gtest"))
    assert case.attrib.get("status") == "run", f"{name} did not run"
