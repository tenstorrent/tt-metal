# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for ttsim error classification.

No kernel and no device. A forked ttsim ``_Exit`` is reported as
``CRASHED with signal 0`` until the error line is parsed; these tests pin
that rewrite and the HTML report's reading of the same line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from helpers.llk_pytest_plugin import pytest_runtest_logreport
from ttsim_error import parse_ttsim_error

_RENDERER_PATH = Path(__file__).resolve().parents[1] / "render_ttsim_report.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_ttsim_report", _RENDERER_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the class module up in sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "[8924] ERROR: UnimplementedFunctionality: tensix_sfpshft2: instr_mod1=4",
            ("UnimplementedFunctionality", "tensix_sfpshft2", "instr_mod1=4"),
        ),
        (
            "[7848] ERROR: UnimplementedFunctionality: tensix_sfplut",
            ("UnimplementedFunctionality", "tensix_sfplut", None),
        ),
        (
            "[7848] ERROR: UnimplementedFunctionality: tensix_sfplut   ",
            ("UnimplementedFunctionality", "tensix_sfplut", None),
        ),
        (
            "[1] ERROR: UnimplementedFunctionality: tensix_elw_op src_fmt=10",
            ("UnimplementedFunctionality", "tensix_elw_op", "src_fmt=10"),
        ),
        (
            "[1] ERROR: UnimplementedFunctionality: tensix_cfg_wr32 reg=281",
            ("UnimplementedFunctionality", "tensix_cfg_wr32", "reg=281"),
        ),
        (
            "[12] ERROR: UnimplementedFunctionality: qsr_unpack_src_value: in_format=5 out_format=4",
            (
                "UnimplementedFunctionality",
                "qsr_unpack_src_value",
                "in_format=5 out_format=4",
            ),
        ),
        (
            "noise\n[9] ERROR: Cat: func: detail with: colon\nmore\n",
            ("Cat", "func", "detail with: colon"),
        ),
        ("[1] ERROR: UnimplementedFunctionality", None),
        ("not an error", None),
        ("", None),
    ],
)
def test_parse_ttsim_error(text, expected):
    assert parse_ttsim_error(text) == expected


class _Report:
    """Enough of a pytest report for ``pytest_runtest_logreport`` to rewrite."""

    def __init__(self, stdout: str):
        self.outcome = "failed"
        self.longrepr = ":-1: running the test CRASHED with signal 0"
        self.when = "???"
        self.sections = [("captured stdout", stdout)]
        self.user_properties = []

    @property
    def capstdout(self) -> str:
        return "".join(
            content
            for name, content in self.sections
            if name.startswith("Captured stdout")
        )


def test_forked_crash_without_detail_is_a_failure():
    report = _Report("[7848] ERROR: UnimplementedFunctionality: tensix_sfplut\n")
    pytest_runtest_logreport(report)
    assert report.when == "call"
    assert report.longrepr == "[ttsim:UnimplementedFunctionality] tensix_sfplut"
    assert ("ttsim_category", "UnimplementedFunctionality") in report.user_properties
    assert ("ttsim_func", "tensix_sfplut") in report.user_properties
    assert report.sections[0][0] == "Captured stdout call"


def test_forked_crash_with_detail_keeps_the_detail():
    report = _Report(
        "[8924] ERROR: UnimplementedFunctionality: tensix_sfpshft2: instr_mod1=4\n"
    )
    pytest_runtest_logreport(report)
    assert (
        report.longrepr
        == "[ttsim:UnimplementedFunctionality] tensix_sfpshft2: instr_mod1=4"
    )


def test_forked_crash_with_unparsed_output_stays_a_crash():
    report = _Report("simulator exited without an ERROR line\n")
    pytest_runtest_logreport(report)
    assert report.when == "call"
    assert "CRASHED with signal" in str(report.longrepr)
    assert report.user_properties == []


def test_report_parses_optional_detail(tmp_path):
    renderer = _load_renderer()
    xml_path = tmp_path / "ttsim.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="3" failures="3" errors="0" skipped="0" time="1">
    <testcase classname="quasar.test_fused_quasar" name="no_detail" time="0.1">
      <failure message="crashed">failed</failure>
      <system-out>[7848] ERROR: UnimplementedFunctionality: tensix_sfplut</system-out>
    </testcase>
    <testcase classname="quasar.test_fused_quasar" name="with_detail" time="0.1">
      <failure message="crashed">failed</failure>
      <system-out>[8924] ERROR: UnimplementedFunctionality: tensix_sfpshft2: instr_mod1=4</system-out>
    </testcase>
    <testcase classname="quasar.test_fused_quasar" name="space_detail" time="0.1">
      <failure message="crashed">failed</failure>
      <system-out>[1] ERROR: UnimplementedFunctionality: tensix_elw_op src_fmt=10</system-out>
    </testcase>
  </testsuite>
</testsuites>
"""
    )
    _files, cases, meta = renderer.parse(xml_path)
    by_name = {case.name: case for case in cases}
    assert by_name["no_detail"].ttsim_category == "UnimplementedFunctionality"
    assert by_name["no_detail"].ttsim_func == "tensix_sfplut"
    assert by_name["no_detail"].ttsim_detail is None
    assert by_name["with_detail"].ttsim_detail == "instr_mod1=4"
    assert by_name["space_detail"].ttsim_func == "tensix_elw_op"
    assert by_name["space_detail"].ttsim_detail == "src_fmt=10"

    html = renderer.render(_files, cases, xml_path, meta)
    # The category is a pill and the function (plus optional detail) is the code body.
    assert (
        "<span class='pill bad'>UnimplementedFunctionality</span> <code>tensix_sfplut</code>"
        in html
    )
    assert "<code>tensix_sfplut</code>:" not in html
    assert "<code>tensix_sfpshft2: instr_mod1=4</code>" in html
    assert "<code>tensix_elw_op: src_fmt=10</code>" in html
    assert renderer.count_nonpassing(1, 1, 1) == 3
