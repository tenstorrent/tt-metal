"""Tests for eval.classify_failures — pure Python, no device needed."""

import json
import tempfile
from pathlib import Path

import pytest

from eval.classify_failures import classify, extract_shape, parse_junit_xml


# --- classify() ---


class TestClassify:
    def test_hang_operation_timeout(self):
        assert classify("RuntimeError: Operation timeout after 5s") == "hang"

    def test_hang_dispatch_timeout(self):
        assert classify("dispatch timeout fired on device 0") == "hang"

    def test_hang_tt_metal_env(self):
        assert classify("TT_METAL_OPERATION_TIMEOUT fired") == "hang"

    def test_oom_out_of_memory(self):
        assert classify("RuntimeError: Out of Memory: Not enough space") == "OOM"

    def test_oom_l1_allocation(self):
        assert classify("L1 allocation failed for buffer of size 2MB") == "OOM"

    def test_oom_circular_buffers(self):
        assert classify("Statically allocated circular buffers exceed L1") == "OOM"

    def test_compilation_error(self):
        assert classify("CompilationError: kernel build failed") == "compilation"

    def test_compilation_linking(self):
        assert classify("Error: linking failed for compute kernel") == "compilation"

    def test_numerical_allclose(self):
        assert classify("AssertionError: torch.allclose(a, b) is False") == "numerical"

    def test_numerical_mismatch(self):
        assert classify("Numerical Mismatch: max_diff=0.5, atol=0.1") == "numerical"

    def test_numerical_pcc(self):
        assert classify("PCC comparison failed: 0.89 < 0.99") == "numerical"

    def test_other_unknown(self):
        assert classify("ValueError: something completely different") == "other"

    def test_empty_string(self):
        assert classify("") == "other"

    def test_priority_hang_over_numerical(self):
        """Hang patterns should win over numerical if both present."""
        text = "Operation timeout after 5s\nallclose failed"
        assert classify(text) == "hang"

    def test_priority_oom_over_numerical(self):
        text = "Out of Memory while checking allclose"
        assert classify(text) == "OOM"


# --- extract_shape() ---


class TestExtractShape:
    def test_brackets(self):
        assert extract_shape("test_foo[32x32]") == "32x32"

    def test_parametrized_id(self):
        assert extract_shape("test_layer_norm_with_affine[minimal_1x1x32x32]") == "minimal_1x1x32x32"

    def test_complex_id(self):
        assert extract_shape("test_foo[b2c3_32x32]") == "b2c3_32x32"

    def test_no_brackets(self):
        assert extract_shape("test_foo") is None

    def test_empty(self):
        assert extract_shape("") is None

    def test_nested_brackets(self):
        assert extract_shape("test_foo[eps_1e-5_default-32x128]") == "eps_1e-5_default-32x128"


# --- parse_junit_xml() ---

SAMPLE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4">
    <testcase classname="eval.golden_tests.ln.test_shapes" name="test_ln[32x32]" time="1.0">
    </testcase>
    <testcase classname="eval.golden_tests.ln.test_shapes" name="test_ln[128x4096]" time="0.8">
      <failure message="Numerical Mismatch: max_diff=0.5">
        AssertionError: Numerical Mismatch: max_diff=0.5, mean_diff=0.1, atol=0.1
      </failure>
    </testcase>
    <testcase classname="eval.golden_tests.ln.test_shapes" name="test_ln[8192x8192]" time="0.3">
      <error message="Out of Memory">
        RuntimeError: Out of Memory: L1 allocation failed
      </error>
    </testcase>
    <testcase classname="eval.golden_tests.ln.test_shapes" name="test_ln[1024x1024]" time="0.0">
      <skipped message="Skipped: previous parametrization of test_ln hung" />
    </testcase>
  </testsuite>
</testsuites>
"""


class TestParseJunitXml:
    @pytest.fixture
    def xml_path(self, tmp_path):
        p = tmp_path / "results.xml"
        p.write_text(SAMPLE_XML)
        return p

    def test_correct_count(self, xml_path):
        results = parse_junit_xml(xml_path)
        assert len(results) == 4

    def test_passed(self, xml_path):
        results = parse_junit_xml(xml_path)
        passed = [r for r in results if r["status"] == "passed"]
        assert len(passed) == 1
        assert passed[0]["test_name"] == "test_ln[32x32]"
        assert passed[0]["shape"] == "32x32"
        assert passed[0]["failure_category"] is None

    def test_failure_classified(self, xml_path):
        results = parse_junit_xml(xml_path)
        failed = [r for r in results if r["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["failure_category"] == "numerical"
        assert failed[0]["shape"] == "128x4096"

    def test_error_classified(self, xml_path):
        results = parse_junit_xml(xml_path)
        errors = [r for r in results if r["status"] == "error"]
        assert len(errors) == 1
        assert errors[0]["failure_category"] == "OOM"

    def test_skipped_hang(self, xml_path):
        results = parse_junit_xml(xml_path)
        skipped = [r for r in results if r["status"] == "skipped"]
        assert len(skipped) == 1
        assert skipped[0]["failure_category"] == "hang"

    def test_test_file_extracted(self, xml_path):
        results = parse_junit_xml(xml_path)
        assert results[0]["test_file"] == "eval.golden_tests.ln"

    def test_message_truncated(self, xml_path):
        results = parse_junit_xml(xml_path)
        for r in results:
            if r["failure_message"]:
                assert len(r["failure_message"]) <= 500


# --- Edge cases ---

EMPTY_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="0">
  </testsuite>
</testsuites>
"""


def test_empty_xml(tmp_path):
    p = tmp_path / "empty.xml"
    p.write_text(EMPTY_XML)
    results = parse_junit_xml(p)
    assert results == []


HANG_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="1">
    <testcase classname="test_mod" name="test_op[big]" time="5.0">
      <failure message="RuntimeError: Operation timeout after 5s">
        RuntimeError: Operation timeout after 5s
        ... dispatch stack trace ...
      </failure>
    </testcase>
  </testsuite>
</testsuites>
"""


def test_hang_detection(tmp_path):
    p = tmp_path / "hang.xml"
    p.write_text(HANG_XML)
    results = parse_junit_xml(p)
    assert len(results) == 1
    assert results[0]["failure_category"] == "hang"
