# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Parser snapshot from tt_ops_code_gen eval/classify_failures.py at
# 034527ad845a7b61596139802c4af567983a14bb. Owned here to keep orchestration
# independent of the evaluator package; recorded runtime plugins stay pinned.

"""Classify pytest failures from JUnit XML into categories.

Categories (checked in priority order):
  hang                - operation timeout / dispatch timeout
  OOM                 - L1 or DRAM allocation failure
  compilation         - kernel build or link failure
  signature           - wrong function signature / missing parameters / import errors
  validation          - op's validate() raised NotImplementedError on a cell
                        that was supposed to be supported (real over-claim bug;
                        xfail-strict cells with NotImplementedError are not
                        recorded as failures in JUnit, so this only triggers
                        on genuine SUPPORTED over-claims)
  numerical-bug       - catastrophic numerical failure: severity=bug in
                        CheckOutputError (PCC <= 0.9, Inf/NaN, or >3x rtol/atol)
  numerical-precision - off but not catastrophic: severity=precision in
                        CheckOutputError (PCC > 0.9, within 3x rtol/atol band)
  numerical           - allclose/PCC mismatch without explicit severity tag
                        (backwards-compat fallback for old-style check_output
                        messages or third-party assertions)
  other               - anything else
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# Classification patterns, checked in priority order (first match wins)
PATTERNS = [
    (
        "hang",
        [
            # tt-metal's actual dispatch-timeout error (system_memory_manager.cpp:
            # 669, 723). These are the high-signal substrings that always appear
            # in real device-hang tracebacks.
            r"device timeout",
            r"potential hang detected",
            # Legacy / speculative patterns — kept as a safety net for older
            # tt-metal versions and host-side timeouts (e.g. descriptor file).
            r"[Oo]peration timeout",
            r"Operation timed out",
            r"TT_METAL_OPERATION_TIMEOUT",
            r"Timeout waiting for",
            r"[Dd]ispatch timeout",
            # craq-sim hang-watchdog firing — same failure mode, same bucket.
            r"hang watchdog fired",
        ],
    ),
    (
        "OOM",
        [
            r"Out of Memory",
            r"out of memory",
            r"Statically allocated circular buffers",
            r"L1 allocation",
            r"not enough space",
            r"DRAM allocation",
            r"\bOOM\b",
            r"Cannot allocate",
        ],
    ),
    (
        "compilation",
        [
            r"CompilationError",
            r"compilation error",
            r"kernel build failed",
            r"CQ Compile",
            r"linking failed",
            r"compile_program_with_kernel",
        ],
    ),
    (
        "signature",
        [
            r"TypeError:.*takes \d+ positional argument",
            r"TypeError:.*missing \d+ required",
            r"TypeError:.*got an unexpected keyword argument",
            r"TypeError:.*got multiple values for argument",
            r"ImportError: cannot import name",
            r"ModuleNotFoundError: No module named",
            r"AttributeError:.*has no attribute",
        ],
    ),
    (
        "validation",
        [
            # validate() rejected an input it shouldn't have (over-claim).
            # xfail-strict catches expected NotImplementedError as xfail
            # (not a failure), so any NotImplementedError reaching JUnit
            # failure output is a real bug: SUPPORTED claims it works.
            r"NotImplementedError",
        ],
    ),
    (
        "numerical-bug",
        [
            # CheckOutputError stamps "severity=bug" into its message for
            # catastrophic cases — see eval/golden_tests/<op>/helpers.py.
            r"severity=bug\b",
        ],
    ),
    (
        "numerical-precision",
        [
            # CheckOutputError stamps "severity=precision" for cases that
            # missed nominal tolerance but stayed within the 3x band.
            r"severity=precision\b",
        ],
    ),
    (
        "numerical",
        [
            r"allclose",
            r"\bPCC\b",
            r"[Nn]umerical [Mm]ismatch",
            r"max_diff=",
            r"mean_diff=",
            r"atol=",
            r"rtol=",
        ],
    ),
]


def classify(traceback_text: str) -> str:
    """Classify a failure traceback into a category."""
    for category, patterns in PATTERNS:
        for pattern in patterns:
            if re.search(pattern, traceback_text):
                return category
    return "other"


def _reconstruct_nodeid(classname: str, name: str) -> str:
    """Reconstruct a pytest path-based nodeid from JUnit XML's classname.

    JUnit emits classname like "eval.golden_tests.softmax.test_golden";
    pytest's nodeid uses "eval/golden_tests/softmax/test_golden.py".
    The axes plugin sidecar uses pytest's form, so we convert here so
    verify_supported can join the two cleanly.
    """
    if not classname:
        return f"::{name}"
    return classname.replace(".", "/") + ".py::" + name


def extract_shape(test_name: str) -> Optional[str]:
    """Extract shape info from a parametrized test name.

    Examples:
        "test_foo[minimal_1x1x32x32]" -> "minimal_1x1x32x32"
        "test_foo[w512]" -> "w512"
        "test_foo[b2c3_32x32]" -> "b2c3_32x32"
        "test_op[1x1x32x64-alignment=tile_aligned-dtype=FLOAT32-...]" -> "1x1x32x64"

    Registry golden case ids are "{shape}-{axis=val-axis=val...}". Strip the
    axes signature (everything from the first "-<name>=") so SHAPE shows just
    the shape — the axes already appear as per-row chips. This is the fallback
    for rows with no recorded shape tag (xfail/skipped golden, regression);
    rows that record a tag (translated, passed golden) override it upstream.
    Ids without an axis signature are returned unchanged.
    """
    match = re.search(r"\[(.+)\]$", test_name)
    if not match:
        return None
    return re.split(r"-[A-Za-z_][\w]*=", match.group(1), maxsplit=1)[0]


_METRIC_KEYS = ("pcc", "rms", "max_abs_diff", "median_abs_diff", "ulp_p99", "device_kernel_ns", "device_num_cores")


def _extract_metrics(tc: ET.Element) -> dict:
    """Pull `metric.*` entries out of a testcase's <properties> block.

    Emitted by `eval/metrics_plugin.py` via `record_property`. Returns a
    dict keyed by metric name (pcc, rms, ...) with float values, or
    {} if no metrics were recorded (xfail/skip, or pre-plugin runs).
    """
    metrics = {}
    properties = tc.find("properties")
    if properties is None:
        return metrics
    for prop in properties.iter("property"):
        name = prop.get("name", "")
        if not name.startswith("metric."):
            continue
        key = name[len("metric.") :]
        if key not in _METRIC_KEYS:
            continue
        raw = prop.get("value", "")
        try:
            metrics[key] = float(raw)
        except ValueError:
            # Unexpected string for a numeric metric — skip.
            continue
    return metrics


def _extract_tag(tc: ET.Element):
    """Pull the `tag` property (set via metrics_plugin.record(tag=...)) out
    of a testcase's <properties>, or None if absent."""
    properties = tc.find("properties")
    if properties is None:
        return None
    for prop in properties.iter("property"):
        if prop.get("name", "") == "tag":
            return prop.get("value", "") or None
    return None


def _extract_observed_axes(tc: ET.Element) -> dict:
    """Pull `axis.*` entries out of a testcase's <properties> block.

    Emitted by `eval/metrics_plugin.py::record_axes` (the observe-only tagging
    path). Each value is JSON of the same serialization the declared axes
    sidecar uses, so `json.loads` recovers the identical type. Returns
    `{axis_name: value}`, or `{}` if the test recorded none.
    """
    observed = {}
    properties = tc.find("properties")
    if properties is None:
        return observed
    for prop in properties.iter("property"):
        name = prop.get("name", "")
        if not name.startswith("axis."):
            continue
        raw = prop.get("value", "")
        try:
            observed[name[len("axis.") :]] = json.loads(raw)
        except (ValueError, TypeError):
            observed[name[len("axis.") :]] = raw  # tolerate non-JSON rather than crash
    return observed


def _empty_metrics() -> dict:
    """Placeholder when no measurement happened (xfail/skip)."""
    return {k: None for k in _METRIC_KEYS}


def parse_junit_xml(xml_path: Path) -> list:
    """Parse a JUnit XML file and classify each test result.

    Returns a list of dicts with keys:
        test_name, test_file, shape, status, failure_category,
        failure_message, plus accuracy metrics
        (pcc, rms, max_abs_diff, median_abs_diff, ulp_p99) when the
        `eval.metrics_plugin` recorded any — None otherwise.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")

        # Keep the full classname so the test module is preserved
        # (e.g. "eval.golden_tests.rms_norm.test_translated" vs
        # ".test_golden" vs ".test_regression"). The trailing module is the
        # only positive signal that distinguishes translated / golden /
        # regression rows downstream (dashboard SOURCE tagging); collapsing
        # it here erased that. Matches the format central-PG runs already use.
        test_file = classname

        shape = extract_shape(name)
        tag = _extract_tag(tc)
        if tag:
            shape = tag
        recorded = _extract_metrics(tc)
        metrics = {k: recorded.get(k) for k in _METRIC_KEYS}

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")

        if failure is not None:
            message = failure.get("message", "")
            traceback = failure.text or ""
            full_text = f"{message}\n{traceback}"
            # Strict xpass: pytest emits a <failure> with "[XPASS(strict)]"
            # in the message. This is a real signal (SUPPORTED under-claims)
            # and deserves its own status, not "failed".
            if "XPASS" in message or "XPASS" in full_text:
                results.append(
                    {
                        "test_name": name,
                        "test_file": test_file,
                        "shape": shape,
                        "nodeid": _reconstruct_nodeid(tc.get("classname", ""), name),
                        "status": "xpass",
                        "failure_category": None,
                        "failure_message": full_text[:2000],
                        **metrics,
                    }
                )
            else:
                category = classify(full_text)
                results.append(
                    {
                        "test_name": name,
                        "test_file": test_file,
                        "shape": shape,
                        "nodeid": _reconstruct_nodeid(tc.get("classname", ""), name),
                        "status": "failed",
                        "failure_category": category,
                        "failure_message": full_text[:2000],
                        **metrics,
                    }
                )
        elif error is not None:
            message = error.get("message", "")
            traceback = error.text or ""
            full_text = f"{message}\n{traceback}"
            category = classify(full_text)
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "nodeid": _reconstruct_nodeid(tc.get("classname", ""), name),
                    "status": "error",
                    "failure_category": category,
                    "failure_message": full_text[:2000],
                    **metrics,
                }
            )
        elif skipped is not None:
            message = skipped.get("message", "")
            skip_type = skipped.get("type", "") or ""
            # pytest emits xfail expected-failures as <skipped type="pytest.xfail">.
            # Distinguish that from a real pytest.mark.skip — the registry
            # model treats xfail (op rejected an unsupported cell) very
            # differently from skip (cell is INVALID, never going to work).
            if "xfail" in skip_type.lower() or "xfail" in message.lower():
                status = "xfail"
                category = None
            else:
                status = "skipped"
                # Setup-phase (S1/S2) OOM skips are tagged INFEASIBLE_L1 by
                # eval.oom — surface that as its own category for visibility.
                if "hung" in message.lower():
                    category = "hang"
                elif "INFEASIBLE_L1" in message:
                    category = "infeasible"
                else:
                    category = None
            # xfail / skip never invoked the op — metrics are always NULL.
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "nodeid": _reconstruct_nodeid(tc.get("classname", ""), name),
                    "status": status,
                    "failure_category": category,
                    "failure_message": message[:2000] if message else None,
                    **_empty_metrics(),
                }
            )
        else:
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "nodeid": _reconstruct_nodeid(tc.get("classname", ""), name),
                    "status": "passed",
                    "failure_category": None,
                    "failure_message": None,
                    **metrics,
                }
            )

        # Attach runtime-captured axes (axis.* props from record_axes) to the
        # row just appended. {} when the test recorded none (xfail/skip that
        # never ran, or a suite off the observe path) — merge_axes treats {}
        # as "no capture", so declared axes still win.
        results[-1]["observed_axes"] = _extract_observed_axes(tc)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classify pytest failures from JUnit XML")
    parser.add_argument("xml_path", help="Path to JUnit XML file")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results = parse_junit_xml(Path(args.xml_path))

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
