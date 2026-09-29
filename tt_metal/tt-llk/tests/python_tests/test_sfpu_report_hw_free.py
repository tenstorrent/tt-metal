# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of the LLK SFPU report (``tt-llk/sfpu_report``).

Everything here runs on the host: which changed files count as device code, which
ops a run measures first, how special-input results are compared, and what the
rendered comment says for a synthetic run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfpu_report"))

import accuracy  # noqa: E402
import cli  # noqa: E402
import overlay  # noqa: E402
import report  # noqa: E402

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    "path, device",
    [
        (
            "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h",
            True,
        ),
        ("tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h", True),
        ("tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h", True),
        ("tt_metal/tt-llk/tests/sources/eltwise_unary_sfpu_test.cpp", True),
        # Host-side code never comes from the PR.
        ("tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py", False),
        ("tt_metal/tt-llk/tests/python_tests/conftest.py", False),
        (".github/workflows/llk-sfpu-report.yaml", False),
        ("tests/ttnn/unit_tests/operations/eltwise/test_unary.py", False),
    ],
)
def test_only_device_code_comes_from_the_pr(path, device):
    assert overlay.is_device_file(path) is device


def test_ops_whose_own_kernel_changed_come_first():
    ops = ["Atan", "Erf", "Gelu", "Reciprocal", "Sigmoid"]
    changed = [
        "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h"
    ]
    assert cli._prioritize(ops, changed) == [
        "Reciprocal",
        "Atan",
        "Erf",
        "Gelu",
        "Sigmoid",
    ]


def _spec(values):
    """{bits: (class, input, result)} as accuracy._specials returns it."""
    return {i: v for i, v in enumerate(values)}


def test_specials_diff_reports_changes_and_nan_propagation():
    base = _spec([("nan", NAN, NAN), ("inf", -INF, -1.0), ("zero", -0.0, -0.0)])
    head = _spec([("nan", NAN, 0.0), ("inf", -INF, NAN), ("zero", -0.0, 0.0)])
    diff = accuracy.specials_diff(base, head)
    assert [(c["class"], c["new"] != c["new"]) for c in diff["changed"]] == [
        ("nan", False),
        ("inf", True),
        ("zero", False),
    ]
    # -0 -> +0 is a change: the sign of a zero is part of the result.
    assert any(c["class"] == "zero" for c in diff["changed"])
    assert diff["nan_propagates"] == {"base": True, "head": False}


def test_specials_diff_ignores_nan_payloads():
    base = _spec([("nan", NAN, NAN)])
    head = _spec([("nan", NAN, -NAN)])
    assert accuracy.specials_diff(base, head)["changed"] == []


def _summary():
    perf_row = {
        "op": "Square",
        "formats": "Float16_b->Float16_b",
        "dest_acc": "No",
        "approx": "No",
        "fast_mode": "No",
        "text_base": 2447,
        "text_head": 2519,
        "MATH_ISOLATE": {
            "base": 191.0,
            "head": 479.0,
            "delta": 1.5,
            "regression": True,
            "improvement": False,
        },
        "L1_TO_L1": {
            "base": 207.0,
            "head": 495.0,
            "delta": 1.4,
            "regression": True,
            "improvement": False,
        },
    }
    stats = lambda mx, mean: {  # noqa: E731
        "lanes": 65279,
        "max": mx,
        "mean": mean,
        "p99": 1.0,
        "exact": 0.5,
        "le1": 0.99,
        "nonfinite": 0,
        "nonfinite_examples": [],
        "worst_input": 1.0,
    }
    return {
        "arch": "wormhole",
        "host": "host",
        "host_board": "n150",
        "run_url": None,
        "mode": "merge-base",
        "tool_sha": "a" * 40,
        "base_sha": "b" * 40,
        "head_sha": "c" * 40,
        "head_moved_to": None,
        "merge_base_age_days": 30,
        "applied": [],
        "not_applied": [],
        "ops": {
            "measured": ["Square"],
            "not_covered": [],
            "why": "requested in the command",
        },
        "iterations": 3,
        "thresholds": cli.THRESHOLDS,
        "perf": {"unary": {"loadmacro": [perf_row], "no-loadmacro": [dict(perf_row)]}},
        "accuracy": [
            {
                "key": ["Square", "Float16", "Float16", "No", "No"],
                "base": stats(1, 0.3),
                "head": stats(1024, 1.4),
                "worse": 22920,
                "better": 2010,
                "bit_identical": False,
                "head_digest": "x",
                "specials": accuracy.specials_diff(
                    _spec([("inf", INF, INF)]), _spec([("inf", INF, NAN)])
                ),
            }
        ],
        "notes": [],
        "commands": ["python3 cli.py ..."],
    }


def test_report_flags_the_regressions():
    text = report.render([_summary()])
    assert text.startswith(report.COMMENT_MARKER)
    assert report.AI_SUMMARY_MARKER in text
    assert "⚠️ 479" in text  # perf
    assert "⚠️ 1,024" in text  # accuracy
    assert "⚠️ `nan`" in text  # edge case changed kind
    assert "0.40x" in text
    assert "merge-base is 30 days old" in text


def test_report_stays_under_the_comment_limit():
    summary = _summary()
    summary["accuracy"] = summary["accuracy"] * 400
    assert len(report.render([summary])) < 65536
