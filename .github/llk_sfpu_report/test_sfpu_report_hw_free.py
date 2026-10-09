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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accuracy  # noqa: E402
import cli  # noqa: E402
import detect  # noqa: E402
import overlay  # noqa: E402
import perf  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

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
    changed = ["tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h"]
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
                "specials": accuracy.specials_diff(_spec([("inf", INF, INF)]), _spec([("inf", INF, NAN)])),
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


def test_binary_ops_are_their_own_family():
    assert cli._family_of("Tanh") == "unary"
    assert cli._family_of("Typecast") == "typecast"
    assert cli._family_of("SfpuLogsigmoid") == "binary"
    assert cli._family_of("SfpuDivInt32") == "binary"


@pytest.mark.parametrize("family", sorted(detect.MODULES))
def test_detection_compiles_modules_and_kernels_that_exist(family):
    assert (runner.PYTHON_TESTS / detect.MODULES[family]).is_file()
    assert (runner.TOOL_LLK / "tests" / "sources" / detect.SOURCES[family]).is_file()


def test_the_accuracy_driver_is_in_the_harness_only_while_it_runs(monkeypatch, tmp_path):
    installed = runner.PYTHON_TESTS / accuracy.DRIVER
    assert accuracy.DRIVER_SOURCE.is_file() and not installed.exists()
    seen = []
    monkeypatch.setattr(runner, "produce_consume", lambda *args, **kwargs: seen.append(installed.is_file()))
    accuracy.measure(None, "wormhole", ["Tanh"], tmp_path / "out", log=None)
    assert seen == [True]
    assert not installed.exists()


def test_the_binary_accuracy_driver_hooks_names_that_exist():
    """The driver replaces functions of test_eltwise_binary_sfpu.py by name; a rename
    there (#57137 dropped ``_assert_against_contract``) would otherwise only show up
    on hardware, as an accuracy run with no results."""
    import ast

    def module_names(path):
        names = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return names

    hooked = {
        call.args[1].value
        for call in ast.walk(ast.parse(accuracy.DRIVER_SOURCE.read_text()))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "setattr"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
    }
    assert hooked == {"assert_against_contract", "generate_stimuli"}
    assert hooked <= module_names(runner.PYTHON_TESTS / "test_eltwise_binary_sfpu.py")


def test_requested_ops_take_any_case_and_skip_unknown_names():
    ops, unknown = cli._requested_ops("tanh,SFPULOGSIGMOID,typecast,tanhh,Tanh")
    assert ops == ["Tanh", "SfpuLogsigmoid", "Typecast"]
    assert unknown == ["tanhh"]


def test_broadcast_variants_get_their_own_perf_row():
    plain = {
        "mathop": "MathOperation.SfpuElwadd",
        "formats.input_A": "Float16_b",
        "formats.output": "Float16_b",
        "dest_acc": "DestAccumulation.No",
        "approx_mode": "ApproximationMode.No",
        "fast_mode": "FastMode.No",
        "sfpu_bcast_dim": "BroadcastType.None_",
    }
    assert perf._row_key(plain)[0] == "SfpuElwadd"
    row = dict(plain, sfpu_bcast_dim="BroadcastType.Row")
    assert perf._row_key(row)[0] == "SfpuElwadd (bcast Row)"
    # Families without the column (unary, typecast) and empty CSV cells keep the bare op.
    unary = {k: v for k, v in plain.items() if k != "sfpu_bcast_dim"}
    assert perf._row_key(unary) == perf._row_key(plain)
    assert perf._row_key(dict(plain, sfpu_bcast_dim=NAN)) == perf._row_key(plain)


def test_reproduce_commands_drop_the_broadcast_label():
    summary = _summary()
    for rows in summary["perf"]["unary"].values():
        for r in rows:
            r["op"] = "Square (bcast Row)"
    text = report.render([summary])
    assert "| Square (bcast Row) |" in text
    assert "--ops Square --formats Float16_b,Float16 --check" in text


def test_binary_enum_names_map_to_math_operations():
    names = cli._binary_enum_to_op()
    assert names["LOGSIGMOID"] == "SfpuLogsigmoid"
    assert names["ATAN2"] == "SfpuAtan2"
    assert names["DIV_INT32"] == "SfpuDivInt32"


def test_exact_ops_count_wrong_lanes():
    import torch

    d = {
        "src": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "src_b": torch.tensor([2.0, 2.0, 2.0, NAN]),
        "golden": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "result": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    stats = accuracy._exact_stats(d)
    assert (stats["lanes"], stats["wrong"]) == (4, 1)
    assert stats["wrong_examples"][0][:2] == (2.0, 2.0)


def test_binary_specials_are_keyed_by_the_operand_pair():
    import torch

    d = {
        "binary": True,
        "src": torch.tensor([INF, INF, 1.0]),
        "src_b": torch.tensor([INF, -INF, 1.0]),
        "result": torch.tensor([INF, NAN, 2.0]),
    }
    spec = accuracy._specials(d)
    assert len(spec) == 3
    assert {v[1] for v in spec.values()} == {(INF, INF), (INF, -INF), (1.0, 1.0)}


def test_report_renders_exact_ops():
    summary = _summary()
    summary["accuracy"] = [
        {
            "key": ["SfpuElwLt", "Float16_b", "Float16_b", "No", "No"],
            "binary": True,
            "base": {
                "metric": "exact",
                "lanes": 16384,
                "wrong": 0,
                "wrong_examples": [],
            },
            "head": {
                "metric": "exact",
                "lanes": 16384,
                "wrong": 12,
                "wrong_examples": [],
            },
            "worse": 12,
            "better": 0,
            "bit_identical": False,
            "specials": {
                "changed": [{"class": "pair", "input": [INF, INF], "old": 0.0, "new": NAN}],
                "nan_propagates": {"base": True, "head": True},
            },
        }
    ]
    text = report.render([summary])
    assert "a lane is right or wrong" in text
    assert "⚠️ 12" in text
    assert "`(inf, inf)`" in text


def test_findings_list_every_regression_once():
    found = report.findings([_summary()])
    kinds = sorted(f["kind"] for f in found)
    # The no-loadmacro twin of the perf row has the same code: listed once.
    assert kinds == ["accuracy", "edge", "perf"]
    text = report.render([_summary()])
    assert "**⚠️ 3 regression(s)**" in text
    assert "--ops Square --formats Float16_b,Float16 --check" in text


def test_a_clean_report_says_so():
    summary = _summary()
    summary["perf"] = {}
    summary["accuracy"] = []
    assert report.findings([summary]) == []
    assert "**No regressions.**" in report.render([summary])


def test_a_small_drift_is_not_a_regression():
    rec = _summary()["accuracy"][0]
    rec["head"] = dict(rec["base"], max=rec["base"]["max"])
    rec["worse"], rec["better"] = 3, 1  # 2 net of 65,279 lanes
    rec["specials"] = {"changed": [], "nan_propagates": {"base": True, "head": True}}
    assert not report._acc_regressed(rec)
