# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_survey.py: one wave diagnosis per implemented kernel, ranked by headroom."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).parent / "perf_survey.py"
_SPEC = importlib.util.spec_from_file_location("perf_survey", _PATH)
survey = importlib.util.module_from_spec(_SPEC)
sys.modules["perf_survey"] = survey  # dataclasses resolve the module by name
_SPEC.loader.exec_module(survey)

UNARY = "perf_eltwise_unary_sfpu_quasar.py"


def _id(module, op, fmt):
    return (
        f"quasar/{module}::test_perf[mathop_formats:(<MathOperation.{op}: OpSpec(x)>, "
        f"QuasarSfpuVariant(formats=InputOutputFormat[A:{fmt},B:{fmt},out:{fmt}]), [64, 64])]"
    )


COLLECTED = {
    UNARY: [
        _id(UNARY, "Exp", "Float32"),
        _id(UNARY, "Exp", "Float16_b"),
        _id(UNARY, "Sigmoid", "Float32"),
        _id(UNARY, "Sigmoid", "Float16_b"),
        _id(UNARY, "Abs", "Float32"),
    ],
    "perf_pack_quasar.py": [
        "quasar/perf_pack_quasar.py::test_perf_pack[Float32-a]",
        "quasar/perf_pack_quasar.py::test_perf_pack[Float16_b-b]",
    ],
}


def test_plan_one_variant_per_op_preferring_float16_b():
    plan = survey.plan(COLLECTED)
    ops = [(e.module, e.op) for e in plan]
    assert ops == [
        (UNARY, "Abs"),
        (UNARY, "Exp"),
        (UNARY, "Sigmoid"),
        ("perf_pack_quasar.py", "perf_pack_quasar"),
    ], "SFPU modules first, ops sorted, one entry per non-SFPU module"
    by_op = {e.op: e for e in plan}
    assert "Float16_b" in by_op["Exp"].test_id
    assert "Float32" in by_op["Abs"].test_id, "falls back to the first variant"
    assert "Float16_b" in by_op["perf_pack_quasar"].test_id


def test_run_type_follows_the_kind_of_kernel():
    by_op = {e.op: e for e in survey.plan(COLLECTED)}
    assert by_op["Sigmoid"].run_type == "MATH_ISOLATE"
    assert by_op["perf_pack_quasar"].run_type == "L1_TO_L1"
    assert by_op["Sigmoid"].sfpu and not by_op["perf_pack_quasar"].sfpu


def _fake_diagnose(results):
    calls = []

    def run(entry, out_dir):
        calls.append(entry.op)
        out_dir.mkdir(parents=True, exist_ok=True)
        status, headroom, top = results[entry.op]
        diag = {"status": status, "run_type": entry.run_type}
        if status in ("ok", "not_reconciled"):
            diag.update(
                measured=200.0,
                floor=200.0 * (1 - headroom / 100),
                stall_headroom_pct=headroom,
                top_causes=[[top, 12.5]],
            )
        else:
            diag["reason"] = "boom"
        (out_dir / "diag.json").write_text(json.dumps(diag))

    return run, calls


RESULTS = {
    "Abs": ("ok", 2.0, "math.idle"),
    "Exp": ("ok", 9.0, "math.bubble.SFPNONLINEAR->SFPSTORE"),
    "Sigmoid": ("ok", 14.0, "math.bubble.SFPNONLINEAR->SFPSTORE"),
    "perf_pack_quasar": ("failed", None, None),
}


def test_survey_ranks_by_headroom_and_flags_candidates(tmp_path):
    run, calls = _fake_diagnose(RESULTS)
    rows = survey.run_survey(survey.plan(COLLECTED), tmp_path, run)
    assert calls == ["Abs", "Exp", "Sigmoid", "perf_pack_quasar"]
    assert [r["op"] for r in rows] == ["Sigmoid", "Exp", "Abs", "perf_pack_quasar"]
    cand = {r["op"]: r["candidate"] for r in rows}
    assert cand == {
        "Sigmoid": "yes",
        "Exp": "yes",
        "Abs": "no",
        "perf_pack_quasar": "no",
    }
    with open(tmp_path / "survey.csv") as f:
        table = list(csv.DictReader(f))
    assert table[0]["op"] == "Sigmoid" and table[0]["top_cause"].startswith(
        "math.bubble.SFPNONLINEAR"
    )
    md = (tmp_path / "survey.md").read_text()
    assert "| Sigmoid |" in md and "perf_pack_quasar" in md


def test_survey_resumes_and_retries_only_failures_when_asked(tmp_path):
    run, calls = _fake_diagnose(RESULTS)
    survey.run_survey(survey.plan(COLLECTED), tmp_path, run)
    calls.clear()
    survey.run_survey(survey.plan(COLLECTED), tmp_path, run)
    assert calls == [], "finished ops are not captured again"
    survey.run_survey(survey.plan(COLLECTED), tmp_path, run, retry_failed=True)
    assert calls == ["perf_pack_quasar"]


def test_the_table_is_rewritten_after_every_op(tmp_path):
    seen = []

    def run(entry, out_dir):
        if (tmp_path / "survey.csv").exists():
            seen.append(len(list(csv.DictReader(open(tmp_path / "survey.csv")))))
        _fake_diagnose(RESULTS)[0](entry, out_dir)

    survey.run_survey(survey.plan(COLLECTED), tmp_path, run)
    assert seen == [0, 1, 2, 3], "written at the start (empty here) and after every op"


def test_collected_ids_are_parsed_from_the_runner_log():
    log = (
        "noise\n"
        + COLLECTED[UNARY][0]
        + "\n  "
        + COLLECTED[UNARY][1]
        + "\n1/170 tests collected\n"
    )
    assert survey.parse_collected(log, UNARY) == COLLECTED[UNARY][:2]


def test_sfpu_ids_without_mathoperation_group_by_the_part_before_the_format():
    m = "perf_eltwise_binary_sfpu_quasar.py"
    ids = [
        f"quasar/{m}::test_b[{p}]"
        for p in (
            "float-add-no-Float16-Float16-No",
            "float-add-no-Float16_b-Float16_b-No",
            "float-atan2-no-Float32-Float32-Yes",
            "bf16-rne-add",
        )
    ]
    plan = {e.op: e for e in survey.plan({m: ids})}
    assert set(plan) == {"float-add-no", "float-atan2-no", "bf16-rne-add"}
    assert "Float16_b" in plan["float-add-no"].test_id


def test_keyed_ids_group_by_their_op_fields():
    m = "perf_sfpu_reduce_quasar.py"
    ids = [
        f"quasar/{m}::test_r[reduce_pool:{p}-formats:{f}->{f}-mathop:{o}-dimension_combinations:[32, 64]]"
        for p in ("Average", "Max")
        for o in ("ReduceColumn", "ReduceRow")
        for f in ("Float32", "Float16_b")
    ]
    plan = {e.op: e for e in survey.plan({m: ids})}
    assert set(plan) == {
        "Average_ReduceColumn",
        "Average_ReduceRow",
        "Max_ReduceColumn",
        "Max_ReduceRow",
    }
    assert "Float16_b" in plan["Max_ReduceRow"].test_id


def test_ids_naming_no_op_fall_back_to_one_entry_per_test():
    m = "perf_sfpu_where_quasar.py"
    ids = [
        f"quasar/{m}::test_w[formats_dest_acc:QuasarSfpuVariant(formats=InputOutputFormat[A:{f}])-vector_mode:{v}]"
        for f in ("Float32", "Float16_b")
        for v in ("None_", "RC")
    ]
    plan = survey.plan({m: ids})
    assert [e.op for e in plan] == ["perf_sfpu_where_quasar"]
    assert "Float16_b" in plan[0].test_id


def test_results_already_on_disk_are_in_the_table_from_the_start(tmp_path):
    run, _ = _fake_diagnose(RESULTS)
    plan = survey.plan(COLLECTED)
    survey.run_survey(
        [e for e in plan if e.op == "perf_pack_quasar"], tmp_path, run
    )  # an earlier run
    seen = []

    def watch(entry, out_dir):
        seen.append([r["op"] for r in csv.DictReader(open(tmp_path / "survey.csv"))])
        run(entry, out_dir)

    survey.run_survey(plan, tmp_path, watch)
    # seen[1] is the table after the first new op was written: the old result must still be there.
    assert (
        "perf_pack_quasar" in seen[1]
    ), "a finished op stays listed before the loop reaches it"
