# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import sys

import pandas as pd

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import regression_accept as ra  # noqa: E402
import regression_compare  # noqa: E402

_HEAD = "| point | test | run type | max delta | reason |\n|---|---|---|--:|---|\n"
_MODULE = "perf_eltwise_unary_sfpu"
_ARCH = "blackhole"
_APPROVERS = {"nstojicTT": "", "ndivnicTT": ""}
_FILTER = (
    f"| mathop=MathOperation.Square | {_MODULE} | L1_TO_L1, MATH_ISOLATE "
    "| +40% | same fix, all Square configs |"
)


def _body(*rows, before="Fixes rounding.\n\n", after="\n\n## Test plan\n- ran it\n"):
    return before + ra.HEADING + "\n\n" + _HEAD + "\n".join(rows) + after


def _point(delta, run_type="L1_TO_L1", **config):
    config.setdefault("mathop", "MathOperation.Square")
    return {
        "marker": "TILE_LOOP",
        "run_type": run_type,
        "test_module": _MODULE,
        "config": tuple(sorted(config.items())),
        "delta": delta,
        "current": 1.0 + delta,
        "baseline": 1.0,
        "abs_delta": 1000.0,
    }


def _id(point, arch=_ARCH):
    return ra.point_id(
        arch, _MODULE, point["marker"], point["run_type"], point["config"]
    )


def _row(point, max_delta="+10%", reason="round to nearest-even"):
    return (
        f"| {_id(point)} | {_MODULE} | {point['run_type']} | {max_delta} | {reason} |"
    )


def _result(*regressions, others=()):
    return {
        "records": [*regressions, *others],
        "regressions": list(regressions),
        "new_points": [],
    }


def _comment(login, body, edited=False):
    return {
        "user": {"login": login},
        "body": body,
        "created_at": "2026-11-03T10:00:00Z",
        "updated_at": "2026-11-04T10:00:00Z" if edited else "2026-11-03T10:00:00Z",
    }


def _table(*rows):
    return ra.table_hash(ra.parse_section(_body(*rows))[0])


def _acceptance(*rows, approved):
    out = ra.read(_body(*rows), [], _APPROVERS, "author")
    out.update(approved=approved, approver="nstojicTT" if approved else None)
    return out


def test_point_id_ignores_how_a_number_is_spelled():
    ids = {
        ra.point_id(_ARCH, "perf_x", "TILE_LOOP", "L1_TO_L1", cfg)
        for cfg in (
            (("tile_cnt", 4), ("mode", "Yes")),
            (("mode", "Yes"), ("tile_cnt", 4.0)),
            (("tile_cnt", "4"), ("mode", "Yes")),
        )
    }
    assert len(ids) == 1 and len(ids.pop()) == 12


def test_point_id_tells_arches_markers_run_types_and_modules_apart():
    cfg = (("tile_cnt", 4),)
    ids = {
        ra.point_id("blackhole", "perf_x", "TILE_LOOP", "L1_TO_L1", cfg),
        ra.point_id("wormhole", "perf_x", "TILE_LOOP", "L1_TO_L1", cfg),
        ra.point_id("blackhole", "perf_x", "KERNEL", "L1_TO_L1", cfg),
        ra.point_id("blackhole", "perf_x", "TILE_LOOP", "MATH_ISOLATE", cfg),
        ra.point_id("blackhole", "perf_y", "TILE_LOOP", "L1_TO_L1", cfg),
    }
    assert len(ids) == 5


def test_no_section_is_none():
    assert ra.parse_section("Just a fix.") == (None, [])


def test_a_point_id_row_and_a_filter_row_parse():
    rules, errors = ra.parse_section(_body(_row(_point(0.08)), _FILTER))
    assert errors == []
    assert rules[0]["id"] == _id(_point(0.08)) and rules[0]["max_pct"] == 10.0
    assert rules[1]["filter"] == {"mathop": "MathOperation.Square"}
    assert rules[1]["run_types"] == ["L1_TO_L1", "MATH_ISOLATE"]


def test_a_bad_row_names_its_row_and_the_problem():
    bad = "| 3fa91c2e7b04 | perf_x | L1_TO_L1 | ten percent | why |"
    _, errors = ra.parse_section(_body(_row(_point(0.08)), bad))
    assert errors == ["row 2: max delta must look like +10%"]


def test_the_placeholder_reason_is_not_a_reason():
    row = "| 3fa91c2e7b04 | perf_x | L1_TO_L1 | +10% | <reason> |"
    assert ra.parse_section(_body(row))[1] == ["row 1: reason is empty"]


def test_a_wrong_header_is_refused():
    body = ra.HEADING + "\n| id | test | type | max | why |\n|---|---|---|---|---|\n"
    assert "header row" in ra.parse_section(body + _row(_point(0.08)))[1][0]


def test_a_point_that_is_neither_an_id_nor_pairs_is_refused():
    row = "| square configs | perf_x | L1_TO_L1 | +10% | why |"
    assert "key=value" in ra.parse_section(_body(row))[1][0]


def test_a_second_section_is_an_error_not_ignored():
    body = _body(_row(_point(0.08))) + "\n" + _body(_FILTER, before="")
    rules, errors = ra.parse_section(body)
    assert rules == [] and "has 2" in errors[0]
    out = ra.read(body, [], _APPROVERS, "author")
    assert out["state"] == "invalid" and out["approved"] is False


def test_the_section_hash_follows_the_sections_only():
    row = _row(_point(0.08))
    base = ra.section_hash(_body(row))
    assert ra.section_hash(_body(row, before="Other text.\n")) == base
    assert ra.section_hash(_body(row.replace("+10%", "+12%"))) != base
    assert ra.section_hash(_body(row) + "\n" + _body(_FILTER, before="")) != base
    assert ra.section_hash("no section") == "none"


def test_an_approval_names_the_hash_of_the_table_it_accepts():
    row = _row(_point(0.08))
    comments = [
        _comment("nstojicTT", f"Looks right.\n/accept-regression {_table(row)}")
    ]
    assert (
        ra.read(_body(row), comments, _APPROVERS, "author")["approver"] == "nstojicTT"
    )


def test_an_edit_to_the_table_after_the_approval_cancels_it():
    row = _row(_point(0.08))
    comments = [_comment("nstojicTT", f"/accept-regression {_table(row)}")]
    changed = _body(row.replace("+10%", "+50%"))
    assert ra.read(changed, comments, _APPROVERS, "author")["approved"] is False


def test_an_approver_comment_that_was_edited_does_not_count():
    row = _row(_point(0.08))
    line = f"/accept-regression {_table(row)}"
    comments = [_comment("nstojicTT", f"Old review text.\n{line}", edited=True)]
    assert ra.read(_body(row), comments, _APPROVERS, "author")["approved"] is False


def test_only_an_approver_who_is_not_the_author_can_approve():
    table = _table(_row(_point(0.08)))
    line = f"/accept-regression {table}"
    for login, author in (
        ("someone", "a"),
        ("github-actions[bot]", "a"),
        ("nstojicTT", "nstojicTT"),
    ):
        assert ra.approval([_comment(login, line)], table, _APPROVERS, author) is None
    no_hash = [_comment("nstojicTT", "/accept-regression")]
    assert ra.approval(no_hash, table, _APPROVERS, "a") is None


def test_an_approved_row_takes_its_point_out_of_the_verdict():
    point = _point(0.08)
    result = _result(point)
    status = ra.apply(result, _acceptance(_row(point), approved=True), _ARCH)
    assert result["regressions"] == [] and len(result["accepted"]) == 1
    assert status["state"] == "approved" and status["accepted"] == 1


def test_a_row_for_one_arch_does_not_cover_the_other():
    point = _point(0.08)
    result = _result(point)
    status = ra.apply(result, _acceptance(_row(point), approved=True), "wormhole")
    assert len(result["regressions"]) == 1 and status["state"] == "approved"
    assert status["missing"] == [_id(point, "wormhole")]


def test_a_filter_row_can_name_the_arch():
    row = _FILTER.replace("mathop=", "arch=wormhole, mathop=")
    for arch, left in (("wormhole", 0), ("blackhole", 1)):
        result = _result(_point(0.3))
        ra.apply(result, _acceptance(row, approved=True), arch)
        assert len(result["regressions"]) == left


def test_a_point_slower_than_its_max_delta_still_fails():
    point = _point(0.15)
    result = _result(point)
    status = ra.apply(result, _acceptance(_row(point), approved=True), _ARCH)
    assert len(result["regressions"]) == 1 and status["missing"] == [_id(point)]


def test_a_covering_table_waits_and_a_partial_one_is_incomplete():
    a, b = _point(0.08), _point(0.08, tile_cnt=8)
    status = ra.apply(_result(a), _acceptance(_row(a), approved=False), _ARCH)
    assert status["state"] == "waiting"
    status = ra.apply(_result(a, b), _acceptance(_row(a), approved=False), _ARCH)
    assert status["state"] == "incomplete" and status["missing"] == [_id(b)]


def test_a_filter_row_covers_every_matching_config():
    exp = _point(0.05, mathop="MathOperation.Exp")
    result = _result(
        _point(0.33, tile_cnt=8), _point(0.30, run_type="MATH_ISOLATE"), exp
    )
    ra.apply(result, _acceptance(_FILTER, approved=True), _ARCH)
    assert result["regressions"] == [exp]


def test_a_whole_percent_row_covers_its_own_point():
    point = _point((1070.0 - 1000.0) / 1000.0)
    row = ra.paste_table([{**point, "point_id": _id(point)}], _ARCH)[-1]
    assert "| +7% |" in row
    result = _result(point)
    row = row.replace(ra.PLACEHOLDER, "on purpose")
    ra.apply(result, _acceptance(row, approved=True), _ARCH)
    assert result["regressions"] == []


def test_many_points_get_one_filter_row_per_test_and_run_type():
    points = [_point(0.05 + i / 1000, tile_cnt=i) for i in range(150)]
    points.append(_point(0.5, run_type="MATH_ISOLATE"))
    result = _result(*points)
    ra.apply(result, {"state": "none"}, _ARCH)
    lines = ra.paste_table(result["regressions"], _ARCH)
    assert lines[4:] == [
        f"| arch={_ARCH} | {_MODULE} | MATH_ISOLATE | +50% | <reason> |",
        f"| arch={_ARCH} | {_MODULE} | L1_TO_L1 | +20% | <reason> |",
    ]
    rows = [line.replace(ra.PLACEHOLDER, "x") for line in lines[4:]]
    ra.apply(result, _acceptance(*rows, approved=True), _ARCH)
    assert result["regressions"] == [] and len(result["accepted"]) == 151


def test_an_id_shared_by_two_points_accepts_neither():
    point = _point(0.08)
    result = _result(point, others=[dict(point)])
    status = ra.apply(result, _acceptance(_row(point), approved=True), _ARCH)
    assert len(result["regressions"]) == 1
    assert "matches 2 points" in status["errors"][0]


def test_without_the_label_the_report_only_names_the_label():
    result = _result(_point(0.08))
    status = ra.apply(result, {"state": "off"}, _ARCH)
    lines = ra.report_lines(result, status, _ARCH)
    assert status == {"state": "off"} and "point_id" not in result["regressions"][0]
    assert any(ra.LABEL in line for line in lines)
    assert not any(ra.HEADING in line or "hash" in line for line in lines)
    assert ra.report_lines(_result(), status, _ARCH) == []


def _pr(tmp_path, body, author="author", labels=(ra.LABEL,)):
    path = tmp_path / "pr.json"
    path.write_text(
        json.dumps(
            {
                "body": body,
                "user": {"login": author},
                "head": {"sha": "abc"},
                "labels": [{"name": x} for x in labels],
            }
        )
    )
    approvers = tmp_path / "approvers.json"
    approvers.write_text(json.dumps(_APPROVERS))
    return str(path), str(approvers)


def _approve(tmp_path, body, comment, commenter="nstojicTT", **pr):
    path, approvers = _pr(tmp_path, body, **pr)
    (tmp_path / "comment.md").write_text(comment)
    args = ["approve", "--pr", path, "--approvers", approvers, "--commenter", commenter]
    return ra.main(args + ["--comment", str(tmp_path / "comment.md")])


def test_approve_accepts_the_hash_of_the_current_table(tmp_path, capsys):
    row = _row(_point(0.08))
    assert _approve(tmp_path, _body(row), f"/accept-regression {_table(row)}") == 0
    assert f"table `{_table(row)}`" in capsys.readouterr().out


def test_approve_refuses_a_stale_hash_and_names_the_current_one(tmp_path, capsys):
    row = _row(_point(0.08))
    stale = _table(row.replace("+10%", "+12%"))
    assert _approve(tmp_path, _body(row), f"/accept-regression {stale}") == 2
    assert f"now `{_table(row)}`" in capsys.readouterr().out


def test_approve_refuses_without_the_label(tmp_path, capsys):
    row = _row(_point(0.08))
    line = f"/accept-regression {_table(row)}"
    assert _approve(tmp_path, _body(row), line, labels=()) == 2
    assert ra.LABEL in capsys.readouterr().out


def test_approve_refuses_others_the_author_no_hash_and_no_table(tmp_path):
    row = _row(_point(0.08))
    line = f"/accept-regression {_table(row)}"
    assert _approve(tmp_path, _body(row), line, commenter="someone") == 2
    assert _approve(tmp_path, _body(row), line, author="nstojicTT") == 2
    assert _approve(tmp_path, _body(row), "/accept-regression") == 2
    assert _approve(tmp_path, "No table here.", line) == 2


def test_read_takes_comments_as_json_lines(tmp_path):
    row = _row(_point(0.08))
    path, approvers = _pr(tmp_path, _body(row))
    comments = tmp_path / "comments.jsonl"
    comments.write_text(
        json.dumps(_comment("nstojicTT", f"/accept-regression {_table(row)}")) + "\n"
    )
    out = tmp_path / "acc.json"
    args = ["read", "--pr", path, "--comments", str(comments), "--approvers", approvers]
    ra.main(args + ["--out", str(out)])
    assert json.loads(out.read_text())["approver"] == "nstojicTT"


def _compare(tmp_path, acceptance):
    for side, value in (("b", 28000.0), ("c", 37400.0)):
        path = tmp_path / side / _MODULE / f"{_MODULE}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "marker": ["TILE_LOOP"],
                "mathop": ["MathOperation.Square"],
                "mean(L1_TO_L1)": [value],
            }
        ).to_csv(path, index=False)
    (tmp_path / "acc.json").write_text(json.dumps(acceptance))
    report = tmp_path / "report.md"
    args = ["--current", str(tmp_path / "c/**/*.csv"), "--baseline"]
    args += [str(tmp_path / "b/**/*.csv"), "--report", str(report), "--arch", _ARCH]
    try:
        regression_compare.main(args + ["--accept", str(tmp_path / "acc.json")])
    except SystemExit as e:
        code = e.code
    status = json.loads((tmp_path / "report.acceptance.json").read_text())
    return code, report.read_text(), status


def test_an_approved_regression_passes_the_comparer(tmp_path):
    code, text, status = _compare(tmp_path, _acceptance(_FILTER, approved=True))
    assert code == 0 and status["accepted"] == 1
    assert "Accepted regressions (1)" in text and "approved by @nstojicTT" in text


def test_an_unapproved_regression_fails_and_offers_the_table(tmp_path):
    code, text, status = _compare(tmp_path, _acceptance(_FILTER, approved=False))
    assert code == 1 and status["state"] == "waiting"
    assert f"/accept-regression {status['table']}" in text and ra.HEADING in text


def test_without_the_label_the_comparer_names_the_label_only(tmp_path):
    code, text, status = _compare(tmp_path, {"state": "off"})
    assert code == 1 and status == {"state": "off"}
    assert ra.LABEL in text and ra.HEADING not in text
