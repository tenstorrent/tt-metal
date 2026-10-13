# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for accepting intentional regressions (tt_metal/tt-llk/perf, #57871)."""

import json
import pathlib
import sys

import pandas as pd

_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
import regression_accept as ra  # noqa: E402
import regression_compare  # noqa: E402

_HEAD = "| point | test | run type | max delta | reason |\n|---|---|---|--:|---|\n"


def _body(*rows, before="Fixes rounding.\n\n", after="\n\n## Test plan\n- ran it\n"):
    return before + ra.HEADING + "\n\n" + _HEAD + "\n".join(rows) + after


_ROW_ID = "| 3fa91c2e7b04 | perf_eltwise_unary_sfpu | L1_TO_L1 | +10% | round to nearest-even (#57408) |"
_ROW_FILTER = (
    "| mathop=MathOperation.Square | perf_eltwise_unary_sfpu | L1_TO_L1, MATH_ISOLATE "
    "| +40% | same fix, all Square configs |"
)


def _point(pid, delta, module="perf_eltwise_unary_sfpu", run_type="L1_TO_L1", **config):
    return {
        "point_id": pid,
        "marker": "TILE_LOOP",
        "run_type": run_type,
        "test_module": module,
        "config": tuple(sorted(config.items())),
        "delta": delta,
        "current": 1.0 + delta,
        "baseline": 1.0,
        "abs_delta": 1000.0,
    }


# --- point ID ---------------------------------------------------------------


def test_point_id_is_12_hex_and_ignores_how_a_number_is_spelled():
    a = ra.point_id(
        "perf_x", "TILE_LOOP", "L1_TO_L1", (("tile_cnt", 4), ("mode", "Yes"))
    )
    b = ra.point_id(
        "perf_x", "TILE_LOOP", "L1_TO_L1", (("mode", "Yes"), ("tile_cnt", 4.0))
    )
    c = ra.point_id(
        "perf_x", "TILE_LOOP", "L1_TO_L1", (("tile_cnt", "4"), ("mode", "Yes"))
    )
    assert a == b == c and len(a) == 12 and int(a, 16) >= 0


def test_point_id_tells_markers_run_types_and_modules_apart():
    cfg = (("tile_cnt", 4),)
    ids = {
        ra.point_id("perf_x", "TILE_LOOP", "L1_TO_L1", cfg),
        ra.point_id("perf_x", "KERNEL", "L1_TO_L1", cfg),
        ra.point_id("perf_x", "TILE_LOOP", "MATH_ISOLATE", cfg),
        ra.point_id("perf_y", "TILE_LOOP", "L1_TO_L1", cfg),
    }
    assert len(ids) == 4


# --- the section ------------------------------------------------------------


def test_no_section_is_none():
    assert ra.parse_section("Just a fix.") == (None, [])


def test_a_point_id_row_and_a_filter_row_parse():
    rules, errors = ra.parse_section(_body(_ROW_ID, _ROW_FILTER))
    assert errors == []
    assert rules[0]["id"] == "3fa91c2e7b04" and rules[0]["max_pct"] == 10.0
    assert rules[1]["filter"] == {"mathop": "MathOperation.Square"}
    assert rules[1]["run_types"] == ["L1_TO_L1", "MATH_ISOLATE"]


def test_a_bad_row_names_its_row_and_the_problem():
    bad = "| 3fa91c2e7b04 | perf_x | L1_TO_L1 | ten percent | why |"
    _, errors = ra.parse_section(_body(_ROW_ID, bad))
    assert errors == ["row 2: max delta must look like +10%"]


def test_the_placeholder_reason_is_not_a_reason():
    row = "| 3fa91c2e7b04 | perf_x | L1_TO_L1 | +10% | <reason> |"
    assert ra.parse_section(_body(row))[1] == ["row 1: reason is empty"]


def test_a_wrong_header_is_refused():
    body = (
        ra.HEADING
        + "\n| id | test | type | max | why |\n|---|---|---|---|---|\n"
        + _ROW_ID
    )
    assert "header row" in ra.parse_section(body)[1][0]


def test_a_point_that_is_neither_an_id_nor_pairs_is_refused():
    row = "| square configs | perf_x | L1_TO_L1 | +10% | why |"
    assert "key=value" in ra.parse_section(_body(row))[1][0]


def test_the_section_hash_follows_the_section_only():
    base = ra.section_hash(_body(_ROW_ID))
    assert ra.section_hash(_body(_ROW_ID, before="Other text.\n")) == base
    assert ra.section_hash(_body(_ROW_ID.replace("+10%", "+12%"))) != base
    assert ra.section_hash("no section") == "none"


# --- approval ---------------------------------------------------------------


def _comment(login, body):
    return {"user": {"login": login}, "body": body}


def _table(*rows):
    rules, _ = ra.parse_section(_body(*rows))
    return ra.table_hash(rules)


_APPROVERS = {"nstojicTT": "U123"}


def test_an_approval_names_the_hash_of_the_table_it_accepts():
    table = _table(_ROW_ID)
    comments = [_comment("nstojicTT", f"Looks right.\n/accept-regression {table}")]
    out = ra.read(_body(_ROW_ID), comments, _APPROVERS, "author")
    assert out["approver"] == "nstojicTT"


def test_an_edit_after_the_approval_cancels_it():
    comments = [_comment("nstojicTT", f"/accept-regression {_table(_ROW_ID)}")]
    changed = _body(_ROW_ID.replace("+10%", "+50%"))
    assert ra.read(changed, comments, _APPROVERS, "author")["approved"] is False


def test_only_an_approver_who_is_not_the_author_can_approve():
    table = _table(_ROW_ID)
    line = f"/accept-regression {table}"
    assert ra.approval([_comment("someone", line)], table, _APPROVERS, "a") is None
    assert (
        ra.approval([_comment("github-actions[bot]", line)], table, _APPROVERS, "a")
        is None
    )
    assert (
        ra.approval([_comment("nstojicTT", line)], table, _APPROVERS, "nstojicTT")
        is None
    )


def test_a_command_without_a_hash_approves_nothing():
    table = _table(_ROW_ID)
    assert (
        ra.approval(
            [_comment("nstojicTT", "/accept-regression")], table, _APPROVERS, "a"
        )
        is None
    )


def test_an_invalid_table_cannot_be_approved():
    body = _body(_ROW_ID, "| x | perf_x | L1_TO_L1 | big | why |")
    comments = [_comment("nstojicTT", "/accept-regression 000000000000")]
    out = ra.read(body, comments, _APPROVERS, "a")
    assert out["state"] == "invalid" and out["approved"] is False


# --- applying the acceptance ---------------------------------------------------


def _acceptance(*rows, approved):
    out = ra.read(_body(*rows), [], _APPROVERS, "author")
    out.update(approved=approved, approver="nstojicTT" if approved else None)
    return out


def test_an_approved_row_takes_its_points_out_of_the_verdict():
    result = {"regressions": [_point("3fa91c2e7b04", 0.08)]}
    status = ra.apply(result, _acceptance(_ROW_ID, approved=True), {"3fa91c2e7b04": 1})
    assert result["regressions"] == [] and len(result["accepted"]) == 1
    assert status["state"] == "approved" and status["accepted"] == 1


def test_a_point_slower_than_its_max_delta_still_fails():
    result = {"regressions": [_point("3fa91c2e7b04", 0.15)]}
    status = ra.apply(result, _acceptance(_ROW_ID, approved=True), {"3fa91c2e7b04": 1})
    assert len(result["regressions"]) == 1 and status["missing"] == ["3fa91c2e7b04"]


def test_a_covering_table_waits_for_an_approver():
    result = {"regressions": [_point("3fa91c2e7b04", 0.08)]}
    status = ra.apply(result, _acceptance(_ROW_ID, approved=False), {"3fa91c2e7b04": 1})
    assert status["state"] == "waiting" and len(result["regressions"]) == 1


def test_a_table_that_misses_a_point_is_incomplete():
    result = {
        "regressions": [_point("3fa91c2e7b04", 0.08), _point("aaaaaaaaaaaa", 0.08)]
    }
    ids = {"3fa91c2e7b04": 1, "aaaaaaaaaaaa": 1}
    status = ra.apply(result, _acceptance(_ROW_ID, approved=False), ids)
    assert status["state"] == "incomplete" and status["missing"] == ["aaaaaaaaaaaa"]


def test_a_filter_row_covers_every_matching_config():
    result = {
        "regressions": [
            _point("111111111111", 0.33, mathop="MathOperation.Square", tile_cnt=8),
            _point(
                "222222222222",
                0.30,
                run_type="MATH_ISOLATE",
                mathop="MathOperation.Square",
            ),
            _point("333333333333", 0.05, mathop="MathOperation.Exp"),
        ]
    }
    ids = {"111111111111": 1, "222222222222": 1, "333333333333": 1}
    ra.apply(result, _acceptance(_ROW_FILTER, approved=True), ids)
    assert [r["point_id"] for r in result["regressions"]] == ["333333333333"]


def test_a_whole_percent_row_covers_its_own_point():
    """1000 -> 1070 cycles is 7.000000000000001% in floats; the pasted +7% must cover it."""
    point = _point("3fa91c2e7b04", (1070.0 - 1000.0) / 1000.0)
    lines = ra.paste_table([point])
    assert "| +7% |" in lines[-1]
    rules, _ = ra.parse_section("\n".join(lines).replace("<reason>", "on purpose"))
    acceptance = {"state": "present", "rules": rules, "approved": True, "approver": "x"}
    result = {"regressions": [point]}
    ra.apply(result, acceptance, {"3fa91c2e7b04": 1})
    assert result["regressions"] == [] and len(result["accepted"]) == 1


def test_an_id_shared_by_two_points_accepts_neither():
    result = {"regressions": [_point("3fa91c2e7b04", 0.08)]}
    status = ra.apply(result, _acceptance(_ROW_ID, approved=True), {"3fa91c2e7b04": 2})
    assert len(result["regressions"]) == 1
    assert "matches 2 points" in status["errors"][0]


def test_the_paste_table_parses_once_a_reason_is_written():
    lines = ra.paste_table([_point("3fa91c2e7b04", 0.0831)])
    assert (
        "| 3fa91c2e7b04 | perf_eltwise_unary_sfpu | L1_TO_L1 | +9% | <reason> |"
        in lines
    )
    rules, errors = ra.parse_section("\n".join(lines).replace("<reason>", "on purpose"))
    assert errors == [] and rules[0]["max_pct"] == 9.0


# --- the command line ---------------------------------------------------------


def _pr(tmp_path, body, author="author"):
    path = tmp_path / "pr.json"
    path.write_text(
        json.dumps({"body": body, "user": {"login": author}, "head": {"sha": "abc"}})
    )
    approvers = tmp_path / "approvers.json"
    approvers.write_text(json.dumps(_APPROVERS))
    return str(path), str(approvers)


def _approve(tmp_path, body, comment, commenter="nstojicTT", author="author"):
    pr, approvers = _pr(tmp_path, body, author=author)
    path = tmp_path / "comment.md"
    path.write_text(comment)
    args = ["approve", "--pr", pr, "--approvers", approvers, "--commenter", commenter]
    return ra.main(args + ["--comment", str(path)])


def test_approve_accepts_the_hash_of_the_current_table(tmp_path, capsys):
    table = _table(_ROW_ID)
    assert _approve(tmp_path, _body(_ROW_ID), f"/accept-regression {table}") == 0
    assert f"table `{table}`" in capsys.readouterr().out


def test_approve_refuses_a_stale_hash_and_names_the_current_one(tmp_path, capsys):
    stale = _table(_ROW_ID.replace("+10%", "+12%"))
    assert _approve(tmp_path, _body(_ROW_ID), f"/accept-regression {stale}") == 2
    assert f"now `{_table(_ROW_ID)}`" in capsys.readouterr().out


def test_approve_refuses_others_the_author_no_hash_and_no_table(tmp_path):
    line = f"/accept-regression {_table(_ROW_ID)}"
    assert _approve(tmp_path, _body(_ROW_ID), line, commenter="someone") == 2
    assert _approve(tmp_path, _body(_ROW_ID), line, author="nstojicTT") == 2
    assert _approve(tmp_path, _body(_ROW_ID), "/accept-regression") == 2
    assert _approve(tmp_path, "No table here.", line) == 2


def test_read_takes_comments_as_json_lines(tmp_path):
    pr, approvers = _pr(tmp_path, _body(_ROW_ID))
    comments = tmp_path / "comments.jsonl"
    approval = _comment("nstojicTT", f"/accept-regression {_table(_ROW_ID)}")
    comments.write_text(json.dumps(approval) + "\nnot json\n")
    out = tmp_path / "acc.json"
    args = ["read", "--pr", pr, "--comments", str(comments), "--approvers", approvers]
    ra.main(args + ["--out", str(out)])
    assert json.loads(out.read_text())["approver"] == "nstojicTT"


# --- end to end through the comparer -----------------------------------------


def _sides(tmp_path, base, cur):
    paths = {}
    for side, value in (("b", base), ("c", cur)):
        path = (
            tmp_path / side / "perf_eltwise_unary_sfpu" / "perf_eltwise_unary_sfpu.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "marker": ["TILE_LOOP"],
                "mathop": ["MathOperation.Square"],
                "mean(L1_TO_L1)": [value],
            }
        ).to_csv(path, index=False)
        paths[side] = str(path)
    return paths


def _compare(tmp_path, acceptance):
    paths = _sides(tmp_path, 28000.0, 37400.0)
    acc = tmp_path / "acc.json"
    acc.write_text(json.dumps(acceptance))
    report = tmp_path / "report.md"
    try:
        regression_compare.main(
            [
                "--current",
                paths["c"],
                "--baseline",
                paths["b"],
                "--report",
                str(report),
                "--accept",
                str(acc),
            ]
        )
    except SystemExit as e:
        code = e.code
    return (
        code,
        report.read_text(),
        json.loads((tmp_path / "report.verdict.json").read_text()),
    )


def test_an_approved_regression_passes_the_comparer(tmp_path):
    code, text, verdict = _compare(tmp_path, _acceptance(_ROW_FILTER, approved=True))
    assert code == 0 and verdict["accepted"] == 1 and verdict["regressions"] == 0
    assert "Accepted regressions (1)" in text and "approved by @nstojicTT" in text


def test_an_unapproved_regression_fails_and_offers_the_table(tmp_path):
    code, text, verdict = _compare(tmp_path, _acceptance(_ROW_FILTER, approved=False))
    assert code == 1 and verdict["accepted"] == 0
    assert "waits for a perf approver" in text and ra.HEADING in text
