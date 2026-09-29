"""Hermetic tests for the bug-audit engine and mining scripts: synthetic run directories, no network, no agents.

  python3 -m pytest tt_metal/tt-llk/.claude/skills/bug-audit/tests -q

Each script runs in-process as __main__ (runpy), exactly as it runs from the command line.
"""

import contextlib
import io
import json
import os
import runpy
import sys

import pytest

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(SKILL, "engine")
MINING = os.path.join(SKILL, "mining")


def run(script, *argv):
    """(exit code, stdout, stderr) of a script run as __main__."""
    out, err, code = io.StringIO(), io.StringIO(), 0
    saved_argv, saved_path = sys.argv, list(sys.path)
    sys.argv = [script, *map(str, argv)]
    sys.path.insert(0, os.path.dirname(script))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if e.code is not None and not isinstance(e.code, int):
            err.write(
                f"{e.code}\n"
            )  # what the interpreter prints for sys.exit("message")
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
    return code, out.getvalue(), err.getvalue()


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        elif path.endswith(".jsonl"):
            fh.write("".join(json.dumps(x) + "\n" for x in obj))
        else:
            json.dump(obj, fh)


def finding(file, line, severity="medium", status="confirmed", **kw):
    return {
        "file": file,
        "line": line,
        "category": kw.pop("category", "race"),
        "severity": severity,
        "summary": kw.pop("summary", f"defect at {file}:{line}"),
        "failure_scenario": "x",
        "evidence": "y",
        "suggested_fix": "z",
        "batch": kw.pop("batch", "B-0000"),
        "status": status,
        "votes": {"confirmed": 3 if status == "confirmed" else 0},
        **kw,
    }


@pytest.fixture
def rundir(tmp_path):
    d = tmp_path / "run"
    write(
        str(d / "state.json"),
        {
            "name": "t",
            "root": str(tmp_path / "tree"),
            "commit": "0" * 12,
            "waves": [],
            "repo": "o/r",
        },
    )
    write(str(d / "batches" / "manifest.json"), [])
    os.makedirs(d / "findings")
    os.makedirs(d / "verdicts")
    return d


# ---- consolidate: merging, severity escalation, dispositions ----------------------------------------------------


def test_consolidate_summary_survives_a_hunt_record_with_classes(rundir):
    write(
        str(rundir / "verdicts" / "A-0000.json"),
        {"findings": [finding("a/k.cpp", 3, "low")]},
    )
    write(
        str(rundir / "findings" / "A-0000.json"),
        {
            "batch": "A-0000",
            "files_read": [{"path": "a/k.cpp"}],
            "classes_checked": ["index-math"],
        },
    )
    code, out, err = run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)
    assert code == 0 and "'low': 1" in out, err


def test_arch_copies_merge_into_one_entry_at_the_worst_severity(rundir):
    wh, bh = "a/wormhole/k.h", "a/blackhole/k.h"
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {"findings": [finding(wh, 10, "medium"), finding(bh, 12, "high")]},
    )
    write(
        str(rundir / "dedup.json"),
        {
            "auto": {},
            "clusters": [
                {
                    "canonical": f"{wh}:10",
                    "duplicates": [f"{bh}:12"],
                    "relation": "arch-copy",
                }
            ],
        },
    )
    code, _, _ = run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)
    assert code == 0
    conf = {
        f"{f['file']}:{f['line']}": f
        for f in json.load(open(rundir / "CONFIRMED.json"))
    }
    canon = conf[f"{wh}:10"]
    assert (
        canon["severity"] == "high"
    ), "a HIGH copy must not hide behind a MEDIUM canonical"
    assert [m["site"] for m in canon["merged_sites"]] == [
        f"{bh}:12"
    ], "the merged site is kept, never dropped"
    assert conf[f"{bh}:12"]["duplicate_of"] == f"{wh}:10"
    open_md = open(rundir / "OPEN.md").read()
    assert (
        open_md.count("### ") == 1 and bh in open_md
    ), "one open entry, listing both sites"


def test_a_disposition_closes_the_finding_and_survives_regeneration(rundir):
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [finding("f.cpp", 5)]})
    write(
        str(rundir / "dispositions.json"),
        {"f.cpp:5": {"state": "already_filed", "note": "x"}},
    )
    for _ in range(2):
        assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    assert "0 open of 1 confirmed" in open(rundir / "OPEN.md").read()


def test_uncertain_outranks_refuted_for_the_same_key(rundir):
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {"findings": [finding("g.cpp", 7, status="uncertain")]},
    )
    write(
        str(rundir / "verdicts" / "B-0001.json"),
        {"findings": [finding("g.cpp", 7, status="refuted", batch="B-0001")]},
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    assert "g.cpp:7" in open(rundir / "UNCERTAIN.md").read()
    assert (
        "g.cpp:7" not in open(rundir / "REFUTED.md").read()
    ), "an unsettled key must never read 'do not re-raise'"


def test_suggested_fixes_replace_the_placeholder(rundir):
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {"findings": [finding("h.cpp", 3, suggested_fix="placeholder")]},
    )
    write(
        str(rundir / "suggested_fixes.json"),
        {"h.cpp:3": "Guard the index. Test: add a case."},
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    open_md = open(rundir / "OPEN.md").read()
    assert "Guard the index" in open_md and "placeholder" not in open_md


# ---- persist_wave: the contract-trace ledger check -------------------------------------------------------------


def test_ledger_site_check(tmp_path):
    tree = tmp_path / "tree"
    write(str(tree / "src" / "k.cpp"), "int a;\nvoid run_kernel() {}\nint c;\n")
    src = open(os.path.join(ENGINE, "persist_wave.py")).read()
    spawn = type(sys)("spawn")
    spawn.__dict__.update(runpy.run_path(os.path.join(ENGINE, "spawn.py")))
    ns = {"os": os, "re": __import__("re"), "spawn": spawn}
    exec(src[src.index("def last_nonblank") : src.index("reread = load(")], ns)
    ok = lambda s: ns["ledger_site_ok"](str(tree), s)  # noqa: E731
    assert ok("src/k.cpp:2") and ok("src/k.cpp:1-3") and ok("src/k.cpp:run_kernel")
    assert not ok("src/k.cpp"), "an in-tree boundary names a line, not just a file"
    assert not ok("src/k.cpp:0") and not ok(
        "src/k.cpp:4"
    ), "the line must be inside the file"
    assert not ok("src/k.cpp:no_such_function") and not ok("src/missing.cpp:1")
    assert (
        ok(str(tmp_path.parent / "external_spec.md")) or True
    )  # absent external doc: unverifiable, either way
    external = tmp_path / "spec.md"
    write(str(external), "spec\n")
    assert ok(
        str(external)
    ), "a document outside the tree is an external reference; it needs no line"


# ---- siblings.py: leads from mined deep reads -------------------------------------------------------------------


def test_sibling_leads_merge_by_location_and_never_collide(rundir, tmp_path):
    deep = tmp_path / "x_deep.jsonl"
    sib = lambda loc, st="unfixed": {
        "location": loc,
        "status": st,
        "why": f"still at {loc}",
    }  # noqa: E731
    write(
        str(deep),
        [
            {
                "id": "I1",
                "primary_class": "race",
                "siblings": [
                    sib("a/b.cpp:10"),
                    sib("the trisc kernel entry"),
                    sib("c.h:1", "fixed"),
                ],
            },
            {
                "id": "I2",
                "primary_class": "race",
                "siblings": [
                    sib("a/b.cpp:10-14 (copy)"),
                    sib("the unpack reconfig path"),
                ],
            },
        ],
    )
    code, out, _ = run(
        os.path.join(ENGINE, "siblings.py"), "--run", rundir, "from-deep", f"{deep}=x"
    )
    assert code == 0, out
    leads = [
        f
        for fn in os.listdir(rundir / "verdicts")
        for f in json.load(open(rundir / "verdicts" / fn))["findings"]
    ]
    keys = [f"{f['file']}:{f['line']}" for f in leads]
    assert (
        len(leads) == 3 and len(set(keys)) == 3
    ), "one lead per location; two worded ones stay distinct"
    merged = next(f for f in leads if f["file"] == "a/b.cpp")
    assert (
        merged["line"] == 10 and len(merged["reasons"]) == 2
    ), "both past cases are kept on the merged lead"
    assert all(
        isinstance(f["line"], int) for f in leads
    ), "lines are ints, as the rest of the engine expects"
    assert all(
        f["status"] == "uncertain" and f["source"] == "history-sibling" for f in leads
    )


# ---- select.py: the holdout pick is reproducible -----------------------------------------------------------------


def _mined(tmp_path, n=20, reverse=False):
    cases, tri = [], []
    for i in range(n):
        cid = f"I{100 + i}"
        cases.append(
            {
                "id": cid,
                "title": f"fix bug {i}",
                "fix": [
                    {
                        "oid": f"{i:040x}",
                        "subject": f"Fix bug {i}",
                        "files": [f"src/f{i}.cpp"],
                        "adds": 3,
                        "dels": 1,
                        "nfiles": 1,
                    }
                ],
                "later": {},
            }
        )
        tri.append(
            {
                "id": cid,
                "verdict": "code-bug",
                "classes": ["race"],
                "mechanism": "m",
                "component": "c",
                "deep_priority": 1,
            }
        )
    if reverse:
        cases, tri = cases[::-1], tri[::-1]
    write(str(tmp_path / "cases.jsonl"), cases)
    write(str(tmp_path / "triage.jsonl"), tri)


PINNED = [
    "I106",
    "I111",
    "I114",
    "I118",
    "I119",
]  # seed 7, n 5, the 20 synthetic cases above


@pytest.mark.parametrize("reverse", [False, True])
def test_seeded_holdout_is_pinned_and_order_independent(tmp_path, reverse):
    _mined(tmp_path, reverse=reverse)
    out = tmp_path / "holdout.jsonl"
    code, _, err = run(
        os.path.join(MINING, "select.py"),
        "holdout",
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--out",
        out,
        "--n",
        5,
        "--seed",
        7,
    )
    assert code == 0, err
    got = sorted(json.loads(x)["id"] for x in open(out))
    assert got == PINNED, "a changed pick silently changes the recall benchmark"


def test_holdout_honours_excluded_ids(tmp_path):
    _mined(tmp_path)
    write(
        str(tmp_path / "old.jsonl"), [{"id": x} for x in PINNED]
    )  # ids only, no fix_commit
    out = tmp_path / "holdout.jsonl"
    run(
        os.path.join(MINING, "select.py"),
        "holdout",
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--out",
        out,
        "--n",
        5,
        "--exclude",
        tmp_path / "old.jsonl",
    )
    assert not set(PINNED) & {json.loads(x)["id"] for x in open(out)}


# ---- dedup: cross-directory groups, and chains across overlapping groups -----------------------------------------


def test_duplicate_chains_resolve_to_the_final_canonical(rundir):
    a, b, c = ("x/a.cpp", 1), ("y/b.cpp", 2), ("z/c.cpp", 3)
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {"findings": [finding(*a), finding(*b), finding(*c, severity="high")]},
    )
    key = lambda s: f"{s[0]}:{s[1]}"  # noqa: E731
    write(
        str(rundir / "dedup.json"),
        {
            "auto": {},
            "clusters": [
                {
                    "canonical": key(b),
                    "duplicates": [key(a)],
                    "relation": "same-defect",
                },
                {
                    "canonical": key(c),
                    "duplicates": [key(b)],
                    "relation": "same-defect",
                },
            ],
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    conf = {
        f"{f['file']}:{f['line']}": f
        for f in json.load(open(rundir / "CONFIRMED.json"))
    }
    sites = {m["site"] for m in conf[key(c)].get("merged_sites", [])}
    assert sites == {
        key(a),
        key(b),
    }, "A must reach C through B, not hang off B, which is itself hidden"
    assert open(rundir / "OPEN.md").read().count("### ") == 1


def test_cross_directory_findings_sharing_two_identifiers_meet_in_a_group(rundir):
    s1 = "Pops the row via `cb_pop_front` before `write_block_sync_granular` flushes"
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {
            "findings": [
                finding("ops/a/k.hpp", 10, summary=s1),
                finding("train/b/k.hpp", 20, summary=s1 + " (fork)"),
                finding(
                    "misc/c/z.cpp", 30, summary="Also calls `cb_pop_front` early"
                ),  # one shared name: not linked
            ]
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    code, out, err = run(os.path.join(ENGINE, "dedup.py"), "--run", rundir, "inputs")
    assert code == 0, err
    groups = [json.load(open(p)) for p in json.loads(out.splitlines()[0])["inputs"]]
    linked = [g for g in groups if g["group"].startswith("~linked:")]
    assert len(linked) == 1 and {f["file"] for f in linked[0]["findings"]} == {
        "ops/a/k.hpp",
        "train/b/k.hpp",
    }


# ---- persist_wave: an invalid ledger entry is recorded, not trusted, and does not re-hunt the batch ---------------


def test_invalid_ledger_entry_is_reported_without_a_rehunt(rundir, tmp_path):
    tree = tmp_path / "tree"
    write(str(tree / "k.cpp"), "int a;\nint b;\n")
    write(
        str(rundir / "batches" / "manifest.json"),
        [{"batch": "B-0000", "files": ["k.cpp"], "prio": "A", "root": str(tree)}],
    )
    os.makedirs(rundir / "done")
    hunt = {
        "files_read": [{"path": "k.cpp", "lines": 2, "last_line": "int b;"}],
        "boundaries": [
            {
                "site": "k.cpp:1",
                "kind": "call",
                "other_side": "k.cpp:2",
                "verdict": "consistent",
                "note": "",
            },
            {
                "site": "k.cpp:1",
                "kind": "call",
                "other_side": "k.cpp",
                "verdict": "consistent",
                "note": "",
            },
        ],
        "findings": [],
    }
    raw = tmp_path / "wave.json"
    write(
        str(raw),
        {"results": [{"batch": "B-0000", "ok": True, "hunt": hunt, "judged": []}]},
    )
    code, out, err = run(os.path.join(ENGINE, "persist_wave.py"), "--run", rundir, raw)
    assert code == 0, out + err
    assert os.path.exists(
        rundir / "done" / "B-0000.done"
    ), "the read check passed: the batch is done"
    saved = json.load(open(rundir / "findings" / "B-0000.json"))
    assert [e["other_side"] for e in saved["ledger_invalid"]] == ["k.cpp"]
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    assert "1 invalid" in open(rundir / "COVERAGE.md").read()


# ---- review fixes: each test pins one failure that the earlier code produced -------------------------------------


def test_rerunning_dedup_keeps_earlier_merges(rundir):
    wh, bh, new = ("a/wormhole/k.h", 10), ("a/blackhole/k.h", 10), ("b/x.cpp", 5)
    key = lambda s: f"{s[0]}:{s[1]}"  # noqa: E731
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {"findings": [finding(*wh), finding(*bh), finding(*new)]},
    )
    write(
        str(rundir / "dedup.json"),
        {
            "auto": {},
            "clusters": [
                {"canonical": key(wh), "duplicates": [key(bh)], "relation": "arch-copy"}
            ],
        },
    )
    raw = rundir / "dedup_raw.json"
    write(
        str(raw), {"results": [{"clusters": []}]}
    )  # a second wave that found nothing new
    assert (
        run(os.path.join(ENGINE, "dedup.py"), "--run", rundir, "persist", raw)[0] == 0
    )
    assert json.load(open(rundir / "dedup.json"))["clusters"][0]["duplicates"] == [
        key(bh)
    ], "the earlier merge survives"


def test_a_recheck_does_not_overrule_a_later_confirmation(rundir):
    k = ("r.cpp", 4)
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {
            "findings": [
                finding(*k, status="uncertain", wave=1),
                finding(*k, status="confirmed", wave=3),
            ]
        },
    )
    write(
        str(rundir / "recheck.json"),
        {
            "r.cpp:4": {
                "outcome": "refuted",
                "votes": {},
                "reasons": [],
                "why": "uncertain",
                "after_wave": 2,
            }
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    assert [
        f"{f['file']}:{f['line']}" for f in json.load(open(rundir / "CONFIRMED.json"))
    ] == ["r.cpp:4"]


def test_deep_selection_excludes_a_held_out_fix_under_another_id(tmp_path):
    _mined(tmp_path, n=2)
    cases = [json.loads(x) for x in open(tmp_path / "cases.jsonl")]
    cases[1]["fix"] = cases[0]["fix"]  # I101 is the PR twin of I100: same fix commit
    write(str(tmp_path / "cases.jsonl"), cases)
    tri = [
        dict(json.loads(x), deep_priority=3) for x in open(tmp_path / "triage.jsonl")
    ]
    write(str(tmp_path / "triage.jsonl"), tri)
    write(
        str(tmp_path / "hold.jsonl"),
        [{"id": "I100", "fix_commit": cases[0]["fix"][0]["oid"]}],
    )
    d = tmp_path / "deep"
    code, _, err = run(
        os.path.join(MINING, "select.py"),
        "deep",
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--exclude",
        tmp_path / "hold.jsonl",
        "--out-dir",
        d,
    )
    assert code == 0, err
    picked = (
        {
            c["id"]
            for p in d.glob("*.json")
            for c in json.load(open(p)).get(
                "cases",
                json.load(open(p)) if isinstance(json.load(open(p)), list) else [],
            )
        }
        if d.exists()
        else set()
    )
    assert "I101" not in picked and "I101" not in err.split("->")[0].split(), (
        picked,
        err,
    )


def test_filed_check_prefers_an_open_match(rundir):
    write(
        str(rundir / "state.json"),
        {
            "name": "t",
            "root": str(rundir),
            "commit": "0" * 12,
            "waves": [],
            "repo": "o/r",
        },
    )
    raw = rundir / "filed_raw.json"
    closed = {
        "number": 5,
        "kind": "issue",
        "state": "CLOSED",
        "same_bug": True,
        "why": "w",
        "url": "https://github.com/o/r/issues/5",
    }
    opened = {
        "number": 9,
        "kind": "issue",
        "state": "OPEN",
        "same_bug": True,
        "why": "w",
        "url": "https://github.com/o/r/issues/9",
    }
    write(str(raw), {"results": [{"key": "f.cpp:1", "matches": [closed, opened]}]})
    assert (
        run(os.path.join(ENGINE, "filed_check.py"), "--run", rundir, "persist", raw)[0]
        == 0
    )
    d = json.load(open(rundir / "dispositions.json"))["f.cpp:1"]
    assert d["state"] == "already_filed" and d.get("issue") == 9, d


def test_filed_check_persist_fails_when_a_judge_died(rundir):
    write(
        str(rundir / "state.json"),
        {
            "name": "t",
            "root": str(rundir),
            "commit": "0" * 12,
            "waves": [],
            "repo": "o/r",
        },
    )
    raw = rundir / "filed_raw.json"
    write(str(raw), {"results": [], "missing": ["/x/c0001.json"]})
    code, out, _ = run(
        os.path.join(ENGINE, "filed_check.py"), "--run", rundir, "persist", raw
    )
    assert code != 0 and "not checked" in out


def _fake_gh(tmp_path, state):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "gh").write_text(
        f'#!/bin/sh\necho \'{{"state":"{state}","mergedAt":null,"url":"u","title":"t"}}\'\n'
    )
    (b / "gh").chmod(0o755)
    return str(b)


@pytest.mark.parametrize(
    "disp",
    [{"state": "not_a_bug", "pr": 1}, {"state": "already_filed", "pr": 1}, {"pr": 1}],
)
def test_disposition_sync_leaves_done_states_and_survives_a_missing_state(
    rundir, tmp_path, monkeypatch, disp
):
    monkeypatch.setenv(
        "PATH", _fake_gh(tmp_path, "CLOSED") + os.pathsep + os.environ["PATH"]
    )
    write(str(rundir / "dispositions.json"), {"f.cpp:1": disp})
    code, out, err = run(
        os.path.join(ENGINE, "disposition.py"), "--run", rundir, "sync"
    )
    assert code == 0, out + err
    after = json.load(open(rundir / "dispositions.json"))["f.cpp:1"]
    if disp.get("state"):
        assert (
            after["state"] == disp["state"]
        ), "a done state is never overwritten by a PR's state"
    else:
        assert (
            after["state"] == "pr_closed"
        ), "an entry with a PR and no state is synced, not a crash"


def test_persisting_the_same_output_twice_is_refused(rundir, tmp_path):
    tree = tmp_path / "tree"
    write(str(tree / "k.cpp"), "int a;\n")
    write(
        str(rundir / "batches" / "manifest.json"),
        [{"batch": "B-0000", "files": ["k.cpp"], "prio": "A", "root": str(tree)}],
    )
    os.makedirs(rundir / "done")
    hunt = {
        "files_read": [{"path": "k.cpp", "lines": 1, "last_line": "int a;"}],
        "boundaries": [],
        "findings": [],
    }
    raw = tmp_path / "wave.json"
    write(
        str(raw),
        {
            "results": [
                {
                    "batch": "B-0000",
                    "ok": True,
                    "hunt": hunt,
                    "judged": [finding("k.cpp", 1, batch="B-0000")],
                }
            ]
        },
    )
    for _ in range(2):
        assert (
            run(os.path.join(ENGINE, "persist_wave.py"), "--run", rundir, raw)[0] == 0
        )
    assert len(json.load(open(rundir / "verdicts" / "B-0000.json"))["findings"]) == 1
    assert len(json.load(open(rundir / "state.json"))["waves"]) == 1
    assert not os.path.exists(
        rundir / "findings" / "B-0000.history.jsonl"
    ), "a second pass must still be possible"


def test_read_check_counts_newline_lines_only(tmp_path):
    src = open(os.path.join(ENGINE, "persist_wave.py")).read()
    ns = {"os": os}
    exec(src[src.index("def last_nonblank") : src.index("_suffix_cache = {}")], ns)
    f = tmp_path / "ff.c"
    f.write_bytes(b"a\x0cb\n\x0c\nc\x1d\nlast\n")
    assert ns["last_nonblank"](str(f)) == (
        4,
        "last",
    ), "form feeds and friends are not line breaks"


def test_marker_counts_each_item_once(tmp_path):
    x = {
        "number": 7,
        "closedAt": "2025-01-05T00:00:00Z",
        "createdAt": "2025-01-01T00:00:00Z",
    }
    write(str(tmp_path / "issue" / "2025-01-01_2025-01-07.jsonl"), [x])
    write(str(tmp_path / "issue" / "closed_2025-01-03_2025-01-09.jsonl"), [x])
    code, out, _ = run(
        os.path.join(MINING, "marker.py"),
        "write",
        tmp_path / "m.json",
        "--repo",
        "o/r",
        "--dumps",
        f"{tmp_path}/issue/*.jsonl",
    )
    assert (
        code == 0 and json.load(open(tmp_path / "m.json"))["counts"]["issues"] == 1
    ), out


def test_init_run_keeps_non_ascii_paths_in_scope(tmp_path):
    import subprocess as sp

    tree = tmp_path / "tree"
    tree.mkdir()
    for name in ("a.c", "café.c"):
        (tree / name).write_text("int x;\n")
    git = lambda *a: sp.run(["git", *a], cwd=tree, check=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    out = tmp_path / "run"
    code, o, e = run(
        os.path.join(ENGINE, "init_run.py"),
        "--root",
        tree,
        "--out",
        out,
        "--repo",
        "o/r",
        "--ext",
        ".c",
    )
    assert code == 0, o + e
    files = {
        f
        for b in json.load(open(out / "batches" / "manifest.json"))
        for f in b["files"]
    }
    assert files == {"a.c", "café.c"}, files


def _git_tree(tmp_path, names):
    import subprocess as sp

    tree = tmp_path / "tree"
    for name in names:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text("int x;\n")
    git = lambda *a: sp.run(["git", *a], cwd=tree, check=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    return tree


def _init(tmp_path, tree, *extra):
    out = tmp_path / "run"
    code, o, e = run(
        os.path.join(ENGINE, "init_run.py"),
        "--root",
        tree,
        "--out",
        out,
        "--repo",
        "o/r",
        "--ext",
        ".c",
        *extra,
    )
    if code:
        return code, o + e, set()
    m = json.load(open(out / "batches" / "manifest.json"))
    return code, o + e, {f for b in m for f in b["files"]}


def test_init_run_include_and_exclude_take_comma_lists_like_prio(tmp_path):
    tree = _git_tree(tmp_path, ["a/x.c", "b/y.c", "c/z.c"])
    code, out, files = _init(
        tmp_path, tree, "--include", "a/*,b/*", "--exclude", "b/*,c/*"
    )
    assert code == 0 and files == {"a/x.c"}, out


def test_init_run_refuses_an_empty_scope(tmp_path):
    tree = _git_tree(tmp_path, ["a/x.c"])
    code, out, _ = _init(tmp_path, tree, "--include", "nowhere/*")
    assert code != 0 and "no file" in out.lower(), out
