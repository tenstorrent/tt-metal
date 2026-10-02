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
    dep = tmp_path / "cache" / "tt-logger" / "init.hpp"
    write(str(dep), "\n" * 80)
    assert ok(
        f"{dep}:64-71 (tt-logger cache, outside the tree)"
    ), "a fetched dependency cited by its absolute path is a real place"
    assert not ok(
        "cache/tt-logger/init.hpp:64-71"
    ), "the same file cited relative to nothing in the tree stays unlocatable"


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


def test_sibling_leads_in_scope_keep_only_the_runs_files(rundir, tmp_path):
    write(
        str(rundir / "batches" / "manifest.json"),
        [{"batch": "A-0000", "prio": "A", "files": ["a/b.cpp"], "lines": 1}],
    )
    deep = tmp_path / "x_deep.jsonl"
    sib = lambda loc: {"location": loc, "status": "unfixed", "why": "w"}  # noqa: E731
    write(
        str(deep),
        [
            {
                "id": "I1",
                "siblings": [sib("a/b.cpp:10"), sib("z/q.cpp:5"), sib("in words")],
            }
        ],
    )
    code, out, _ = run(
        os.path.join(ENGINE, "siblings.py"),
        "--run",
        rundir,
        "from-deep",
        f"{deep}=x",
        "--in-scope",
    )
    assert code == 0, out
    leads = [
        f
        for fn in os.listdir(rundir / "verdicts")
        for f in json.load(open(rundir / "verdicts" / fn))["findings"]
    ]
    assert [f["file"] for f in leads] == ["a/b.cpp"], leads
    assert "1 of 3 leads" in out and "1 unlocated" in out, out


def test_sibling_leads_in_scope_refuses_a_run_without_batches(rundir, tmp_path):
    deep = tmp_path / "x_deep.jsonl"
    write(
        str(deep),
        [{"id": "I1", "siblings": [{"location": "a.c:1", "status": "unfixed"}]}],
    )
    code, out, err = run(
        os.path.join(ENGINE, "siblings.py"),
        "--run",
        rundir,
        "from-deep",
        f"{deep}=x",
        "--in-scope",
    )
    assert code != 0 and "manifest is empty" in out + err, out + err


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


def test_a_second_finding_on_the_same_line_is_kept_not_dropped(rundir):
    write(
        str(rundir / "verdicts" / "A-0000.json"),
        {
            "findings": [
                finding(
                    "k.cpp",
                    9,
                    "medium",
                    category="arg-binding",
                    summary="arg 2 bound to the wrong slot",
                ),
                finding(
                    "k.cpp",
                    9,
                    "high",
                    category="arg-binding",
                    summary="arg 4 bound to the wrong slot",
                    votes={"confirmed": 2},
                ),
                finding(
                    "k.cpp", 9, "low", summary="arg 2 bound to the wrong slot"
                ),  # an exact repeat
            ]
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    conf = json.load(open(rundir / "CONFIRMED.json"))
    assert len(conf) == 1, conf  # file:line stays the identity
    f = conf[0]
    assert [m["summary"] for m in f["same_line"]] == [
        "arg 4 bound to the wrong slot"
    ], f
    assert (
        f["severity"] == "high"
    ), "a HIGH finding on the line must not hide behind the entry's own severity"
    md = open(rundir / "CONFIRMED.md").read()
    assert (
        "arg 2 bound to the wrong slot" in md and "arg 4 bound to the wrong slot" in md
    )


def test_an_unsettled_finding_on_a_confirmed_line_is_kept_but_does_not_raise_severity(
    rundir,
):
    write(
        str(rundir / "verdicts" / "A-0000.json"),
        {
            "findings": [
                finding("k.cpp", 9, "low", summary="count off by one"),
                finding(
                    "k.cpp",
                    9,
                    "high",
                    status="uncertain",
                    summary="stride uses rows, not tiles",
                ),
            ]
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    (f,) = json.load(open(rundir / "CONFIRMED.json"))
    assert [(m["summary"], m["status"]) for m in f["same_line"]] == [
        ("stride uses rows, not tiles", "uncertain")
    ], f
    assert (
        f["severity"] == "low"
    ), "an unsettled finding must not raise a confirmed entry's severity"
    assert (
        "uncertain] stride uses rows, not tiles" in open(rundir / "CONFIRMED.md").read()
    )


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


def test_recheck_queues_a_candidate_deferred_at_the_agent_limit(rundir):
    deferred = finding(
        "d.cpp",
        7,
        status="needs_recheck",
        reasons=[
            "[deferred] the wave reached its 950-agent limit before verifying this"
        ],
    )
    write(str(rundir / "verdicts" / "A-0000.json"), {"findings": [deferred]})
    code, out, err = run(
        os.path.join(ENGINE, "recheck.py"),
        "--run",
        rundir,
        "queue",
        "--to-dir",
        rundir / "items",
    )
    assert code == 0, out + err
    rc = json.load(open(rundir / "recheck.json"))
    (entry,) = [v for v in rc.values() if v["finding"]["file"] == "d.cpp"]
    assert entry["why"] == "deferred at the wave's agent limit", rc


def _refresh_twin(tmp_path):
    """A refresh: P100 (a fix PR) was deep-read long ago; its issue I200 closed after the watermark and carries the
    same fix commit. The refresh's case file holds only I200; the deep store's rows carry no fix commit.
    """
    fix = lambda: {  # noqa: E731
        "oid": "a" * 40,
        "subject": "Fix the race (#100)",
        "files": ["src/k.cpp"],
        "adds": 3,
        "dels": 1,
        "nfiles": 1,
    }
    write(
        str(tmp_path / "old_cases.jsonl"),
        [{"id": "P100", "title": "fix race", "fix": [fix()], "later": {}}],
    )
    write(
        str(tmp_path / "cases.jsonl"),
        [{"id": "I200", "title": "race in k", "fix": [fix()], "later": {}}],
    )
    write(
        str(tmp_path / "triage.jsonl"),
        [
            {
                "id": "I200",
                "verdict": "code-bug",
                "classes": ["race"],
                "mechanism": "m",
                "component": "c",
                "deep_priority": 3,
            }
        ],
    )
    write(str(tmp_path / "deep.jsonl"), [{"id": "P100", "is_real_bug": "yes"}])


def _select(tmp_path, *extra):
    return run(
        os.path.join(MINING, "select.py"),
        *extra,
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--exclude",
        tmp_path / "deep.jsonl",
        "--deep",
        tmp_path / "deep.jsonl",
    )


def test_a_refresh_holdout_never_shares_a_fix_with_an_old_deep_read(tmp_path):
    _refresh_twin(tmp_path)
    # without the case file the deep store came from, the twin cannot be recognised: refuse rather than leak it
    code, out, err = _select(tmp_path, "holdout", "--out", tmp_path / "h.jsonl")
    assert code != 0 and "--deep-cases" in out + err, out + err
    code, out, err = _select(
        tmp_path,
        "holdout",
        "--out",
        tmp_path / "h.jsonl",
        "--deep-cases",
        tmp_path / "old_cases.jsonl",
    )
    assert code == 0, out + err
    assert (
        open(tmp_path / "h.jsonl").read() == ""
    ), "I200 shares P100's fix, which is in the pack"


def test_a_refresh_deep_selection_skips_a_case_already_read_under_another_id(tmp_path):
    _refresh_twin(tmp_path)
    code, out, err = _select(
        tmp_path,
        "deep",
        "--out-dir",
        tmp_path / "deep_out",
        "--deep-cases",
        tmp_path / "old_cases.jsonl",
    )
    assert code == 0, out + err
    batched = (
        "".join(
            open(os.path.join(tmp_path / "deep_out", f)).read()
            for f in os.listdir(tmp_path / "deep_out")
        )
        if os.path.isdir(tmp_path / "deep_out")
        else ""
    )
    assert "I200" not in batched, batched


def test_a_refresh_deep_selection_still_reads_a_case_with_a_fix_never_read(tmp_path):
    _refresh_twin(tmp_path)
    cases = [json.loads(x) for x in open(tmp_path / "cases.jsonl")]
    refix = dict(cases[0]["fix"][0], oid="b" * 40, subject="Fix the race properly")
    cases[0]["fix"].append(refix)  # I200 also carries a later fix nobody has read
    write(str(tmp_path / "cases.jsonl"), cases)
    code, out, err = _select(
        tmp_path,
        "deep",
        "--out-dir",
        tmp_path / "deep_out",
        "--deep-cases",
        tmp_path / "old_cases.jsonl",
    )
    assert code == 0, out + err
    batched = "".join(
        open(os.path.join(tmp_path / "deep_out", f)).read()
        for f in os.listdir(tmp_path / "deep_out")
    )
    assert "I200" in batched, "a fix nobody has read is a defect the pack has not seen"
    code, out, err = _select(
        tmp_path,
        "holdout",
        "--out",
        tmp_path / "h.jsonl",
        "--deep-cases",
        tmp_path / "old_cases.jsonl",
    )
    assert (
        code == 0 and open(tmp_path / "h.jsonl").read() == ""
    ), "but it is never a holdout"


def test_deep_selection_never_rereads_a_deep_read_case_or_samples_its_twin(tmp_path):
    _refresh_twin(tmp_path)
    # P100 itself in the case file, triaged one below --min-priority so only --sample could pick it or its twin
    cases = [json.loads(x) for x in open(tmp_path / "cases.jsonl")]
    old = [json.loads(x) for x in open(tmp_path / "old_cases.jsonl")]
    write(str(tmp_path / "cases.jsonl"), cases + old)
    tri = [json.loads(x) for x in open(tmp_path / "triage.jsonl")]
    tri[0]["deep_priority"] = 1
    tri.append(dict(tri[0], id="P100"))
    write(str(tmp_path / "triage.jsonl"), tri)
    code, out, err = run(
        os.path.join(MINING, "select.py"),
        "deep",
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--deep",
        tmp_path / "deep.jsonl",
        "--sample",
        "5",
        "--out-dir",
        tmp_path / "deep_out",
    )
    assert code == 0, out + err
    d = tmp_path / "deep_out"
    batched = "".join(open(d / f).read() for f in os.listdir(d)) if d.is_dir() else ""
    assert "P100" not in batched and "I200" not in batched, batched


def _deep_ids(tmp_path, *extra):
    code, out, err = run(
        os.path.join(MINING, "select.py"),
        "deep",
        "--cases",
        tmp_path / "cases.jsonl",
        "--triage",
        tmp_path / "triage.jsonl",
        "--out-dir",
        tmp_path / "deep_out",
        *extra,
    )
    assert code == 0, out + err
    d = tmp_path / "deep_out"
    return "".join(open(d / f).read() for f in os.listdir(d)) if d.is_dir() else ""


def test_deep_selection_never_rereads_a_deep_read_case_even_without_a_fix(tmp_path):
    tri = {
        "verdict": "code-bug",
        "classes": ["race"],
        "mechanism": "m",
        "component": "c",
        "deep_priority": 3,
    }
    write(
        str(tmp_path / "cases.jsonl"),
        [{"id": "P7", "title": "t", "fix": [], "later": {}}],
    )
    write(str(tmp_path / "triage.jsonl"), [dict(tri, id="P7")])
    write(str(tmp_path / "deep.jsonl"), [{"id": "P7", "is_real_bug": "yes"}])
    assert "P7" not in _deep_ids(tmp_path, "--deep", tmp_path / "deep.jsonl")


def test_a_deep_store_passed_only_as_exclude_still_counts_its_fixes_as_read(tmp_path):
    _refresh_twin(tmp_path)
    batched = _deep_ids(
        tmp_path,
        "--exclude",
        tmp_path / "deep.jsonl",
        "--deep-cases",
        tmp_path / "old_cases.jsonl",
    )
    assert "I200" not in batched, "I200's only fix is P100's, which was read"


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


def _git_tree(tmp_path, names, contents=None):
    import subprocess as sp

    tree = tmp_path / "tree"
    for name in names:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text((contents or {}).get(name, "int x;\n"))
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


def test_init_run_batches_hold_at_most_300_lines_and_every_file_once(tmp_path):
    sizes = {
        "a/f1.c": 1,
        "a/f50.c": 50,
        "a/f120.c": 120,
        "a/f200.c": 200,
        "b/f290.c": 290,
        "b/f299.c": 299,
        "b/f310.c": 310,
        "c/f1200.c": 1200,
    }
    tree = _git_tree(
        tmp_path, list(sizes), {f: "int x;\n" * n for f, n in sizes.items()}
    )
    code, out, files = _init(tmp_path, tree)
    assert code == 0, out
    man = json.load(open(tmp_path / "run" / "batches" / "manifest.json"))
    listed = [f for b in man for f in b["files"]]
    assert sorted(listed) == sorted(
        sizes
    ), "every in-scope file is in exactly one batch"
    for b in man:
        lines = sum(sizes[f] for f in b["files"])
        # a file longer than the budget is a batch of its own; otherwise the default budget is 300 lines
        assert lines <= 300 or len(b["files"]) == 1, (b["files"], lines)
    assert (
        json.load(open(tmp_path / "run" / "state.json"))["batching"]["max_lines"] == 300
    )
    assert [b["files"] for b in man if "c/f1200.c" in b["files"]] == [["c/f1200.c"]]


def _commit_all(tree, msg):
    import subprocess as sp

    sp.run(["git", "add", "-A"], cwd=tree, check=True)
    sp.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg],
        cwd=tree,
        check=True,
    )
    return sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_init_run_diff_mode_keeps_changed_files_with_non_ascii_names(tmp_path):
    odd = "c/ünï cödé.c"
    tree = _git_tree(tmp_path, ["b/g1.c", "b/same.c", odd])
    import subprocess as sp

    base = sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tree / "b/g1.c").write_text("int y;\n")
    (tree / odd).write_text("int y;\n")
    _commit_all(tree, "change")
    code, out, files = _init(tmp_path, tree, "--since", base)
    assert code == 0, out
    assert files == {"b/g1.c", odd}, files


def test_init_run_diff_mode_reaches_changed_files_inside_a_submodule(tmp_path):
    import subprocess as sp

    sub = tmp_path / "subrepo"
    sub.mkdir()
    (sub / "s.c").write_text("int s;\n")
    (sub / "t.c").write_text("int t;\n")
    sp.run(["git", "init", "-q"], cwd=sub, check=True)
    _commit_all(sub, "sub base")
    tree = _git_tree(tmp_path, ["top.c", "other.c"])
    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(sub),
            "sub",
        ],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    base = _commit_all(tree, "add sub")
    (tree / "sub" / "s.c").write_text(
        "int s2;\n"
    )  # change one file inside the submodule checkout
    _commit_all(tree / "sub", "sub change")
    (tree / "top.c").write_text("int top2;\n")
    _commit_all(tree, "bump sub, change top")
    code, out, files = _init(tmp_path, tree, "--since", base, "--recurse-submodules")
    assert code == 0, out
    assert files == {"top.c", "sub/s.c"}, files


def test_init_run_diff_mode_reaches_a_change_inside_a_nested_submodule(tmp_path):
    import subprocess as sp

    allow = ["-c", "protocol.file.allow=always"]

    def repo(path, files):
        path.mkdir()
        for f, body in files.items():
            (path / f).write_text(body)
        sp.run(["git", "init", "-q"], cwd=path, check=True)
        _commit_all(path, "init")

    repo(tmp_path / "inner", {"a.c": "int a;\n", "b.c": "int b;\n"})
    repo(tmp_path / "mid", {"m.c": "int m;\n"})
    sp.run(
        ["git", *allow, "submodule", "add", "-q", str(tmp_path / "inner"), "inner"],
        cwd=tmp_path / "mid",
        check=True,
        capture_output=True,
    )
    _commit_all(tmp_path / "mid", "add inner")
    tree = _git_tree(tmp_path, ["top.c"])
    sp.run(
        ["git", *allow, "submodule", "add", "-q", str(tmp_path / "mid"), "mid"],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    sp.run(
        ["git", *allow, "submodule", "update", "-q", "--init", "--recursive"],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    base = _commit_all(tree, "add mid")
    (tree / "mid" / "inner" / "a.c").write_text(
        "int a2;\n"
    )  # the only change, two levels down
    _commit_all(tree / "mid" / "inner", "change a")
    _commit_all(tree / "mid", "bump inner")
    _commit_all(tree, "bump mid")
    code, out, files = _init(tmp_path, tree, "--since", base, "--recurse-submodules")
    assert code == 0, out
    assert files == {"mid/inner/a.c"}, files


def test_init_run_diff_mode_takes_a_whole_submodule_whose_old_pin_is_gone(tmp_path):
    import subprocess as sp

    sub = tmp_path / "subrepo"
    sub.mkdir()
    (sub / "s.c").write_text("int s;\n")
    (sub / "t.c").write_text("int t;\n")
    sp.run(["git", "init", "-q"], cwd=sub, check=True)
    _commit_all(sub, "sub base")
    tree = _git_tree(tmp_path, ["top.c"])
    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(sub),
            "sub",
        ],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    inner = tree / "sub"
    (inner / "s.c").write_text("int pinned;\n")
    _commit_all(inner, "the old pin, about to vanish")
    base = _commit_all(tree, "pin it")
    # rewrite the submodule's history so the old pin is unreachable, then drop it from the clone
    sp.run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=inner, check=True)
    (inner / "t.c").write_text("int t2;\n")
    _commit_all(inner, "new history")
    for cmd in (
        ["reflog", "expire", "--expire=now", "--all"],
        ["gc", "-q", "--prune=now"],
    ):
        sp.run(["git", *cmd], cwd=inner, check=True)
    _commit_all(tree, "bump sub")
    code, out, files = _init(tmp_path, tree, "--since", base, "--recurse-submodules")
    assert code == 0, out
    assert files == {"sub/s.c", "sub/t.c"} and "not fetched" in out, (files, out)


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


def _init_knowledge(tmp_path, repo, *extra):
    tree = _git_tree(tmp_path, ["a/x.c"])
    out = tmp_path / "run"
    code, o, e = run(
        os.path.join(ENGINE, "init_run.py"),
        "--root",
        tree,
        "--out",
        out,
        "--repo",
        repo,
        "--ext",
        ".c",
        *extra,
    )
    k = json.load(open(out / "state.json"))["knowledge"] if code == 0 else None
    return code, o + e, k, out


def test_init_run_hands_hunters_the_class_lists_by_default(tmp_path):
    code, out, k, _ = _init_knowledge(tmp_path, "tenstorrent/tt-metal")
    assert code == 0 and k == [
        "references/classes-universal.md",
        "references/classes-tenstorrent.md",
    ], out


def test_init_run_default_is_universal_only_outside_tenstorrent(tmp_path):
    code, out, k, _ = _init_knowledge(tmp_path, "o/r")
    assert code == 0 and k == ["references/classes-universal.md"], out


def test_init_run_knowledge_none_is_explicit_and_warns(tmp_path):
    code, out, k, _ = _init_knowledge(tmp_path, "o/r", "--knowledge", "none")
    assert code == 0 and k == [] and "NO bug-class list" in out, out


def test_init_run_refuses_a_missing_knowledge_file_before_writing(tmp_path):
    code, out, _, run_dir = _init_knowledge(
        tmp_path, "o/r", "--knowledge", "references/nope.md"
    )
    assert code != 0 and "nope.md" in out, out
    assert not run_dir.exists() or not os.listdir(run_dir), os.listdir(run_dir)


def _init_prio(tmp_path, *extra, repo="o/r"):
    tree = _git_tree(tmp_path, ["hot/x.c", "hot/sub/y.c", "cold/z.c", "mine/w.c"])
    out = tmp_path / "run"
    code, o, e = run(
        os.path.join(ENGINE, "init_run.py"),
        "--root",
        tree,
        "--out",
        out,
        "--repo",
        repo,
        "--ext",
        ".c",
        *extra,
    )
    if code:
        return code, o + e, {}, {}
    prio = {
        f: b["prio"]
        for b in json.load(open(out / "batches" / "manifest.json"))
        for f in b["files"]
    }
    return code, o + e, prio, json.load(open(out / "state.json"))


def _pack(tmp_path, body="`hot` (9); `mine` (4)"):
    p = tmp_path / "pack.md"
    p.write_text(
        f"# pack\n\n## Hot areas\n\nDirectories:\n\n{body}\n\n## Classes, by weight\n"
    )
    return p


def test_init_run_pack_hot_areas_lift_only_unclaimed_files_in_that_exact_dir(tmp_path):
    code, out, prio, st = _init_prio(
        tmp_path, "--pack", _pack(tmp_path), "--prio", "C=mine/*"
    )
    assert code == 0, out
    assert prio == {
        "hot/x.c": "A",  # in a hot area, no --prio glob
        "hot/sub/y.c": "C",  # a subdirectory is not the hot area itself
        "cold/z.c": "C",
        "mine/w.c": "C",  # an explicit --prio glob wins over the pack
    }, prio
    assert st["hot_promoted"] == 1 and "1 file(s) in 1 hot area(s)" in out, out


def test_init_run_pack_none_and_no_pack_leave_priorities_alone(tmp_path):
    code, out, prio, st = _init_prio(tmp_path, "--pack", "none")
    assert code == 0 and set(prio.values()) == {"C"} and st["pack"] is None, out


def test_init_run_finds_the_repo_pack_by_default(tmp_path):
    code, out, _, st = _init_prio(tmp_path, repo="tenstorrent/tt-metal")
    assert code == 0 and st["pack"] == "packs/tt-metal.md", out


def test_init_run_refuses_a_pack_without_hot_areas(tmp_path):
    p = tmp_path / "pack.md"
    p.write_text("# pack\n\n## Classes, by weight\n")
    code, out, _, _ = _init_prio(tmp_path, "--pack", p)
    assert code != 0 and "Hot areas" in out, out


def test_headless_sessions_deny_builds_tests_devices_and_tree_changes():
    sys.path.insert(0, ENGINE)
    from common import STATIC_DENY, headless_flags

    flags = headless_flags()
    assert (
        flags[:2] == ["--permission-mode", "auto"] and flags[2] == "--disallowedTools"
    )
    for rule in [
        "Bash(make *)",
        "Bash(pytest *)",
        "Bash(tt-smi *)",
        "Bash(rm *)",
        "Bash(sed -i *)",
        "Bash(git checkout *)",
        "Bash(git -C * checkout *)",
    ]:
        assert rule in STATIC_DENY, rule
    assert not any(
        r.startswith(("Bash(cp", "Bash(grep", "Bash(git log", "Bash(sed -n *"))
        for r in STATIC_DENY
    )
    for driver in ("run_headless.py", "run_workflow_headless.py"):
        src = open(os.path.join(ENGINE, driver)).read()
        assert "*headless_flags()" in src and '"auto"' not in src, driver


def test_blocked_actions_counts_refused_calls_in_workflow_agent_transcripts(tmp_path):
    sys.path.insert(0, ENGINE)
    from common import blocked_actions

    sub = (
        tmp_path
        / "projects"
        / "-some-cwd"
        / "sid-1"
        / "subagents"
        / "workflows"
        / "wf_x"
    )
    sub.mkdir(parents=True)
    denied = '{"content":"Permission to use Bash with command make --version has been denied."}'
    (sub / "agent-1.jsonl").write_text(
        f"{denied}\n" + '{"content":"ok"}\n' + f"{denied}\n"
    )
    (tmp_path / "projects" / "-some-cwd" / "other").mkdir()
    assert blocked_actions("sid-1", str(tmp_path)) == 2
    assert blocked_actions("sid-2", str(tmp_path)) == 0


def test_severity_rerating_round_trip_overrides_the_hunters_rating(rundir, tmp_path):
    write(
        str(rundir / "verdicts" / "A-0000.json"),
        {"findings": [finding("a.cpp", 3, "high"), finding("b.cpp", 5, "low")]},
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    code, out, err = run(
        os.path.join(ENGINE, "severity.py"),
        "--run",
        rundir,
        "prepare",
        "--to-dir",
        tmp_path / "sev",
    )
    assert code == 0 and json.loads(out.splitlines()[0])["n"] == 1, out + err
    items = json.load(open(tmp_path / "sev" / "b0000.json"))["items"]
    assert sorted(i["key"] for i in items) == ["a.cpp:3", "b.cpp:5"]
    write(
        str(tmp_path / "out.json"),
        {
            "result": {
                "ratings": [
                    {"key": "a.cpp:3", "severity": "medium", "why": "narrow config"},
                    {"key": "b.cpp:5", "severity": "low", "why": "diagnostic only"},
                ],
                "missing": [],
            }
        },
    )
    code, out, err = run(
        os.path.join(ENGINE, "severity.py"),
        "--run",
        rundir,
        "persist",
        tmp_path / "out.json",
    )
    assert code == 0, out + err
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    a = next(
        f for f in json.load(open(rundir / "CONFIRMED.json")) if f["file"] == "a.cpp"
    )
    assert a["severity"] == "medium" and a["severity_audit"] == "high", a
    code, out, _ = run(
        os.path.join(ENGINE, "severity.py"),
        "--run",
        rundir,
        "prepare",
        "--to-dir",
        tmp_path / "sev",
    )
    assert (
        json.loads(out.splitlines()[0])["n"] == 0
    ), "rated findings are skipped unless --all"


def test_severity_rater_sees_every_merged_site_and_the_worst_one_it_must_cover(
    rundir, tmp_path
):
    wh, bh = "a/wormhole/k.h", "a/blackhole/k.h"
    write(
        str(rundir / "verdicts" / "B-0000.json"),
        {
            "findings": [
                finding(wh, 10, "medium", summary="WH copy reads a stale bank"),
                finding(bh, 12, "high", summary="BH copy hangs every matmul"),
            ]
        },
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
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    code, out, err = run(
        os.path.join(ENGINE, "severity.py"),
        "--run",
        rundir,
        "prepare",
        "--to-dir",
        tmp_path / "sev",
    )
    assert code == 0, out + err
    (item,) = json.load(open(tmp_path / "sev" / "b0000.json"))["items"]
    # one rating replaces the merged entry's severity, so the rater must see the HIGH copy's own claim, not only
    # its location; it is not shown any claimed severity (a sibling lead's is a placeholder), to rate from the code
    sites = {s["site"]: s for s in item["other_sites"]}
    assert sites[f"{bh}:12"]["claim"] == "BH copy hangs every matmul", item
    assert "severity" not in sites[f"{bh}:12"] and "hunters_worst" not in item, item


def _recheck_all(rundir, tmp_path, outcome_of):
    """queue -> a fake recheck wave that answers outcome_of(summary) -> persist -> consolidate; returns queued items."""
    code, out, err = run(os.path.join(ENGINE, "recheck.py"), "--run", rundir, "queue")
    assert code == 0, out + err
    items = json.loads(out.splitlines()[0])["items"]
    write(
        str(tmp_path / "rc_out.json"),
        {
            "items": [
                {
                    "finding": it["finding"],
                    "why": it["why"],
                    "outcome": outcome_of(it["finding"]["summary"]),
                    "votes": {"confirmed": 3},
                    "reasons": ["[x] y"],
                }
                for it in items
            ]
        },
    )
    code, out, err = run(
        os.path.join(ENGINE, "recheck.py"),
        "--run",
        rundir,
        "persist",
        tmp_path / "rc_out.json",
    )
    assert code == 0, out + err
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    return items


def test_a_recheck_verdict_applies_only_to_the_finding_it_judged(rundir, tmp_path):
    refuted = finding(
        "z.cpp", 9, "high", status="refuted", summary="hunt claim, refuted"
    )
    refuted["votes"] = {"refuted": 3}
    lead = finding("z.cpp", 9, "low", status="uncertain", summary="sibling lead")
    lead["votes"] = {}
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [refuted, lead]})
    _recheck_all(rundir, tmp_path, lambda s: "confirmed")
    conf = json.load(open(rundir / "CONFIRMED.json"))
    # the confirmed entry is the lead the recheck judged, at its own severity, not the refuted HIGH claim
    assert [(f["summary"], f["severity"]) for f in conf] == [
        ("sibling lead", "low")
    ], conf


def test_two_unsettled_findings_on_one_line_are_each_rechecked(rundir, tmp_path):
    a = finding("y.cpp", 7, status="uncertain", summary="first claim")
    b = finding("y.cpp", 7, status="uncertain", summary="second claim")
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [a, b]})
    items = _recheck_all(
        rundir, tmp_path, lambda s: "refuted" if s == "first claim" else "confirmed"
    )
    assert sorted(i["finding"]["summary"] for i in items) == [
        "first claim",
        "second claim",
    ]
    conf = json.load(open(rundir / "CONFIRMED.json"))
    assert [f["summary"] for f in conf] == ["second claim"], conf


def test_a_line_keyed_recheck_from_an_older_run_still_applies_to_its_own_finding(
    rundir,
):
    a = finding("x.cpp", 4, status="uncertain", summary="judged claim")
    b = finding("x.cpp", 4, status="uncertain", summary="other claim")
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [a, b]})
    write(
        str(rundir / "recheck.json"),
        {
            "x.cpp:4": {
                "why": "no verifier could settle it",
                "outcome": "confirmed",
                "finding": {"file": "x.cpp", "line": 4, "summary": "judged claim"},
                "votes": {"confirmed": 3},
                "reasons": [],
            }
        },
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    conf = json.load(open(rundir / "CONFIRMED.json"))
    assert [f["summary"] for f in conf] == ["judged claim"], conf
    # the other claim on the line was never judged: it stays unsettled, riding on the entry
    assert [(m["summary"], m["status"]) for m in conf[0]["same_line"]] == [
        ("other claim", "uncertain")
    ], conf


def test_a_queued_recheck_outcome_keeps_the_wave_verdict_and_is_queued_again(
    rundir, tmp_path
):
    ref = finding("r.cpp", 2, "high", status="refuted", summary="refuted claim")
    ref["votes"] = {"refuted": 3}
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [ref]})
    code, out, err = run(
        os.path.join(ENGINE, "recheck.py"),
        "--run",
        rundir,
        "queue",
        "--refuted-sample",
        "1",
    )
    assert code == 0, out + err
    (it,) = json.loads(out.splitlines()[0])["items"]
    # the persist/consolidate side of a recheck whose verifiers died (recheck-wave.js returning "queued" is tested
    # under quickjs below)
    died = {"confirmed": 0, "refuted": 0, "uncertain": 0, "died": 3}
    write(
        str(tmp_path / "o.json"),
        {
            "items": [
                {
                    "finding": it["finding"],
                    "why": it["why"],
                    "outcome": "queued",
                    "votes": died,
                    "reasons": [],
                }
            ]
        },
    )
    run(
        os.path.join(ENGINE, "recheck.py"),
        "--run",
        rundir,
        "persist",
        tmp_path / "o.json",
    )
    assert run(os.path.join(ENGINE, "consolidate.py"), "--run", rundir)[0] == 0
    assert (
        "refuted claim" in open(rundir / "REFUTED.md").read()
    ), "the wave verdict stands"
    code, out, _ = run(os.path.join(ENGINE, "recheck.py"), "--run", rundir, "queue")
    assert [
        i["finding"]["summary"] for i in json.loads(out.splitlines()[0])["items"]
    ] == ["refuted claim"], "and the next recheck wave gets it again"


def test_a_recheck_whose_verifiers_died_twice_is_not_handed_out_again(rundir, tmp_path):
    lead = finding(
        "l.cpp", 3, status="uncertain", summary="a lead that kills verifiers"
    )
    write(str(rundir / "verdicts" / "B-0000.json"), {"findings": [lead]})
    died = {"confirmed": 0, "refuted": 0, "uncertain": 0, "died": 3}
    for attempt in range(2):
        code, out, err = run(
            os.path.join(ENGINE, "recheck.py"), "--run", rundir, "queue"
        )
        (it,) = json.loads(out.splitlines()[0])["items"]
        write(
            str(tmp_path / "o.json"),
            {
                "items": [
                    {
                        "finding": it["finding"],
                        "why": it["why"],
                        "outcome": "queued",
                        "votes": died,
                        "reasons": [],
                    }
                ]
            },
        )
        run(
            os.path.join(ENGINE, "recheck.py"),
            "--run",
            rundir,
            "persist",
            tmp_path / "o.json",
        )
    code, out, err = run(os.path.join(ENGINE, "recheck.py"), "--run", rundir, "queue")
    assert json.loads(out.splitlines()[0])["items"] == [], "a looping wave must end"
    code, out, err = run(os.path.join(ENGINE, "recheck.py"), "--run", rundir, "report")
    assert "given up" in out and "l.cpp:3" in out, out


def test_recheck_wave_keeps_an_item_queued_when_a_verifier_dies():
    quickjs = pytest.importorskip(
        "quickjs"
    )  # a JS engine, to run the Workflow script with stub agents
    plan = {
        "all-dead": [None, None, None],
        "dead-two-refute": [None, "refuted", "refuted"],
        "dead-two-confirm": [None, "confirmed", "confirmed"],
        "split": ["confirmed", "refuted", "uncertain"],
    }
    items = [
        {"finding": {"file": "a.c", "line": i, "summary": k}, "why": "x"}
        for i, k in enumerate(plan)
    ]
    src = (
        open(os.path.join(ENGINE, "recheck-wave.js"))
        .read()
        .replace("export const meta", "const meta", 1)
    )
    ctx = quickjs.Context()
    ctx.eval(
        f"""
globalThis.args = {json.dumps({"run": "/r", "root": "/t", "items": items})};
const plan = {json.dumps(plan)};
globalThis.log = () => {{}};
globalThis.agent = async (prompt) => {{
  const k = Object.keys(plan).find((s) => prompt.includes("Claim: " + s + "\\n"));
  const lens = prompt.includes("Lens: reachability") ? 0 : prompt.includes("Lens: semantics") ? 1 : 2;
  return plan[k][lens] === null ? null : {{ verdict: plan[k][lens], reason: "r" }};
}};
globalThis.parallel = (ts) => Promise.all(ts.map((t) => t()));
globalThis.pipeline = (xs, f) => Promise.all(xs.map(f));
(async function () {{
{src}
}})().then((r) => {{ globalThis.out = JSON.stringify(r) }}, (e) => {{ globalThis.out = "ERR " + e }});
"""
    )
    while ctx.execute_pending_job():
        pass
    res = ctx.eval("globalThis.out")
    assert not res.startswith("ERR"), res
    got = {i["finding"]["summary"]: i["outcome"] for i in json.loads(res)["items"]}
    assert got == {
        "all-dead": "queued",
        "dead-two-refute": "queued",
        "dead-two-confirm": "confirmed",
        "split": "uncertain",
    }, got


def test_every_program_and_helper_spawn_allows_is_used():
    # spawn.py is the allowlist of what the skill may start; an entry nothing uses only widens it
    import ast
    import glob

    sys.path.insert(0, ENGINE)
    import spawn

    used_programs, used_helpers = set(), set()
    for path in glob.glob(os.path.join(SKILL, "engine", "*.py")) + glob.glob(
        os.path.join(SKILL, "mining", "*.py")
    ):
        for n in ast.walk(ast.parse(open(path).read())):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "spawn"
            ):
                used_helpers.add(n.func.attr)
                if n.args and isinstance(n.args[0], ast.Constant):
                    used_programs.add(n.args[0].value)
    if "shell_run" in used_helpers:
        used_programs.add("sh")
    assert set(spawn.PROGRAMS) == used_programs, set(spawn.PROGRAMS) ^ used_programs
    helpers = {
        n for n in dir(spawn) if callable(getattr(spawn, n)) and not n.startswith("_")
    }
    assert helpers - {"subprocess"} == used_helpers, helpers ^ used_helpers


def test_every_spawn_user_imports_it_before_first_use():
    import ast
    import glob

    for path in glob.glob(os.path.join(SKILL, "engine", "*.py")) + glob.glob(
        os.path.join(SKILL, "mining", "*.py")
    ):
        tree = ast.parse(open(path).read())
        uses = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "spawn"
        ]
        if not uses:
            continue
        imports = [
            n.lineno
            for n in tree.body
            if isinstance(n, ast.Import) and any(a.name == "spawn" for a in n.names)
        ]
        assert imports and min(imports) < min(
            uses
        ), f"{path}: spawn used at {min(uses)} before any import"
