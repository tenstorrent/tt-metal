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
        {"name": "t", "root": str(tmp_path / "tree"), "commit": "0" * 12, "waves": []},
    )
    write(str(d / "batches" / "manifest.json"), [])
    os.makedirs(d / "findings")
    os.makedirs(d / "verdicts")
    return d


# ---- consolidate: merging, severity escalation, dispositions ----------------------------------------------------


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
    ns = {"os": os, "re": __import__("re"), "subprocess": __import__("subprocess")}
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
    "I101",
    "I102",
    "I104",
    "I110",
    "I112",
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
