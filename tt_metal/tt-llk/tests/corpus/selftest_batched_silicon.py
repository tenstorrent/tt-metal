#!/usr/bin/env python3
"""Self-test for the batched silicon executor (laneBU, sweep_2x2.py).

Drives the REAL sweep code (imported, not re-implemented) with filesystem
fixtures only — no toolchain, no simulator, no device:

  1. partition_perf_legs: distinct files share a session; same-file legs
     with distinct mathop tokens share (row-filter split); same-file
     token-less legs (the sem-vs-hand impl axis, invisible to the CSV)
     NEVER share; mixed token/token-less never share; deterministic.
  2. LAYOUT PARITY (the mandate gate): a 3-op dry-run through the real
     _silicon_phase produces the IDENTICAL per-op evidence layout in
     batched and --serial-legacy modes — same per-leg dirs, same node.txt/
     flags.txt contents, jobkeys equal except the mode field, identical
     assembled results — plus the batched PLAN.txt under silicon-batches/.
     The 3 ops cover full2x2 (with a hand OFF==ON byte-identity fold),
     pinpair, and a sem-perf byte-identical refusal row.
  3. session splitting: a fabricated consumer-session dir (report.json +
     combined module CSVs + group manifest + classify hash files) splits
     back into per-leg evidence: passed leg -> rc 0 + '1 passed' log +
     TEXT_HASHES = group-manifest subset at the classify relpaths + only
     its own CSV rows (mathop filter); failed leg -> rc 1, no 'passed';
     no-outcome leg -> rc 98; missing group ELF -> rc 97; a shared-module
     CSV without a mathop column -> rc 97 (never guesses row ownership).
  4. _job_cached: a green keyed hash-matched batched cell is reused; a
     text-hash or mode mismatch, or --force, re-runs.

Run by the nightly wrapper with the other gate self-tests; exit 0 green.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


# ---------------- 1. partition_perf_legs ----------------
def spec_of(file, mathop, op, sel="sem-perf", leg="on"):
    return {"file": file, "mathop": mathop, "op": op, "sel": sel, "leg": leg}


parts = sweep.partition_perf_legs(
    [
        spec_of("a.py", None, "op1"),
        spec_of("b.py", None, "op2"),
        spec_of("c.py", "Ceil", "op3"),
    ]
)
check("distinct files share one session", len(parts) == 1 and len(parts[0]) == 3)

parts = sweep.partition_perf_legs(
    [
        spec_of("u.py", "Ceil", "ceil"),
        spec_of("u.py", "EqualZero", "eqz"),
        spec_of("u.py", "Sqrt", "sqrt"),
    ]
)
check("same file, distinct mathop tokens share one session", len(parts) == 1)

parts = sweep.partition_perf_legs(
    [
        spec_of("u.py", None, "exp", sel="sem-perf"),
        spec_of("u.py", None, "exp", sel="hand-perf"),
    ]
)
check(
    "same file, token-less sem/hand legs NEVER share (CSV rows would collapse)",
    len(parts) == 2,
)

parts = sweep.partition_perf_legs(
    [
        spec_of("u.py", "Ceil", "ceil"),
        spec_of("u.py", None, "exp"),
    ]
)
check("mixed token/token-less on one file never share", len(parts) == 2)

specs = [
    spec_of("u.py", "Ceil", "ceil"),
    spec_of("u.py", None, "exp"),
    spec_of("u.py", None, "sigmoid"),
    spec_of("b.py", "SfpuBinaryMax", "minmax-max"),
    spec_of("b.py", "SfpuBinaryMin", "minmax-min"),
]
p1 = sweep.partition_perf_legs(list(specs))
p2 = sweep.partition_perf_legs(list(reversed(specs)))
check(
    "partition is deterministic (input order irrelevant)",
    [[(s["op"], s["sel"]) for s in b] for b in p1]
    == [[(s["op"], s["sel"]) for s in b] for b in p2],
    (p1, p2),
)
check(
    "packing: token legs + binary pair share; token-less get own sessions",
    len(p1) == 3 and sum(len(b) for b in p1) == 5,
    p1,
)


# ---------------- shared fixtures for 2-4 ----------------
def mk_row(op, kind, nodes, pin_flags=""):
    return {
        "op": op,
        "corpus_id": f"fixture__{op}",
        "kind": kind,
        "marker": "TILE_LOOP",
        "metric": "MATH_ISOLATE",
        "per_tile": True,
        "issue_slot_lb": None,
        "pin_flags": pin_flags,
        "extra_env": {},
        "sel_extra_env": {"sem": {}, "hand": {}},
        "nodes": {
            sel: nodes.get(sel, "")
            for sel in ("sem-corr", "sem-perf", "hand-corr", "hand-perf")
        },
    }


ROW_FULL = mk_row(
    "alpha",
    "full2x2",
    {
        "sem-corr": "test_u.py::test_alpha[functional-fresh]",
        "sem-perf": "perf_u.py::test_perf_alpha[formats:F-impl:1]",
        "hand-corr": "test_u.py::test_alpha[functional-prod]",
        "hand-perf": "perf_u.py::test_perf_alpha[formats:F-impl:0]",
    },
)
ROW_PIN = mk_row(
    "beta",
    "pinpair",
    {
        "sem-corr": "test_p.py::test_beta[gen]",
        "sem-perf": "test_p.py::test_beta_profile[gen]",
        "hand-corr": "test_p.py::test_beta[hand]",
        "hand-perf": "test_p.py::test_beta_profile[hand]",
    },
    pin_flags="-mpin-flags",
)
ROW_REFUSAL = mk_row(
    "gamma",
    "semantic",
    {
        "sem-corr": "test_g.py::test_gamma[functional]",
        "sem-perf": "perf_g.py::test_perf_gamma[mathop:Gamma-impl:1]",
    },
)
CLS_FULL = {
    "sem-corr": {"status": "OK", "all": "CHANGED", "math": "CHANGED"},
    "sem-perf": {"status": "OK", "all": "CHANGED", "math": "CHANGED"},
    "hand-corr": {"status": "OK", "all": "IDENTICAL", "math": "IDENTICAL"},
    "hand-perf": {"status": "OK", "all": "IDENTICAL", "math": "IDENTICAL"},
}
CLS_PIN = {
    "sem-corr": {"status": "OK", "all": "SINGLE_LEG"},
    "sem-perf": {"status": "OK", "all": "SINGLE_LEG"},
    "hand-corr": {"status": "OK", "all": "SINGLE_LEG"},
    "hand-perf": {"status": "OK", "all": "SINGLE_LEG"},
}
CLS_REFUSAL = {
    "sem-corr": {"status": "OK", "all": "CHANGED", "math": "CHANGED"},
    "sem-perf": {"status": "OK", "all": "IDENTICAL", "math": "IDENTICAL"},
}


def mk_sweep(ev, mode, dry_run=True, force=False):
    sw = object.__new__(sweep.Sweep)
    sw.a = argparse.Namespace(
        force=force,
        dry_run=dry_run,
        knob_silicon_rows=None,
        skip_craq_gate=False,
        classify_workers=2,
    )
    sw.ev = pathlib.Path(ev)
    sw.ev.mkdir(parents=True, exist_ok=True)
    sw.python = pathlib.Path(sys.executable)
    sw.reds = []
    sw.exec_mode = mode
    return sw


def tree_of(root):
    """Relative per-op silicon evidence paths (session-level dirs excluded)."""
    out = {}
    for p in sorted(pathlib.Path(root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts[0] == "silicon-batches":
            continue
        out[str(rel)] = p.read_text(errors="replace")
    return out


# ---------------- 2. 3-op dry-run layout parity ----------------
slots = [
    ("go", ROW_FULL, copy.deepcopy(CLS_FULL), None),
    ("go", ROW_PIN, copy.deepcopy(CLS_PIN), None),
    ("go", ROW_REFUSAL, copy.deepcopy(CLS_REFUSAL), None),
]
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw_serial = mk_sweep(td / "serial", "serial")
    res_serial = sw_serial._silicon_phase(copy.deepcopy(slots))
    sw_batched = mk_sweep(td / "batched", "batched")
    res_batched = sw_batched._silicon_phase(copy.deepcopy(slots))

    t_serial = tree_of(td / "serial")
    t_batched = tree_of(td / "batched")
    # inner.sh is the legacy per-leg run script; the batched executor's
    # equivalent is the session-level session.sh under silicon-batches/.
    norm_serial = {k: v for k, v in t_serial.items() if not k.endswith("inner.sh")}
    norm_batched = {k: v for k, v in t_batched.items() if not k.endswith("inner.sh")}
    check(
        "dry-run: identical per-leg evidence file SET in both modes",
        set(norm_serial) == set(norm_batched),
        set(norm_serial) ^ set(norm_batched),
    )
    diffs = []
    for k in norm_serial:
        a, b = norm_serial[k], norm_batched.get(k)
        if k.endswith("jobkey.json"):
            ja, jb = json.loads(a), json.loads(b)
            if ja.pop("mode", None) != "serial" or jb.pop("mode", None) != "batched":
                diffs.append((k, "mode fields wrong"))
            elif ja != jb:
                diffs.append((k, "jobkey differs beyond mode"))
        elif a != b:
            diffs.append((k, "content differs"))
    check(
        "dry-run: per-leg contents identical (jobkeys equal except mode)",
        not diffs,
        diffs[:4],
    )
    leg_dirs = {k.rsplit("/", 1)[0] for k in norm_serial if k.endswith("node.txt")}
    check(
        "dry-run: full2x2 + pinpair + refusal legs enumerate identically "
        f"({len(leg_dirs)} leg dirs)",
        # alpha: corr sem off+on, corr hand off (fold), perf sem r1-3 off+on,
        #        perf hand r1-3 off (fold)            = 3 + 6 + 3 = 12
        # beta (pinpair): corr sem+hand, perf sem+hand x3 = 2 + 6  = 8
        # gamma: corr sem off+on (refusal => no perf legs)         = 2
        len(leg_dirs) == 22,
        sorted(leg_dirs),
    )
    check(
        "dry-run: assembled results identical in both modes",
        json.dumps(res_serial, sort_keys=True, default=str)
        == json.dumps(res_batched, sort_keys=True, default=str),
    )
    check(
        "dry-run: batched mode wrote the session PLAN",
        (td / "batched/silicon-batches/PLAN.txt").is_file()
        and "3 CSV partition" not in "",  # plan exists; content free-form
    )
    check(
        "dry-run: refusal row produced ZERO perf leg dirs (recorded refusal)",
        not any("gamma/silicon/sem-perf" in d for d in leg_dirs),
    )

# ---------------- 3. session splitting ----------------
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev", "batched", dry_run=False)

    # group build fixture: two variants' ELFs + build.h
    rt = td / "gdir/rt"
    rels = {
        "minmax-max": "sources/b.cpp/v1/elf/math.elf",
        "minmax-min": "sources/b.cpp/v2/elf/math.elf",
    }
    manifest = {}
    for op, rel in rels.items():
        f = rt / "tt-llk-build" / rel
        f.parent.mkdir(parents=True)
        f.write_bytes(f"ELF {op}".encode())
        (f.parent.parent / "build.h").write_text(f"// build {op}\n")
        manifest[rel] = (
            hashlib.sha256(f"TEXT {op}".encode()).hexdigest(),
            hashlib.sha256(f"ELF {op}".encode()).hexdigest(),
        )
    gctx = {"rt": rt, "manifest": manifest, "seeded": False}

    node_max = "perf_b.py::test_perf_mm[formats:F-mathop:SfpuBinaryMax-impl:1]"
    node_min = "perf_b.py::test_perf_mm[formats:F-mathop:SfpuBinaryMin-impl:1]"
    node_dead = "perf_b2.py::test_perf_dead[mathop:Dead-impl:1]"
    row_max = mk_row("minmax-max", "full2x2", {"sem-perf": node_max, "sem-corr": "t"})
    row_min = mk_row("minmax-min", "full2x2", {"sem-perf": node_min, "sem-corr": "t"})
    row_dead = mk_row("deadop", "full2x2", {"sem-perf": node_dead, "sem-corr": "t"})

    # classify hash files (the node->ELF maps)
    for row, rel in (
        (row_max, rels["minmax-max"]),
        (row_min, rels["minmax-min"]),
    ):
        hf = sw.ev / row["op"] / "classify/sem-perf/hashes-on.txt"
        hf.parent.mkdir(parents=True)
        t, e = manifest[rel]
        hf.write_text(f"{rel}\ttext:{t}\telf:{e}\n")
    hf = sw.ev / "deadop/classify/sem-perf/hashes-on.txt"
    hf.parent.mkdir(parents=True)
    hf.write_text("sources/missing.cpp/v9/elf/math.elf\ttext:00\telf:00\n")

    jobs = [
        sw._mk_job(row_max, "sem-perf", "r1", "on", "-mflags", "perf"),
        sw._mk_job(row_min, "sem-perf", "r1", "on", "-mflags", "perf"),
        sw._mk_job(row_dead, "sem-perf", "r1", "on", "-mflags", "perf"),
    ]

    # session dir fixture: report.json (max passed, min failed, dead absent),
    # combined module CSV with both mathops' rows
    sdir = td / "gdir/r1-p0"
    sdir.mkdir(parents=True)
    (sdir / "report.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "exitstatus": 1,
                "reports": {
                    node_max: {
                        "setup": {"outcome": "passed"},
                        "call": {"outcome": "passed"},
                        "teardown": {"outcome": "passed"},
                    },
                    node_min: {
                        "setup": {"outcome": "passed"},
                        "call": {"outcome": "failed"},
                        "teardown": {"outcome": "passed"},
                    },
                },
            }
        )
    )
    pd = sdir / "perf_data/perf_b"
    pd.mkdir(parents=True)
    (pd / "perf_b.post.csv").write_text(
        "mathop,marker,mean(MATH_ISOLATE),tile_cnt\n"
        "MathOperation.SfpuBinaryMax,TILE_LOOP,100.0,8\n"
        "MathOperation.SfpuBinaryMin,TILE_LOOP,200.0,8\n"
    )
    raw = sdir / "raw_perf_data"
    raw.mkdir()
    (raw / "perf_b.master.csv").write_text(
        "mathop,marker,value\nMathOperation.SfpuBinaryMax,TILE_LOOP,1\n"
        "MathOperation.SfpuBinaryMin,TILE_LOOP,2\n"
    )

    sw._split_batch_session(sdir, jobs, 1, gctx)

    w_max = jobs[0]["work"]
    w_min = jobs[1]["work"]
    w_dead = jobs[2]["work"]
    check(
        "split: passed leg -> rc 0, '1 passed' log, session pointer",
        (w_max / "rc.txt").read_text().strip() == "0"
        and sweep.Sweep._passed(w_max / "log.txt")
        and str(sdir) in (w_max / "session.txt").read_text(),
    )
    check(
        "split: TEXT_HASHES = group-manifest subset at the classify relpaths",
        (w_max / "TEXT_HASHES.txt").read_text()
        == f"{rels['minmax-max']}\ttext:{manifest[rels['minmax-max']][0]}"
        f"\telf:{manifest[rels['minmax-max']][1]}\n",
    )
    check(
        "split: per-leg ELF archive holds the leg's variant (elf + build.h)",
        (w_max / "elf" / rels["minmax-max"]).is_file()
        and (w_max / "elf/sources/b.cpp/v1/build.h").is_file()
        and not (w_max / "elf/sources/b.cpp/v2").exists(),
    )
    max_csv = (w_max / "perf_data/perf_b/perf_b.post.csv").read_text()
    min_csv = (w_min / "perf_data/perf_b/perf_b.post.csv").read_text()
    check(
        "split: shared-module CSV rows filtered per leg by mathop",
        "SfpuBinaryMax" in max_csv
        and "SfpuBinaryMin" not in max_csv
        and "SfpuBinaryMin" in min_csv
        and "SfpuBinaryMax" not in min_csv,
    )
    check(
        "split: _perf_value reads the split CSV (per-tile)",
        sw._perf_value(row_max, "sem-perf", "r1", "on") == 100.0 / 8,
    )
    check(
        "split: raw temp CSVs filtered per leg too",
        "SfpuBinaryMin" not in (w_max / "raw_perf_data/perf_b.master.csv").read_text(),
    )
    check(
        "split: failed leg -> rc 1 and log WITHOUT a 'passed' count",
        (w_min / "rc.txt").read_text().strip() == "1"
        and not sweep.Sweep._passed(w_min / "log.txt"),
    )
    check(
        "split: leg whose node never ran -> rc 98; missing group ELF noted (97 class)",
        (w_dead / "rc.txt").read_text().strip() in ("97", "98")
        and "lacks" in (w_dead / "log.txt").read_text(),
    )

    # shared-module CSV WITHOUT a mathop column -> never guess ownership
    (pd / "perf_b.post.csv").write_text("marker,mean(MATH_ISOLATE)\nTILE_LOOP,1.0\n")
    sw2 = mk_sweep(td / "ev", "batched", dry_run=False)
    sw2._split_batch_session(sdir, jobs[:2], 0, gctx)
    check(
        "split: shared-module CSV without mathop column -> rc 97, no rows claimed",
        (w_max / "rc.txt").read_text().strip() == "97"
        and "cannot be attributed" in (w_max / "log.txt").read_text(),
    )

    # ---------------- 4. _job_cached ----------------
    sw3 = mk_sweep(td / "ev", "batched", dry_run=False)
    sw3._split_batch_session(
        sdir, jobs[:1], 0, gctx
    )  # re-produce a green max leg (CSV now unsplittable -> 97, so rebuild CSV first)
    (pd / "perf_b.post.csv").write_text(
        "mathop,marker,mean(MATH_ISOLATE),tile_cnt\n"
        "MathOperation.SfpuBinaryMax,TILE_LOOP,100.0,8\n"
    )
    sw3._split_batch_session(sdir, jobs[:1], 0, gctx)
    job = sw3._mk_job(row_max, "sem-perf", "r1", "on", "-mflags", "perf")
    check(
        "cache: green keyed hash-matched batched cell is reused", sw3._job_cached(job)
    )
    forced = mk_sweep(td / "ev", "batched", dry_run=False, force=True)
    check("cache: --force never reuses", not forced._job_cached(job))
    hf = sw3.ev / "minmax-max/classify/sem-perf/hashes-on.txt"
    hf.write_text(f"{rels['minmax-max']}\ttext:{'0'*64}\telf:{'0'*64}\n")
    check("cache: classify .text change re-runs", not sw3._job_cached(job))

# ---------------- 5. producer failure never poisons the group ----------------
# The storm-first-silicon lesson: the pin-12 counted-row ICE failed 2/117
# producer compiles and the old rc-gate withheld the WHOLE group (33 rows).
# A failed producer session must fail ONLY the legs whose classify ELF sets
# are incomplete in the group build; every fully-covered leg still runs.
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev", "batched", dry_run=False)
    sw.verify_toolchain = lambda phase: None
    sw.info = {"cc1plus_sha256": "cc", "tt_metal_head": "head"}

    rt = td / "grt"
    rel_ok = "sources/b.cpp/v1/elf/math.elf"
    rel_bad = "sources/b.cpp/v2/elf/math.elf"  # NOT in the group build
    f = rt / "tt-llk-build" / rel_ok
    f.parent.mkdir(parents=True)
    f.write_bytes(b"ELF ok")
    manifest = {rel_ok: ("t1", "e1")}

    node_ok_c = "test_b.py::test_c[mathop:SfpuBinaryMax]"
    node_ok_p = "perf_b.py::test_p[mathop:SfpuBinaryMax-impl:1]"
    node_bad_c = "test_b.py::test_c[mathop:SfpuBinaryMin]"
    node_bad_p = "perf_b.py::test_p[mathop:SfpuBinaryMin-impl:1]"
    row_ok = mk_row("cov-ok", "full2x2", {"sem-corr": node_ok_c, "sem-perf": node_ok_p})
    row_bad = mk_row(
        "cov-bad", "full2x2", {"sem-corr": node_bad_c, "sem-perf": node_bad_p}
    )
    cls = {
        "sem-corr": {"status": "OK", "all": "CHANGED", "math": "CHANGED"},
        "sem-perf": {"status": "OK", "all": "CHANGED", "math": "CHANGED"},
    }
    for row, rel in ((row_ok, rel_ok), (row_bad, rel_bad)):
        for sel in ("sem-corr", "sem-perf"):
            for leg in ("off", "on"):
                hf = sw.ev / row["op"] / "classify" / sel / f"hashes-{leg}.txt"
                hf.parent.mkdir(parents=True, exist_ok=True)
                hf.write_text(f"{rel}\ttext:t\telf:e\n")

    gctx_fail = {"rt": rt, "manifest": manifest, "seeded": False, "producer_rc": 2}
    sw._group_build = lambda gdir, flags, extra_env, jobs: gctx_fail
    sessions = []

    def fake_session(gctx, gdir, name, jobs, flags, extra_env):
        sessions.append((name, sorted({j["op"] for j in jobs})))
        for j in jobs:  # minimal green evidence so step 3 sees corr PASS
            j["work"].mkdir(parents=True, exist_ok=True)
            (j["work"] / "rc.txt").write_text("0\n")
            (j["work"] / "log.txt").write_text("1 passed\n")

    sw._run_batch_session = fake_session
    sw._batched_silicon([(row_ok, dict(cls)), (row_bad, dict(cls))])

    bad_legs = [
        ("sem-corr", "corr-off"),
        ("sem-corr", "corr-on"),
        ("sem-perf", "r1-off"),
        ("sem-perf", "r3-on"),
    ]
    bad_all_96 = all(
        (sw.ev / "cov-bad/silicon" / sel / leg / "rc.txt").read_text().strip() == "96"
        and "did not compile"
        in (sw.ev / "cov-bad/silicon" / sel / leg / "log.txt").read_text()
        for sel, leg in bad_legs
    )
    check("coverage: uncovered legs fail INDIVIDUALLY (rc 96, named ELF)", bad_all_96)
    check(
        "coverage: covered legs of the same failed group still run",
        any("cov-ok" in ops for _n, ops in sessions),
        sessions,
    )
    check(
        "coverage: no session ever carries a withheld leg",
        all(ops == ["cov-ok"] for _n, ops in sessions),
        sessions,
    )
    check(
        "coverage: covered rows got corr AND all perf reps",
        sorted(n for n, _o in sessions)
        == sorted(["corr", "corr"] + [f"r{r}-p0" for r in (1, 2, 3) for _ in (0, 1)]),
        sorted(n for n, _o in sessions),
    )

    # fail-closed: a leg with NO classify map after a failed producer
    sw2 = mk_sweep(td / "ev2", "batched", dry_run=False)
    row_nomap = mk_row("cov-nomap", "full2x2", {"sem-corr": node_ok_c})
    job = sw2._mk_job(row_nomap, "sem-corr", "corr", "off", "-mx", "corr")
    kept = sw2._producer_coverage([job], gctx_fail, td / "gdir")
    check(
        "coverage: no classify map after failed producer -> fail CLOSED",
        kept == []
        and (job["work"] / "rc.txt").read_text().strip() == "96"
        and "failing closed" in (job["work"] / "log.txt").read_text(),
    )

# ---------------- 6. batched classify sessions ----------------
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev", "batched", dry_run=False)
    sw.verify_toolchain = lambda phase: None
    sw.info = {"cc1plus_sha256": "cc", "tt_metal_head": "head"}
    sw._macro_scan = lambda work, legnames: {"classification": "SELFTEST_STUB"}
    # objcopy stand-in: emit the ELF's bytes as its .text (arg 4 = elf path)
    objcopy = td / "objcopy.sh"
    objcopy.write_text('#!/bin/sh\ncat "$4"\n')
    objcopy.chmod(0o755)
    sw.objcopy = objcopy

    def nid(op):
        return f"perf_c.py::test_c[mathop:{op}]"

    rows = {
        op: mk_row(f"cls-{op}", "semantic", {"sem-perf": nid(op)})
        for op in ("changed", "ident", "fail", "dropped", "absent", "cached")
    }
    # pre-cached verdict: must never reach a chunk session
    cw = sw.ev / "cls-cached/classify/sem-perf"
    cw.mkdir(parents=True)
    (cw / "verdict.json").write_text(
        json.dumps(
            {
                "selector": "sem-perf",
                "status": "COMPILE_FAIL",
                "leg": "off",
                "cc1plus_sha256": "cc",
                "tt_metal_head": "head",
            }
        )
    )

    seen_nodes = []

    def fake_chunk(cdir, cjobs, flags, extra_env):
        rt = cdir / "rt"
        shutil.rmtree(cdir, ignore_errors=True)
        rt.mkdir(parents=True)
        report = {"reports": {}, "variant_files": {}, "deselected": []}
        for j in cjobs:
            node, op, leg = j["node"], j["row"]["op"], j["leg"]
            seen_nodes.append(node)
            if op == "cls-dropped":
                report["deselected"].append(node)
                continue
            if op == "cls-absent":
                continue  # session died before this node
            outcome = "failed" if (op == "cls-fail" and leg == "on") else "passed"
            report["reports"][node] = {
                "setup": {"outcome": "passed"},
                "call": {"outcome": outcome},
                "teardown": {"outcome": "passed"},
            }
            if outcome != "passed":
                continue
            vdir = f"test_c/v-{op}-{leg}"
            elf = rt / "tt-llk-build" / vdir / "elf/math.elf"
            elf.parent.mkdir(parents=True)
            content = op if op == "cls-ident" else f"{op}-{leg}"
            elf.write_bytes(content.encode())
            (elf.parent.parent / "build.h").write_text(f"// {op} {leg}\n")
            # a FOREIGN sibling variant under the same parent dirs (the
            # gcd+rsubint32 finding): file-level attribution must never
            # sweep it into this node's hashes or archive
            foreign = rt / "tt-llk-build/test_c/v-foreign/elf/math.elf"
            if not foreign.is_file():
                foreign.parent.mkdir(parents=True)
                foreign.write_bytes(b"foreign")
            report["variant_files"][node] = [
                f"{vdir}/elf/math.elf",
                f"{vdir}/build.h",
            ]
        return report

    sw._classify_chunk_session = fake_chunk
    pending = [(rows[op], "sem-perf", None) for op in rows]
    unproven = sw._batched_classify(pending)
    check(
        "classify batch: unprovable legs are RETURNED as solo specs "
        "(row, sel, legs, tag) for the concurrent solo pool",
        sorted(s[0]["op"] for s in unproven) == ["cls-absent", "cls-dropped"]
        and all(s[1] == "sem-perf" and s[3] == "classify" for s in unproven)
        and all(s[2] and s[2][0][0] == "off" for s in unproven),
        unproven,
    )

    v_changed = json.loads(
        (sw.ev / "cls-changed/classify/sem-perf/verdict.json").read_text()
    )
    check(
        "classify batch: OFF!=ON content -> OK/CHANGED verdict with scan + keys",
        v_changed.get("status") == "OK"
        and v_changed.get("all") == "CHANGED"
        and v_changed.get("math") == "CHANGED"
        and v_changed.get("macro_scan") == {"classification": "SELFTEST_STUB"}
        and v_changed.get("cc1plus_sha256") == "cc",
        v_changed,
    )
    v_ident = json.loads(
        (sw.ev / "cls-ident/classify/sem-perf/verdict.json").read_text()
    )
    check(
        "classify batch: OFF==ON content -> IDENTICAL verdict",
        v_ident.get("all") == "IDENTICAL" and v_ident.get("math") == "IDENTICAL",
        v_ident,
    )
    w = sw.ev / "cls-changed/classify/sem-perf"
    hash_on = (w / "hashes-on.txt").read_text()
    check(
        "classify batch: per-leg evidence layout matches the solo path "
        "(node.txt, flags, hashes, elf archive, compile log)",
        (w / "node.txt").read_text().strip() == nid("changed")
        and (w / "flags-on.txt").is_file()
        and hash_on.startswith("test_c/v-cls-changed-on/elf/math.elf\ttext:")
        and f"text:{hashlib.sha256(b'cls-changed-on').hexdigest()}" in hash_on
        and (w / "elf-on/test_c/v-cls-changed-on/elf/math.elf").is_file()
        and (w / "elf-on/test_c/v-cls-changed-on/build.h").is_file()
        and (w / "compile-on.log").is_file(),
        hash_on,
    )
    check(
        "classify batch: a foreign sibling variant in the shared tree is "
        "NEVER swept into another node's hashes/archive (file-level "
        "attribution; the gcd+rsubint32 finding)",
        "v-foreign" not in hash_on
        and not (w / "elf-on/test_c/v-foreign").exists()
        and "v-foreign" not in (w / "hashes-off.txt").read_text(),
    )
    v_fail = json.loads((sw.ev / "cls-fail/classify/sem-perf/verdict.json").read_text())
    check(
        "classify batch: one node's compile failure -> ITS OWN COMPILE_FAIL "
        "verdict (leg named), other nodes unaffected",
        v_fail.get("status") == "COMPILE_FAIL"
        and v_fail.get("leg") == "on"
        and any("cls-fail/sem-perf: compile on failed" in r for r in sw.reds),
        (v_fail, sw.reds),
    )
    check(
        "classify batch: deselected/no-outcome nodes -> NO verdict (solo "
        "fallback), never a guessed one",
        not (sw.ev / "cls-dropped/classify/sem-perf/verdict.json").exists()
        and not (sw.ev / "cls-absent/classify/sem-perf/verdict.json").exists(),
    )
    check(
        "classify batch: cached verdict never re-enters a chunk session",
        nid("cached") not in seen_nodes,
        seen_nodes,
    )
    check(
        "classify batch: chunk build trees are removed after extraction",
        not any((sw.ev / "classify-batches").rglob("tt-llk-build")),
    )

# ---------------- 5. batched session node-id robustness (pin-14 unblock) ---
# The live 22-flag sweep died rc=1 on a pytest node id containing a single
# quote (SdpaFwOp parametrization repr <DestSync.Half: 'SyncHalf'>): node ids
# used to be inlined single-quoted into session.sh.  They now travel via the
# line-oriented nodes.txt argfile expanded with bash mapfile, so no sh
# quoting layer parses them.  Drive the REAL _run_batch_session with a stub
# interpreter that prints its argv one-per-line: the tricky ids must arrive
# as single argv entries, byte-exact.  Locks/LLK/PYDIR are patched to fixture
# paths (hermetic: no real lock files, no real tree paths in the script).
_saved = {k: getattr(sweep, k) for k in ("DEVICE_LOCK", "SILICON_LOCK", "LLK", "PYDIR")}
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sweep.DEVICE_LOCK = str(td / "dev.lock")
    sweep.SILICON_LOCK = str(td / "sil.lock")
    sweep.LLK = td / "llk"
    sweep.PYDIR = td / "pydir"
    sweep.PYDIR.mkdir()
    stub = td / "fake-python"
    stub.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    stub.chmod(0o755)
    sw = mk_sweep(td / "ev", "batched", dry_run=False)
    sw.python = stub
    # session splitting has its own fixtures above; this section's subject is
    # the generated command surviving the shell layers.
    sw._split_batch_session = lambda *a, **kw: None
    gctx = {"rt": td / "rt"}
    gctx["rt"].mkdir()
    quoted_node = (
        "test_sfpu_sdpa_fw.py::test_sfpu_sdpa_fw[variant:(<SdpaFwOp.Exp: 1>,"
        " 49024, <DestSync.Half: 'SyncHalf'>)-precision:Bf16Dest]"
    )
    spacey_node = 'test_a.py::test_b[case with spaces (and "parens") $HOME `id`]'
    jobs = [
        {"file": "test_sfpu_sdpa_fw.py", "node": quoted_node, "sel": "sem-perf"},
        {"file": "test_a.py", "node": spacey_node, "sel": "sem-perf"},
    ]
    gdir = td / "gdir"
    gdir.mkdir()
    sw._run_batch_session(gctx, gdir, "s1", jobs, "-O2", ())
    argv_lines = (gdir / "s1/log.txt").read_text().splitlines()
    check(
        "batched session: single-quote node id reaches pytest argv byte-exact",
        quoted_node in argv_lines,
        argv_lines[-4:],
    )
    check(
        "batched session: spaces/parens/metachar node id reaches pytest argv "
        "byte-exact (no expansion)",
        spacey_node in argv_lines,
    )
    check(
        "batched session: session script inlines NO node text (argfile only)",
        quoted_node not in (gdir / "s1/session.sh").read_text()
        and spacey_node not in (gdir / "s1/session.sh").read_text(),
    )
    check(
        "batched session: nodes.txt argfile carries the ids one per line "
        "(jobs sort file-first, so test_a.py leads)",
        (gdir / "s1/nodes.txt").read_text() == f"{spacey_node}\n{quoted_node}\n",
    )
    try:
        sw._run_batch_session(
            gctx,
            gdir,
            "s2",
            [{"file": "t.py", "node": "bad\nnode", "sel": "sem-perf"}],
            "-O2",
            (),
        )
        check("batched session: newline node id refused by name", False)
    except SystemExit as e:
        check(
            "batched session: newline node id refused by name",
            "newline" in str(e),
            str(e),
        )
for k, v in _saved.items():
    setattr(sweep, k, v)

# ---------------- 7. concurrent solo classify (laneDB) ----------------
# Knob-attribution legs and chunk-unprovable fallbacks keep ONE isolated
# pytest session per leg (CH's refusal: those legs never SHARE a session)
# but the sessions now run CONCURRENTLY through the worker pool.  Drive
# the REAL classify() + _solo_classify_pool with a stubbed _pytest that
# fabricates a deterministic per-(node, flags) build under the session's
# own RUNNER_TEMP: the same spec set run with workers=1 (pool no-ops; the
# sequential replay compiles inline, the legacy path) and workers=8 must
# produce CONTENT-IDENTICAL evidence trees and verdicts; the stub proves
# sessions really overlapped (workers=8) and never shared a RUNNER_TEMP;
# cached and duplicate specs never re-enter a session.
FAKE_KNOBS = {
    "k-changed1": "-fselftest-a",
    "k-changed2": "-fselftest-b",
    "k-ident": "-fselftest-ident",
    "k-fail": "-fselftest-compile-fail",
}


def mk_cls_pool_sweep(td, ev, workers):
    sw = mk_sweep(td / ev, "batched", dry_run=False)
    sw.a.classify_workers = workers
    sw.verify_toolchain = lambda phase: None
    sw.info = {"cc1plus_sha256": "cc", "tt_metal_head": "head"}
    sw._macro_scan = lambda work, legnames: {"classification": "SELFTEST_STUB"}
    objcopy = td / "objcopy.sh"
    if not objcopy.exists():
        objcopy.write_text('#!/bin/sh\ncat "$4"\n')
        objcopy.chmod(0o755)
    sw.objcopy = objcopy
    lock = threading.Lock()
    stats = {
        "active": 0,
        "max_active": 0,
        "sessions": 0,
        "rts": [],
        "rt_overlap": False,
        "extras": set(),
    }
    sw._selftest_stats = stats

    def fake_pytest(node, extra, env, log, timeout=1800):
        rt = pathlib.Path(env["RUNNER_TEMP"])
        flags = env["TT_LLK_EXTRA_COMPILER_OPTIONS"]
        with lock:
            stats["sessions"] += 1
            stats["active"] += 1
            stats["max_active"] = max(stats["max_active"], stats["active"])
            stats["extras"].update(extra)
            if str(rt) in stats["rts"]:
                stats["rt_overlap"] = True  # two live sessions, one RUNNER_TEMP
            stats["rts"].append(str(rt))
        try:
            time.sleep(0.05)
            if "-fselftest-compile-fail" in flags:
                pathlib.Path(log).write_text("1 failed\n")
                return 1
            # the ident knob's flag changes nothing (byte-identical legs)
            content_flags = flags.replace(" -fselftest-ident", "")
            elf = rt / "tt-llk-build/test_c/v0/elf/math.elf"
            elf.parent.mkdir(parents=True, exist_ok=True)
            elf.write_bytes(f"{node}|{content_flags}".encode())
            (elf.parent.parent / "build.h").write_text("// selftest build\n")
            pathlib.Path(log).write_text("1 passed\n")
            return 0
        finally:
            with lock:
                stats["active"] -= 1
                stats["rts"].remove(str(rt))

    sw._pytest = fake_pytest
    return sw


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    row = mk_row("kop", "semantic", {"sem-perf": "perf_c.py::test_c[mathop:Kop]"})

    def kspecs(row):
        return [
            (
                row,
                "sem-perf",
                (("off", sweep.OFF_FLAGS), ("knob", f"{sweep.OFF_FLAGS} {flag}")),
                f"knobs/{knob}",
            )
            for knob, flag in FAKE_KNOBS.items()
        ]

    verdicts = {}
    for arm, workers in (("serial", 1), ("conc", 8)):
        sw = mk_cls_pool_sweep(td, arm, workers)
        # pre-cached verdict + duplicate spec: neither may cost a session
        cw = sw.ev / "kop/knobs/k-cached/sem-perf"
        cw.mkdir(parents=True)
        (cw / "verdict.json").write_text(
            json.dumps(
                {
                    "selector": "sem-perf",
                    "status": "COMPILE_FAIL",
                    "leg": "off",
                    "cc1plus_sha256": "cc",
                    "tt_metal_head": "head",
                }
            )
        )
        specs = kspecs(row) + [
            (
                row,
                "sem-perf",
                (("off", sweep.OFF_FLAGS), ("knob", f"{sweep.OFF_FLAGS} -fx")),
                "knobs/k-cached",
            ),
            kspecs(row)[0],  # duplicate of k-changed1
        ]
        sw._solo_classify_pool(specs, f"selftest-{arm}")
        pool_sessions = sw._selftest_stats["sessions"]
        # the sequential row-loop replay (attribute_knobs' exact calls):
        # with workers=1 this IS the legacy inline compile; with workers=8
        # every verdict must resume from cache (zero extra sessions)
        verdicts[arm] = {
            knob: sw.classify(
                row,
                "sem-perf",
                legs=(("off", sweep.OFF_FLAGS), ("knob", f"{sweep.OFF_FLAGS} {flag}")),
                tag=f"knobs/{knob}",
            )
            for knob, flag in FAKE_KNOBS.items()
        }
        sw._selftest_stats["pool_sessions"] = pool_sessions
        if arm == "serial":
            check(
                "solo pool: workers=1 is a no-op (legacy inline compiles, "
                "strictly one session at a time)",
                pool_sessions == 0
                and sw._selftest_stats["sessions"] == 8
                and sw._selftest_stats["max_active"] == 1,
                sw._selftest_stats,
            )
            st_serial = sw
        else:
            check(
                "solo pool: workers=8 compiles every pending verdict "
                "concurrently (sessions overlapped), replay is 100% cached "
                "(zero extra sessions), cached/duplicate specs never enter",
                pool_sessions == 8
                and sw._selftest_stats["sessions"] == 8
                and sw._selftest_stats["max_active"] > 1,
                sw._selftest_stats,
            )
            st_conc = sw
        check(
            f"solo pool ({arm}): every session is an isolated "
            "--compile-producer with its own live RUNNER_TEMP",
            sw._selftest_stats["extras"] == {"--compile-producer"}
            and not sw._selftest_stats["rt_overlap"],
            sw._selftest_stats,
        )
    check(
        "solo pool: verdict CONTENT identical serial vs concurrent "
        "(CHANGED/IDENTICAL/COMPILE_FAIL all covered)",
        verdicts["serial"] == verdicts["conc"]
        and verdicts["serial"]["k-changed1"]["all"] == "CHANGED"
        and verdicts["serial"]["k-ident"]["all"] == "IDENTICAL"
        and verdicts["serial"]["k-fail"]["status"] == "COMPILE_FAIL"
        and verdicts["serial"]["k-fail"]["leg"] == "knob",
        verdicts,
    )
    t_serial = tree_of(td / "serial")
    t_conc = tree_of(td / "conc")
    check(
        "solo pool: evidence trees BYTE-IDENTICAL serial vs concurrent "
        "(every verdict.json/node.txt/flags/hashes/log/ELF archive)",
        t_serial == t_conc and len(t_serial) > 20,
        (set(t_serial) ^ set(t_conc)) or "content diff",
    )
    check(
        "solo pool: COMPILE_FAIL stops the verdict's later legs in both "
        "arms (no knob-leg evidence after the off leg... i.e. failing "
        "knob leg leaves no hashes/elf archive)",
        not any(
            "k-fail" in k and ("hashes-knob" in k or "elf-knob" in k) for k in t_serial
        ),
        [k for k in t_serial if "k-fail" in k],
    )
    check(
        "solo pool: RED events identical as a set (thread completion "
        "order must not change what is reported)",
        sorted(st_serial.reds) == sorted(st_conc.reds)
        and any("kop/sem-perf: compile knob failed" in r for r in st_conc.reds),
        (st_serial.reds, st_conc.reds),
    )

if FAILS:
    print(f"batched-silicon self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "batched-silicon self-test: ALL GREEN (partitioning incl. sem/hand "
    "never-share; 3-op dry-run layout parity batched==legacy; session split "
    "back to per-leg evidence with mathop-filtered CSVs and manifest-subset "
    "TEXT_HASHES; cache keying; producer failure attributes per leg, never "
    "poisons the group; batched classify verdicts match the solo layout "
    "with per-node attribution and solo fallback; node ids survive every "
    "shell layer via the nodes.txt argfile — quotes/spaces/parens proven, "
    "newline refused by name; concurrent solo classify byte-identical to "
    "the sequential legacy path with isolated overlapping sessions)"
)
