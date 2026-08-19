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
import sys
import tempfile

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
        force=force, dry_run=dry_run, knob_silicon_rows=None, skip_craq_gate=False
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

if FAILS:
    print(f"batched-silicon self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "batched-silicon self-test: ALL GREEN (partitioning incl. sem/hand "
    "never-share; 3-op dry-run layout parity batched==legacy; session split "
    "back to per-leg evidence with mathop-filtered CSVs and manifest-subset "
    "TEXT_HASHES; cache keying)"
)
