#!/usr/bin/env python3
"""witness_preflight.py — union fire-witness check (conf-lint rule R9's
compile half; the pin-11 lesson turned into a gate).

Pin 9 shipped two ON-set flags that never engaged; pin 11 shipped
prgm-const whose fire witness holds on its lane's build but NOT on the
shipped nine-lane union (flag interactions).  The rule of pin 10 — "every
ON-set entry fire-witnessed" — was still being verified per-lane, by hand,
against the lane's own toolchain.  This helper verifies the witnesses ON
THE UNION, at the PINNED toolchain, mechanically:

For each entry of _REVIEWED_FIRE_WITNESSES in sweep_2x2.conf
(`flag|pytest-node|dump-flag|required-line-regex`), compile JUST that node
(pytest --compile-producer, BH) with the full reviewed ON set plus the
entry's dump flag, then search the produced GCC dump files for the required
line.  A missing line is RED naming the flag:

    fire witness stale on the union: <flag>

meaning the flag no longer demonstrably fires on the shipped flag UNION —
either the union interaction killed the fire (the pin-11 prgm-const case)
or the witness row went stale (node/line changed).  Both demand review, not
a silent nightly.

Fast by design: entries sharing a pytest node are compiled ONCE with the
union of their dump flags (the seeded table = 3 compiles).  Structure-only
validation of the table itself is conf_lint.sh R9; this helper re-checks
the same structure (a malformed table refuses, exit 2) and adds the
compile+grep half.

Exit codes: 0 all witnesses present; 1 at least one witness missing (RED);
2 configuration/environment error (malformed table, unpinned toolchain,
missing venv, compile failure).

Wired into nightly_bh_sweep.sh preflight; `--skip-witness` on the wrapper
is the loudly-logged emergency escape.  Self-tested (fixture dumps, no
toolchain) by selftest_witness_preflight.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONF = HERE / "sweep_2x2.conf"


def _load_sweep(path=HERE / "sweep_2x2.py"):
    spec = importlib.util.spec_from_file_location("sweep_2x2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_witness_table(conf_text):
    """Parse _REVIEWED_FIRE_WITNESSES from the conf TEXT (never sourced).

    Returns (rows, errors).  Each row: dict(flag, node, dump_flag, line).
    Errors are structural — conf_lint.sh R9 enforces the same rules; this
    parser refuses on them too so the compile half can never run against a
    malformed table."""
    m = re.search(r'^_REVIEWED_FIRE_WITNESSES="\n(.*?)^"\s*$', conf_text, re.M | re.S)
    if not m:
        return None, ['conf lacks a multi-line _REVIEWED_FIRE_WITNESSES="..." table']
    rows, errors, seen = [], [], set()
    for i, raw in enumerate(m.group(1).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4 or not all(parts):
            errors.append(
                f"witness table line {i}: need 4 non-empty '|' fields "
                f"(flag|pytest-node|dump-flag|required-line): {line!r}"
            )
            continue
        flag, node, dump_flag, need = parts
        if not re.fullmatch(r"-mtt-tensix-[a-z0-9-]+", flag):
            errors.append(
                f"witness table line {i}: '{flag}' is not a -mtt-tensix- flag"
            )
        if not re.fullmatch(r"-fdump-(rtl|tree)-rvtt_[a-z_]+", dump_flag):
            errors.append(
                f"witness table line {i}: '{dump_flag}' is not a "
                "-fdump-{rtl,tree}-rvtt_* dump flag"
            )
        key = (flag, node, dump_flag, need)
        if key in seen:
            errors.append(f"witness table line {i}: exact duplicate entry for {flag}")
        seen.add(key)
        rows.append({"flag": flag, "node": node, "dump_flag": dump_flag, "line": need})
    return rows, errors


def check_on_set(rows, on_flags):
    """Every witnessed flag must be in the reviewed ON set — a witness for a
    flag that left the union is a stale table (config error, not a RED)."""
    on = set(on_flags.split())
    return [r["flag"] for r in rows if r["flag"] not in on]


def dump_pass_name(dump_flag):
    """'-fdump-rtl-rvtt_macro_planner' -> 'rvtt_macro_planner'."""
    return re.sub(r"^-fdump-(rtl|tree)-", "", dump_flag)


def scan_dumps(build_root, rows):
    """Grep each row's required line across the dump files its pass
    produced under one build tree.  Returns {flag: (found, [files])} keyed
    by (flag, line) rows — pure; selftest-covered with fixture trees."""
    build_root = pathlib.Path(build_root)
    verdicts = []
    for r in rows:
        passname = dump_pass_name(r["dump_flag"])
        files = sorted(p for p in build_root.rglob(f"*.{passname}") if p.is_file())
        found_in = []
        for f in files:
            try:
                if re.search(r["line"], f.read_text(errors="replace"), re.M):
                    found_in.append(str(f))
            except OSError:
                continue
        verdicts.append(
            {
                "flag": r["flag"],
                "node": r["node"],
                "dump_flag": r["dump_flag"],
                "line": r["line"],
                "found": bool(found_in),
                "dump_files_scanned": len(files),
                "found_in": found_in[:5],
            }
        )
    return verdicts


def stale_message(v):
    """The RED line for one missing witness (selftest-covered)."""
    return (
        f"witness-preflight: RED — fire witness stale on the union: "
        f"{v['flag']} (required dump line {v['line']!r} MISSING from "
        f"{v['dump_files_scanned']} {dump_pass_name(v['dump_flag'])} "
        f"dump file(s) of node {v['node']} compiled at the pinned "
        "toolchain with the full reviewed ON set — the flag no "
        "longer demonstrably fires on the shipped union; fix the "
        "compiler interaction or re-review the witness row)"
    )


def group_rows(rows):
    """Entries sharing a node compile once with the union of dump flags."""
    groups = {}
    for r in rows:
        groups.setdefault(r["node"], {"node": r["node"], "dump_flags": [], "rows": []})
        g = groups[r["node"]]
        if r["dump_flag"] not in g["dump_flags"]:
            g["dump_flags"].append(r["dump_flag"])
        g["rows"].append(r)
    return list(groups.values())


def find_python(tests_dir):
    for c in (
        tests_dir / ".venv-laneE",
        tests_dir / ".venv",
        tests_dir / "python_tests/.venv",
    ):
        if (c / "bin/python").is_file():
            return c / "bin/python"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conf", type=pathlib.Path, default=DEFAULT_CONF)
    ap.add_argument(
        "--work",
        type=pathlib.Path,
        required=True,
        help="work/evidence dir for compile trees, logs and verdicts.json",
    )
    ap.add_argument(
        "--tt-metal-home",
        type=pathlib.Path,
        # HERE is .../tt_metal/tt-llk/tests/corpus -> checkout root is 4 up.
        default=HERE.parents[3],
        help="checkout whose harness compiles the witness nodes (default: "
        "the checkout containing this script)",
    )
    ap.add_argument(
        "--allow-pin-mismatch",
        action="store_true",
        help="LOUD escape: run against a non-pinned toolchain (lane "
        "verification of a candidate compiler); the nightly never passes this",
    )
    ap.add_argument(
        "--only-flag",
        action="append",
        default=[],
        help="check only these flags' witnesses (repeatable)",
    )
    a = ap.parse_args(argv)

    sweep = _load_sweep()
    conf_text = a.conf.read_text()
    rows, errors = parse_witness_table(conf_text)
    if errors:
        for e in errors:
            print(f"witness-preflight: CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    if a.only_flag:
        rows = [r for r in rows if r["flag"] in set(a.only_flag)]
    if not rows:
        print("witness-preflight: no witness entries to check — GREEN (vacuous)")
        return 0
    stale = check_on_set(rows, sweep.ON_FLAGS)
    if stale:
        print(
            "witness-preflight: CONFIG ERROR: witnessed flag(s) not in the "
            f"reviewed ON set (stale table): {', '.join(sorted(set(stale)))}",
            file=sys.stderr,
        )
        return 2

    # Pinned-toolchain requirement: the witness proves the fire at the PIN.
    pin = re.search(r"^_REVIEWED_CC1PLUS_SHA256=([0-9a-f]{64})$", conf_text, re.M)
    if not pin:
        print(
            "witness-preflight: CONFIG ERROR: conf lacks a full "
            "_REVIEWED_CC1PLUS_SHA256 pin",
            file=sys.stderr,
        )
        return 2
    llk = a.tt_metal_home / "tt_metal/tt-llk"
    compiler = llk / "tests/sfpi/compiler/bin/riscv-tt-elf-g++"
    if not compiler.is_file():
        print(
            f"witness-preflight: ENV ERROR: no harness toolchain at {compiler}",
            file=sys.stderr,
        )
        return 2
    cc1 = subprocess.run(
        [str(compiler), "-print-prog-name=cc1plus"], capture_output=True, text=True
    ).stdout.strip()
    if not cc1 or not pathlib.Path(cc1).is_file():
        print(
            f"witness-preflight: ENV ERROR: cannot resolve cc1plus via {compiler}",
            file=sys.stderr,
        )
        return 2
    cc1_sha = sweep.sha256(pathlib.Path(cc1))
    if cc1_sha != pin.group(1):
        msg = (
            f"resolved cc1plus {cc1_sha} != reviewed pin {pin.group(1)} " f"(at {cc1})"
        )
        if a.allow_pin_mismatch:
            print(
                f"witness-preflight: WARNING — PIN MISMATCH HONORED "
                f"(--allow-pin-mismatch): {msg}; verdicts below are NOT "
                "statements about the reviewed pin",
                file=sys.stderr,
            )
        else:
            print(
                f"witness-preflight: ENV ERROR: {msg} — the witness gate "
                "proves fires at the PIN; repoint tests/sfpi or pass "
                "--allow-pin-mismatch for a lane-candidate run",
                file=sys.stderr,
            )
            return 2
    python = find_python(llk / "tests")
    if python is None:
        print(
            "witness-preflight: ENV ERROR: no tt-llk virtualenv under "
            f"{llk / 'tests'} (.venv-laneE/.venv/python_tests/.venv)",
            file=sys.stderr,
        )
        return 2

    a.work.mkdir(parents=True, exist_ok=True)
    all_verdicts = []
    for g in group_rows(rows):
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", g["node"])[:80]
        rt = a.work / f"rt-{tag}"
        shutil.rmtree(rt, ignore_errors=True)
        rt.mkdir(parents=True)
        log = a.work / f"compile-{tag}.log"
        env = dict(
            os.environ,
            CHIP_ARCH="blackhole",
            LLK_HOME=str(llk),
            RUNNER_TEMP=str(rt),
            TT_LLK_EXTRA_COMPILER_OPTIONS=sweep.ON_FLAGS
            + " "
            + " ".join(g["dump_flags"]),
        )
        with open(log, "w") as f:
            rc = subprocess.run(
                [str(python), "-m", "pytest", "-q", "--compile-producer", g["node"]],
                cwd=llk / "tests/python_tests",
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800,
            ).returncode
        if rc != 0:
            print(
                f"witness-preflight: ENV ERROR: witness node compile failed "
                f"(rc={rc}) for {g['node']} — see {log}",
                file=sys.stderr,
            )
            return 2
        all_verdicts += scan_dumps(rt / "tt-llk-build", g["rows"])
    (a.work / "verdicts.json").write_text(
        json.dumps({"cc1plus_sha256": cc1_sha, "verdicts": all_verdicts}, indent=2)
        + "\n"
    )
    red = 0
    for v in all_verdicts:
        if v["found"]:
            print(
                f"witness-preflight: GREEN {v['flag']} — {v['line']!r} present "
                f"({len(v['found_in'])}+ dump file(s), node {v['node']})"
            )
        else:
            red = 1
            print(stale_message(v))
    if red:
        print(
            "witness-preflight: FAILED — at least one ON-set flag has a "
            "stale union fire witness (the pin-11 lesson; see RED lines)",
        )
    else:
        print(
            "witness-preflight: ALL GREEN — every declared witness fires on the union"
        )
    return red


if __name__ == "__main__":
    sys.exit(main())
