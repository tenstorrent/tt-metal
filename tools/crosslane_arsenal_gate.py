#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Cross-lane arsenal ACCEPTANCE GATE (lane FB, 2026-08-21; the
# lreg_arsenal_gate pattern).
#
# Runs the whole cross-lane battery against an arbitrary compiler build +
# pinned simulator and writes VERDICTS.tsv.  Three legs:
#
#   host      pure-host oracle identity battery
#             (tests/python_tests/test_crosslane_oracle_identities.py)
#   fixtures  demand-golden regeneration + byte-identity gate
#             (test_crosslane_demand_goldens.py + crosslane_fixtures/)
#   sim       permutation-identity lane tracer on the pinned simulator
#             (test_crosslane_lane_tracer.py + sources/sfpu_crosslane_probe.cpp)
#
# Modes:
#   --mode today   validate the arsenal itself against the current pin
#                  (tests/sfpi as wired in the tree).
#   --mode future  the BUILDER-ACCEPTANCE contract (lane FA's surface):
#                  run the same battery under the builder's toolchain.
#                  Wire the builder's compiler with --gcc-exec-prefix
#                  <hybrid>/compiler/lib/gcc/ (flags-off/cc1plus-only
#                  builds) or repoint $LLK_HOME/tests/sfpi at a full
#                  hybrid (new -m flags need the hybrid DRIVER too --
#                  lane DT recipe); pass new flags via --extra-flags.
#                  Every deviation from the recorded battery verdicts is
#                  a gate FAIL; FACT rows (swap-role, tie-divergence,
#                  aliasing, BF16-RMW) must reproduce identically.
#
# The sim must be the PINNED oracle (bh 32489dda4fd6... lineage, craq-sim
# 9f324140) or its reviewed successor; the gate records sha256 of the sim,
# driver and cc1plus in the TSV header. soc_descriptor.yaml must sit beside
# the .so (TT_METAL_SIMULATOR takes the FILE path).
#
# Usage:
#   tools/crosslane_arsenal_gate.py --mode today \
#       --llk-home <repo>/tt_metal/tt-llk \
#       --sim <simstage>/bh/libttsim.so [--out VERDICTS.tsv]
#   tools/crosslane_arsenal_gate.py --mode future ... \
#       [--gcc-exec-prefix <hybrid>/compiler/lib/gcc/] \
#       [--extra-flags "-mtt-tensix-..."]

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

FACT_RE = re.compile(
    r"(SWAP-ROLE-FACT|INDEXED-SWAP-ROLE|ALIAS-FACT|TIE-DIVERGENCE|"
    r"BF16-RMW-FACT|ORACLE MISMATCH)[:\s].*")
TEST_RE = re.compile(r"(\S+\.py::\S+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL)")


def sha(path):
    if not path or not os.path.exists(path):
        return "absent"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_pytest(pydir, venv_py, args, env, timeout=1800):
    cmd = [venv_py, "-m", "pytest", "-q", "-s", "--tb=line"] + args
    p = subprocess.run(cmd, cwd=pydir, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p


def parse_results(output):
    tests = TEST_RE.findall(output)
    facts = [m.group(0).strip() for m in FACT_RE.finditer(output)]
    return tests, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["today", "future"], required=True)
    ap.add_argument("--llk-home", required=True,
                    help="tt_metal/tt-llk of the tree under test")
    ap.add_argument("--sim", help="pinned libttsim.so (file path, "
                                  "soc_descriptor.yaml beside it)")
    ap.add_argument("--arch", default="blackhole")
    ap.add_argument("--gcc-exec-prefix",
                    help="hybrid <toolchain>/compiler/lib/gcc/ to select a "
                         "builder cc1plus without repointing tests/sfpi")
    ap.add_argument("--extra-flags", default="",
                    help="TT_LLK_EXTRA_COMPILER_OPTIONS for the probe builds")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--runner-temp")
    ap.add_argument("--out", default="VERDICTS.tsv")
    args = ap.parse_args()

    llk = os.path.abspath(args.llk_home)
    pydir = os.path.join(llk, "tests", "python_tests")
    venv_py = os.path.join(llk, "tests", ".venv", "bin", "python")
    if not os.path.exists(venv_py):
        print(f"FATAL: venv python not found at {venv_py}", file=sys.stderr)
        return 2
    sfpi = os.path.join(llk, "tests", "sfpi")
    driver = os.path.join(sfpi, "compiler", "bin", "riscv-tt-elf-g++")
    cc1plus = os.path.join(sfpi, "compiler", "libexec", "gcc",
                           "riscv-tt-elf", "15.1.0", "cc1plus")
    if args.gcc_exec_prefix:
        cand = os.path.join(os.path.dirname(
            os.path.dirname(args.gcc_exec_prefix.rstrip("/"))),
            "libexec", "gcc", "riscv-tt-elf", "15.1.0", "cc1plus")
        if os.path.exists(cand):
            cc1plus = cand

    env = dict(os.environ)
    env["CHIP_ARCH"] = args.arch
    env["LLK_HOME"] = llk
    rt = args.runner_temp or tempfile.mkdtemp(prefix="crosslane-gate-")
    env["RUNNER_TEMP"] = rt
    if args.gcc_exec_prefix:
        env["GCC_EXEC_PREFIX"] = args.gcc_exec_prefix
    if args.extra_flags:
        env["TT_LLK_EXTRA_COMPILER_OPTIONS"] = args.extra_flags
    if args.sim:
        env["TT_METAL_SIMULATOR"] = os.path.abspath(args.sim)

    rows = []
    facts_all = []
    fails = 0

    def leg(name, pytest_args, need_sim=False):
        nonlocal fails
        if need_sim and (args.skip_sim or not args.sim):
            rows.append((name, "-", "SKIP", "no --sim / --skip-sim"))
            return
        p = run_pytest(pydir, venv_py, pytest_args, env)
        tests, facts = parse_results(p.stdout + p.stderr)
        facts_all.extend((name, f) for f in facts)
        if not tests:
            rows.append((name, "-", "FAIL",
                         "no test results parsed (rc=%d)" % p.returncode))
            fails += 1
            sys.stderr.write(p.stdout[-4000:] + p.stderr[-2000:])
            return
        for tid, verdict in tests:
            ok = verdict in ("PASSED", "SKIPPED", "XFAIL")
            rows.append((name, tid, "PASS" if verdict == "PASSED"
                         else verdict, ""))
            if not ok:
                fails += 1
        if p.returncode != 0 and all(v == "PASSED" for _, v in tests):
            rows.append((name, "-", "FAIL",
                         f"pytest rc={p.returncode} despite green tests"))
            fails += 1

    leg("host", ["test_crosslane_oracle_identities.py"])
    leg("fixtures", ["test_crosslane_demand_goldens.py"])
    leg("sim", ["--run-simulator", "test_crosslane_lane_tracer.py"],
        need_sim=True)

    with open(args.out, "w") as f:
        f.write(f"# crosslane_arsenal_gate mode={args.mode} "
                f"arch={args.arch}\n")
        f.write(f"# driver_sha={sha(os.path.realpath(driver))} "
                f"cc1plus_sha={sha(os.path.realpath(cc1plus))} "
                f"sim_sha={sha(args.sim)}\n")
        f.write(f"# gcc_exec_prefix={args.gcc_exec_prefix or '-'} "
                f"extra_flags={args.extra_flags or '-'}\n")
        f.write("leg\tcase\tverdict\tdetail\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
        for leg_name, fact in facts_all:
            f.write(f"{leg_name}\tFACT\tINFO\t{fact}\n")

    n_pass = sum(1 for r in rows if r[2] == "PASS")
    print(f"crosslane_arsenal_gate: {n_pass} PASS, {fails} FAIL "
          f"-> {args.out}")
    for leg_name, fact in facts_all:
        print(f"  [{leg_name}] {fact}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
