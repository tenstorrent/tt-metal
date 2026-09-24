#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce every table from the #57444 evaluation.

Configures and builds each standard-library configuration, then runs all seven items and prints
the results as markdown, ready to paste into the issue. Nothing is cached between runs except the
build directories.

    ./reproduce.py                  # everything, 5 repetitions
    ./reproduce.py --repetitions 3  # faster, noisier
    ./reproduce.py --only gcc       # a single configuration
    ./reproduce.py --skip-build     # reuse existing build directories
"""

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# (key, build dir, compiler, extra flags, label)
CONFIGS = [
    ("gcc", "out-gcc", "g++-12", [], "gcc-12 / libstdc++"),
    ("clang", "out-clang", "clang++-20", [], "clang-20 / libstdc++"),
    ("libcxx", "out-libcxx", "clang++-20", ["-stdlib=libc++"], "clang-20 / libc++"),
]
CANDIDATES = ["std", "zoo", "fu2"]
RUNTIME_CANDIDATES = [*CANDIDATES, "ttsl"]
COLUMNS = {"std": "StdFn", "zoo": "ZooFn", "fu2": "Fu2Fn", "ttsl": "TtslFn"}


def run(cmd, **kw):
    # cmd is always a list and shell=False, so no shell parses these arguments and there is no
    # injection path; every value is hardcoded in CONFIGS or constrained by argparse. SAST tools
    # tend to flag the non-literal argument anyway; that has been reviewed and dismissed.
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def configure_and_build(cfg, skip_build):
    _, out, cxx, extra, label = cfg
    path = os.path.join(HERE, out)
    if not skip_build:
        r = run(
            [
                "cmake",
                "-S",
                HERE,
                "-B",
                path,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CXX_COMPILER={cxx}",
                f"-DCMAKE_CXX_FLAGS={' '.join(extra)}",
            ]
        )
        if r.returncode:
            sys.exit(f"configure failed for {label}:\n{r.stderr[-2000:]}")
    r = run(["cmake", "--build", path, "-j", str(os.cpu_count() or 8)])
    if r.returncode:
        sys.exit(f"build failed for {label}:\n{r.stderr[-2000:]}")
    return path


def check_capacity(path, label):
    """sbo_probe exits non-zero if the pinned capacity no longer matches std::function."""
    r = run([os.path.join(path, "sbo_probe")])
    if r.returncode:
        sys.exit(f"capacity mismatch in {label} -- the comparison would not be fair:\n{r.stdout}")
    return r.stdout.strip().splitlines()


def parse_bench_csv(text):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("name,"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    med = {r["name"][:-7]: float(r["real_time"]) for r in rows if r["name"].endswith("_median")}
    cv = {r["name"][:-3]: float(r["real_time"]) * 100 for r in rows if r["name"].endswith("_cv")}
    allocs = {r["name"][:-7]: r.get("allocs/iter", "") for r in rows if r["name"].endswith("_median")}
    return med, cv, allocs


def run_bench(path, reps):
    r = run(
        [
            os.path.join(path, "bench"),
            f"--benchmark_repetitions={reps}",
            "--benchmark_report_aggregates_only=true",
            "--benchmark_format=csv",
        ]
    )
    if r.returncode:
        sys.exit(f"bench failed:\n{r.stderr[-2000:]}")
    return parse_bench_csv(r.stdout)


def object_sizes(path):
    """Item 6: .text bytes of the same TU compiled once per candidate."""
    out = {}
    for cand in CANDIDATES:
        hits = [
            os.path.join(dp, f)
            for dp, _, fs in os.walk(path)
            for f in fs
            if f == "codegen_tu.cpp.o" and f"codegen_{cand}" in dp
        ]
        if not hits:
            continue
        s = run(["size", "-A", hits[0]]).stdout
        out[cand] = sum(int(l.split()[1]) for l in s.splitlines() if l.startswith(".text"))
    return out


def compile_times(cfg, best_of=3):
    """Item 7: parse + instantiation cost, one library per translation unit."""
    _, out, cxx, extra, _ = cfg
    dep = os.path.join(HERE, out, "_deps")
    fu2 = os.path.join(dep, "function2-src", "include")
    zoo = os.path.join(dep, "zoo-src", "inc")
    res = {}
    for cand in CANDIDATES:
        best = None
        for _ in range(best_of):
            cmd = [
                cxx,
                "-std=c++20",
                "-O2",
                *extra,
                f"-DCANDIDATE_{cand.upper()}",
                f"-I{fu2}",
                f"-I{zoo}",
                f"-I{HERE}",
                "-c",
                "-o",
                os.devnull,
                os.path.join(HERE, "compile_time_tu.cpp"),
            ]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True)
            dt = time.perf_counter() - t0
            if r.returncode:
                sys.exit(f"compile-time probe failed for {cand}:\n{r.stderr.decode()[-2000:]}")
            best = dt if best is None else min(best, dt)
        res[cand] = best
    return res


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--only", choices=[c[0] for c in CONFIGS], action="append")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    for tool in ("cmake", "size"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    configs = [c for c in CONFIGS if not args.only or c[0] in args.only]
    configs = [c for c in configs if shutil.which(c[2]) or sys.exit(f"{c[2]} not found on PATH")]

    bench, sizes, ctimes, banners = {}, {}, {}, {}
    for cfg in configs:
        key, _, _, _, label = cfg
        print(f"# {label} ...", file=sys.stderr)
        path = configure_and_build(cfg, args.skip_build)
        banners[key] = check_capacity(path, label)
        bench[key] = run_bench(path, args.repetitions)
        sizes[key] = object_sizes(path)
        ctimes[key] = compile_times(cfg)

    print(f"Median ns over {args.repetitions} repetitions, +/- coefficient of variation.\n")
    for key, _, _, _, label in configs:
        print(f"- **{label}** -- {'; '.join(banners[key][:3])}")
    print()

    # Runtime scenarios, grouped as in the issue.
    scen = defaultdict(dict)
    for key, _, _, _, label in configs:
        med, cv, allocs = bench[key]
        for base, v in med.items():
            m = re.match(r"(\w+)<(\w+)(?:, (\w+))?>(?:/(\d+))?", base)
            if m:
                k = (m.group(1), m.group(3) or "", m.group(4) or "")
                scen[k].setdefault(label, {})[m.group(2)] = (v, cv.get(base, 0.0), allocs.get(base, ""))
            elif base.startswith("BM_MoveOnlyCapture_Std"):
                scen[("BM_MoveOnlyCapture", "", "")].setdefault(label, {})["StdFn"] = (
                    v,
                    cv.get(base, 0.0),
                    allocs.get(base, ""),
                )

    for (name, cap, arg), per in scen.items():
        title = name + (f" [{cap}]" if cap else "") + (f"/{arg}" if arg else "")
        print(f"### {title}\n")
        rows = []
        for _, _, _, _, label in configs:
            got = per.get(label, {})
            cells = []
            for c in RUNTIME_CANDIDATES:
                col = COLUMNS[c]
                if col in got:
                    v, cvv, al = got[col]
                    cells.append(f"{v:.2f} ±{cvv:.1f}%" + (f" ({al} allocs)" if al not in ("", "0") else ""))
                else:
                    cells.append("—")
            rows.append([label, *cells])
        print(md_table(["config", *RUNTIME_CANDIDATES], rows) + "\n")

    print("### Item 6 — object size, `.text` bytes\n")
    print(
        md_table(
            ["config", "std", "zoo", "fu2"],
            [[l, *[str(sizes[k].get(c, "—")) for c in CANDIDATES]] for k, _, _, _, l in configs],
        )
        + "\n"
    )

    print("### Item 7 — compile time, 300 instantiations, best of 3\n")
    rows = []
    for k, _, _, _, l in configs:
        base = ctimes[k]["std"]
        rows.append([l, *[f"{ctimes[k][c]:.2f}s ({ctimes[k][c]/base:.2f}x)" for c in CANDIDATES]])
    print(md_table(["config", "std", "zoo", "fu2"], rows))


if __name__ == "__main__":
    main()
