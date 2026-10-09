#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Rebuild perf_split_costs.json from the JUnit reports of LLK perf CI runs.

usage: perf_split_costs.py <arch>=<run id>[,<run id>...] [...]
   e.g. perf_split_costs.py wormhole=37617426927,37617449196 blackhole=37617444663

Cost per module = compile worker-seconds / COMPILE_WORKERS + measure worker-seconds / MEASURE_WORKERS,
averaged over the runs: the time the module adds to a shard. The cost is per module, not per test,
because which test of a module compiles a shared ELF depends on the order (see the split in llk_pytest_plugin.py).
"""
import collections, glob, json, pathlib, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

COMPILE_WORKERS, MEASURE_WORKERS = (
    10,
    15,
)  # -n of the two passes in run_llk_perf_<arch>.sh
OUT = pathlib.Path(__file__).with_name("perf_split_costs.json")

table = json.loads(OUT.read_text()) if OUT.exists() else {}
for spec in sys.argv[1:]:
    arch, ids = spec.split("=")
    ids = ids.split(",")
    cost, tests = collections.Counter(), collections.Counter()
    for rid in ids:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    rid,
                    "-R",
                    "tenstorrent/tt-metal",
                    "-p",
                    f"perf-junit-report-{arch}-*",
                    "-D",
                    tmp,
                ],
                check=True,
            )
            for kind, workers in (
                ("compile", COMPILE_WORKERS),
                ("run", MEASURE_WORKERS),
            ):
                for path in glob.glob(f"{tmp}/*/*-{kind}.xml"):
                    for case in ET.parse(path).getroot().iter("testcase"):
                        module = case.get("classname", "").split(".")[0]
                        cost[module] += float(case.get("time", 0)) / workers / len(ids)
                        if kind == "compile":
                            tests[module] += 1 / len(ids)
    table[arch] = {
        m: {"cost_s": round(cost[m], 1), "tests": round(tests[m])} for m in sorted(cost)
    }
OUT.write_text(json.dumps(table, indent=1) + "\n")
