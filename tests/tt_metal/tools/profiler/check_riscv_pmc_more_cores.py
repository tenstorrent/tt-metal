# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-card Blackhole check of the RISC-V counter capture on idle ethernet cores and the DRAM tile RISC.

Runs build/test/tt_metal/tools/profiler/test_riscv_pmc_more_cores (one timestamped-events kernel per idle
ethernet core and one DRISC kernel, two launches each) twice and checks profile_log_device.csv:

  off  TT_METAL_DEVICE_PROFILER=1: no capture marker and no ERISC row in the log, the behaviour without the option.
  on   TT_METAL_PROFILE_RISCV_PMC set: every idle ethernet core has one 9092/9093/9094 triple per launch on
       profiler RISC ERISC, the selects are echoed in 9094 and 0 < minstret <= mcycle. ERISC is the only profiler
       RISC in the log, so neither ERISC1 nor the DRAM tile RISC contributes a row, and both runs complete.

Run it alone on an idle card: python3 tests/tt_metal/tools/profiler/check_riscv_pmc_more_cores.py
"""

import csv
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BINARY = os.path.join(REPO, "build/test/tt_metal/tools/profiler/test_riscv_pmc_more_cores")
LOG = os.path.join(REPO, "generated/profiler/.logs/profile_log_device.csv")
LAUNCHES = 2
EV3, EV4 = 0x10110111, 0xA0000000
IDS = {9092: "counts", 9093: "cycles", 9094: "events"}


def run(env_extra):
    if os.path.exists(LOG):
        os.remove(LOG)
    env = dict(os.environ, TT_METAL_SLOW_DISPATCH_MODE="1", TT_METAL_DEVICE_PROFILER="1", **env_extra)
    result = subprocess.run([BINARY], cwd=REPO, env=env, capture_output=True, text=True)
    sys.stdout.write(result.stdout[-2000:])
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"{BINARY} failed with {result.returncode}")
    with open(LOG, newline="") as f:
        f.readline()
        rows = list(csv.DictReader((line.replace(", ", ",") for line in f)))
    return rows


def pmc_by_core(rows):
    cores = {}
    for r in rows:
        tid = int(r["timer_id"])
        if tid in IDS:
            cores.setdefault((r["core_x"], r["core_y"]), []).append(
                (r["RISC processor type"], IDS[tid], int(r["data"]))
            )
    return cores


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    return cond


def main():
    ok = True

    rows = run({})
    ok &= check(not pmc_by_core(rows), "off: no capture marker in the log")
    ok &= check(not any(r["RISC processor type"].strip() == "ERISC" for r in rows), "off: no idle ethernet row")

    rows = run({"TT_METAL_PROFILE_RISCV_PMC": f"{EV3:#x},{EV4:#x}"})
    ok &= check({r["RISC processor type"].strip() for r in rows} <= {"ERISC"}, "on: ERISC is the only profiler RISC")
    cores = pmc_by_core(rows)
    ok &= check(len(cores) > 0, f"on: {len(cores)} idle ethernet cores have capture markers")
    for core, markers in sorted(cores.items()):
        kinds = [k for _, k, _ in markers]
        good = all(kinds.count(k) == LAUNCHES for k in IDS.values())
        good &= all(risc.strip() == "ERISC" for risc, _, _ in markers)
        for _, kind, data in markers:
            if kind == "events":
                good &= data == (EV4 << 32) | EV3
            if kind == "cycles":
                mcycle, minstret = data & 0xFFFFFFFF, data >> 32
                good &= 0 < minstret <= mcycle
        ok &= check(good, f"on: core {core}: {LAUNCHES} launches, selects echoed, 0 < minstret <= mcycle")
    zones = {r["zone name"].strip() for r in rows}
    ok &= check("ERISC-IDLE-FW" in zones, "on: idle firmware zone present")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
