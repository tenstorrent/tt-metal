#!/usr/bin/env python3
"""galaxy-kit per-chip benchmark worker — work-stealing, execute-only.

Runs on a galaxy compute node.  ELFs were compiled on quietbox (stage.sh)
and shipped in builds/<group>/; every pytest session runs with
--compile-consumer and needs no riscv toolchain (only the shipped
riscv-tt-elf-size/objdump for the TEXT_SIZE metric).

Honesty rules baked in:
  - CORR GATE: an arm's perf reps run only after that arm's correctness
    node PASSED on this same chip (<arm>-CORR-FAIL.txt otherwise).
  - SAME-CHIP PAIRS: both arms of an (op, leg) item run back-to-back on
    the claiming worker's chip; the chip id is recorded per cell.
  - REPS: LK_REPS solo perf sessions per arm (default 5); reps are
    expected cycle-identical (the collector reports spreads).
  - NO DEVICE RESETS ever in a worker (EXABOX.md §7 wall 2): recovery is
    one retry after a kill-tree session timeout, nothing else.

Work-stealing: queue/<op>__<leg>__c<k> items, claimed by atomic mkdir in
claims/.  A worker skips copies of an (op,leg) it already ran, so K copies
land on K distinct chips.  Resume-safe: session dirs with rc.txt are kept.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("LK_BASE", "/data/nkapre/craq-laneLK"))
NREPS = int(os.environ.get("LK_REPS", "5"))
SESSION_TIMEOUT = int(os.environ.get("LK_SESSION_TIMEOUT", "900"))


def read_arms():
    arms = {}
    with (BASE / "ARMS.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            arms.setdefault((r["op"], r["leg"]), []).append(r)
    return arms


def group_of(arm):
    g = arm["flagskey"]
    if arm["env"]:
        g += "+" + re.sub(r"[^A-Za-z0-9]+", "_", arm["env"])[:30]
    return g


class Worker:
    def __init__(self, chip, device=None):
        self.chip = chip
        self.device = str(chip) if device is None else device
        self.tag = f"chip{chip:02d}"
        self.wroot = BASE / "workers" / self.tag
        self.farm = self.wroot / "farm/tt_metal/tt-llk"
        self.python = str(BASE / "venv/bin/python")
        self.rt_cache = {}
        self.done_ops = set()
        self.wroot.mkdir(parents=True, exist_ok=True)
        self._setup_farm()
        d = self.wroot / "done-ops.txt"
        if d.is_file():
            for line in d.read_text().splitlines():
                if line.strip():
                    self.done_ops.add(tuple(line.split("\t")[:2]))

    def _setup_farm(self):
        # private farm copy: perf_data isolation between 32 workers
        if (self.farm / "tests/python_tests/conftest.py").is_file():
            return
        self.farm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE / "farm/tt_metal/tt-llk", self.farm, symlinks=True)

    def _runner_temp(self, group):
        # private hardlink clone of the shipped build: consumer sessions only
        # ADD files (temp_perf_data, order_records) — never edit in place
        if group in self.rt_cache:
            return self.rt_cache[group]
        rt = self.wroot / "rt" / group
        if not (rt / "tt-llk-build").is_dir():
            rt.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "cp",
                    "-al",
                    str(BASE / "builds" / group / "tt-llk-build"),
                    str(rt / "tt-llk-build"),
                ],
                check=True,
            )
        self.rt_cache[group] = rt
        return rt

    def _session(self, sdir, node, flags, env_extra, timeout=SESSION_TIMEOUT):
        rcf = sdir / "rc.txt"
        if rcf.is_file():
            return int(rcf.read_text().strip() or 99)
        sdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for k in ("TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR_PATH"):
            env.pop(k, None)
        group = env_extra.pop("__group")
        rt = self._runner_temp(group)
        env.update(
            CHIP_ARCH="blackhole",
            SHORT_ARCH="bh",
            LLK_HOME=str(self.farm),
            RUNNER_TEMP=str(rt),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
            TT_VISIBLE_DEVICES=self.device,
            PYTHONUNBUFFERED="1",
        )
        env.update(env_extra)
        shutil.rmtree(self.farm / "perf_data", ignore_errors=True)
        shutil.rmtree(rt / "tt-llk-build/temp_perf_data", ignore_errors=True)
        (sdir / "node.txt").write_text(node + "\n")
        (sdir / "flags.txt").write_text(flags + "\n")
        cmd = [
            self.python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--compile-consumer",
            node,
        ]
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd,
                cwd=self.farm / "tests/python_tests",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            out, rc = r.stdout, r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"") + b"\nLK-SESSION-TIMEOUT\n"
            rc = 98
        dt = time.time() - t0
        (sdir / "log.txt").write_bytes(out)
        if b" passed" not in out and rc == 0:
            rc = 97  # green rc with no pass line = silently-empty session
        pd = self.farm / "perf_data"
        if pd.is_dir():
            shutil.copytree(pd, sdir / "perf_data", dirs_exist_ok=True)
            shutil.rmtree(pd, ignore_errors=True)
        (sdir / "secs.txt").write_text(f"{dt:.1f}\n")
        rcf.write_text(f"{rc}\n")
        return rc

    def run_item(self, op, leg, arms):
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        res.mkdir(parents=True, exist_ok=True)
        (res / "chip.txt").write_text(
            f"{self.chip}\tdevice={self.device}\thost={os.uname().nodename}\n"
        )
        fails = 0
        for arm in arms:
            flags = (BASE / "flags" / (arm["flagskey"] + ".txt")).read_text().strip()
            env_extra = {"__group": group_of(arm)}
            for kv in filter(None, arm["env"].split(";")):
                k, _, v = kv.partition("=")
                env_extra[k] = v
            aname = arm["arm"]
            if arm["corr_node"]:
                rc = self._session(
                    res / f"{aname}-corr", arm["corr_node"], flags, dict(env_extra)
                )
                if rc != 0:
                    (res / f"{aname}-CORR-FAIL.txt").write_text(f"rc={rc}\n")
                    print(
                        f"[{self.tag}] {op}/{leg} {aname}: CORR FAIL rc={rc}",
                        flush=True,
                    )
                    fails += 1
                    continue
            for k in range(1, NREPS + 1):
                sdir = res / f"{aname}-perf-r{k}"
                rc = self._session(sdir, arm["perf_node"], flags, dict(env_extra))
                if rc != 0 and not (sdir / "rc.first.txt").is_file():
                    # one retry (transient hiccup); kill-tree only, no reset
                    (sdir / "rc.txt").rename(sdir / "rc.first.txt")
                    if (sdir / "log.txt").is_file():
                        (sdir / "log.txt").rename(sdir / "log.first.txt")
                    shutil.rmtree(sdir / "perf_data", ignore_errors=True)
                    rc = self._session(sdir, arm["perf_node"], flags, dict(env_extra))
                if rc != 0:
                    fails += 1
                print(f"[{self.tag}] {op}/{leg} {aname} r{k}: rc={rc}", flush=True)
        with (self.wroot / "done-ops.txt").open("a") as f:
            f.write(f"{op}\t{leg}\n")
        self.done_ops.add((op, leg))
        return fails

    def steal_loop(self):
        arms = read_arms()
        qdir = BASE / "queue"
        cdir = BASE / "claims"
        cdir.mkdir(exist_ok=True)
        idle_passes = 0
        while True:
            progressed = False
            for item in sorted(p.name for p in qdir.iterdir()):
                op, leg, _copy = item.rsplit("__", 2)
                if (op, leg) in self.done_ops:
                    continue
                try:
                    (cdir / item).mkdir()
                except FileExistsError:
                    continue
                (cdir / item / "owner.txt").write_text(self.tag + "\n")
                print(f"[{self.tag}] claimed {item}", flush=True)
                self.run_item(op, leg, arms[(op, leg)])
                (cdir / item / "done.txt").write_text("done\n")
                progressed = True
            if not progressed:
                idle_passes += 1
                if idle_passes >= 2:
                    break
                time.sleep(10)
        print(f"LK-WORKER-DONE {self.tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--device", help="TT_VISIBLE_DEVICES override (pilot/local)")
    ap.add_argument("--item", help="run one queue item directly (pilot mode)")
    args = ap.parse_args()
    w = Worker(args.chip, device=args.device)
    if args.item:
        op, leg = args.item.split("__")[:2]
        rc = w.run_item(op, leg, read_arms()[(op, leg)])
        sys.exit(0 if rc == 0 else 2)
    w.steal_loop()


if __name__ == "__main__":
    main()
