# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Hang watchdog for the M3 prefill stress test. On a stall it captures tt-triage state of every rank through the
live inspector RPC (port 50051 + rank; no reset, no kill) into <run>/hang_<utc>/, moe_fused_swiglu state first,
then writes CAPTURE_DONE.

  watch:   python3 stall_watchdog.py watch <run> --stall-s 240 [--until-epoch T]
           Stall = the producer has pushed chunks the last rank has not started, and no rank logged a
           CHUNK_START for --stall-s. Idle gaps of a paced producer are not stalls. Stops watching after an
           orderly SHUTDOWN or at --until-epoch. At most one capture per run.
  capture: python3 stall_watchdog.py capture <run> --why "..."

dump_callstacks halts RISCs: never point this at a healthy live run.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(HERE, *[".."] * 5)))
STEPS = [
    ("00_moe_swiglu", ["--run=dump_moe_swiglu", "-v"]),
    ("01_callstacks", ["--run=dump_running_operations", "--full-cores", "--run=dump_callstacks", "-vv", "--full-callstack"]),
    ("02_semaphores", ["--run=dump_semaphores", "--include-global-unchanged"]),
    ("03_cbs", ["--run=dump_circular_buffers", "--dump-cb-content", "-vv"]),
    ("04_noc_eth", ["--run=check_noc_status", "--run=check_eth_status", "--run=dump_risc_debug_signals",
                    "--run=dump_lightweight_asserts"]),
    ("05_callstacks_again", ["--run=dump_callstacks", "-vv"]),
    ("06_mesh_sockets", ["--run=dump_mesh_sockets"]),
    ("07_moe_swiglu_again", ["--run=dump_moe_swiglu", "-v"]),  # static vs. still moving
]  # fmt: skip
CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+)")
ORDERLY_END = re.compile(r"shutdown complete|SHUTDOWN sentinel")


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Run:
    def __init__(self, run_dir, events):
        self.dir = os.path.abspath(run_dir)
        with open(os.path.join(self.dir, "run.json")) as f:
            self.devices = json.load(f)["devices"]
        self.ranks = list(range(len(self.devices)))
        self.events = events or os.path.join(self.dir, "events.log")

    def event(self, msg):
        with open(self.events, "a") as f:
            f.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} [watchdog] {msg}\n")

    def triage(self, rank, hang_dir, name, args, timeout=900):
        cmd = ["python3", os.path.join(TT, "tools", "tt-triage.py"), "--disable-progress", "--disable-colors",
               f"--inspector-rpc-port={50051 + rank}", "--inspector-disable-rank",
               f"--additional-scripts-directory={os.path.join(HERE, 'triage')}",
               f"--sqlite-output-path={hang_dir}/r{rank}_{name}.sqlite"] + args  # fmt: skip
        env = dict(os.environ, TT_VISIBLE_DEVICES=self.devices[rank], TT_METAL_HOME=TT, COLUMNS="400",
                   PYTHONPATH=f"{TT}:{TT}/ttnn:{TT}/tools")  # fmt: skip
        t0 = time.time()
        with open(os.path.join(hang_dir, f"r{rank}_{name}.txt"), "w") as f:
            f.write(f"# TT_VISIBLE_DEVICES={self.devices[rank]} {' '.join(cmd)}\n")
            f.flush()
            try:
                rc = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=TT, env=env
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        return rc, time.time() - t0

    def capture(self, why, last_lines=None):
        hd = os.path.join(self.dir, f"hang_{utc()}")
        os.makedirs(hd, exist_ok=True)
        self.event(f"HANG suspected: {why}; capturing ranks {self.ranks} to {hd}")
        with open(os.path.join(hd, "why.txt"), "w") as f:
            f.write(why + "\n")
            for r, line in sorted((last_lines or {}).items()):
                f.write(f"rank {r}: {line}\n")
        for log in ("runner.log", "producer.log"):
            try:
                with open(os.path.join(self.dir, log), errors="replace") as f:
                    tail = f.readlines()[-60:]
                with open(os.path.join(hd, log + ".tail"), "w") as g:
                    g.writelines(tail)
            except OSError:
                pass
        for name, args in STEPS:  # cheapest and most telling first, every rank per step
            for r in self.ranks:
                rc, el = self.triage(r, hd, name, args)
                self.event(f"  rank {r} {name}: rc={rc} {el:.0f}s")
        try:
            subprocess.run(["python3", os.path.join(HERE, "classify_moe.py"), hd],
                           stdout=open(os.path.join(hd, "classify.txt"), "w"), stderr=subprocess.STDOUT, timeout=300)  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            pass
        open(os.path.join(hd, "CAPTURE_DONE"), "w").close()
        self.event(f"capture done: {hd}")
        return hd


def _read_new(path, pos):
    try:
        with open(path, errors="replace") as f:
            f.seek(pos)
            data = f.read()
            return data, f.tell()
    except OSError:
        return "", pos


def watch(run, stall_s, until):
    runner_log, producer_log = os.path.join(run.dir, "runner.log"), os.path.join(run.dir, "producer.log")
    last_rank = run.ranks[-1]
    rpos = ppos = pushes = done_last = 0
    last_any, out_since, last_line = None, None, {}
    while True:
        pdata, ppos = _read_new(producer_log, ppos)
        pushes += pdata.count("[producer] push slot")
        rdata, rpos = _read_new(runner_log, rpos)
        for line in rdata.splitlines():
            m = CHUNK_START.search(line)
            if m:
                last_any = time.time()
                last_line[int(m.group(1))] = line[m.start() :][:300]
                done_last += int(m.group(1)) == last_rank
            if ORDERLY_END.search(line):
                run.event(f"orderly end seen ({line.strip()[-80:]}); stop watching")
                return
        outstanding = pushes - done_last
        if outstanding <= 0:
            out_since = None
        elif out_since is None:
            out_since = time.time()
        quiet_s = time.time() - max(last_any or 0, out_since or 0)
        if outstanding > 0 and quiet_s > stall_s:
            run.capture(
                f"no CHUNK_START for {quiet_s:.0f}s with {outstanding} chunks outstanding "
                f"(producer pushes {pushes}, last-rank CHUNK_STARTs {done_last})",
                last_line,
            )
            return
        if until and time.time() > until:
            run.event("deadline reached; stop watching")
            return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["watch", "capture"])
    ap.add_argument("run_dir")
    ap.add_argument("--stall-s", type=float, default=240)
    ap.add_argument("--until-epoch", type=float, default=None)
    ap.add_argument("--events", default=None, help="event log (default <run>/events.log)")
    ap.add_argument("--why", default="manual capture")
    a = ap.parse_args()
    run = Run(a.run_dir, a.events)
    if a.mode == "capture":
        run.capture(a.why)
    else:
        watch(run, a.stall_s, a.until_epoch)
    if glob.glob(os.path.join(run.dir, "hang_*", "CAPTURE_DONE")):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
