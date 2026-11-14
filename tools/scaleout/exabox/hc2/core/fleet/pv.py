#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""pv.py: the power virus, the bus-bar stage's other max load.

run-ttx --broadcast --timeout inf --interface pci:<i> single-core-matrix-pv-fp32-noc.ttx on every chip at once, each in its
own session (start_new_session) with its own TMPDIR (run-ttx is a PyInstaller onefile), for --load-s s. A load only: no
numeric check. --tdp does not cap it (~167 W per chip at a 160 W limit), so it stops early, all chips at once, on:
  a power brick at or above --brick-c (rows "epoch name C" in --brick, the read-only sdr poll fleet.sh runs),
  a feed above --feed-a (the HSC sampler's rows "epoch tray addr VIN_raw IOUT_raw" in --hsc, IOUT at 62.5 mA/LSB),
each only where its file has rows. The chips' power never stops it:
their summed telemetry power (--telem, c8_telemetry.txt rows "epoch bdf
temp_mC power_uW ...", latest per chip) is only logged as a peak. Ends with SIGINT, SIGTERM, SIGKILL to each of our own process groups (by pgid, never by
name), confirms none is left and removes its scratch dir. The pgids go to --pgids first, one "pgid starttime" line each
(/proc/<pid>/stat field 22, 0 when unreadable), so fleet.sh can end any left if this process is killed, and only while the
leader still has that start time (a reused pgid is someone else's). rc 0 load done, 3 setup error, 4 stopped early, 5 a process group left, 124 --limit-s.
It runs the full --load-s with no other stop (if the box can't handle the load, that is
the box's problem). --stream <dir>: the rows of --hsc, --brick, --telem and --psu (a PSU poll's "epoch name W" rows) copied
from the load start on, as they come, into <dir>/hsc.txt, brick.txt, telem.txt and psu.txt, plus one samples.txt line per poll ("epoch elapsed_s
feed_A brick_C sum_W", "-" when not read), each line flushed and every file fsync'd at least every 1 s, so a host that
dies under the load leaves its last seconds on disk (lib/pv_crash.sh reads samples.txt after the reboot).
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def say(m):
    print(m, flush=True)


def pin_check(pin, d):
    """'' when every file the pin names is in d with its sha256 and size, else the why"""
    try:
        rows = [l.split() for l in open(pin) if l.strip() and not l.startswith("#")]
    except OSError as e:
        return f"no pin {pin} ({e.strerror})"
    for r in rows:
        kv = dict(x.split("=", 1) for x in r[1:] if "=" in x)
        p = os.path.join(d, r[0])
        try:
            data = open(p, "rb").read()
        except OSError as e:
            return f"{r[0]} missing in {d} ({e.strerror})"
        if len(data) != int(kv.get("bytes", -1)) or hashlib.sha256(data).hexdigest() != kv.get("sha256"):
            return f"{r[0]} in {d} does not match {pin}"
    return ""


def rows(path, since):
    if not path:
        return []
    try:
        with open(path, errors="replace") as f:
            out = [l.split() for l in f]
    except OSError:
        return []
    return [r for r in out if r and r[0].isdigit() and int(r[0]) >= since]


def feed_peak(path, since):
    """(amps, 'tray T HSC A') of the highest IOUT read since since, or None"""
    best = None
    for r in rows(path, since):
        if len(r) >= 5 and r[4].isdigit():
            a = int(r[4]) / 16.0
            if best is None or a > best[0]:
                best = (a, f"tray {r[1]} HSC {r[2]}")
    return best


def brick_peak(path, since):
    best = None
    for r in rows(path, since):
        if len(r) >= 3:
            try:
                c = float(r[2])
            except ValueError:
                continue
            if best is None or c > best[0]:
                best = (c, r[1])
    return best


def sum_w(path, since):
    """the chips' summed power, W, from each chip's latest telemetry row since since, or None"""
    last = {}
    for r in rows(path, since):
        if len(r) >= 4 and r[3].isdigit():
            last[r[1]] = int(r[3])
    return (sum(last.values()) / 1e6, len(last)) if last else None


def chips_default():
    return sorted(int(os.path.basename(p)) for p in glob.glob("/dev/tenstorrent/[0-9]*"))


def bdf(i, sysroot="/sys"):
    """PCI BDF of chip index i from tt-kmd's sysfs node, else '?' (the WARN names it; lib/fleet.sh adds the tray)"""
    b = os.path.basename(os.path.realpath(os.path.join(sysroot, "class", "tenstorrent", f"tenstorrent!{i}", "device")))
    return b if re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]$", b) else "?"


class Stop(Exception):
    pass


def size(p):
    try:
        return os.path.getsize(p) if p else 0
    except OSError:
        return 0


class Stream:
    """copies new whole lines of each source file into <dir>/<name>.txt and writes samples.txt; fsyncs every FSYNC_S"""

    FSYNC_S = 1.0

    def __init__(self, d, srcs):
        os.makedirs(d, exist_ok=True)
        self.src = [(p, open(os.path.join(d, n + ".txt"), "a")) for n, p in srcs]
        self.off = [size(p) for _, p in srcs]  # only rows from the load start on (c8_telemetry.txt holds the whole run)
        self.part = [b""] * len(self.src)
        self.smp = open(os.path.join(d, "samples.txt"), "a")
        self.synced = 0.0

    def poll(self, el, f, b, w):
        for k, (p, out) in enumerate(self.src):
            if not p:
                continue
            if size(p) < self.off[k]:  # truncated or replaced: read it again from its start
                self.off[k], self.part[k] = 0, b""
            try:
                with open(p, "rb") as fh:
                    fh.seek(self.off[k])
                    data = fh.read()
                    self.off[k] = fh.tell()
            except OSError:
                continue
            lines = (self.part[k] + data).split(b"\n")
            self.part[k] = lines.pop()
            for l in lines:
                out.write(l.decode(errors="replace") + "\n")
                out.flush()
        fa = f"{f[0]:.1f}" if f else "-"
        bc = f"{b[0]:g}" if b else "-"
        sw = f"{w[0]:.0f}" if w else "-"
        self.smp.write(f"{int(time.time())} {el:.0f} {fa} {bc} {sw}\n")
        self.smp.flush()
        if time.time() - self.synced >= self.FSYNC_S:
            self.sync()

    def sync(self):
        for fh in [o for _, o in self.src] + [self.smp]:
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        self.synced = time.time()

    def close(self):
        self.sync()
        for fh in [o for _, o in self.src] + [self.smp]:
            fh.close()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "pv"))
    ap.add_argument("--pin", default="")
    ap.add_argument("--ttx", default="single-core-matrix-pv-fp32-noc.ttx")
    ap.add_argument("--load-s", type=float, required=True)
    ap.add_argument("--limit-s", type=float, default=0)
    ap.add_argument("--chips", default="")
    ap.add_argument("--sys", default="/sys", help="sysfs root for the chip index -> BDF map (tests)")
    ap.add_argument("--hsc", default="")
    ap.add_argument("--feed-a", type=float, default=219)
    ap.add_argument("--brick", default="")
    ap.add_argument("--brick-c", type=float, default=88)
    ap.add_argument("--telem", default="")
    ap.add_argument("--psu", default="")
    ap.add_argument("--stream", default="")
    ap.add_argument("--poll-s", type=float, default=1)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--pgids", required=True)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args(argv)
    pin = a.pin or os.path.join(a.dir, "pv.pin")
    why = pin_check(pin, a.dir)
    chips = [int(c) for c in a.chips.split(",") if c != ""] if a.chips else chips_default()
    if not why and not chips:
        why = "no chip under /dev/tenstorrent"
    if why:
        say(f"PV ERROR setup: {why}")
        return 3
    t0 = int(time.time())

    def on_sig(n, _f):
        raise Stop(f"signal {n}")

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_sig)
    os.makedirs(a.log_dir, exist_ok=True)
    procs, rc, stop = {}, 0, ""
    st = (
        Stream(a.stream, (("hsc", a.hsc), ("brick", a.brick), ("telem", a.telem), ("psu", a.psu))) if a.stream else None
    )
    pk = {"feed": None, "brick": None, "sum": None}
    try:
        open(a.pgids, "w").close()
        for i in chips:
            tmp = os.path.join(a.scratch, str(i))
            os.makedirs(tmp, exist_ok=True)
            env = dict(os.environ, TMPDIR=tmp)
            log = open(os.path.join(a.log_dir, f"chip{i}.log"), "wb")
            p = subprocess.Popen(
                [
                    os.path.join(a.dir, "run-ttx"),
                    "--broadcast",
                    "--timeout",
                    "inf",
                    "--interface",
                    f"pci:{i}",
                    os.path.join(a.dir, a.ttx),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            procs[i] = p
            with open(a.pgids, "a") as f:
                f.write(f"{p.pid} {start_time(p.pid)}\n")
        say(
            f"PV LOAD start: run-ttx {a.ttx} on {len(chips)} chips ({','.join(map(str, chips))}) for {a.load_s:g} s; "
            f"stops: brick >= {a.brick_c:g} C, feed > {a.feed_a:g} A (chips' power is logged, never a stop)"
        )
        end = time.time() + a.load_s
        while time.time() < end:
            if a.limit_s and time.time() - t0 > a.limit_s:
                stop, rc = f"limit {a.limit_s:g} s", 124
                break
            f, b, w = feed_peak(a.hsc, t0), brick_peak(a.brick, t0), sum_w(a.telem, t0)
            if st:
                st.poll(time.time() - t0, f, b, w)
            for k, v in (("feed", f), ("brick", b), ("sum", w)):
                if v and (pk[k] is None or v[0] > pk[k][0]):
                    pk[k] = v
            if b and b[0] >= a.brick_c:
                stop = f"brick {b[1]} at {b[0]:g} C (>= {a.brick_c:g} C)"
            elif f and f[0] > a.feed_a:
                stop = f"feed {f[1]} at {f[0]:.1f} A (> {a.feed_a:g} A)"
            if stop:
                rc = 4
                break
            done = {i: p.poll() for i, p in procs.items() if p.poll() is not None}
            if len(done) == len(procs):
                break
            time.sleep(min(a.poll_s, Stream.FSYNC_S) if st else a.poll_s)
    except Stop as e:
        stop, rc = str(e), 4
    finally:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        el = time.time() - t0
        if st:
            st.close()
        early = {i: p.poll() for i, p in procs.items() if p.poll() is not None}
        for i, r in sorted(early.items()):
            say(
                f"PV WARN chip {i} ({bdf(i, a.sys)}): run-ttx exited rc {r} at {el:.0f} s, before the load end (see {a.log_dir}/chip{i}.log)"
            )
        left = []
        for sig, wait in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            live = [p for p in procs.values() if alive(p.pid)]
            for p in live:
                try:
                    os.killpg(p.pid, sig)  # our own group: pgid = the run-ttx we started (start_new_session)
                except ProcessLookupError:
                    pass
            t = time.time() + wait
            while time.time() < t and any(alive(p.pid) for p in live):
                time.sleep(0.2)
                for p in live:
                    p.poll()
            left = [p.pid for p in procs.values() if alive(p.pid)]
            if not left:
                break
        if a.scratch and os.path.isdir(a.scratch) and os.path.abspath(a.scratch) not in ("/", os.path.expanduser("~")):
            shutil.rmtree(a.scratch, ignore_errors=True)
    pks = "; ".join(
        f"peak {n} {v[0]:.1f}{u} ({v[1]})"
        for n, u in (("feed", " A"), ("brick", " C"), ("sum", " W"))
        for v in [pk[n]]
        if v
    )
    if procs and len(early) == len(procs) and not stop and el < a.load_s - 1:
        say(f"PV ERROR setup: every run-ttx exited before the load end ({el:.0f} of {a.load_s:g} s; see {a.log_dir})")
        rc = 3
    if stop:
        say(f"PV STOP {stop} at {el:.0f} s: every run-ttx stopped")
    if left:
        say(f"PV LEFT process groups {' '.join(map(str, left))} still running after SIGKILL")
        rc = 5
    say(
        f"PV LOAD done: {len(procs)} chips, {el:.0f} of {a.load_s:g} s{', ' + pks if pks else ''}; no run-ttx of ours left"
        if not left
        else f"PV LOAD done: {len(procs)} chips, {el:.0f} s{', ' + pks if pks else ''}"
    )
    return rc


def start_time(pid):
    """/proc/<pid>/stat field 22 (start time in clock ticks since boot), "0" when unreadable."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return "0"


def alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
