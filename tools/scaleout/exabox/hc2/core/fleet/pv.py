#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""pv.py: the power virus, the bus-bar stage's other max load.

--runner run-ttx (the default; pv_umd was removed) runs run-ttx --broadcast --timeout inf --interface pci:<i> single-core-matrix-pv-fp32-noc.ttx
on every chip at once, one process per chip in its own session (start_new_session, its own TMPDIR: a PyInstaller onefile),
for --load-s s. It programs fixed TLB windows itself and hung chips beside tt_telemetry_collector, so lib/fleet.sh's
fl_pv_guard runs it only where no other process holds a device (a root telemetry collector service stopped for it);
it ran 32 chips clean on a galaxy (11-13).
--runner run-ttx-kmdtlb is the same run-ttx with one more PyInstaller runtime hook, pv/kmdtlb.py: every TLB
window it uses is allocated from the driver (tt-kmd) and freed at its exit, and a write to any other window's config is
refused, so it can run beside the collector (whose windows it never touches). pv.py builds it into --scratch from the pinned
run-ttx and pv/kmdtlb.code with pv/repack.py and runs it only when it matches the pin's run-ttx-kmdtlb row. A load only: no numeric check. --tdp does not cap it (~167 W per chip at a 160 W limit), so it stops early, all chips at once, on:
  a power brick at or above --brick-c (rows "epoch name C" in --brick, the read-only sdr poll fleet.sh runs),
  a feed above --feed-a (the HSC sampler's rows "epoch tray addr VIN_raw IOUT_raw" in --hsc, IOUT at 62.5 mA/LSB),
each only where its file has rows. The chips' power never stops it:
their summed telemetry power (--telem, c8_telemetry.txt rows "epoch bdf
temp_mC power_uW ...", latest per chip) is only logged as a peak, with how many chips it covers: a row with power_uW >= 4e9 or
temp_mC >= 65535000 is an all-ones read (a hung chip's telemetry), counted as unread, never as power, and a chip that reads
all-ones for 10 s or more gets one "PV WARN chip <i> (<bdf>): telemetry unreadable (all-ones) from <t> s into the load" line.
A runner that logs an ERROR line in its <log-dir>/chip<i>.log (not run-ttx's own line when it is interrupted at our stop) or exits non-zero on its own before the load end gets one "PV ERROR chip <i> (<bdf>): <first error
line>" line, and pv.py returns 6 (when nothing else set its rc). Ends with SIGINT, SIGTERM, SIGKILL to each of our own process groups (by pgid, never by
name), confirms none is left and removes its scratch dir. The pgids go to --pgids first, one "pgid starttime" line each
(/proc/<pid>/stat field 22, 0 when unreadable), so fleet.sh can end any left if this process is killed, and only while the
leader still has that start time (a reused pgid is someone else's). rc 0 load done, 3 setup error, 4 stopped early, 5 a process group left, 6 a runner error, 124 --limit-s.
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
sys.path.insert(1, HERE)
from chunk_sched import timing_text  # noqa: E402   (the LOAD line's measured burst on-time and pause)


def say(m):
    print(m, flush=True)


def pin_check(pin, d, at=None, skip=("run-ttx-kmdtlb",), only=None):
    """'' when every file the pin names is in d (or at the path at[name] gives) with its sha256 and size, else the why;
    rows named in skip (by default run-ttx-kmdtlb: pv.py builds it, it is never in d) are left out, and with only, every row
    but those; a name in only with no row in the pin is a why too"""
    try:
        rows = [l.split() for l in open(pin) if l.strip() and not l.startswith("#")]
    except OSError as e:
        return f"no pin {pin} ({e.strerror})"
    for n in only or ():
        if n not in skip and not any(r[0] == n for r in rows):
            return f"no {n} row in {pin}"
    for r in rows:
        if r[0] in skip or (only is not None and r[0] not in only):
            continue
        kv = dict(x.split("=", 1) for x in r[1:] if "=" in x)
        p = (at or {}).get(r[0]) or os.path.join(d, r[0])
        try:
            data = open(p, "rb").read()
        except OSError as e:
            return f"{r[0]} missing at {p} ({e.strerror})"
        if len(data) != int(kv.get("bytes", -1)) or hashlib.sha256(data).hexdigest() != kv.get("sha256"):
            return f"{r[0]} at {p} does not match {pin}"
    return ""


def runner_rows(pin):
    """the pin rows the default runner uses: run-ttx and every .ttx (not kmdtlb.py, kmdtlb.code or run-ttx-kmdtlb)"""
    try:
        names = [l.split()[0] for l in open(pin) if l.strip() and not l.startswith("#")]
    except OSError:
        names = []
    return ("run-ttx",) + tuple(n for n in names if n.endswith(".ttx"))


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


ONES_UW = 4000000000  # power_uW at or above it: an all-ones read (0xFFFFFFFF less a scale offset), not power
ONES_MC = 65535000  # temp_mC at or above it: an all-ones read
ONES_S = 10  # all-ones this long or more: one PV WARN line for the chip
RAMP_S = 2  # a telemetry row this many s or more after the load started shows its power; earlier rows may be idle power


def telem_rows(path, since, until=None, bdfs=None):
    """[(epoch, bdf, power_uW, ok)] of the telemetry rows since since (to until), only bdfs when given; ok False for an all-ones read"""
    out = []
    for r in rows(path, since):
        if len(r) < 4 or not r[3].isdigit() or (bdfs and r[1] not in bdfs) or (until is not None and int(r[0]) > until):
            continue
        try:
            t = int(r[2])
        except ValueError:
            t = 0
        out.append((int(r[0]), r[1], int(r[3]), int(r[3]) < ONES_UW and t < ONES_MC))
    return out


def sum_w(path, since, bdfs=None, n=0):
    """(W, 'k of n chips read'): the summed power of the chips whose latest telemetry row since since is a valid read (an
    all-ones latest row is unread, never power), or None when none is"""
    last = {}
    for _e, b, uw, ok in telem_rows(path, since, bdfs=bdfs):
        last[b] = uw if ok else None
    good = [v for v in last.values() if v is not None]
    return (sum(good) / 1e6, f"{len(good)} of {n or len(last)} chips read") if good else None


def ramped_rows(path, since, t_on, bdfs=None):
    """True when a valid telemetry row is RAMP_S s or more after the load started (t_on); read during the load only, so the
    rows it sees are the ones the peak sum could use"""
    return any(ok and e >= t_on + RAMP_S for e, _b, _uw, ok in telem_rows(path, since, bdfs=bdfs))


def ones_runs(path, since, until, bdfs=None, min_s=ONES_S):
    """{bdf: s into the load} of the chips whose telemetry read all-ones for ONES_S s or more: from a run's first all-ones row
    to its last, or to until when no valid read follows it"""
    runs, start, last = {}, {}, {}
    for e, b, _uw, ok in telem_rows(path, since, until, bdfs):
        if ok:
            if b in start and last[b] - start[b] >= min_s and b not in runs:
                runs[b] = start[b] - since
            start.pop(b, None)
        else:
            start.setdefault(b, e)
            last[b] = e
    for b, s0 in start.items():
        if until - s0 >= min_s and b not in runs:
            runs[b] = s0 - since
    return runs


# run-ttx's own line when it is interrupted at our stop (KeyboardInterrupt before its ttx_run_end): no error of the load
RUNTTX_STOP = re.compile(r"TTX errored out before start not running ttx_run_end")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ERR = re.compile(r"\bERROR\b")


def first_error(log, runner, stopped):
    """the first ERROR line of a runner's chip<i>.log (ANSI colours stripped), '' when none; stopped: we interrupted it"""
    try:
        with open(log, errors="replace") as f:
            for l in f:
                l = ANSI.sub("", l).strip()
                if ERR.search(l) and not (runner.startswith("run-ttx") and stopped and RUNTTX_STOP.search(l)):
                    return l
    except OSError:
        pass
    return ""


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


KMDTLB = "run-ttx-kmdtlb"


def build_kmdtlb(pin, d, rtx, scratch):
    """(path, '') of run-ttx-kmdtlb built into scratch from the pinned run-ttx and kmdtlb.code, checked against the pin's
    run-ttx-kmdtlb row; ('', why) when it cannot be built or does not match"""
    sys.path.insert(0, os.path.join(HERE, "pv"))
    try:
        import repack
    finally:
        sys.path.pop(0)
    out = os.path.join(scratch, KMDTLB)
    try:
        os.makedirs(scratch, exist_ok=True)
        with open(rtx, "rb") as f, open(os.path.join(d, "kmdtlb.code"), "rb") as g:
            data = repack.repack(f.read(), g.read())
        with open(out, "wb") as f:
            f.write(data)
        os.chmod(out, 0o755)
    except (OSError, repack.Bad) as e:
        why = f"{KMDTLB} not built: {e}"
    else:
        why = pin_check(pin, d, {KMDTLB: out}, skip=(), only=(KMDTLB,))
    if why:  # nothing of it left: the file, and scratch when it made it empty
        for f in (out, out + ".tmp"):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(scratch)
        except OSError:
            pass
        return "", why
    return out, ""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "pv"))
    ap.add_argument("--pin", default="")
    # run-ttx's path (the core finds it through HC_RUN_TTX, else in --dir); it must still match the pin
    ap.add_argument("--run-ttx", default="")
    ap.add_argument("--ttx", default="single-core-matrix-pv-fp32-noc.ttx")
    ap.add_argument("--runner", default="run-ttx", choices=("run-ttx", KMDTLB))
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
    ap.add_argument(
        "--ones-s", type=float, default=ONES_S, help="all-ones telemetry this long: a PV WARN (tests shorten it)"
    )
    ap.add_argument(
        "--stop-file",
        default="",
        help="lib/fleet.sh's holder watcher writes it when another process takes a device node",
    )
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--pgids", required=True)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args(argv)
    pin = a.pin or os.path.join(a.dir, "pv.pin")
    rtx = a.run_ttx or os.path.join(a.dir, "run-ttx")
    if a.runner == KMDTLB:  # every row but run-ttx-kmdtlb, which build_kmdtlb checks once built
        why = pin_check(pin, a.dir, {"run-ttx": rtx})
        if not why:
            rtx, why = build_kmdtlb(pin, a.dir, rtx, a.scratch)
    else:  # only its own rows: a kmdtlb.py or kmdtlb.code change never disables the default power virus
        why = pin_check(pin, a.dir, {"run-ttx": rtx}, only=runner_rows(pin))
    chips = [int(c) for c in a.chips.split(",") if c != ""] if a.chips else chips_default()
    if not why and not chips:
        why = "no chip under /dev/tenstorrent"
    rn = a.runner
    if why:
        say(f"PV ERROR setup: {why}")
        return 3
    t0 = int(time.time())
    bmap = {bdf(i, a.sys): i for i in chips}
    bdfs = set(bmap) if "?" not in bmap else None  # every chip's BDF known: only their telemetry rows count

    def on_sig(n, _f):
        raise Stop(f"signal {n}")

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_sig)
    os.makedirs(a.log_dir, exist_ok=True)
    procs, rc, stop, t_on, t_off, ramped = {}, 0, "", None, None, False
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
            cmd = [rtx, "--broadcast", "--timeout", "inf", "--interface", f"pci:{i}", os.path.join(a.dir, a.ttx)]
            p = subprocess.Popen(
                cmd,
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
            f"PV LOAD start: {rn} {a.ttx} on {len(chips)} chips ({','.join(map(str, chips))}) for {a.load_s:g} s; "
            f"stops: brick >= {a.brick_c:g} C, feed > {a.feed_a:g} A (chips' power is logged, never a stop)"
        )
        t_on = time.time()  # the load's on-time counts from every runner started to the first stop signal
        end = t_on + a.load_s
        while time.time() < end:
            if a.limit_s and time.time() - t0 > a.limit_s:
                stop, rc = f"limit {a.limit_s:g} s", 124
                break
            f, b, w = feed_peak(a.hsc, t0), brick_peak(a.brick, t0), sum_w(a.telem, t0, bdfs, len(chips) if bdfs else 0)
            ramped = ramped or bool(a.telem) and ramped_rows(a.telem, t0, t_on, bdfs)
            if st:
                st.poll(time.time() - t0, f, b, w)
            for k, v in (("feed", f), ("brick", b), ("sum", w)):
                if v and (pk[k] is None or v[0] > pk[k][0]):
                    pk[k] = v
            if b and b[0] >= a.brick_c:
                stop = f"brick {b[1]} at {b[0]:g} C (>= {a.brick_c:g} C)"
            elif f and f[0] > a.feed_a:
                stop = f"feed {f[1]} at {f[0]:.1f} A (> {a.feed_a:g} A)"
            elif a.stop_file and os.path.exists(a.stop_file):  # the watcher saw another holder
                try:
                    with open(a.stop_file) as sf:
                        h = sf.readline().strip()
                except OSError:
                    h = ""
                stop = f"another process holds a Tenstorrent device: {h or a.stop_file}"
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
        t_off = time.time()
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        el = time.time() - t0
        if st:
            st.close()
        early = {i: p.poll() for i, p in procs.items() if p.poll() is not None}
        for i, r in sorted(early.items()):
            if r == 0:
                say(
                    f"PV WARN chip {i} ({bdf(i, a.sys)}): {rn} exited rc {r} at {el:.0f} s, before the load end (see {a.log_dir}/chip{i}.log)"
                )
        left = []
        for sig, wait in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            live = [p for p in procs.values() if alive(p.pid)]
            for p in live:
                try:
                    os.killpg(p.pid, sig)  # our own group: pgid = the runner we started (start_new_session)
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
    errs = 0
    for i in sorted(procs):  # after the stop
        e = first_error(os.path.join(a.log_dir, f"chip{i}.log"), rn, i not in early)
        if not e and early.get(i):
            e = f"{rn} exited rc {early[i]} at {el:.0f} s, before the load end"
        if e:
            errs += 1
            say(f"PV ERROR chip {i} ({bdf(i, a.sys)}): {e} (see {a.log_dir}/chip{i}.log)")
    inv = {b: i for b, i in bmap.items() if b != "?"}
    for b, s0 in sorted(
        ones_runs(a.telem, t0, t0 + el, bdfs, a.ones_s).items(), key=lambda x: (inv.get(x[0], 1 << 30), x[0])
    ):
        say(f"PV WARN chip {inv.get(b, '?')} ({b}): telemetry unreadable (all-ones) from {s0} s into the load")
    if a.telem and t_on is not None and not ramped:
        # a short load (4 s smoke) can end before the collector writes a row of it: the peak sum is then idle power or none
        nr = "no telemetry row after the load ramped"
        pk["sum"] = (pk["sum"][0], f"{pk['sum'][1]}; {nr}, so idle power, not the load's") if pk["sum"] else None
        nosum = f"peak sum not read ({nr})" if not pk["sum"] else ""
    else:
        nosum = ""
    pks = "; ".join(
        f"peak {n} {v[0]:.0f}{u} ({v[1]})" if n == "sum" else f"peak {n} {v[0]:.1f}{u} ({v[1]})"
        for n, u in (("feed", " A"), ("brick", " C"), ("sum", " W"))
        for v in [pk[n]]
        if v
    ) + (f"{'; ' if pk['feed'] or pk['brick'] else ''}{nosum}" if nosum else "")
    if errs and rc == 0:
        rc = 6
    if procs and len(early) == len(procs) and not stop and el < a.load_s - 1:
        say(f"PV ERROR setup: every {rn} exited before the load end ({el:.0f} of {a.load_s:g} s; see {a.log_dir})")
        rc = 3
    if stop:
        say(f"PV STOP {stop} at {el:.0f} s: every {rn} stopped")
    if left:
        say(f"PV LEFT process groups {' '.join(map(str, left))} still running after SIGKILL")
        rc = 5
    if errs:
        pks += f"{'; ' if pks else ''}{rn} errors on {errs} chips (PV ERROR lines above)"
    on_s = round(t_off - t_on, 3) if t_on is not None else None
    tm = timing_text(
        dict(on=dict(n=1, mean=on_s, min=on_s, max=on_s), pause=None) if on_s is not None else None
    )  # one burst
    say(
        f"PV LOAD done: {len(procs)} chips, {el:.0f} of {a.load_s:g} s, {tm}{', ' + pks if pks else ''}; no {rn} of ours left"
        if not left
        else f"PV LOAD done: {len(procs)} chips, {el:.0f} s, {tm}{', ' + pks if pks else ''}"
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
