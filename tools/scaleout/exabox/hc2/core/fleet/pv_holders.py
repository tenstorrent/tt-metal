# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pv_holders.py: the power virus guard's holder scan. run-ttx maps BAR0 and programs fixed TLB
windows itself, so beside another TLB user (tt_telemetry_collector maps windows 0 and 1) it hangs chips. Just before fl_bb_pv
launches run-ttx, this scans every process's open fds for a Tenstorrent device node (the /dev/tenstorrent* glob, as
checkspill's DEVICE_NODES), leaving out only run.sh's own process tree (--self and its descendants by parent pid). Not its
process group or session: a collector started beside run.sh in the same ssh command shares both. Nothing is exempt by name:
the fabric-manager agent and telemetry collectors count too. A process whose fd dir cannot be read (another user's, with no
sudo), or a /proc mounted hidepid=2 for a non-root scan, makes the scan incomplete; kernel threads (empty cmdline) hold no
fds and are left out.
  pv_holders.py --self <run.sh pid> [--proc DIR] [--glob GLOB] [--sudo]
prints one line: the holders as '<pid> (<comm>) holds <node>[ (+N more)], ...', then '; ' and 'holder scan incomplete
(no sudo): ...' when some fd dirs could not be read; nothing when the scan is complete and found no holder.
rc 0: complete, no holder; 10: a holder; 11: no holder seen, but incomplete (the caller skips the virus on 10 and 11).
--proc (default /proc) and --glob (default /dev/tenstorrent*; fnmatch, so * also matches /) are args, not env, so they reach a
scan under sudo: lib/fleet.sh passes FLEET_PROC and FLEET_DEVD (gate.sh's test hooks)."""
import argparse
import fnmatch
import os
import sys


def stat(proc, pid):
    """(ppid, pgrp, session) of pid, or None once it is gone"""
    try:
        with open(f"{proc}/{pid}/stat") as f:
            s = f.read()
    except OSError:
        return None
    w = s.rsplit(")", 1)[-1].split()
    try:
        return int(w[1]), int(w[2]), int(w[3])
    except (IndexError, ValueError):
        return None


def ours(proc, pid, me):
    st = stat(proc, pid)
    if st is None:
        return True  # gone: holds nothing now
    if pid == me:
        return True
    p = st[0]
    for _ in range(64):  # run.sh's descendants, also those that left its group and session (setsid loads)
        if p <= 1:
            return False
        if p == me:
            return True
        st = stat(proc, p)
        if st is None:
            return False
        p = st[0]
    return False


def kthread(proc, pid):
    try:
        with open(f"{proc}/{pid}/cmdline", "rb") as f:
            return f.read() == b""
    except OSError:
        return True


def hidden(proc):
    """True when a non-root scan of /proc cannot see other users' processes (mounted hidepid=2 or invisible)"""
    if proc != "/proc" or os.geteuid() == 0:
        return False
    try:
        with open("/proc/self/mountinfo") as f:
            for l in f:
                w = l.split()
                if len(w) > 4 and w[4] == "/proc" and " - proc " in l:
                    return any(o in ("hidepid=2", "hidepid=invisible") for o in w[-1].split(","))
    except OSError:
        pass
    return False


def scan(proc, glob, me):
    """{pid: [nodes]} of the other processes holding a node, and the pids whose fds could not be read"""
    held, unread = {}, []
    try:
        pids = sorted(int(p) for p in os.listdir(proc) if p.isdigit())
    except OSError:
        return held, ["(no proc)"]
    if hidden(proc):
        unread.append("(proc hidepid=2)")
    for pid in pids:
        if pid == os.getpid() or ours(proc, pid, me):
            continue
        fd = f"{proc}/{pid}/fd"
        try:
            ents = os.listdir(fd)
        except FileNotFoundError:
            continue
        except OSError:
            if not kthread(proc, pid):
                unread.append(pid)
            continue
        nodes = set()
        for e in ents:
            try:
                t = os.readlink(f"{fd}/{e}")
            except OSError:
                continue
            if fnmatch.fnmatchcase(t, glob):
                nodes.add(t)
        if nodes:
            held[pid] = sorted(nodes)
    return held, unread


def comm(proc, pid):
    try:
        with open(f"{proc}/{pid}/comm") as f:
            return f.read().strip() or "?"
    except OSError:
        return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self", dest="me", type=int, required=True, help="run.sh's pid: it and its descendants are ours")
    ap.add_argument("--proc", default="/proc", help="the proc dir (test hook)")
    ap.add_argument("--glob", default="/dev/tenstorrent*", help="the device node glob")
    ap.add_argument(
        "--sudo", action="store_true", help="the scan runs under sudo -n (an unreadable fd dir is not 'no sudo')"
    )
    a = ap.parse_args(argv)
    proc, glob = a.proc, a.glob
    held, unread = scan(proc, glob, a.me)
    parts = []
    for pid, nodes in sorted(held.items()):
        parts.append(
            f"{pid} ({comm(proc, pid)}) holds {nodes[0]}" + (f" (+{len(nodes) - 1} more)" if len(nodes) > 1 else "")
        )
    out = ", ".join(parts)
    if unread:
        why = "no sudo" if not a.sudo else "fds unreadable under sudo"
        lst = " ".join(str(p) for p in unread[:8]) + (" ..." if len(unread) > 8 else "")
        out += (
            "; " if out else ""
        ) + f"holder scan incomplete ({why}): the fds of {len(unread)} process(es) could not be read ({lst})"
    if out:
        print(out)
    return 10 if held else 11 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
