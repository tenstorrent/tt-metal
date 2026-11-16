# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pv_holders.py: the power virus guard's holder scan. run-ttx maps BAR0 and programs fixed TLB
windows itself, so beside another TLB user (tt_telemetry_collector maps windows 0 and 1) it hangs chips. Just before fl_bb_pv
launches the virus (run-ttx), this scans every process's open fds for a Tenstorrent device node (the /dev/tenstorrent* glob, as
checkspill's DEVICE_NODES), leaving out only run.sh's own process tree (--self and its descendants by parent pid). Not its
process group or session: a collector started beside run.sh in the same ssh command shares both. Nothing is exempt by name:
the fabric-manager agent and telemetry collectors count too. A process whose fd dir cannot be read (another user's, with no
sudo), or a /proc mounted hidepid=1, 2, noaccess or invisible for a non-root scan, makes the scan incomplete; kernel threads
(empty cmdline) hold no fds and are left out (an unreadable cmdline is not one).
  pv_holders.py --self <run.sh pid> [--proc DIR] [--glob GLOB] [--sudo]
prints one line: the holders as '<pid> (<comm>) holds <node>[ (+N more)], ...', then '; ' and 'holder scan incomplete
(no sudo): ...' when some fd dirs could not be read; nothing when the scan is complete and found no holder.
rc 0: complete, no holder; 10: a holder; 11: no holder seen, but incomplete (the caller skips the virus on 10 and 11).
--proc (default /proc) and --glob (default /dev/tenstorrent*; fnmatch, so * also matches /) are args, not env, so they reach a
scan under sudo: lib/fleet.sh passes FLEET_PROC and FLEET_DEVD (gate.sh's test hooks).
--podlog DIR: the kubelet's own pod log dirs (<namespace>_<pod>_<uid>), as fleet/gate.sh reads wan
pods; a pod whose name matches --pod-re (default tt-telemetry|telemetry-collector: the collector's k3s pod) is a holder too, even
in a gap between its process's restarts (it restarts every few seconds). Local reads only, no kubectl.
  pv_holders.py --self <run.sh pid> --names [--proc DIR]
The check lib/fleet.sh runs before it trusts an incomplete scan (rc 11) inside an
exclusive Slurm job: a root telemetry service holds TLB windows with fds a no-sudo scan cannot read, but its comm and cmdline
are readable. It reads /proc/<pid>/comm and cmdline of every process it can list but ours (run.sh's tree, its ancestors, and
the processes of its own Slurm job by cgroup) and prints one line. rc 0: all read, none a known TLB user by name; 12: one
or more are (each named '<pid> (user <u>) <comm>: <what>'); 13: fail closed, when a comm or cmdline cannot be read, or when the
listing cannot hold every process (no kthreadd at pid 2: /proc hidepid=2 or invisible hides other users' processes, or a pid
namespace).
  pv_holders.py --self <run.sh pid> --units [--proc DIR] [--glob GLOB] [--podlog DIR] [--sudo]
The pause check: rc 0 and '<unit> [<unit>...]|<pid> (<comm>) in <unit>, ...' when the scan is
complete and every holder is a root process in a telemetry collector's systemd system service (cgroup /system.slice/<unit>,
the unit or comm tt-telemetry or telemetry-collector), which lib/fleet.sh may stop for the virus; rc 14 and why not: a
collector pod, an incomplete scan, or any other holder (another user's, outside a system service, the tt-metrics-exporter or
the fabric-manager agent).
  --beside-collector (scan and --names; lib/fleet.sh passes it with the run-ttx-kmdtlb runner, whose TLB windows come from
the driver): a telemetry collector (comm, cmdline or systemd unit tt-telemetry or telemetry-collector), the
tt-metrics-exporter and the fabric-manager agent (by comm only: tt-metrics-expo, tt-fabric-manag; all three take their windows
from the driver too) and a collector pod are no holder and no name hit; they are named after
'beside the telemetry collector: ' (scan: rc 0 when they are all it found). Every other holder or name counts as before."""
import argparse
import errno
import fnmatch
import os
import re
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
    except OSError as e:
        if gone(e, f"{proc}/{pid}/cmdline"):
            return True  # gone: holds nothing now
        return False  # unreadable (another user's, hidepid=1): not shown to be a kernel thread


def hidepid(mountinfo):
    """the hidepid option of the /proc mount in these mountinfo lines that hides other users' processes or their fds
    (hidepid=1, 2, noaccess or invisible), else ''"""
    for l in mountinfo:
        w = l.split()
        if len(w) > 4 and w[4] == "/proc" and " - proc " in l:
            return next(
                (o for o in w[-1].split(",") if o.startswith("hidepid=") and o not in ("hidepid=0", "hidepid=off")), ""
            )
    return ""


def hidden(proc):
    """the hidepid option ('hidepid=2' ...) when a non-root scan of /proc cannot see other users' processes or their fds, else ''"""
    if proc != "/proc" or os.geteuid() == 0:
        return ""
    try:
        with open("/proc/self/mountinfo") as f:
            return hidepid(f)
    except OSError:
        return ""


def scan(proc, glob, me):
    """{pid: [nodes]} of the other processes holding a node, and the pids whose fds could not be read"""
    held, unread = {}, []
    try:
        pids = sorted(int(p) for p in os.listdir(proc) if p.isdigit())
    except OSError:
        return held, ["(no proc)"]
    h = hidden(proc)
    if h:
        unread.append(f"(proc {h})")
    for pid in pids:
        if pid == os.getpid() or ours(proc, pid, me):
            continue
        fd = f"{proc}/{pid}/fd"
        try:
            ents = os.listdir(fd)
        except OSError as e:
            if gone(e, fd):
                continue
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


# known TLB users by name (comm, or anywhere in cmdline), case-insensitive; comm is cut at 15 characters
# (tt_telemetry_co, tt-metrics-expo, tt-fabric-manag), so each pattern also matches the cut name
B, E = r"(?<![a-z0-9])", r"(?![a-z0-9])"
TLB_NAMES = (
    ("telemetry collector", r"tt[_-]telemetry|telemetry[_-]collector"),
    ("tt-metrics-exporter", r"tt[_-]metrics[_-]exp"),
    ("fabric-manager agent", r"fabric[_-]manag"),
    ("tt-smi", B + r"tt[_-]smi" + E),
    ("tt-flash", B + r"tt[_-]flash" + E),
    ("tt-exalens", r"exalens"),
    ("luwen", r"luwen"),
    ("tray-test", r"tray[_-]?test" + "|" + B + r"run-ttx" + E),
    ("ttnn", B + r"ttnn" + E),
    ("tt-metal", r"tt[_-]metal"),
    ("umd", B + r"(tt[_-])?umd" + E),
)


def tlb_name(comm_s, cmd):
    """the name of the first known TLB user comm_s or cmd matches, else ''"""
    t = f"{comm_s}\n{cmd}".lower()
    return next((n for n, rx in TLB_NAMES if re.search(rx, t)), "")


def gone(e, path):
    """whether OSError e on <proc>/<pid>/<file> means the process is gone: ENOENT, ESRCH (a process that exits
    mid-read), or any other error after which <proc>/<pid> no longer exists; EACCES and EPERM never (fail closed)"""
    if isinstance(e, (FileNotFoundError, ProcessLookupError)):
        return True
    return e.errno not in (errno.EACCES, errno.EPERM) and not os.path.lexists(os.path.dirname(path))


def readf(path):
    """the file's bytes; None when it or its process is gone (gone()); OSError when it cannot be read"""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        if gone(e, path):
            return None
        raise


def job_of(proc, pid):
    """'job_<n>' of pid's Slurm cgroup (cgroup v1 or v2), else ''"""
    try:
        b = readf(f"{proc}/{pid}/cgroup")
    except OSError:
        return ""
    m = re.search(r"/(job_[0-9]+)(/|$)", (b or b"").decode(errors="replace"), re.M)
    return m.group(1) if m else ""


def ancestors(proc, me):
    out, p = set(), me
    for _ in range(64):
        st = stat(proc, p)
        if st is None or st[0] <= 0:
            break
        p = st[0]
        out.add(p)
        if p == 1:
            break
    return out


def user_of(proc, pid):
    try:
        u = os.stat(f"{proc}/{pid}").st_uid
    except OSError:
        return "?"
    try:
        import pwd

        return pwd.getpwuid(u).pw_name
    except (KeyError, ImportError):
        return str(u)


def names(proc, me, beside=False):
    """(rc, line) of the name check: 0 none, 12 a known TLB user by name, 13 not everything could be read or listed;
    beside: a telemetry collector, the tt-metrics-exporter or the fabric-manager agent is no hit, only named"""
    try:
        pids = sorted(int(p) for p in os.listdir(proc) if p.isdigit())
    except OSError:
        return 13, f"cannot list {proc}"
    try:
        kt = readf(f"{proc}/2/comm")
    except OSError:
        kt = None
    why = []
    if (kt or b"").strip() != b"kthreadd":
        h = hidden(proc)
        hid = f"/proc {h} hides other users' processes" if h else "a pid namespace, or other users' processes hidden"
        why.append(f"cannot list every process: no kthreadd at pid 2 ({hid})")
    job, anc = job_of(proc, me), ancestors(proc, me)
    hits, unread, n, col = [], [], 0, []
    for pid in pids:
        if pid == os.getpid() or pid in anc:
            continue
        try:  # read first: ours() takes a process whose stat it cannot read (hidepid=1) for gone
            c, a = readf(f"{proc}/{pid}/comm"), readf(f"{proc}/{pid}/cmdline")
        except OSError:
            unread.append(pid)
            continue
        if c is None or a is None or ours(proc, pid, me) or (job and job_of(proc, pid) == job):
            continue  # gone, or ours
        n += 1
        cs = c.decode(errors="replace").strip()
        cmd = a.replace(b"\0", b" ").decode(errors="replace")
        hit = tlb_name(cs, cmd)
        if hit and beside and (hit == "telemetry collector" and collector(proc, pid, cs, cmd) or plumbing(cs)):
            col.append(f"{pid} (user {user_of(proc, pid)}) {cs or '?'}")
            continue
        if hit:
            hits.append(f"{pid} (user {user_of(proc, pid)}) {cs or '?'}: {hit}")
    if unread:
        lst = " ".join(str(p) for p in unread[:8]) + (" ..." if len(unread) > 8 else "")
        why.append(f"the comm or cmdline of {len(unread)} process(es) could not be read ({lst})")
    bs = f"; beside the telemetry collector: {', '.join(col)}" if col else ""
    if hits:
        more = f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""
        return (
            12,
            f"known TLB user(s) by name: {', '.join(hits[:4])}{more}" + (f"; {'; '.join(why)}" if why else "") + bs,
        )
    if why:
        return 13, "process name check incomplete: " + "; ".join(why) + bs
    other = " other than the telemetry collector" if col else ""
    return 0, f"the comm and cmdline of all {n} other process(es) read: none is a known TLB user by name{other}{bs}"


# the holders the guard may pause: a root systemd system service of a telemetry collector; never
# a pod (kubepods), a person's process (user.slice), the k8s tt-metrics-exporter or the fabric-manager agent
SVC_RX = r"tt[_-]telemetry|telemetry[_-]collector"
NOT_SVC_RX = r"metrics[_-]exp|fabric[_-]manag"
UNIT_RX = r"[A-Za-z0-9@._-]+\.service"


def unit_of(proc, pid):
    """'<name>.service' of pid's systemd system service (cgroup v2 '0::/system.slice/<name>.service', or v1
    'name=systemd:'), else ''"""
    try:
        b = readf(f"{proc}/{pid}/cgroup")
    except OSError:
        return ""
    for ln in (b or b"").decode(errors="replace").splitlines():
        w = ln.split(":", 2)
        if len(w) == 3 and ((w[0] == "0" and w[1] == "") or w[1] == "name=systemd"):
            m = re.fullmatch(r"/system\.slice/(" + UNIT_RX + r")", w[2])
            return m.group(1) if m else ""
    return ""


def units(proc, glob, me, podlog, pod_re, uid):
    """(rc, line) of the pause check: 0 and '<unit> [<unit>...]|<what>' when every holder is a process of uid (root) in a
    telemetry collector's systemd system service; 14 and why not (a pod, an incomplete scan, any other holder)"""
    pl = pods(podlog, pod_re) if podlog else []
    if pl:
        return 14, f"a telemetry collector pod (Kubernetes, not ours to stop): {' '.join(pl)}"
    held, unread = scan(proc, glob, me)
    if unread:
        return 14, f"holder scan incomplete: {len(unread)} process(es) unread"
    if not held:
        return 14, "no holder to pause"
    us, what, why = [], [], []
    for pid in sorted(held):
        c, u = comm(proc, pid), unit_of(proc, pid)
        try:
            cmd = (readf(f"{proc}/{pid}/cmdline") or b"").replace(b"\0", b" ").decode(errors="replace")
            pu = os.stat(f"{proc}/{pid}").st_uid
        except OSError:
            why.append(f"{pid} ({c}): unreadable")
            continue
        t = f"{u}\n{c}\n{cmd}".lower()
        if pu != uid:
            why.append(f"{pid} (user {user_of(proc, pid)}) {c}: not root")
        elif not u:
            why.append(f"{pid} {c}: not in a systemd system service")
        elif re.search(NOT_SVC_RX, t) or not re.search(SVC_RX, f"{u}\n{c}".lower()):
            why.append(f"{pid} {c} ({u}): not a telemetry collector service")
        else:
            if u not in us:
                us.append(u)
            what.append(f"{pid} ({c}) in {u}")
    if why:
        return 14, "not only a pausable telemetry service: " + "; ".join(why)
    return 0, f"{' '.join(us)}|{', '.join(what)}"


def collector(proc, pid, c, cmd=None):
    """whether pid is a telemetry collector by comm, cmdline or systemd unit (never the metrics exporter or fabric manager)"""
    if cmd is None:
        try:
            cmd = (readf(f"{proc}/{pid}/cmdline") or b"").replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = ""
    t = f"{unit_of(proc, pid)}\n{c}\n{cmd}".lower()
    return bool(re.search(SVC_RX, t)) and not re.search(NOT_SVC_RX, t)


# device plumbing that maps its TLB windows through the driver beside a driver-window runner: matched by comm only
# (tt-metrics-expo, tt-fabric-manag), never by cmdline, so a tt-metal test whose arguments name a fabric manager still counts
PLUMBING_RX = r"^tt[_-]metrics[_-]exp|^tt[_-]fabric[_-]manag"


def plumbing(c):
    """whether comm c is the tt-metrics-exporter or the fabric-manager agent (--beside-collector only)"""
    # A comm matching PLUMBING_RX (^tt[_-]metrics[_-]exp|^tt[_-]fabric[_-]manag) is no holder under --beside-collector, only
    # because those processes use driver-allocated TLB windows only and run-ttx-kmdtlb (kmdtlb.py) writes only the config
    # registers of windows the driver gave it, never another process' windows. A process of theirs that programs fixed
    # windows itself would make this wrong; tests/test_pv_guard.sh case 16 pins the regex.
    return bool(re.search(PLUMBING_RX, (c or "").strip().lower()))


def pods(podlog, rx):
    """'<namespace>/<pod>' of each pod log dir under podlog whose pod name matches rx; [] with no such dir"""
    try:
        ents = os.listdir(podlog)
    except OSError:
        return []
    out = set()
    for e in ents:
        w = e.split("_")
        if len(w) >= 3 and re.search(rx, w[1]):
            out.add(f"{w[0]}/{w[1]}")
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self", dest="me", type=int, required=True, help="run.sh's pid: it and its descendants are ours")
    ap.add_argument("--proc", default="/proc", help="the proc dir (test hook)")
    ap.add_argument("--glob", default="/dev/tenstorrent*", help="the device node glob")
    ap.add_argument(
        "--sudo", action="store_true", help="the scan runs under sudo -n (an unreadable fd dir is not 'no sudo')"
    )
    ap.add_argument(
        "--podlog",
        default="",
        help="the kubelet pod log dir (lib/fleet.sh: FLEET_PODLOG, /var/log/pods); none: no pod read",
    )
    ap.add_argument("--pod-re", default="tt-telemetry|telemetry-collector", help="pod names that count as a holder")
    ap.add_argument(
        "--names", action="store_true", help="the name check of every other process (comm, cmdline): rc 0, 12 or 13"
    )
    ap.add_argument("--units", action="store_true", help="the pause check: rc 0 '<units>|<holders>' or 14 and why not")
    ap.add_argument("--svc-uid", type=int, default=0, help="the uid a pausable service runs as (test hook)")
    ap.add_argument(
        "--beside-collector",
        action="store_true",
        help="the runner's TLB windows come from the driver: a telemetry collector (process or pod) is no holder",
    )
    a = ap.parse_args(argv)
    proc, glob = a.proc, a.glob
    if a.units:
        rc, line = units(proc, glob, a.me, a.podlog, a.pod_re, a.svc_uid)
        print(line)
        return rc
    if a.names:
        rc, line = names(proc, a.me, a.beside_collector)
        print(line)
        return rc
    pl = pods(a.podlog, a.pod_re) if a.podlog else []
    held, unread = scan(proc, glob, a.me)
    parts, col = [], []
    for pid, nodes in sorted(held.items()):
        c = comm(proc, pid)
        h = f"{pid} ({c}) holds {nodes[0]}" + (f" (+{len(nodes) - 1} more)" if len(nodes) > 1 else "")
        (col if a.beside_collector and (collector(proc, pid, c) or plumbing(c)) else parts).append(h)
    if pl:
        (col if a.beside_collector else parts).append(
            f"telemetry collector pod(s) on this node (kubelet pod logs {a.podlog}): {' '.join(pl)}"
        )
    out = ", ".join(parts)
    if unread:
        why = "no sudo" if not a.sudo else "fds unreadable under sudo"
        lst = " ".join(str(p) for p in unread[:8]) + (" ..." if len(unread) > 8 else "")
        out += (
            "; " if out else ""
        ) + f"holder scan incomplete ({why}): the fds of {len(unread)} process(es) could not be read ({lst})"
    if col:
        out += ("; " if out else "") + "beside the telemetry collector: " + ", ".join(col)
    if out:
        print(out)
    return 10 if parts else 11 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
