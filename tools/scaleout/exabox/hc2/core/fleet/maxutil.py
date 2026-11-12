#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""maxutil.py: the bus-bar stage's max-util part.

tt-metal's unit_tests_didt_max_util (tests/didt/max_util_workload) at FPU 92 %, DRAM 92 %, duty 100 % on every chip, run
from the prebuilt archive fleet/maxutil.pin names (fleet/maxutil_pack.sh; no tt-metal build at run time). One process:
device open, init and kernel compile come first; the timed load counts from the last "Main program ready" line, so every
second of it is steady max load. The mode's table gives the bursts: plain and full one 240 s burst (full's old table, fixed seed,
30-90 s bursts with 10-30 s idle between, is the test fixture tests/fleet/maxutil_full.tsv), smoke two FLEET_SMOKE_S bursts with as much idle between. An idle
is the test process stopped (SIGSTOP of our own child, SIGCONT after): its queued programs drain and the chips sit idle.
NUM_ITERATIONS is the bursts' seconds times the enqueue rate measured on this host (state file), else DEFAULT_RATE.

Usage: maxutil.py size <seconds> <rate>                    iterations for that many seconds of load
       maxutil.py rate <state.json>                        "<rate> <source>": the host's measured rate, else the default
       maxutil.py table <quick|full|smoke> [<smoke s>]     the mode's bursts, one "<burst s> <idle s>" row each
       maxutil.py check <table.tsv>                        exit 0 when a full table keeps its rules
       maxutil.py gen <seed> <total s>                     the old full table for that seed (tests/fleet/maxutil_full.tsv)
       maxutil.py run [--pin P | --bin B --home H --lib L --cache C [--runtime-root R] [--opal O]] --mode M [--smoke-s S] [--state F] --log G --result J
                      [--limit-s N]                         --pin (default fleet/maxutil.pin): unpack the archive first
       maxutil.py unpack [--pin P]                         unpack (or reuse) the pinned archive node-local; print its paths
       maxutil.py parse <log>                              the summary run would print for that log (tests)
The archive: fleet/maxutil.pin holds "sha256=<sha256 of the archive file>" and "file=<name>" lines
(fleet/maxutil_pack.sh; a sha256sum-style "<sha256> [<file name>]" line also works); the file is looked up in TRAY_MAXUTIL_DIST (colon-separated dirs), else ~/tray-test-dist
and /data/$USER/tray-test-dist, checked against the pin, and unpacked to node-local disk ($SLURM_TMPDIR, else $TMPDIR,
else /tmp; never under /data) once per pin. No pin, no archive or a sha256 mismatch is
"MAXUTIL ERROR setup: <why>" and exit 3: fleet.sh makes the run INCOMPLETE, never a silent skip.
Lines for fleet.sh start with "MAXUTIL ": "MAXUTIL FAIL <chip> <why>" names a chip by its tt-kmd number
(the test's device id is its PCIe id, UMD's "Opening local chip ids/PCIe ids" line); fleet.sh adds BDF and tray.
"""
import json
import math
import os
import pty
import random
import re
import select
import shutil
import socket
import signal
import stat
import subprocess
import hashlib
import sys
import time

DEFAULT_RATE = 145.0   # enqueues/s, measured in testing (7242 in 49.9 s, NUM_WL_LOOPS 5000)
WL_LOOPS = 5000
PLAIN_S = 240          # 240 s at max power at the bus-bar stage end, in plain and full; it takes plain's 4 min chunk
FULL_S = 600           # about 10 min in full
BURST = (30, 90)
IDLE = (10, 30)
SEED = 50
SETUP_S = 60           # device open, init and kernel compile before the timed load (52 s measured), in the plan
KILL_GRACE_S = 5        # SIGTERM at --limit-s, then SIGKILL of our own child this long after
STOP_GRACE_S = 2        # our own TERM/INT: SIGKILL of the child this long after its SIGTERM, inside run.sh's 4 s reap
MIN_RATE_S = 20        # a run shorter than this does not update the host's rate (smoke)
ENV = {"MAX_UTIL_FPU_UTILIZATION_PCT": "92", "MAX_UTIL_DRAM_UTILIZATION_PCT": "92", "MAX_UTIL_DUTY_CYCLE": "100",
       "MAX_UTIL_SUPER_SYNC": "0", "MAX_UTIL_NUM_WL_LOOPS": str(WL_LOOPS)}
HERE = os.path.dirname(os.path.abspath(__file__))
READY = re.compile(rb"Main program ready: device=(\d+)")
PASS = re.compile(rb"Device (\d+): compute output validation passed")
BADV = re.compile(rb"Device (\d+)[: ][^\n]*?(validation (?:failed|FAILED|mismatch)|mismatch|NaN|not finite)[^\n]*")
UTIL = re.compile(rb"Device (\d+) FPU utilization: [^\n]*?measured=([0-9.]+)%")
OPEN = re.compile(rb"Opening local chip ids/PCIe ids: \{([0-9, ]*)\}")
ITER = re.compile(rb"Iteration (\d+)/(\d+)")


def size(seconds, rate):
    return max(1, math.ceil(float(seconds) * float(rate)))


def rate(state):
    try:
        r = float(json.load(open(state))["rate"])
        if 1 <= r <= 100000:
            return r, "measured " + state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return DEFAULT_RATE, "default (measured in testing)"


class SetupError(Exception):
    pass


# test-infra (seen in testing: rc 127 in 2 s, libevent_pthreads-2.1.so.7 not found, then "no chip passed"): a binary
# that cannot load is "MAXUTIL NOT_RUN test-infra: <why>" and rc 127, never a chip FAIL. The preflight reads ldd; a missing
# lib found elsewhere on the box (TRAY_MAXUTIL_LIB_SEARCH dirs, else DEFAULT_LIB_SEARCH) goes on LD_LIBRARY_PATH: nothing
# is installed or shipped (no outside packages on lab boxes)
LOADER = re.compile(rb"error while loading shared libraries[^\n]*|[^\n]*cannot open shared object file[^\n]*")
DEFAULT_LIB_SEARCH = ("~/tt-metal", "~/.venv", "~/venv", "~/tt-metal-venv", "/opt", "/usr/local/lib")
LIB_SEARCH_DEPTH, LIB_SEARCH_S = 6, 20


def ldd_missing(binary, env, ldd="ldd"):
    """the libs ldd says are 'not found' for binary under env (no ldd, or it fails: none, the run goes on as before)"""
    try:
        r = subprocess.run([ldd, binary], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({m.group(1) for m in re.finditer(r"^\s*(\S+)\s+=>\s+not found", r.stdout.decode(errors="replace"), re.M)})


def find_on_box(name, roots=None):
    """the first file called name under the search roots (depth LIB_SEARCH_DEPTH, LIB_SEARCH_S s in all), else None"""
    if roots is None:
        e = os.environ.get("TRAY_MAXUTIL_LIB_SEARCH")
        roots = e.split(":") if e is not None else DEFAULT_LIB_SEARCH
    end = time.time() + LIB_SEARCH_S
    for r in roots:
        r = os.path.expanduser(r)
        if not r or not os.path.isdir(r):
            continue
        base = r.rstrip(os.sep).count(os.sep)
        for d, dirs, files in os.walk(r):
            if time.time() > end:
                return None
            if name in files and os.path.isfile(os.path.join(d, name)):
                return os.path.join(d, name)
            if d.count(os.sep) - base >= LIB_SEARCH_DEPTH:
                dirs[:] = []
    return None


def preflight(binary, env, ldd="ldd", roots=None):
    """(lines, error): WARN lines for libs taken from elsewhere on the box (env's LD_LIBRARY_PATH gets their dirs); error
    names the libs still missing, else "" """
    miss = ldd_missing(binary, env, ldd)
    if not miss:
        return [], ""
    host, out, found = socket.gethostname().split(".")[0], [], []
    for lib in miss:
        f = find_on_box(lib, roots)
        if f:
            found.append(os.path.dirname(f))
            out.append(f"MAXUTIL WARN test-infra: {os.path.basename(binary)} on {host}: the loader does not find {lib}; "
                       f"using the on-box copy {f}")
    if found:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(found)) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        miss = ldd_missing(binary, env, ldd)
    if not miss:
        return out, ""
    return out, (f"{os.path.basename(binary)} cannot load {', '.join(miss)} on {host} (ldd: not found; no usable on-box copy; "
                 f"nothing is installed on lab boxes)")


def dist_dirs():
    e = os.environ.get("TRAY_MAXUTIL_DIST")
    if e:
        return [d for d in e.split(":") if d]
    u = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    return [os.path.expanduser("~/tray-test-dist")] + ([f"/data/{u}/tray-test-dist"] if u else [])


def read_pin(path):
    """(sha256, file name or None): maxutil_pack.sh's sha256=/file= lines, else the first sha256sum-style line"""
    try:
        lines = open(path).read().splitlines()
    except OSError as e:
        raise SetupError(f"no pin {path} ({e.strerror}): the prebuilt max-util archive is not pinned (fleet/maxutil_pack.sh)")
    kv = dict(ln.strip().split("=", 1) for ln in lines if re.match(r"\s*(sha256|file|bytes)=", ln))
    if "sha256" in kv:
        if not re.fullmatch(r"[0-9a-f]{64}", kv["sha256"]):
            raise SetupError(f"pin {path}: '{kv['sha256']}' is not a sha256")
        return kv["sha256"], (os.path.basename(kv["file"]) if kv.get("file") else None)
    for ln in lines:
        f = ln.split("#")[0].split()
        if f:
            if not re.fullmatch(r"[0-9a-f]{64}", f[0]):
                raise SetupError(f"pin {path}: '{f[0]}' is not a sha256")
            return f[0], (os.path.basename(f[1].lstrip("*")) if len(f) > 1 else None)
    raise SetupError(f"pin {path} is empty")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_archive(sha, name, dirs):
    """the archive file whose sha256 is the pin: by its pinned name, else a maxutil*.tar* name carrying the sha's first 12"""
    seen = []
    for d in dirs:
        if name:
            c = [os.path.join(d, name)]
        else:
            try:
                c = sorted(os.path.join(d, x) for x in os.listdir(d) if x.startswith("maxutil") and ".tar" in x and sha[:12] in x)
            except OSError:
                c = []
        for p in c:
            if not os.path.isfile(p):
                continue
            got = sha256(p)
            if got == sha:
                return p
            seen.append(f"{p} has sha256 {got[:12]}")
    want = name or f"maxutil*{sha[:12]}*.tar*"
    raise SetupError(f"prebuilt max-util archive {want} (sha256 {sha[:12]}) not found in {', '.join(dirs)}"
                     + (f"; {'; '.join(seen)}" if seen else "") + " (fleet/maxutil_pack.sh builds and copies it)")


def node_root():
    r = os.environ.get("TRAY_MAXUTIL_ROOT") or os.path.join(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp",
                                                           "tray-test-maxutil")
    r = os.path.abspath(r)
    if r == "/data" or r.startswith("/data/"):
        raise SetupError(f"unpack dir {r} is under /data: max-util unpacks to node-local disk only")
    return r


def layout(top):
    """bin, home, runtime_root, lib and opal (OPAL_PREFIX, "" without mpi) of an unpacked tree, from its maxutil/MANIFEST
    (maxutil_pack.sh). Never inferred from the binary's path: the first ancestor with a tt_metal/ dir is
    maxutil/tt-metal/build_Release, which has no tt_metal/hw or root soc_descriptors."""
    m = os.path.join(top, "maxutil")
    kv = {}
    try:
        for ln in open(os.path.join(m, "MANIFEST")):
            k, eq, v = ln.rstrip("\n").partition("=")
            if eq:
                kv[k] = v
    except OSError as e:
        raise SetupError(f"unpacked archive {top} has no maxutil/MANIFEST ({e.strerror})")
    for k in ("bin", "home", "lib"):
        if not kv.get(k):
            raise SetupError(f"unpacked archive {top}: maxutil/MANIFEST has no {k}=")
    b = os.path.join(m, kv["bin"])
    if not os.access(b, os.X_OK):
        raise SetupError(f"unpacked archive {top} has no executable {kv['bin']} (MANIFEST bin=)")
    h = os.path.join(m, kv["home"])
    rr = os.path.join(m, kv.get("runtime_root") or kv["home"])   # an archive packed before runtime_root= (abaecf815feb)
    if not os.path.isdir(os.path.join(rr, "tt_metal")):
        raise SetupError(f"unpacked archive {top} has no tt_metal/ dir under {kv.get('runtime_root') or kv['home']} "
                         f"(MANIFEST runtime_root=)")
    lib = ":".join(os.path.join(m, x) for x in kv["lib"].split(":") if x)
    opal = os.path.join(m, kv["opal_prefix"]) if kv.get("opal_prefix") else ""
    return {"bin": b, "home": h, "runtime_root": rr, "lib": lib, "opal": opal, "sfpi": kv.get("sfpi", "")}


def link_sfpi(rr):
    """an archive packed with MANIFEST sfpi=external (the offline bundle, tools/bundle/cut.py) has no runtime/sfpi of its
    own: <runtime_root>/runtime/sfpi becomes a link to the same SFPI release in TRAY_MAXUTIL_SFPI (the bundle's ttnn),
    renewed on each call so a moved bundle still works"""
    src = os.environ.get("TRAY_MAXUTIL_SFPI", "")
    if not src or not os.path.isdir(os.path.join(src, "compiler", "bin")):
        raise SetupError(f"archive packed with sfpi=external needs TRAY_MAXUTIL_SFPI naming an SFPI dir with compiler/bin "
                         f"(got '{src}')")
    src = os.path.abspath(src)
    dst = os.path.join(rr, "runtime", "sfpi")
    if os.path.islink(dst) and os.readlink(dst) == src:
        return
    if os.path.exists(dst) and not os.path.islink(dst):
        raise SetupError(f"{dst} is a real dir in an archive packed with sfpi=external")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp.{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(src, tmp)
    os.replace(tmp, dst)


FRESH_S = 6 * 3600     # an older pin's tree touched in the last 6 h is kept


def private(root):
    """True when root is a dir of our own uid with no group or other access (mode 0700): no process of another uid
    can have entered it, so prune may skip another uid's pids it cannot read"""
    try:
        st = os.lstat(root)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and not st.st_mode & 0o077


def make_private(root):
    """an existing root of ours made earlier (0755: group or other bits) is set to 0700, so private() holds
    and prune can remove idle trees as a non-root user; a symlink or a root of another uid is left alone. Opened with
    O_NOFOLLOW and changed by fd, so the checked dir is the one changed. True when it changed the mode."""
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or not st.st_mode & 0o077:
            return False
        os.fchmod(fd, 0o700)
    except OSError as e:
        print(f"MAXUTIL unpack root {root} not made private ({e}); prune may then remove nothing", flush=True)
        return False
    finally:
        os.close(fd)
    print(f"MAXUTIL unpack root {root} was mode {stat.S_IMODE(st.st_mode):04o}, set to 0700 so prune can remove idle trees", flush=True)
    return True


def foreign(d):
    """a pid that denied our read and is not plainly ours, judged from what any user may read: another uid or gid in
    its status, or not dumpable (its fd dir then belongs to root: our sshd). A pid with exactly our ids that denies
    the read is ours (False). Max-util jobs run as the plain user (run.sh AS), so they are always readable."""
    me, gid = str(os.geteuid()), str(os.getegid())
    try:
        ids = {}
        with open(os.path.join(d, "status")) as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in ("Uid", "Gid"):
                    ids[k] = set(v.split())
        if ids.get("Uid") != {me} or ids.get("Gid") != {gid}:
            return True
        return str(os.stat(os.path.join(d, "fd")).st_uid) != me
    except FileNotFoundError:
        return not os.path.isdir(d)   # exited mid-scan: uses nothing
    except OSError:
        return False


def in_use(proc="/proc", skip_others=False):
    """every path a live process uses: cwd, exe, open fds and mapped files. None when any read fails (permission, I/O):
    then nothing counts as unused. A process gone mid-scan, or a link it does not have (a kernel thread's exe), is no failure.
    skip_others (a private root only): a pid that denies the read and is not plainly ours (foreign) is skipped, not a
    failure; a denied read of one of our own pids is still None (a non-root user cannot read /proc/1/cwd)."""
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return None
    seen = set()
    for p in pids:
        d = os.path.join(proc, p)
        try:
            for k in ("cwd", "exe"):
                try:
                    seen.add(os.readlink(os.path.join(d, k)))
                except FileNotFoundError:
                    pass
            try:
                fds = os.listdir(os.path.join(d, "fd"))
            except FileNotFoundError:
                fds = []
            for f in fds:
                try:
                    seen.add(os.readlink(os.path.join(d, "fd", f)))
                except FileNotFoundError:   # closed since the listing
                    pass
            try:
                with open(os.path.join(d, "maps")) as m:
                    for ln in m:
                        x = ln.split(None, 5)
                        if len(x) == 6 and x[5].startswith("/"):
                            seen.add(x[5].rstrip("\n"))
            except FileNotFoundError:
                pass
        except ProcessLookupError:
            continue
        except PermissionError:
            if skip_others and foreign(d):
                continue
            return None
        except OSError:
            return None
    return seen


def prune(root, keep, proc="/proc", now=None):
    """older pins' trees of ours directly under root (names a sha's first 12, or an unpack's .tmp.<pid>), removed only
    when untouched for FRESH_S and no live process uses anything under them; nothing when a /proc read fails.
    Under a private root (mode 0700, our uid; unpack makes it so) a pid we may not read
    that is foreign (another uid or gid, or not dumpable) is skipped: another uid cannot reach the trees; uid 0 and
    our own sshd or setgid shells could, but max-util jobs run as the plain user (run.sh AS) and stay readable.
    Returns the names removed."""
    root = os.path.realpath(root)
    now = time.time() if now is None else now
    old = []
    for x in os.listdir(root):
        if x == keep or not re.fullmatch(r"[0-9a-f]{12}(\.tmp\.[0-9]+)?", x):
            continue
        t = os.path.join(root, x)
        try:
            if os.path.islink(t) or not os.path.isdir(t):
                continue
            newest = max([os.lstat(t).st_mtime] + [os.lstat(os.path.join(t, e)).st_mtime for e in os.listdir(t)])
        except OSError:
            continue
        if now - newest >= FRESH_S:
            old.append(x)
    if not old:
        return []
    used = in_use(proc, skip_others=private(root))
    if used is None:
        return []
    gone = []
    for x in old:
        t = os.path.join(root, x)
        if any(u == t or u.startswith(t + "/") for u in used):
            continue
        shutil.rmtree(t, ignore_errors=True)
        gone.append(x)
    return gone


def unpack(pin, dirs=None, root=None):
    """the pinned archive unpacked node-local (reused when its .ok marker names the pin); its layout and kernel cache"""
    sha, name = read_pin(pin)
    root = root or node_root()
    tree = os.path.join(root, sha[:12])
    ok = os.path.join(tree, ".ok")
    reused = False
    try:
        reused = open(ok).read().split()[0] == sha
    except (OSError, IndexError):
        pass
    if not reused:
        arc = find_archive(sha, name, dirs or dist_dirs())
        os.makedirs(root, mode=0o700, exist_ok=True)   # private: prune may then skip other uids' pids
        make_private(root)   # a root made 0755 earlier
        tmp = f"{tree}.tmp.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        r = subprocess.run(["tar", "-xf", arc, "-C", tmp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise SetupError(f"tar -xf {arc} failed (rc {r.returncode}): {r.stdout.decode(errors='replace').strip()[-200:]}")
        layout(tmp)   # refuses a tree with no binary before it replaces anything
        shutil.rmtree(tree, ignore_errors=True)
        os.rename(tmp, tree)
        with open(ok, "w") as f:
            f.write(f"{sha} {arc}\n")
        prune(root, sha[:12])
    out = layout(tree)
    if out.pop("sfpi") == "external":
        link_sfpi(out["runtime_root"])
    out.update(cache=os.path.join(tree, ".tt-metal-cache"), tree=tree, sha=sha, reused=reused)
    return out


def default_state():
    return os.path.expanduser(f"~/.cache/tray-test-maxutil/rate-{socket.gethostname().split('.')[0]}.json")


def gen(seed=SEED, total=FULL_S):
    g, rows, t = random.Random(seed), [], 0
    while total - t >= BURST[0]:
        b = min(g.randint(*BURST), total - t)
        i = g.randint(*IDLE)
        if total - t - b - i < BURST[0]:
            i = 0
            b = min(BURST[1], total - t)
        rows.append((b, i))
        t += b + i
        if i == 0:
            break
    return rows


def load_table(path):
    rows = []
    for ln in open(path):
        f = ln.split("#")[0].split()
        if f:
            rows.append((int(f[0]), int(f[1])))
    return rows


def check(rows):
    bad = []
    for n, (b, i) in enumerate(rows, 1):
        if not BURST[0] <= b <= BURST[1]:
            bad.append(f"row {n}: burst {b} s outside {BURST[0]}-{BURST[1]}")
        last = n == len(rows)
        if (last and i != 0) or (not last and not IDLE[0] <= i <= IDLE[1]):
            bad.append(f"row {n}: idle {i} s (between bursts {IDLE[0]}-{IDLE[1]}, 0 after the last)")
    tot = sum(b + i for b, i in rows)
    if abs(tot - FULL_S) > 60:
        bad.append(f"total {tot} s, not about {FULL_S}")
    return bad


def table(mode, smoke_s=2, load_s=None):
    """load_s: the burst's s when max-util is the weaker of the bus-bar stage's two max loads (180 s)"""
    if mode == "smoke":
        s = max(1, int(smoke_s))
        return [(s, s), (s, 0)]
    return [(int(load_s) if load_s else PLAIN_S, 0)]


def summary(log, marks=None):
    """per-chip result from the log bytes; marks: wall times run recorded (ready, first iteration, first pass)"""
    chips = [int(c) for c in (OPEN.search(log).group(1).decode().replace(" ", "").split(",") if OPEN.search(log) else []) if c]
    ok = sorted({int(m.group(1)) for m in PASS.finditer(log)})
    badv = {}
    for m in BADV.finditer(log):
        badv.setdefault(int(m.group(1)), m.group(0).decode(errors="replace")[:160])
    util = {int(m.group(1)): float(m.group(2)) for m in UTIL.finditer(log)}
    it = [(int(a), int(b)) for a, b in ITER.findall(log)]
    return {"chips": chips, "passed": ok, "failed": badv, "util": util,
            "iterations": it[-1][1] if it else None, "last_iteration": it[-1][0] if it else None, "marks": marks or {}}


def judge(s, rc, stopped=None):
    """MAXUTIL lines: one FAIL per chip its output names, else per chip missing its pass line after a clean exit;
    stopped (a signal name): one INCOMPLETE line, no verdict on a load that was cut"""
    if stopped:
        return [f"MAXUTIL INCOMPLETE - stopped by {stopped} after {len(s['passed'])} chip pass line(s), rc {rc}"]
    out, want = [], s["chips"] or sorted(set(s["passed"]) | set(s["failed"]))
    for c, why in sorted(s["failed"].items()):
        out.append(f"MAXUTIL FAIL {c} {why}")
    if rc == 0:
        for c in want:
            if c not in s["passed"] and c not in s["failed"]:
                out.append(f"MAXUTIL FAIL {c} no 'compute output validation passed' line after a clean exit")
    if not want or not s["passed"]:
        out.append("MAXUTIL FAIL - no chip passed its output check")
    u = sorted(s["util"].values())
    out.append(f"MAXUTIL {len(s['passed'])}/{len(want) or '?'} chips passed their output check"
               + (f", FPU measured {u[0]:g}-{u[-1]:g} %" if u else ", FPU utilization not reported") + f", rc {rc}")
    return out


def run(a):
    if not a.bin:
        try:
            u = unpack(a.pin)
        except SetupError as e:
            print(f"MAXUTIL ERROR setup: {e}", flush=True)
            json.dump({"maxutil": {"mode": a.mode, "error": str(e)}}, open(a.result, "w"), indent=1)
            return 3
        a.bin, a.home, a.runtime_root, a.lib, a.opal, a.cache = u["bin"], u["home"], u["runtime_root"], u["lib"], u["opal"], u["cache"]
        print(f"MAXUTIL archive {u['sha'][:12]} {'reused' if u['reused'] else 'unpacked'} at {u['tree']}", flush=True)
    a.state = a.state or default_state()
    rows = table(a.mode, a.smoke_s, a.load_s)
    load_s = sum(b for b, _ in rows)
    r, src = rate(a.state)
    n = size(load_s, r)
    # TT_METAL_RUNTIME_ROOT only here, for the test binary (MANIFEST runtime_root=, else --runtime-root, else --home):
    # exported for all of run.sh, it broke every other tt-metal load (t9600)
    env = dict(os.environ, TT_METAL_HOME=a.home, TT_METAL_RUNTIME_ROOT=a.runtime_root or a.home, ARCH_NAME="blackhole", TT_METAL_CACHE=a.cache,
               LD_LIBRARY_PATH=a.lib + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
               MAX_UTIL_NUM_ITERATIONS=str(n), **ENV)
    if a.opal:
        env["OPAL_PREFIX"] = a.opal
    for k in ("TT_METAL_TDP_LIMIT_WATTS",):   # TDP item: never a power-limit knob
        env.pop(k, None)
    os.makedirs(a.cache, exist_ok=True)
    warn, err = preflight(a.bin, env)
    for ln in warn:
        print(ln, flush=True)
    if err:
        print(f"MAXUTIL NOT_RUN test-infra: {err}", flush=True)
        json.dump({"maxutil": {"mode": a.mode, "error": f"test-infra: {err}", "rc": 127}}, open(a.result, "w"), indent=1)
        return 127
    print(f"MAXUTIL plan: mode {a.mode}, bursts {' '.join(f'{b}+{i}' for b, i in rows)} s, {load_s} s of load,"
          f" NUM_ITERATIONS {n} at {r:g}/s ({src}), FPU 92 DRAM 92 duty 100, NUM_WL_LOOPS {WL_LOOPS}", flush=True)
    # TERM/INT (the broker's preemption, run.sh's pg_reap, reliability.sh's stop): no more bursts, SIGCONT and
    # SIGTERM of the child, SIGKILL STOP_GRACE_S later. unit_tests_didt_max_util has no stop input (its iteration count is
    # fixed at start by MAX_UTIL_NUM_ITERATIONS) and no close on SIGTERM: it ends without its own device close, and the
    # next tt-smi -glx_reset (run.sh's C9 reset, reliability.sh's step-end reset) brings the chips back.
    stop = {}

    def on_stop(signum, _frame):
        stop.setdefault("sig", signal.Signals(signum).name)
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, on_stop)
    mfd, sfd = pty.openpty()   # a tty: the test's log and progress lines come as written, not block-buffered
    p = subprocess.Popen([a.bin, "--gtest_filter=*All*"], stdout=sfd, stderr=sfd, stdin=subprocess.DEVNULL, env=env, cwd=a.home)
    os.close(sfd)
    buf, marks, pauses = bytearray(), {"start": time.time()}, []
    t_ready = None
    plan, k, t_next, paused_until = None, 0, None, None
    deadline = time.time() + a.limit_s
    limited, kill_at = False, None
    exited = False   # the child exited: drain the tty with no wait, then end
    with open(a.log, "ab") as lg:
        while True:
            now = time.time()
            if stop and "t" not in stop:
                stop["t"] = now
                print(f"MAXUTIL stopped by {stop['sig']}: no more bursts; the test process gets SIGTERM, SIGKILL {STOP_GRACE_S} s later", flush=True)
                if paused_until is not None:
                    pauses[-1].append(now)
                    paused_until = None
                t_next, kill_at = None, now + STOP_GRACE_S
                if p.poll() is None:
                    p.send_signal(signal.SIGCONT)
                    p.terminate()
            if not stop and not limited and now > deadline and p.poll() is None:
                print(f"MAXUTIL FAIL - its own limit {a.limit_s} s passed; the test process was stopped", flush=True)
                limited, kill_at = True, now + KILL_GRACE_S
                p.send_signal(signal.SIGCONT)
                p.terminate()
            elif kill_at is not None and now >= kill_at and p.poll() is None:
                p.kill()   # it ignored SIGTERM: SIGKILL of our own child, by its pid
                kill_at = None
            if paused_until is not None and now >= paused_until:
                p.send_signal(signal.SIGCONT)
                pauses[-1].append(now)
                paused_until = None
                t_next = now + rows[k][0] if k < len(rows) else None
            elif plan and paused_until is None and t_next is not None and now >= t_next and k < len(rows) and rows[k][1] > 0 \
                    and "pass" not in marks and not limited and p.poll() is None:
                p.send_signal(signal.SIGSTOP)
                pauses.append([now])
                paused_until = now + rows[k][1]
                k += 1
            try:
                rd, _, _ = select.select([mfd], [], [], 0 if exited else 0.2)
                ch = os.read(mfd, 65536) if rd else b""
            except OSError:
                ch = b""   # the child closed the tty
            if ch:
                lg.write(ch)
                lg.flush()
                tail = bytes(buf[-200:]) + ch
                buf += ch
                if READY.search(tail):
                    t_ready = time.time()
                if "load" not in marks and b"Iteration " in tail and t_ready:
                    marks["load"] = t_ready   # the last "Main program ready" before the first progress line
                    plan, k, t_next = True, 0, t_ready + rows[0][0]
                if "pass" not in marks and PASS.search(tail):
                    marks["pass"] = time.time()
            elif exited:
                break   # a no-wait read after the exit gave EOF/EIO or nothing: all its output is read
            elif p.poll() is not None:
                exited = True   # lines written just before the exit may still be in the tty
    os.close(mfd)
    rc = p.wait()
    if limited:
        rc = 124   # a limit kill fails the run whatever the child's own exit code
    if stop:
        rc = 143 if stop["sig"] == "SIGTERM" else 130
    marks["end"] = time.time()
    if paused_until is not None:
        pauses[-1].append(marks["end"])
    paused = sum(b - a_ for a_, b in pauses if b)
    s = summary(bytes(buf), marks)
    t0 = marks.get("load")
    t1 = marks.get("pass") or marks["end"]
    act = (t1 - t0 - paused) if t0 else None
    meas = round(n / act, 1) if act and act > 0 and rc == 0 and s["passed"] else None
    res = {"burst": {"t0": t0, "seconds": round(t1 - t0, 1) if t0 else None, "tflops_per_chip": None,
                     "op": "unit_tests_didt_max_util FPU 92 DRAM 92 duty 100", "load": f"max-util {a.mode} ({len(rows)} burst(s))"},
           "maxutil": {"mode": a.mode, "rows": rows, "iterations": n, "rate_used": r, "rate_source": src,
                       "load_s_measured": round(act, 1) if act else None, "rate_measured": meas, "paused_s": round(paused, 1),
                       "pauses": pauses, "setup_s": round(t0 - marks["start"], 1) if t0 else None, "rc": rc,
                       "passed": s["passed"], "failed": s["failed"], "util": s["util"], "chips": s["chips"],
                       "stopped": stop.get("sig"), "stop_s": round(marks["end"] - stop["t"], 1) if stop.get("t") else None}}
    json.dump(res, open(a.result, "w"), indent=1)
    if meas and act >= MIN_RATE_S and not stop:
        try:
            os.makedirs(os.path.dirname(a.state) or ".", exist_ok=True)
            json.dump({"rate": meas, "iterations": n, "seconds": round(act, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "wl_loops": WL_LOOPS}, open(a.state + ".tmp", "w"))
            os.replace(a.state + ".tmp", a.state)
        except OSError as e:
            print(f"MAXUTIL rate not saved ({e})", flush=True)
    print(f"MAXUTIL timed load {res['maxutil']['load_s_measured']} s of planned {load_s} s (setup before it"
          f" {res['maxutil']['setup_s']} s, idle {round(paused, 1)} s in {len(pauses)} pause(s)), measured rate {meas}/s", flush=True)
    ld = LOADER.search(bytes(buf))
    if not s["passed"] and not s["failed"] and not limited and (rc in (126, 127) or ld):   # it never ran: test-infra, not a chip FAIL
        why = ld.group(0).decode(errors="replace").strip()[-300:] if ld else "the test binary did not start"
        print(f"MAXUTIL NOT_RUN test-infra: {os.path.basename(a.bin)} rc {rc} on {socket.gethostname().split('.')[0]}: {why}", flush=True)
        return 127
    for ln in judge(s, rc, stop.get("sig")):
        print(ln, flush=True)
    return rc


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    c = argv[0]
    if c == "size":
        print(size(argv[1], argv[2]))
    elif c == "rate":
        r, src = rate(argv[1])
        print(f"{r:g} {src}")
    elif c == "table":
        for b, i in table(argv[1], argv[2] if len(argv) > 2 else 2, argv[3] if len(argv) > 3 else None):
            print(b, i)
    elif c == "gen":
        for b, i in gen(int(argv[1]), int(argv[2])):
            print(f"{b}\t{i}")
    elif c == "check":
        bad = check(load_table(argv[1]))
        for b in bad:
            print("BAD", b)
        return 1 if bad else 0
    elif c == "parse":
        for ln in judge(summary(open(argv[1], "rb").read()), int(argv[2]) if len(argv) > 2 else 0):
            print(ln)
    elif c == "run":
        import argparse
        ap = argparse.ArgumentParser()
        for k in ("bin", "home", "runtime-root", "lib", "opal", "cache", "state"):
            ap.add_argument("--" + k)
        for k in ("log", "result"):
            ap.add_argument("--" + k, required=True)
        ap.add_argument("--pin", default=os.path.join(HERE, "maxutil.pin"))
        ap.add_argument("--mode", default="quick")
        ap.add_argument("--smoke-s", type=int, default=2)
        ap.add_argument("--limit-s", type=int, default=900)
        ap.add_argument("--load-s", type=int, default=0)
        a = ap.parse_args(argv[1:])
        if a.bin and not (a.home and a.lib and a.cache):
            ap.error("--bin needs --home, --lib and --cache")
        return run(a)
    elif c == "unpack":
        pin = argv[2] if len(argv) > 2 and argv[1] == "--pin" else os.path.join(HERE, "maxutil.pin")
        try:
            u = unpack(pin)
        except SetupError as e:
            print(f"MAXUTIL ERROR setup: {e}")
            return 3
        for k in ("bin", "home", "runtime_root", "lib", "opal", "cache", "tree", "sha", "reused"):
            print(f"{k} {u[k]}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
