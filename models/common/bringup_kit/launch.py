#!/usr/bin/env python3
"""Pin-the-commit launcher for device jobs (generalizes DeepSeek tools/grid.py + grid_exec.sh + run_main.sh).

    launch.py --worktree WT --commit HASH --host 45 --log LOG --env DSV41_SESSION=isl4k_b16 [--env K=V ...] -- pytest -x -s demo/text_demo.py -k session

Refuses to start unless the worktree's tracked files are clean (except --ignore-dirty prefixes) and HEAD == --commit. Then, on the host
(10.82.97.<host>, or 'local'), DETACHED (setsid nohup, survives a session restart):
  flock -w <wait> /tmp/dsv4_dev.lock  (one device job per host)
    unset every variable starting with --unset-prefix (default DSV41_,MOE_COMPUTE_,TT_METAL_: leftovers of short runs poison measurements)
    export only: the --env pairs, TT_METAL_HOME/PYTHONPATH=<worktree>, TT_METAL_CACHE, PYTHONFAULTHANDLER=1,
                 <slot-prefix>BUILD_SLOTS (cap on concurrent full model builds, default 5) and <slot-prefix>BUILD_STAGGER_S (min seconds between
                 two build starts cluster-wide, default 480); both are read by the model's tt/build_slots.py (copy it from the DeepSeek package)
    print a GRID_ENV header (host, date, commit, branch, every exported variable) into the log
    run the command in its own process group, start hangwatch.py on it (stall limit 45 min), wait, print 'LAUNCH_RC=<rc> HANGWATCH_RC=<0|3>'.
--dry-run prints the script and starts nothing. The same script text is what runs, so a dry run is the documentation of a run.
"""
import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitlib  # noqa: E402
from host_health import load_bad  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.environ.get("KIT_DEVICE_LOCK", "/tmp/dsv4_dev.lock")  # one device job per host; override only in tests


def check_tree(worktree, commit, ignore_dirty):
    if not kitlib.resolve(worktree, commit):
        kitlib.die(f"commit {commit} not found in {worktree}")
    if not kitlib.same_commit(worktree, "HEAD", commit):
        kitlib.die(f"HEAD {kitlib.head(worktree)} != --commit {commit}: refusing to start")
    dirty = kitlib.dirty_files(worktree, ignore_prefixes=ignore_dirty)
    if dirty:
        kitlib.die(f"worktree dirty ({', '.join(dirty[:5])}): commit first, refusing to start")


def build_script(a, head_short):
    env = dict(kv.split("=", 1) for kv in a.env)
    exports = {
        "TT_METAL_HOME": a.worktree,
        "PYTHONPATH": a.worktree,
        "TT_METAL_CACHE": a.cache_dir,
        "PYTHONFAULTHANDLER": "1",
        f"{a.slot_prefix}BUILD_SLOTS": str(a.build_slots),
        f"{a.slot_prefix}BUILD_STAGGER_S": str(a.build_stagger_s),
        **env,
    }
    prefixes = "|".join(p for p in a.unset_prefix.split(",") if p)
    exp = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(exports.items()))
    grid_vars = "|".join(sorted({p for p in a.unset_prefix.split(",") if p} | {a.slot_prefix}))
    cmd = " ".join(shlex.quote(c) for c in a.command)
    hw = os.path.join(HERE, "hangwatch.py")
    inner = f"""set -u
for v in $(compgen -e | grep -E '^({prefixes})'); do unset "$v"; done
cd {shlex.quote(a.worktree)}
[ -f {shlex.quote(a.python_env)}/bin/activate ] && source {shlex.quote(a.python_env)}/bin/activate
{exp}
KIT_CMD={shlex.quote(cmd)}
echo "GRID_ENV $(hostname -s) $(date +%FT%T) head={head_short} branch={kitlib.git(a.worktree, 'rev-parse', '--abbrev-ref', 'HEAD')} cmd=$KIT_CMD"
env | grep -E '^({grid_vars}|PYTHONPATH|PYTHONFAULTHANDLER)' | sort | sed 's/^/GRID_ENV /'
setsid {cmd} &
PID=$!
python3 {shlex.quote(hw)} --pid $PID --log {shlex.quote(a.log)} --stall-min {a.stall_min} --report-dir {shlex.quote(a.report_dir)} {a.hangwatch_args} &
HW=$!
wait $PID; RC=$?
wait $HW; HRC=$?   # hangwatch exits by itself once the job is gone; if it was mid-handling (triage, kill, reset) we must let it finish
echo "LAUNCH_RC=$RC HANGWATCH_RC=$HRC"
"""
    outer = f"""setsid nohup flock -w {a.lock_wait_s} {LOCK} bash -c {shlex.quote(inner)} >> {shlex.quote(a.log)} 2>&1 < /dev/null &
echo launched
"""
    return outer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worktree", required=True)
    p.add_argument("--commit", required=True, help="hash HEAD must equal")
    p.add_argument("--host", default="local", help="last octet (10.82.97.N), full address, or 'local'")
    p.add_argument("--log", required=True)
    p.add_argument(
        "--env", action="append", default=[], metavar="K=V", help="the documented per-run variables (repeatable)"
    )
    p.add_argument(
        "--unset-prefix",
        default="DSV41_,MOE_COMPUTE_,TT_METAL_",
        help="comma list of variable prefixes unset before the run",
    )
    p.add_argument(
        "--slot-prefix",
        default="DSV41_",
        help="prefix of the model's BUILD_SLOTS / BUILD_STAGGER_S variables (tt/build_slots.py)",
    )
    p.add_argument("--build-slots", type=int, default=5)
    p.add_argument("--build-stagger-s", type=int, default=480)
    p.add_argument("--cache-dir", default="/mnt/tt-data/ssinghal/tt-metal-cache/kit")
    p.add_argument("--python-env", default=None, help="venv dir (default <worktree>/python_env)")
    p.add_argument("--stall-min", type=float, default=45)
    p.add_argument("--report-dir", default="/mnt/tt-data/ssinghal/dsv4-logs/triage")
    p.add_argument(
        "--hangwatch-args", default="", help="extra args for hangwatch.py, e.g. '--reset-cmd none --triage-cmd none'"
    )
    p.add_argument("--lock-wait-s", type=int, default=14400)
    p.add_argument(
        "--ignore-dirty", action="append", default=[], help="path prefix allowed to be dirty, repeatable (e.g. tools/)"
    )
    p.add_argument("--allow-bad-host", action="store_true", help="launch even if the host is listed in bad_hosts.txt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER, help="after --")
    a = p.parse_args(argv)
    if a.command and a.command[0] == "--":
        a.command = a.command[1:]
    if not a.command:
        kitlib.die("no command (put it after --)")
    for kv in a.env:
        if "=" not in kv:
            kitlib.die(f"--env {kv}: expected K=V")
    a.worktree = os.path.abspath(a.worktree)
    a.log = os.path.abspath(a.log)
    a.python_env = a.python_env or os.path.join(a.worktree, "python_env")
    check_tree(a.worktree, a.commit, a.ignore_dirty)
    bad = load_bad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bad_hosts.txt"))
    if a.host.split(".")[-1] in bad and not a.allow_bad_host:
        kitlib.die(f"host {a.host} is in bad_hosts.txt: {bad[a.host.split('.')[-1]]} (--allow-bad-host to override)")
    script = build_script(a, kitlib.head(a.worktree))
    if a.dry_run:
        print(script)
        return 0
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    open(a.log, "a").close()
    if a.host == "local":
        r = subprocess.run(["bash", "-s"], input=script, text=True)
    else:
        addr = a.host if "." in a.host else f"10.82.97.{a.host}"
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", addr, "bash -s"], input=script, text=True
        )
    print(f"log: {a.log}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
