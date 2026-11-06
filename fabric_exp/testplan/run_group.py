#!/usr/bin/env python3
"""Regression groups for landing the new fabric router (TEST_PLAN.md §4): plan / run / job / stage / submit / collect /
status / compare / resolve. The groups live in manifest.yaml next to this file; README.md has the workflow.

  plan     list groups, steps, timeouts and how each runs (broker commands for a09u32, submit lines for Slurm)
  resolve  print one group's resolved step commands for a tree (from_ci entries read from that tree)
  run      run a group's steps for one router and pass, here: on a09u32 inside a broker job (budgeted, --resume), or
           inside a Slurm job (called by `job`)
  job      the Slurm side (regress.sbatch calls it): node guard, busy check, per-job TT_METAL_HOME farm, reset and
           health check, every router / pass of the group, results to /data
  stage    copy a built tree to its /data path for Slurm nodes, rewrite its venv's paths, verify it from a09u32
  submit   sbatch a group for a Slurm system (dry run with sbatch --test-only unless --go)
  collect  copy a label's Slurm results from /data into fabric_exp/testplan/results
  status   one line per step of a label
  compare  base vs test (labels and/or routers): coverage per test case, fabric CSV rows, metrics, benchmark data

The target tree is a parameter (--tree NAME from the manifest's trees, or a path), so the same groups run on main's
build and later on the new router's build.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "manifest.yaml")
DONE = {"PASS", "FAIL", "HANG", "TIMEOUT", "SKIP", "NOT_RUN", "KNOWN_FAIL", "PERF_WINDOW", "ENV"}
# a FAIL case whose message says the measured value left the test's own perf window (too fast or too slow)
PERF_WINDOW_RX = r"outside (?:of )?(?:the )?(?:performance |expected )?range|outside \[|outside \("
WINDOW_NUM_RX = re.compile(r"([-+]?\d+(?:\.\d+)?)[^\d\[\(]*?(?:is )?outside[^\[\(]*[\[\(]\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)")  # a step with one of these is not run again by --resume
STOPPERS = {"HANG", "TIMEOUT", "CAPPED", "BUSY"}  # after these the device may be wedged: a broker job stops there
DROP_ENV = {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH", "GTEST_OUTPUT", "PYTEST_ADDOPTS",
            "CONDA_PREFIX", "ARCH_NAME", "PYTHON_ENV_DIR", "VLLM_TARGET_DEVICE"}  # the a09u32 broker injects study-tree values
KEEP_TT = ("TT_VISIBLE_DEVICES",)  # kept if the caller set it (device selection belongs to the host, not the test)
STEP_ENV_STRIP = ("SLURM_", "PMI_", "PMIX_", "OMPI_", "PRTE_")  # mpirun inside a step must not read the allocation


class RegressError(Exception):
    pass


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def iso(t=None):
    return datetime.datetime.fromtimestamp(t or time.time()).astimezone().isoformat(timespec="seconds")


def say(*a):
    print(f"[regress {ts()}]", *a, flush=True)


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class _NoDupLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate mapping keys (YAML would silently keep the last one)."""


def _no_dup_mapping(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dup = sorted({str(k) for k in keys if keys.count(k) > 1})
    if dup:
        raise RegressError(f"duplicate keys {dup} in the manifest at line {node.start_mark.line + 1}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_NoDupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup_mapping)


def load_manifest(path):
    with open(path) as f:
        man = yaml.load(f, Loader=_NoDupLoader)
    if man.get("version") != 1:
        raise RegressError(f"{path}: unknown manifest version {man.get('version')}")
    ids = [g["id"] for g in man["groups"]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise RegressError(f"duplicate group ids: {sorted(dup)}")
    return man


def get_group(man, gid):
    for g in man["groups"]:
        if g["id"] == gid:
            return g
    raise RegressError(f"no group {gid!r}; groups: {', '.join(x['id'] for x in man['groups'])}")


def short_host():
    return socket.gethostname().split(".")[0]


def tree_info(man, name_or_path, where):
    """(name, path, sha) of a tree. where = 'local' (a09u32 path) or 'data' (the /data copy)."""
    trees = man.get("trees", {})
    if name_or_path in trees:
        t = trees[name_or_path]
        path = t["path"] if where == "local" else t["data_path"]
        return name_or_path, path, t.get("sha")
    path = os.path.abspath(name_or_path)
    return os.path.basename(path.rstrip("/")), path, None


def tree_patches(path):
    try:
        with open(os.path.join(path, ".regress_tree.json")) as f:
            return json.load(f).get("patches") or []
    except (OSError, ValueError):
        return []


def tree_sha(path):
    meta = os.path.join(path, ".regress_tree.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get("sha")
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------- step resolution

def resolve_step(man, step, tree):
    """The step as it runs on `tree`: cmd, timeout, CI source, auto_triage, list."""
    r = {"id": step["id"], "result": as_list(step.get("result")), "needs": as_list(step.get("needs")),
         "env": {k: str(v) for k, v in (step.get("env") or {}).items()}, "status": step.get("status", "ready"),
         "marker": step.get("marker"), "metrics": step.get("metrics"), "est_min": step.get("est_min"),
         "on_missing": step.get("on_missing", "skip"), "perf_window": step.get("perf_window"),
         "known_fail": step.get("known_fail"), "env_fail": step.get("env_fail"),
         "known_fail_statuses": step.get("known_fail_statuses") or ["FAIL"],
         "known_fail_cases": step.get("known_fail_cases"), "timeout_note": step.get("timeout_note"),
         "deviation": step.get("deviation"), "note": step.get("note")}
    src = step.get("from_ci")
    if src:
        lst = os.path.join(tree, man.get("ci_lists", "tests/pipeline_reorg"), src["list"])
        if not os.path.exists(lst):
            raise RegressError(f"step {step['id']}: {lst} does not exist")
        with open(lst) as f:
            entries = yaml.safe_load(f)
        e = next((x for x in entries if isinstance(x, dict) and x.get("name") == src["name"]), None)
        if e is None:
            raise RegressError(f"step {step['id']}: {src['list']} has no entry named {src['name']!r}")
        skus = e.get("skus") or {}
        sku = src.get("sku")
        if sku not in skus:
            raise RegressError(f"step {step['id']}: entry {src['name']!r} has no sku {sku!r} (has {sorted(skus)})")
        skud = skus[sku] or {}
        cmd = e["cmd"]
        subs = {k: v for k, v in e.items()
                if isinstance(v, (str, int, float)) and k not in ("name", "cmd", "team", "owner_id")}
        subs.update({k: v for k, v in skud.items() if k != "timeout" and isinstance(v, (str, int, float))})
        for k, v in subs.items():
            cmd = cmd.replace("{%s}" % k, str(v))
        left = sorted(set(re.findall(r"(?<![$\\])\{([a-z_]+)\}", cmd)))
        if left:
            raise RegressError(f"step {step['id']}: unresolved placeholders {left} in {src['name']!r}")
        for a, b in src.get("replace") or []:  # literal edits, e.g. a test directory narrowed to a file subset
            if a not in cmd:
                raise RegressError(f"step {step['id']}: replace target {a!r} not in {src['name']!r}")
            cmd = cmd.replace(a, b)
        lines = cmd.splitlines()
        if src.get("lines_matching"):
            lines = [l for l in lines if re.search(src["lines_matching"], l)]
        for pat in as_list(src.get("drop_lines")):
            lines = [l for l in lines if not re.search(pat, l)]
        cmd = "\n".join(lines).strip() + "\n"
        if not cmd.strip():
            raise RegressError(f"step {step['id']}: the line filters left no command")
        r.update(cmd=cmd, timeout_min=step.get("timeout_min", skud.get("timeout")), list=src["list"],
                 auto_triage=e.get("auto_triage", True),
                 ci={"list": src["list"], "name": src["name"], "sku": sku, "ci_timeout_min": skud.get("timeout"),
                     "filtered": bool(src.get("lines_matching") or src.get("drop_lines") or src.get("replace"))})
    else:
        ref = step.get("ci_ref") or {}
        r.update(cmd=step["cmd"].rstrip() + "\n", timeout_min=step["timeout_min"], list=ref.get("list"),
                 auto_triage=step.get("auto_triage", True), ci=dict(ref, mirrored=True) if ref else None)
    if step.get("pre_cmd"):  # diagnostics before the command (logged in the step), never part of the CI command
        r["cmd"] = "# pre_cmd (diagnostics)\n" + step["pre_cmd"].rstrip() + "\n# the step's command\n" + r["cmd"]
    if not r["timeout_min"]:
        raise RegressError(f"step {step['id']}: no timeout")
    return r


def mnt_paths(cmd):
    return sorted(set(m.rstrip("/'\";,)") for m in re.findall(r"/mnt/[A-Za-z0-9_./+=-]+", cmd)))


def active_path_map(man):
    """The manifest's path_map entries that apply on this node: left side missing, right side present."""
    return {a: b for a, b in (man.get("path_map") or {}).items() if not os.path.exists(a) and os.path.exists(b)}


def map_paths(text, pm):
    for a, b in pm.items():
        text = text.replace(a, b)
    return text


# ------------------------------------------------------------------------------------------------- context

class Ctx:
    """Where and how a run executes."""

    def __init__(self, man, group, args, mode=None):
        self.man, self.group, self.args = man, group, args
        self.system = man["systems"].get(group["system"], {})
        self.node = short_host()
        broker = self.system.get("broker") or {}
        if mode:
            self.mode = mode
        elif getattr(args, "mode", None):
            self.mode = args.mode
        elif broker and self.node == broker.get("node"):
            self.mode = "broker"
        elif os.environ.get("SLURM_JOB_ID"):
            self.mode = "slurm"
        else:
            self.mode = "local"
        where = "data" if self.mode == "slurm" else "local"
        self.tree_name, self.tree, self.tree_sha_expected = tree_info(man, args.tree, where)
        self.tree_sha = tree_sha(self.tree)
        self.tree_patches = tree_patches(self.tree)
        self.out_root = args.out or (man["results"]["data"] if self.mode == "slurm" else man["results"]["local"])
        self.label = args.label
        self.home = self.tree  # TT_METAL_HOME; a Slurm job replaces it with its own farm
        self.cache_root = (broker.get("cache") if self.mode == "broker" else None) or \
            os.path.join(os.path.expanduser("~"), ".tt-metal-cache-regress")
        b = getattr(args, "budget_s", None)
        self.budget_s = (broker.get("budget_s", 1380) if self.mode == "broker" else 0) if b is None else b
        self.data_tree = self.tree.startswith("/data/")
        self.t0 = time.time()

    def cache(self, router):
        return os.path.join(self.cache_root, f"{self.tree_name}-{router}")

    def step_dir(self, router, k, sid):
        return os.path.join(self.out_root, self.label, self.group["id"], router, f"pass{k}", sid)


def router_env(man, router):
    env = man.get("router_env", {}).get(router)
    if env is None:
        raise RegressError(f"router {router!r} has no entry in router_env")
    bad = [k for k, v in env.items() if str(v).upper() == "TBD"]
    if bad:
        raise RegressError(f"router {router!r}: {bad} still TBD in router_env (set the port's switch first)")
    return {k: str(v) for k, v in env.items()}


def build_env(ctx, rs, router, sdir):
    man, tree, home = ctx.man, ctx.tree, ctx.home
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TT_") and not k.startswith("REGRESS_") and k not in DROP_ENV}
    for k in KEEP_TT:
        if k in os.environ:
            env[k] = os.environ[k]
    if ctx.mode == "slurm":
        env = {k: v for k, v in env.items() if not k.startswith(STEP_ENV_STRIP)}
    env.update({k: str(v) for k, v in (man.get("ci_env") or {}).items()})
    if not rs.get("auto_triage", True):
        env.pop("TT_METAL_OPERATION_TIMEOUT_SECONDS", None)
    env.update({k: str(v) for k, v in (man.get("ci_env_by_list") or {}).get(rs.get("list"), {}).items()})
    env.update(router_env(man, router))
    for need in rs["needs"]:
        w = (man.get("weights") or {}).get(need)
        if w:
            env.update({k: str(v) for k, v in (w.get("env") or {}).items()})
    venv = os.path.join(tree, "python_env")
    env.update({
        "TT_METAL_HOME": home,
        "PYTHONPATH": f"{home}:{HERE}",
        "LD_LIBRARY_PATH": os.path.join(tree, "build", "lib"),
        "PATH": f"{venv}/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        "VIRTUAL_ENV": venv,
        "PYTHON_ENV_DIR": venv,
        "TT_METAL_CACHE": ctx.cache(router),
        "GTEST_OUTPUT": f"xml:{sdir}/gtest/",
        "PYTEST_ADDOPTS": "-p regress_junit_plugin",
        "REGRESS_JUNIT_DIR": os.path.join(sdir, "junit"),
        "REGRESS_STEP_DIR": sdir,
        "REGRESS_HARNESS_DIR": HERE,  # helper scripts next to run_group.py (dit_ring_joint_perf.py)
        "PYTHONUNBUFFERED": "1",
    })
    if ctx.data_tree:
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # the /data tree is shared by every job
    if getattr(ctx, "tmpdir", None):
        env["TMPDIR"] = ctx.tmpdir  # mpirun's session dir: the job's own, never a shared full /tmp
    env.update(rs["env"])
    env.update(json.loads(os.environ.get("REGRESS_STEP_ENV_JSON") or "{}"))  # an A/B's override for every step
    pm = active_path_map(man)
    if pm:
        env = {k: map_paths(v, pm) for k, v in env.items()}
    return env


# ------------------------------------------------------------------------------------------------- execution

def tail_text(path, n=4096):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def last_line(path):
    for l in reversed(tail_text(path).splitlines()):
        l = re.sub(r"\x1b\[[0-9;]*m", "", l).strip()
        if l:
            return l[:150]
    return ""


def kill_group(p, grace=15):
    """TERM then KILL the step's process group; done when the leader has exited (polled, so a zombie counts)."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        t = time.time()
        while time.time() - t < wait:
            if p.poll() is not None:
                time.sleep(1)  # let the rest of the group follow the leader
                break
            time.sleep(0.5)
        if p.poll() is not None and sig == signal.SIGTERM:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return


def hang_capture(rs, env, cwd, sdir):
    """REGRESS_HANG_CAPTURE (a hang_capture.py the node can read): after a step's timeout kill, a read-only capture of
    every router on the node into the step's hangcap/ (the routers keep their state until the reset that follows)."""
    cap = os.environ.get("REGRESS_HANG_CAPTURE")
    if not cap or not os.path.isfile(cap):
        return
    py = os.path.join(env.get("VIRTUAL_ENV", ""), "bin", "python")
    out = os.path.join(sdir, "hangcap")
    say(f"   {rs['id']}: hang capture into {out}")
    try:
        r = subprocess.run([py if os.path.exists(py) else sys.executable, cap, out, "--time-budget", "540",
                            "--meta", f"step={rs['id']}"], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
        with open(os.path.join(sdir, "hangcap.log"), "w") as f:
            f.write(r.stdout + r.stderr)
        say(f"   {rs['id']}: hang capture rc={r.returncode}")
    except Exception as e:  # the reset still follows
        say(f"   {rs['id']}: hang capture failed: {e}")


def execute(rs, env, cwd, sdir, timeout_s, stop_on_fail=False, heartbeat_s=60, busy_s=180):
    """Runs the step's command; returns (rc, verdict, start, end). verdict: '' | 'timeout' | 'busy'."""
    script = os.path.join(sdir, "cmd.sh")
    with open(script, "w") as f:
        f.write("#!/bin/bash\n# generated by run_group.py\n")
        if stop_on_fail:
            f.write("set -eo pipefail\n")
        else:  # keep going after a failing line (CI stops there); every failure is logged and fails the step
            # REGRESS_CAPTURE_ON_ERR=1 (with REGRESS_HANG_CAPTURE in the step env): a read-only router capture right
            # after each failing line, before the next line re-initializes the fabric (a fabric-init failure leaves the
            # routers as they stopped until the next init or reset)
            cap = ""
            if env.get("REGRESS_CAPTURE_ON_ERR") == "1" and env.get("REGRESS_HANG_CAPTURE"):
                cap = ("\"${VIRTUAL_ENV:-.}/bin/python\" \"$REGRESS_HANG_CAPTURE\" \"$REGRESS_STEP_DIR/errcap_$__regress_fails\" "
                       "--time-budget 300 --meta step_line=$__regress_fails > \"$REGRESS_STEP_DIR/errcap_$__regress_fails.log\" 2>&1; "
                       "echo \"[regress] error capture $__regress_fails rc=$?\" >&2; ")
            f.write("set -o pipefail\n__regress_fails=0\n"
                    "trap '__rc=$?; echo \"[regress] command failed (rc=$__rc): $BASH_COMMAND\" >&2; "
                    "__regress_fails=$((__regress_fails + 1)); " + cap + "' ERR\n")
        f.write(rs["cmd"])
        if not stop_on_fail:
            f.write("\nexit $(( __regress_fails > 0 ? 1 : 0 ))\n")
    log = os.path.join(sdir, "log.txt")
    start = time.time()
    with open(log, "wb") as lf:
        p = subprocess.Popen(["bash", script], cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    verdict, last_hb = "", start
    while True:
        rc = p.poll()
        if rc is not None:
            break
        now = time.time()
        if now - start > timeout_s:
            age = now - os.path.getmtime(log)
            verdict = "timeout" if age > 120 else "timeout_progressing"
            say(f"   {rs['id']}: past its {timeout_s} s timeout (log last written {int(age)} s ago): killing it")
            kill_group(p)
            hang_capture(rs, env, cwd, sdir)
            break
        if now - start > busy_s and "Waiting for KMD lock" in last_line(log) and now - os.path.getmtime(log) > 60:
            verdict = "busy"
            say(f"   {rs['id']}: waiting on the KMD lock for > {busy_s} s (another process holds the device)")
            kill_group(p)
            break
        if now - last_hb >= heartbeat_s:
            say(f"   {rs['id']} {int(now - start)} s / {timeout_s} s | {last_line(log)}")
            last_hb = now
        time.sleep(5)
    rc = p.wait()
    try:  # leftovers of the step's session (servers, stray children)
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return rc, verdict, start, time.time()


# ------------------------------------------------------------------------------------------------- collectors

def parse_junit_files(files):
    """Test cases of gtest / pytest junit XML: {key: {status, time, file, message}}."""
    out = {}
    rank = {"PASS": 0, "SKIP": 1, "FAIL": 2}
    for path in sorted(files, key=os.path.getmtime):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            out[f"<unparsable>::{os.path.basename(path)}"] = {"status": "FAIL", "message": str(e)[:300]}
            continue
        for tc in root.iter("testcase"):
            cls, name = tc.get("classname", ""), tc.get("name", "")
            status, msg = "PASS", ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    status, msg = "FAIL", (child.get("message") or child.text or "")[:300]
                    break
                if child.tag == "skipped":
                    status, msg = "SKIP", (child.get("message") or "")[:200]
            if tc.get("result") == "skipped" or tc.get("status") == "notrun":
                status = "SKIP"
            key = f"{cls}::{name}" if cls else name
            n, k = 2, key
            while k in out and out[k]["file"] != os.path.basename(path):
                k = f"{key}#{n}"
                n += 1
            prev = out.get(k)
            if prev is None or rank[status] > rank[prev["status"]]:
                out[k] = {"status": status, "time": float(tc.get("time") or 0), "file": os.path.basename(path),
                          "message": msg}
    return out


def parse_pytest_log(text):
    """Case outcomes from a pytest -v log (no junit, e.g. a step killed at its timeout); messages are not available."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^(\S+\.py)::(\S+?)(?: |$)", line)
        if m:
            cur = f"{m.group(1)[:-3].replace('/', '.')}::{m.group(2)}"
        if cur:
            for st, mapped in (("PASSED", "PASS"), ("FAILED", "FAIL"), ("ERROR", "FAIL"), ("SKIPPED", "SKIP")):
                if line.startswith(st) or line.rstrip().endswith(" " + st):
                    out[cur] = {"status": mapped, "time": 0.0, "file": "log.txt",
                                "message": "from the -v log (no junit: the step was killed before the summary)"}
                    break
    return out


def copy_new(src_dir, dst_dir, since, pattern="**/*"):
    n = 0
    for p in glob.glob(os.path.join(src_dir, pattern), recursive=True):
        if os.path.isfile(p) and os.path.getmtime(p) >= since - 1:
            d = os.path.join(dst_dir, os.path.relpath(p, src_dir))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(p, d)
            n += 1
    return n


def collect(rs, ctx, sdir, start, copy=True):
    """Collector outputs into sdir; returns a dict merged into status.json. copy=False: only what is in sdir."""
    res = {}
    log = os.path.join(sdir, "log.txt")
    text = ""
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            text = re.sub(r"\x1b\[[0-9;]*m", "", f.read())
    except OSError:
        pass
    kinds = rs["result"]
    xmls = []
    if "gtest" in kinds:
        xmls += glob.glob(os.path.join(sdir, "gtest", "*.xml"))
    if "pytest" in kinds:
        xmls += glob.glob(os.path.join(sdir, "junit", "*.xml"))
    if xmls or "gtest" in kinds or "pytest" in kinds:
        tests = parse_junit_files(xmls)
        if not xmls and "pytest" in kinds:  # killed before pytest wrote its junit: case outcomes from the -v log
            tests = parse_pytest_log(text)
            res["tests_from_log"] = True
        if rs.get("perf_window"):  # in-test perf windows: the value is data (metrics), not a functional failure
            rx = PERF_WINDOW_RX if rs["perf_window"] is True else rs["perf_window"]
            for t in tests.values():
                if t["status"] == "FAIL" and re.search(rx, t.get("message", "")):
                    t["status"] = "PERF_WINDOW"
                    m = WINDOW_NUM_RX.search(t["message"])
                    if m:
                        t["measured"], t["window"] = float(m.group(1)), [float(m.group(2)), float(m.group(3))]
        for rx, why in (rs.get("known_fail_cases") or {}).items():  # case-level KNOWN_FAIL (TEST_PLAN §4.5)
            for name, t in tests.items():
                if t["status"] == "FAIL" and re.search(rx, name):
                    t["status"], t["known_fail"] = "KNOWN_FAIL", why
        counts = {s: sum(1 for t in tests.values() if t["status"] == s)
                  for s in ("PASS", "FAIL", "SKIP", "PERF_WINDOW", "KNOWN_FAIL")}
        with open(os.path.join(sdir, "tests.json"), "w") as f:
            json.dump({"counts": counts, "files": sorted(os.path.basename(x) for x in xmls), "tests": tests}, f,
                      indent=1, sort_keys=True)
        res["tests"] = counts
        res["junit_files"] = len(xmls)
    gen = os.path.join(ctx.home, "generated")
    if "fabric_csv" in kinds and copy:
        res["fabric_csv"] = copy_new(os.path.join(gen, "fabric"), os.path.join(sdir, "fabric"), start, "*.csv")
    if "benchmark_json" in kinds and copy:
        res["benchmark_files"] = copy_new(os.path.join(gen, "benchmark_data"), os.path.join(sdir, "benchmark_data"),
                                          start)
    if "metrics" in kinds and rs.get("metrics"):
        rows = []
        for m in (mm for rx in as_list(rs["metrics"]) for mm in re.finditer(rx, text)):
            d = {k: v for k, v in m.groupdict().items() if k != "value" and v is not None}
            rows.append((d, float(m.group("value").replace(",", ""))))  # "10,384,185 ns" style values too
        keys = sorted({k for d, _ in rows for k in d})
        with open(os.path.join(sdir, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["value"])
            for d, v in rows:
                w.writerow([d.get(k, "") for k in keys] + [v])
        res["metrics"] = len(rows)
    if "log_marker" in kinds and rs.get("marker"):
        res["marker_found"] = len(re.findall(rs["marker"], text))
    # what test_tt_fabric reports, whatever the collector
    m = re.findall(r"Tests: (\d+) executed, (\d+) failed", text)
    if m:
        res["tt_fabric"] = {"runs": len(m), "executed": sum(int(a) for a, _ in m), "failed": sum(int(b) for _, b in m)}
    res["failed_lines"] = re.findall(r"^\[regress\] command failed \(rc=\d+\): .*$", text, re.M)[:20]
    return res


def verdict_status(rc, verdict, capped, res, rs):
    st = _verdict(rc, verdict, capped, res, rs)
    if st == "FAIL" and rs.get("env_fail"):
        return "ENV"  # fails because of the node's environment (recorded, not scored)
    if rs.get("known_fail") and st in (rs.get("known_fail_statuses") or ["FAIL"]):
        return "KNOWN_FAIL"
    return st


def _verdict(rc, verdict, capped, res, rs):
    if verdict in ("timeout", "timeout_progressing"):
        if capped:
            return "CAPPED"
        return "TIMEOUT" if verdict == "timeout_progressing" else "HANG"  # TIMEOUT: still logging when killed
    if verdict == "busy":
        return "BUSY"
    t = res.get("tests") or {}
    test_line = re.compile(r"pytest|\./build/test/|unit_tests|test_tt_fabric|tt-run")
    if rc != 0 and not t.get("FAIL") and (t.get("PERF_WINDOW") or t.get("KNOWN_FAIL")) and \
            all(test_line.search(l) for l in res.get("failed_lines") or []):
        # every failing case only missed its own perf window, or is a recorded main KNOWN_FAIL case
        return "PERF_WINDOW" if t.get("PERF_WINDOW") else "PASS"
    if rc != 0:
        return "FAIL"
    if (res.get("tests") or {}).get("FAIL"):
        return "FAIL"
    if "log_marker" in rs["result"] and rs.get("marker") and not res.get("marker_found"):
        return "FAIL"
    if (res.get("tt_fabric") or {}).get("failed"):
        return "FAIL"
    return "PASS"


# ------------------------------------------------------------------------------------------------- run

def read_status(sdir):
    try:
        with open(os.path.join(sdir, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def estimate_s(ctx, rs, router):
    """Expected duration: the longest earlier run of this step in the label, else est_min, else half the timeout."""
    best = 0
    for p in glob.glob(os.path.join(ctx.out_root, ctx.label, ctx.group["id"], "*", "pass*", rs["id"], "status.json")):
        try:
            with open(p) as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        if st.get("status") in ("PASS", "FAIL"):
            best = max(best, st.get("duration_s", 0))
    if best:
        return best * 1.2 + 30
    if rs.get("est_min"):
        return rs["est_min"] * 60
    return rs["timeout_min"] * 60 * 0.5


def check_needs(ctx, rs):
    """'' or the reason the step can't run here."""
    man = ctx.man
    for need in rs["needs"]:
        w = (man.get("weights") or {}).get(need)
        if w and not os.path.isdir(map_paths(w["mount"], active_path_map(man))):
            return f"weights: {w['mount']} not mounted on {ctx.node} (nor a path_map target)"
        if need == "mpi" and not (shutil.which("mpirun") or os.path.exists(os.path.join(ctx.tree, "python_env/bin/tt-run"))):
            return "mpi: no mpirun"
        if need == "ring_mapping" and ctx.node in ((ctx.system.get("slurm") or {}).get("exclude_for_ring") or []):
            return f"ring_mapping: {ctx.node} cannot map ring/torus"
        if need == "study_tree":
            return "study_tree: planned"
        if need == "no_bw_telemetry_switch":  # on main the env is a no-op: refuse rather than measure telemetry-on
            b = os.path.join(ctx.tree, "build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric")
            try:
                with open(b, "rb") as f:
                    has = b"TT_FABRIC_NO_BW_TELEMETRY" in f.read()
            except OSError:
                has = False
            if not has:
                return "no_bw_telemetry_switch: this tree's test_tt_fabric ignores TT_FABRIC_NO_BW_TELEMETRY (patches/main_no_bw_telemetry.patch)"
    missing = [p for p in mnt_paths(rs["cmd"]) if not os.path.exists(p)]
    if missing and rs.get("on_missing", "skip") == "skip":
        return "missing paths: " + ", ".join(missing[:6])
    return ""


def write_status(sdir, st):
    with open(os.path.join(sdir, "status.json"), "w") as f:
        json.dump(st, f, indent=1, sort_keys=True)


def prepare_step_dir(sdir):
    if os.path.exists(sdir):  # an earlier attempt (CAPPED / BUSY / ERROR): keep it aside
        n = 1
        while os.path.exists(f"{sdir}.attempt{n}"):
            n += 1
        os.rename(sdir, f"{sdir}.attempt{n}")
    os.makedirs(sdir)


def run_steps(ctx, router, k, steps, resume=True, stop_on_fail=False, reset_cb=None):
    """Runs `steps` (resolved) for one router and pass. Returns (results list, remaining count, stop reason)."""
    out, ran, stop = [], 0, ""
    pm = active_path_map(ctx.man)
    for i, rs in enumerate(steps):
        if pm:
            rs = dict(rs, cmd=map_paths(rs["cmd"], pm), path_map=pm)
        sdir = ctx.step_dir(router, k, rs["id"])
        prev = read_status(sdir)
        if resume and prev and prev.get("status") in DONE:
            say(f"{rs['id']}: done already ({prev['status']}), skipped")
            out.append(prev)
            continue
        if rs.get("status") == "not_run":
            if not (prev and prev.get("status") == "NOT_RUN"):
                prepare_step_dir(sdir)
                write_status(sdir, {"group": ctx.group["id"], "step": rs["id"], "router": router, "pass": k,
                                    "label": ctx.label, "node": ctx.node, "status": "NOT_RUN", "reason": rs["why"],
                                    "ci": rs.get("ci"), "start": iso(), "end": iso(), "duration_s": 0, "rc": None})
            say(f"{rs['id']}: NOT RUN ({rs['why']})")
            continue
        why = check_needs(ctx, rs)
        timeout_s, capped = int(rs["timeout_min"] * 60), False
        cap_min = getattr(ctx.args, "max_timeout_min", None)
        if cap_min and timeout_s > cap_min * 60:  # an A/B's stall timeout (shorter than CI's)
            timeout_s = int(cap_min * 60)
        if not why and ctx.budget_s:
            left = ctx.budget_s - (time.time() - ctx.t0) - 20
            if timeout_s > left:
                est = estimate_s(ctx, rs, router)
                # start only if the estimate fits well inside what is left: a budget kill mid-test can leave the
                # fabric faulted (job 604); the first step of a job runs capped anyway (progress)
                if est > left * 0.6 and (ran or left < 300):
                    stop = f"budget: {int(left)} s left, {rs['id']} needs ~{int(est)} s (timeout {timeout_s} s)"
                    break
                timeout_s, capped = int(left), True
        prepare_step_dir(sdir)
        env = build_env(ctx, rs, router, sdir)
        st = {"group": ctx.group["id"], "step": rs["id"], "router": router, "pass": k, "label": ctx.label,
              "node": ctx.node, "mode": ctx.mode, "tree": ctx.tree, "tree_name": ctx.tree_name, "tree_sha": ctx.tree_sha,
              "tree_patches": ctx.tree_patches,
              "ci": rs.get("ci"), "timeout_s": timeout_s, "ci_timeout_s": int(rs["timeout_min"] * 60),
              "capped": capped, "slurm_job": os.environ.get("SLURM_JOB_ID"), "path_map": rs.get("path_map"),
              "missing_paths": [p for p in mnt_paths(rs["cmd"]) if not os.path.exists(p)] or None,
              "telemetry": ctx.group.get("telemetry", "CI default"), "rerun_of": rs.get("rerun_of"),
              "jit_cache": ctx.cache(router), "timeout_note": rs.get("timeout_note"), "deviation": rs.get("deviation"),
              "note": rs.get("note"), "step_env": rs.get("env") or None,
              "data_inodes_at_submit": os.environ.get("REGRESS_DATA_INODES_AT_SUBMIT"),
              "data_inodes_free": data_inodes_free() if ctx.mode == "slurm" else None}
        with open(os.path.join(sdir, "cmd.txt"), "w") as f:
            f.write(rs["cmd"])
        with open(os.path.join(sdir, "env.txt"), "w") as f:
            f.write("".join(f"{a}={b}\n" for a, b in sorted(env.items())
                            if a.startswith(("TT_", "ARCH", "PYTHON", "LD_", "GTEST", "PYTEST", "HF_", "REGRESS_",
                                             "LOGURU", "TTNN", "TRACY", "MESH", "VIRTUAL", "TMPDIR", "OMPI_"))))
        if why:
            st.update(status="SKIP", reason=why, start=iso(), end=iso(), duration_s=0, rc=None)
            write_status(sdir, st)
            say(f"{rs['id']}: SKIP ({why})")
            out.append(st)
            continue
        os.makedirs(env["TT_METAL_CACHE"], exist_ok=True)
        say(f"{rs['id']}: start (router {router}, pass {k}, timeout {timeout_s} s{', capped by the budget' if capped else ''})")
        rc, verdict, t1, t2 = execute(rs, env, ctx.home, sdir, timeout_s, stop_on_fail)
        ran += 1
        res = collect(rs, ctx, sdir, t1)
        status = verdict_status(rc, verdict, capped, res, rs)
        st.update(res, status=status, rc=rc, start=iso(t1), end=iso(t2), duration_s=round(t2 - t1, 1))
        write_status(sdir, st)
        extra = []
        if res.get("tests"):
            extra.append("tests " + " ".join(f"{a}={b}" for a, b in res["tests"].items() if b))
        if res.get("tt_fabric"):
            extra.append(f"test_tt_fabric {res['tt_fabric']['executed']} executed, {res['tt_fabric']['failed']} failed")
        if res.get("metrics") is not None:
            extra.append(f"{res['metrics']} metric rows")
        if res.get("failed_lines"):
            extra.append(f"{len(res['failed_lines'])} failed lines")
        say(f"{rs['id']}: {status} rc={rc} in {int(t2 - t1)} s; {'; '.join(extra)}")
        out.append(st)
        if status in STOPPERS:
            if reset_cb and status != "BUSY" and reset_cb(f"after {status} in {rs['id']}"):
                continue
            stop = f"{status} in {rs['id']}"
            break
        if reset_cb and (status == "FAIL" or (ctx.group.get("reset_between_steps") and i < len(steps) - 1)):
            if not reset_cb(f"after {status} in {rs['id']}"):  # A/B groups reset between arms whatever the result
                stop = f"reset failed after {rs['id']}"
                break
    remaining = sum(1 for rs in steps
                    if (read_status(ctx.step_dir(router, k, rs["id"])) or {}).get("status") not in DONE)
    return out, remaining, stop


def nodeid_of(key, tree):
    """pytest node id from a junit key 'dotted.module[.Class]::name' (the module part is the longest existing file)."""
    cls, _, name = key.partition("::")
    name = re.sub(r"#\d+$", "", name)
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):
        rel = os.path.join(*parts[:i]) + ".py"
        if os.path.exists(os.path.join(tree, rel)):
            return "::".join([rel] + parts[i:] + [name])
    return None


def rerun_step(ctx, rs, router, k):
    """A derived step that reruns the FAIL cases of `rs` (pass k) one pytest process per case, with the step's exports."""
    sdir = ctx.step_dir(router, k, rs["id"])
    try:
        with open(os.path.join(sdir, "tests.json")) as f:
            tests = json.load(f)["tests"]
    except (OSError, ValueError, KeyError):
        raise RegressError(f"{rs['id']}: no tests.json in {sdir}")
    fails = sorted(n for n, t in tests.items() if t["status"] == "FAIL")
    if not fails:
        raise RegressError(f"{rs['id']}: no failing cases in pass {k}")
    exports = [l for l in rs["cmd"].splitlines() if re.match(r"\s*export\s", l)]
    timeout = "--timeout=600" if "--timeout=600" in rs["cmd"] else ""
    lines = ["# rerun of each FAIL case of %s pass %d on its own (one process per case)" % (rs["id"], k)] + exports
    for n in fails:
        nid = nodeid_of(n, ctx.tree)
        if not nid:
            raise RegressError(f"{rs['id']}: cannot map {n} to a pytest node id")
        lines.append(f"pytest {timeout} {shlex.quote(nid)}".replace("  ", " "))
    return dict(rs, id=rs["id"] + ".rerun", cmd="\n".join(lines) + "\n", result=["pytest"],
                timeout_min=min(rs["timeout_min"], 23), rerun_of={"step": rs["id"], "pass": k, "cases": len(fails)},
                est_min=min(rs["timeout_min"], 2 + 1.5 * len(fails)))


def group_steps(man, g, tree, only=None):
    """Ready steps resolved for `tree`, plus not_run steps (recorded as NOT_RUN, never executed)."""
    steps = []
    for s in g.get("steps", []):
        st = s.get("status", "ready")
        if st == "ready":
            steps.append(resolve_step(man, s, tree))
        elif st == "not_run":
            steps.append({"id": s["id"], "status": "not_run", "why": s.get("why", ""), "timeout_min": 0, "needs": [],
                          "result": [], "cmd": "", "env": {}, "ci": s.get("from_ci") or s.get("ci_ref")})
    if only:
        want = only.split(",")
        unknown = set(want) - {s["id"] for s in steps}
        if unknown:
            raise RegressError(f"unknown steps {sorted(unknown)} in {g['id']}")
        steps = [s for s in steps if s["id"] in want]
    return steps


def write_summary(ctx):
    gdir = os.path.join(ctx.out_root, ctx.label, ctx.group["id"])
    rows = []
    for p in sorted(glob.glob(os.path.join(gdir, "*", "pass*", "*", "status.json"))):
        if ".attempt" in p.split(os.sep)[-2]:
            continue
        try:
            with open(p) as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        rows.append({k: st.get(k) for k in ("router", "pass", "step", "status", "duration_s", "node", "tree_sha",
                                            "tests", "tt_fabric", "metrics", "reason", "start")})
    with open(os.path.join(gdir, "summary.json"), "w") as f:
        json.dump({"group": ctx.group["id"], "label": ctx.label, "updated": iso(), "steps": rows}, f, indent=1)


def preflight(ctx, routers):
    g = ctx.group
    if g.get("status", "ready") != "ready" and not ctx.args.force:
        raise RegressError(f"group {g['id']} is {g.get('status')}: {g.get('why', '')} (--force overrides)")
    for r in routers:
        router_env(ctx.man, r)
        allowed = (ctx.man.get("router_trees") or {}).get(r)
        if allowed and not any(ctx.tree_name == a or (a.endswith("_") and ctx.tree_name.startswith(a)) for a in allowed):
            raise RegressError(f"router {r!r} runs on trees {allowed}, not {ctx.tree_name!r} (router_trees)")
    for p in ("build/lib/libtt_metal.so", "python_env/bin/python"):
        if not os.path.exists(os.path.join(ctx.tree, p)):
            raise RegressError(f"tree {ctx.tree}: no {p}")
    if ctx.tree_sha_expected and ctx.tree_sha and ctx.tree_sha != ctx.tree_sha_expected:
        raise RegressError(f"tree {ctx.tree} is at {ctx.tree_sha}, the manifest says {ctx.tree_sha_expected}")
    sysname = g["system"]
    sl = ctx.system.get("slurm") or {}
    if ctx.mode == "slurm":
        ok = (sl.get("node_regex") and re.match(galaxy_node_regex(sl), ctx.node)) or ctx.node in (sl.get("nodes") or {}) or \
            (sl.get("overflow_node_regex") and re.match(sl["overflow_node_regex"], ctx.node))
        if not ok and not ctx.args.any_node:
            raise RegressError(f"node {ctx.node} is not allowed for {sysname}")
    if ctx.mode == "local" and not ctx.args.any_node:
        raise RegressError(f"{ctx.node} is neither {sysname}'s broker host nor a Slurm job (--any-node overrides)")


def cmd_run(args):
    man = load_manifest(args.manifest)
    g = get_group(man, args.group)
    ctx = Ctx(man, g, args)
    preflight(ctx, [args.router])
    steps = group_steps(man, g, ctx.tree, args.steps)
    if args.rerun_failed:
        want = args.rerun_failed.split(",")
        steps = [rerun_step(ctx, rs, args.router, args.pass_) for rs in group_steps(man, g, ctx.tree, ",".join(want))]
    say(f"run {g['id']} on {ctx.node} ({ctx.mode}), tree {ctx.tree_name} {ctx.tree} @ {(ctx.tree_sha or '?')[:12]}, "
        f"router {args.router}, pass {args.pass_}, label {ctx.label}, {len(steps)} steps"
        f"{f', budget {ctx.budget_s} s' if ctx.budget_s else ''}")
    dropped = sorted(k for k in os.environ if k.startswith("TT_") and k not in KEEP_TT)
    if dropped:
        say(f"inherited TT_* variables dropped: {' '.join(dropped)}")
    if args.dry_run:
        pm = active_path_map(man)
        steps = [dict(rs, cmd=map_paths(rs["cmd"], pm)) for rs in steps]
        if pm:
            say(f"path_map active here: {pm}")
        for rs in steps:
            print(f"--- {rs['id']} (timeout {rs['timeout_min']} min, needs {rs['needs'] or '-'})"
                  f"{' SKIP: ' + check_needs(ctx, rs) if check_needs(ctx, rs) else ''}")
            print(rs["cmd"].rstrip())
        return 0
    res, remaining, stop = run_steps(ctx, args.router, args.pass_, steps, resume=args.resume,
                                     stop_on_fail=args.stop_on_fail)
    write_summary(ctx)
    counts = {}
    for st in res:
        counts[st["status"]] = counts.get(st["status"], 0) + 1
    say(f"{g['id']}: {' '.join(f'{a}={b}' for a, b in sorted(counts.items()))}; "
        f"{remaining} step(s) left{' (' + stop + ')' if stop else ''}")
    if remaining:
        say("REGRESS: not finished; run the same command again (--resume skips finished steps)")
    else:
        say("REGRESS: ALL STEPS DONE")
    if any(st["status"] in ("HANG", "TIMEOUT", "CAPPED", "BUSY") for st in res):
        say("note: a step was killed; the device may need tt_device_reset before the next job")
        return 3
    return 1 if counts.get("FAIL") else 0


# ------------------------------------------------------------------------------------------------- Slurm job side

def other_users_busy():
    me = os.environ.get("USER", "")
    try:
        ps = subprocess.run(["ps", "-eo", "user:32,uid,comm,args", "--no-headers"], capture_output=True, text=True,
                            timeout=30).stdout
    except Exception:
        return []
    busy = []
    for l in ps.splitlines():
        f = l.split(None, 3)
        if len(f) < 4 or f[0] in (me, "root"):
            continue
        try:
            uid = int(f[1])
        except ValueError:
            continue
        if 1000 <= uid < 60000 and re.match(r"(python|pytest|test_|unit_tests|tt-smi|tt_|ttnn|prte|prted|mpirun|orted|run_)",
                                           f[2]) and not re.search(r"tt-fabric-manager|tt-metrics-exporter|tt_telemetry", l):
            busy.append(l[:200])
    return busy[:8]


def make_farm(tree, farm):
    """TT_METAL_HOME for one job: symlinks to the tree's top-level entries, its own generated/ and .pytest_cache."""
    os.makedirs(farm, exist_ok=True)
    for name in os.listdir(tree):
        if name in ("generated", ".pytest_cache", "__pycache__"):
            continue
        dst = os.path.join(farm, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(tree, name), dst)
    os.makedirs(os.path.join(farm, "generated"), exist_ok=True)


ENV_REPORT = r"""
echo "host $(hostname) kernel $(uname -r) cpus $(nproc) mem $(free -g | awk '/Mem:/ {print $2}') GB"
df -h /tmp /data /home 2>/dev/null; df -i /data 2>/dev/null | tail -1
for d in /mnt/MLPerf /mnt/MLPerf/huggingface /mnt/MLPerf/tt_dnn-models /mnt/models /mnt/models/huggingface /mnt/weka /mnt/weka/model-weights; do
  if [ -e $d ]; then echo "mount $d: present ($(ls $d 2>/dev/null | head -8 | tr '\n' ' '))"; else echo "mount $d: MISSING"; fi
done
for t in tt-smi mpirun mpirun-ulfm python3 git; do echo "tool $t: $(command -v $t || echo MISSING)"; done
tt-smi --version 2>&1 | head -1
ls /dev/tenstorrent 2>/dev/null | wc -l | sed 's/^/devices: /'
"""


def galaxy_node_regex(sl):
    """The system's node_regex; REGRESS_GLX_SC36_2=1 (one-time per-invocation exception, coordinator 2026-11-04) also
    accepts the bh_sc36_2 single Galaxies (bh-glx-120-c*). The manifest default stays 110-a only."""
    rx = sl.get("node_regex") or "."
    if os.environ.get("REGRESS_GLX_SC36_2") == "1" and rx == "^bh-glx-110-a":
        return "^bh-glx-(110-a|120-c)"
    return rx


def data_inodes_free(path="/data"):
    try:
        return os.statvfs(path).f_favail
    except OSError:
        return -1


def env_report(ctx, on_broker_host=False):
    script = ENV_REPORT
    if on_broker_host:  # no tt-smi and no device listing on a broker host, even version-only (coordinator 2026-11-04)
        script = "\n".join(l for l in ENV_REPORT.splitlines() if "tt-smi" not in l and "/dev/tenstorrent" not in l)
        script = script.replace("for t in tt-smi mpirun", "for t in mpirun")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=300)
    path = os.path.join(ctx.job_dir, "env_report.txt")
    with open(path, "w") as f:
        f.write(r.stdout + r.stderr)
    for l in r.stdout.splitlines():
        print("   ", l)
    return path


def device_reset(ctx, why, env):
    sl = ctx.system.get("slurm") or {}
    cmd = sl.get("reset", "tt-smi -r")
    chips = ctx.system.get("chips", 1)
    health = os.path.join(ctx.tree, "build/test/tt_metal/tt_fabric/test_system_health")
    for attempt in range(1, 4):
        t = time.time()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
        ls = subprocess.run("tt-smi -ls", shell=True, capture_output=True, text=True, timeout=300).stdout
        n = ls.count("Blackhole")
        say(f"reset ({why}, attempt {attempt}): `{cmd}` rc={r.returncode} in {int(time.time() - t)} s, "
            f"{n} Blackhole lines in tt-smi -ls (want >= {chips})")
        if r.returncode != 0 or n < chips:
            time.sleep(10)
            continue
        mode = sl.get("health", "required")  # required (CI's ensure-bh-links-online on boxes) | advisory | off
        if mode == "off":
            return True
        h = subprocess.run([health], cwd=ctx.home, env=env, capture_output=True, text=True, timeout=600)
        say(f"   test_system_health rc={h.returncode} ({mode})")
        if h.returncode == 0:
            return True
        with open(os.path.join(ctx.job_dir, f"health_fail_{int(time.time())}.log"), "w") as f:
            f.write(h.stdout + h.stderr)
        if mode == "advisory":
            return True
        time.sleep(5)
    return False


def node_check(ctx, nc, env):
    """First job on a node: run the system's check row once; record ok / noring / fail under results/_nodes/."""
    d = os.path.join(ctx.man["results"]["data"], "_nodes")
    os.makedirs(d, exist_ok=True)
    for st in ("ok", "noring", "fail"):
        if os.path.exists(os.path.join(d, f"{ctx.node}.{st}")):
            say(f"node check: {ctx.node} is '{st}' from an earlier job")
            if st == "ok":
                return 0
            say(f"!!! {ctx.node} failed its node check before: stopping (add it to the excludes)")
            return 76
    log = os.path.join(ctx.job_dir, "node_check.log")
    say(f"node check on {ctx.node} (first job here): {nc['cmd'][:120]}")
    with open(log, "wb") as f:
        try:
            r = subprocess.run(["bash", "-c", nc["cmd"]], cwd=ctx.home, env=env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=int(nc.get("timeout_min", 15)) * 60)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    text = open(log, errors="replace").read()
    if rc == 0 and re.search(nc["pass_rx"], text):
        st = "ok"
    elif nc.get("noring_rx") and re.search(nc["noring_rx"], text):
        st = "noring"
    else:
        st = "fail"
    with open(os.path.join(d, f"{ctx.node}.{st}"), "w") as f:
        f.write(f"{iso()} job {os.environ.get('SLURM_JOB_ID')} rc={rc}\n")
    shutil.copy2(log, os.path.join(d, f"{ctx.node}.{st}.log"))
    say(f"node check on {ctx.node}: {st} (rc={rc}); log {log}")
    if st != "ok":
        say(f"!!! node check {st}: stopping; exclude {ctx.node} from Galaxy submits")
        return 76
    return 0


def cmd_job(args):
    """Inside a Slurm allocation (regress.sbatch)."""
    man = load_manifest(args.manifest)
    g = get_group(man, args.group)
    ctx = Ctx(man, g, args, mode="slurm")
    routers = args.routers.split(",") if args.routers else g.get("routers", ["main"])
    preflight(ctx, routers)
    job = os.environ.get("SLURM_JOB_ID", "manual")
    brokers = {(sd.get("broker") or {}).get("node") for sd in man["systems"].values()} - {None}
    if ctx.node in brokers and not args.no_reset:  # never touch a broker host's device outside its broker (2026-11-04)
        raise RegressError(f"{ctx.node} is a tt-device broker host: `job` may only run there with --no-reset "
                           "(a no-device test of the harness); device work goes through the broker")
    say(f"job {job}: {g['id']} on {ctx.node}, tree {ctx.tree} @ {(ctx.tree_sha or '?')[:12]}, routers {routers}, "
        f"label {ctx.label}")
    busy = other_users_busy()
    mcp = "/usr/local/bin/tt-device-mcp"  # LoudBoxes run their own tt-device broker (lb2.sbatch's check)
    if os.path.exists(mcp):
        try:
            st = subprocess.run([mcp, "status", "5"], capture_output=True, text=True, timeout=30).stdout
            run = st.split("RUNNING", 1)[1].split("\n\n", 1)[0] if "RUNNING" in st else ""
            if re.search(r"^\s+\d{3,}\s", run, re.M):
                busy.append("the node's tt-device broker is running a job")
        except Exception as e:
            say(f"tt-device-mcp status failed: {e}")
    if busy and not args.ignore_busy:
        say("other users' processes that may hold the device; stopping:")
        for l in busy:
            print("   ", l)
        return 75
    # per-job TT_METAL_HOME farm and JIT cache on /data (coordinator rule 2026-11-03: nothing on the nodes outside
    # /data/aagarwal), removed at the end unless --keep-scratch
    scratch = args.scratch or os.path.join(man["results"]["data"], "_scratch", job)
    ctx.job_dir = os.path.join(man["results"]["data"], "_jobs", job)
    os.makedirs(ctx.job_dir, exist_ok=True)
    env_report(ctx, on_broker_host=ctx.node in brokers)
    # Open MPI / PRRTE put their session dir (and PMIx sockets: short paths needed) under TMPDIR; a full /tmp kills
    # mpirun at init (bh-glx-110-a01u14, job 171630). Use the job's own short dir there, and stop early if /tmp is full.
    tmp_free = shutil.disk_usage("/tmp").free / 1e9
    if tmp_free < 2:
        say(f"!!! /tmp on {ctx.node} has {tmp_free:.1f} GB free: stopping (resubmit with --exclude={ctx.node})")
        return 75
    ctx.tmpdir = f"/tmp/{os.environ.get('USER', 'regress')}-regress/{job}/tmp"
    os.makedirs(ctx.tmpdir, exist_ok=True)
    ctx.home = os.path.join(scratch, "home")
    make_farm(ctx.tree, ctx.home)
    ctx.cache_root = os.path.join(scratch, "cache")
    local_cache = None
    if args.local_cache:  # JIT cache on the node's own /tmp (coordinator OK 2026-11-03: the job's /tmp, removed at the end)
        local_cache = f"/tmp/{os.environ.get('USER', 'regress')}-regress/{job}"
        os.makedirs(local_cache, exist_ok=True)
        ctx.cache_root = os.path.join(local_cache, "cache")
        say(f"JIT cache on node-local {ctx.cache_root} ({shutil.disk_usage('/tmp').free / 1e9:.0f} GB free on /tmp)")
    steps = group_steps(man, g, ctx.tree, args.steps)
    passes = g.get("passes") if g.get("kind") == "perf" else None
    if passes:  # A/B orders of the manifest, restricted to the routers of this job, repeated to --passes
        orders = [o for o in ([r for r in p if r in routers] for p in passes) if o]
        order = [orders[i % len(orders)] for i in range(args.passes or len(passes))]
    else:
        order = [routers] * (args.passes or 1)
    say(f"order: {order}; {len(steps)} steps; scratch {scratch}")
    base_env = build_env(ctx, {"needs": [], "env": {}, "list": None}, routers[0], ctx.job_dir)
    reset = (lambda why: device_reset(ctx, why, base_env)) if not args.no_reset else None
    if reset and not reset("job start"):
        say("!!! reset / health check failed at the start: stopping")
        return 2
    nc = (ctx.system.get("slurm") or {}).get("node_check")
    if nc and not args.no_reset and ctx.node not in brokers:
        r = node_check(ctx, nc, base_env)
        if r:
            return r
    rc = 0
    for k, prs in enumerate(order, start=1):
        for r in prs:
            say(f"=== pass {k}, router {r}")
            res, remaining, stop = run_steps(ctx, r, k, steps, resume=True, stop_on_fail=args.stop_on_fail,
                                             reset_cb=reset)
            write_summary(ctx)
            if any(s["status"] == "FAIL" for s in res):
                rc = max(rc, 1)
            if stop:
                say(f"!!! stopped: {stop}")
                rc = 3
                if "BUSY" in stop or "reset failed" in stop:
                    break
    if not args.keep_scratch and "/_scratch/" in scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    if not args.keep_scratch:
        shutil.rmtree(f"/tmp/{os.environ.get('USER', 'regress')}-regress/{job}", ignore_errors=True)
    say(f"job done rc={rc}")
    return rc


# ------------------------------------------------------------------------------------------------- stage

# build intermediates only: runtime/hw/lib/*/*.o and the sfpi toolchain's crt*.o are linked by the kernel JIT (a global
# "*.o" exclude broke every kernel build on Galaxy, jobs 171351 / 171352)
STAGE_EXCLUDES = ["/.git", "/.cpmcache", "/generated", "/.pytest_cache", "/.regress_build.log", "/build_Release/CMakeFiles",
                  "/build_Release/_deps/*-build", "/build_Release/_deps/*-subbuild", "/build_Release/**/*.o",
                  "/build_Release/**/*.o.d", "/build_Release/Testing",
                  "/build_Release/.ninja_*", "/build_Release/build.ninja", "/build_Release/compile_commands.json",
                  "/build_Release/**/CMakeFiles", "/build_Release/**/*.a"]


def rewrite_text_files(root, old, new):
    """Replace `old` by `new` in text files under root (no binaries); returns the files changed."""
    changed = []
    out = subprocess.run(["grep", "-rlIF", "--exclude=*.pyc", old, root], capture_output=True, text=True).stdout
    for p in out.splitlines():
        if os.path.islink(p):
            continue
        with open(p, encoding="utf-8", errors="surrogateescape") as f:
            s = f.read()
        st = os.stat(p)
        with open(p, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(s.replace(old, new))
        os.utime(p, (st.st_atime, st.st_mtime))
        changed.append(p)
    return changed


def honours_no_bw_telemetry(tree):
    """Whether the tree's test_tt_fabric reads TT_FABRIC_NO_BW_TELEMETRY (every job here sets it to 1). From
    ce2dee3cd6e on, the port leaves it out; the switch is the local patch port_tune/local_patches/
    test_tt_fabric_no_bw_telemetry.patch, without which a benchmark row silently runs telemetry-on."""
    f = os.path.join(tree, "tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp")
    try:
        return "TT_FABRIC_NO_BW_TELEMETRY" in open(f, errors="replace").read()
    except OSError:
        return False


def cmd_stage(args):
    man = load_manifest(args.manifest)
    t = man["trees"][args.tree]
    src, dst = t["path"].rstrip("/"), t["data_path"].rstrip("/")
    sha = tree_sha(src)
    if t.get("sha") and sha != t["sha"]:
        raise RegressError(f"{src} is at {sha}, the manifest says {t['sha']}")
    venv_py = os.path.realpath(os.path.join(src, "python_env/bin/python"))
    uv_src = os.path.dirname(os.path.dirname(venv_py))  # .../cpython-3.10.19-linux-x86_64-gnu
    uv_dst = os.path.join(os.path.dirname(dst), "..", "uv", os.path.basename(uv_src))
    uv_dst = os.path.normpath(uv_dst)
    say(f"stage {args.tree}: {src} @ {sha} -> {dst}; venv python {venv_py} -> {uv_dst}")
    if not honours_no_bw_telemetry(src):
        say(f"WARNING: {src}'s test_tt_fabric does not read TT_FABRIC_NO_BW_TELEMETRY (the switch left the PR at "
            "ce2dee3cd6e): its benchmark rows run telemetry-on. Apply fabric_exp/port_tune/local_patches/"
            "test_tt_fabric_no_bw_telemetry.patch before staging it for perf or telemetry-off runs.")
    ex = sum((["--exclude", e] for e in STAGE_EXCLUDES), [])
    if args.dry_run:
        r = subprocess.run(["rsync", "-a", "-n", "--stats", "--delete"] + ex + [src + "/", dst + "/"],
                           capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if re.search(r"Number of|Total file size", l)))
        return 0
    if not args.verify_only:
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        r = subprocess.run(["rsync", "-a", "--delete", "--stats"] + ex + [src + "/", dst + "/"], capture_output=True, text=True)
        say(f"rsync tree rc={r.returncode} in {int(time.time() - t0)} s: " +
            "; ".join(l.strip() for l in r.stdout.splitlines() if re.search(r"files transferred|Total transferred", l)))
        if r.returncode:
            print(r.stderr[-3000:])
            return 1
        os.makedirs(os.path.dirname(uv_dst), exist_ok=True)
        r = subprocess.run(["rsync", "-a", "--delete", uv_src + "/", uv_dst + "/"], capture_output=True, text=True)
        say(f"rsync uv python rc={r.returncode}")
        # the venv: python symlink, pyvenv.cfg, scripts' shebangs, editable-install paths
        venv = os.path.join(dst, "python_env")
        changed = rewrite_text_files(venv, src, dst)
        changed += rewrite_text_files(venv, uv_src, uv_dst)
        n = 0
        for root, dirs, files in os.walk(dst):
            for name in dirs + files:
                p = os.path.join(root, name)
                if os.path.islink(p):
                    tgt = os.readlink(p)
                    for a, b in ((src, dst), (uv_src, uv_dst)):
                        if tgt.startswith(a + "/") or tgt == a:
                            os.remove(p)
                            os.symlink(b + tgt[len(a):], p)
                            n += 1
                            break
        say(f"venv rewrite: {len(changed)} text files, {n} symlinks re-pointed")
        with open(os.path.join(dst, ".regress_tree.json"), "w") as f:
            json.dump({"sha": sha, "source": src, "staged": iso(), "uv_python": uv_dst,
                       "patches": tree_patches(src)}, f, indent=1)
    return verify_stage(dst, src)


def verify_stage(dst, src):
    """From a09u32: the /data tree must not reach back into the source tree or /home."""
    bad = 0
    env = {"PATH": f"{dst}/python_env/bin:/usr/bin:/bin", "LD_LIBRARY_PATH": f"{dst}/build/lib", "TT_METAL_HOME": dst,
           "PYTHONPATH": dst, "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1",
           "ARCH_NAME": "blackhole"}
    for b in ("build/test/tt_metal/tt_fabric/fabric_unit_tests", "build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric",
              "build/test/tt_metal/unit_tests_eth", "build/test/ttnn/unit_tests_ttnn_udm", "build/tools/scaleout/run_cluster_validation",
              "build/test/tt_metal/tt_fabric/test_system_health"):
        p = os.path.join(dst, b)
        if not os.path.exists(p):
            say(f"verify: missing {b}")
            bad += 1
            continue
        out = subprocess.run(["ldd", p], env=env, capture_output=True, text=True).stdout
        home = [l.strip() for l in out.splitlines() if "/home/" in l or "not found" in l]
        if home:
            bad += 1
            say(f"verify: {b} resolves outside {dst}: {home[:4]}")
    code = ("import sys, ttnn; mods = [m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)];"
            "bad = [m for m in mods if m.startswith('/home/')]; print('ttnn', ttnn.__file__); print('home modules', len(bad), bad[:5])")
    r = subprocess.run([f"{dst}/python_env/bin/python", "-c", code], env=env, capture_output=True, text=True,
                       timeout=600, cwd=dst)
    say(f"verify: import ttnn rc={r.returncode}: {r.stdout.strip()[-400:]} {r.stderr.strip()[-600:] if r.returncode else ''}")
    if r.returncode or "home modules 0" not in r.stdout:
        bad += 1
    maps = subprocess.run(["bash", "-c", f"{dst}/python_env/bin/python -c 'import ttnn; print(open(\"/proc/self/maps\").read())' "
                           "| grep -o '/[^ ]*\\.so[^ ]*' | sort -u | grep '^/home/' | head"], env=env, capture_output=True,
                          text=True, cwd=dst, timeout=600).stdout.strip()
    if maps:
        bad += 1
        say(f"verify: shared objects loaded from /home: {maps}")
    for tool in ("pytest", "tt-run"):
        p = os.path.join(dst, "python_env/bin", tool)
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                first = f.readline().strip()
            if "/home/" in first:
                bad += 1
                say(f"verify: {tool} shebang {first}")
    say(f"verify: {'OK' if not bad else f'{bad} problem(s)'}")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------- submit / collect

def slurm_count_galaxy():
    """Study Galaxy jobs that count toward the cap: running AND pending, except multi-Galaxy pod jobs (the quad,
    e.g. 171264 on the bh_sc36_6_110a_pod* partitions), which may stay queued (user rule via coordinator 2026-11-04).
    A Galaxy pair group (glxq_run.sh --pair: sibling jobs glxp-<group>-<pair>) counts as ONE, whatever its siblings'
    states (user rule via coordinator 2026-11-05)."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", "aagarwal"), "-h", "-o", "%i|%P|%j|%T|%D"],
                         capture_output=True, text=True).stdout
    jobs, groups = [], set()
    for l in out.splitlines():
        i, p, j, s, n = (l.split("|") + ["", "", "", "", ""])[:5]
        if not re.match(r"bh_sc36|bh_110a", p) or j == "interactive":
            continue
        if j.startswith("glxp-"):
            g = j.split("-")[1]
            if g not in groups:
                groups.add(g)
                jobs.append((i, p, f"glxp-{g} (pair group)", s))
            continue
        if re.search(r"_pod\d", p) or (n.isdigit() and int(n) > 1):
            continue  # multi-Galaxy quad: excepted
        jobs.append((i, p, j, s))
    return jobs


def fmt_time(minutes):
    m = int(minutes)
    return f"{m // 60:02d}:{m % 60:02d}:00"


def group_minutes(man, g, tree, routers, passes, only=None):
    steps = group_steps(man, g, tree, only)
    per = sum(s["timeout_min"] for s in steps)
    if g.get("kind") == "perf" and g.get("passes"):
        n = passes or len(g["passes"])
        runs = n * len(routers)
    else:
        runs = len(routers) * (passes or 1)
    return steps, per, per * runs


def cmd_submit(args):
    man = load_manifest(args.manifest)
    g = get_group(man, args.group)
    if g.get("status", "ready") != "ready" and not args.force:
        raise RegressError(f"group {g['id']} is {g.get('status')}: {g.get('why', '')}")
    system = man["systems"][g["system"]]
    sl = system.get("slurm")
    if not sl or system.get("status") in ("not_approved", "multi_galaxy"):
        raise RegressError(f"system {g['system']} has no approved Slurm setup")
    t = man["trees"][args.tree]
    meta = os.path.join(t["data_path"], ".regress_tree.json")
    if not os.path.exists(meta):
        raise RegressError(f"{t['data_path']} is not staged (run_group.py stage --tree {args.tree})")
    with open(meta) as f:
        staged = json.load(f)
    if t.get("sha") and staged.get("sha") != t["sha"]:
        raise RegressError(f"staged tree is {staged.get('sha')}, the manifest says {t['sha']}")
    routers = args.routers.split(",") if args.routers else g.get("routers", ["main"])
    for r in routers:
        router_env(man, r)
    steps, per, total = group_minutes(man, g, t["data_path"], routers, args.passes, args.steps)
    minutes = args.time_min or (total + 20)
    if minutes > 240 and not args.force:
        raise RegressError(f"{g['id']}: {minutes} min > 4 h; split it (--steps / --routers)")
    if sl.get("max_time_min") and minutes > sl["max_time_min"]:
        raise RegressError(f"{g['id']}: {minutes} min > the partition limit {sl['max_time_min']} min")
    sb = ["sbatch", "--nodes=1", "--exclusive", f"--time={fmt_time(minutes)}",
          f"--job-name=regress-{args.label}-{g['id']}", "--output=/data/aagarwal/regress/logs/%x-%j.log",
          "--chdir=/data/aagarwal/regress"]
    if sl.get("nodes") and args.overflow:
        sb += [f"--partition={sl['overflow_partition']}"]
    elif sl.get("nodes") and args.node == "any":  # lb2_run.sh -n any: approved nodes' partitions, all others excluded
        parts = sorted(set(sl["nodes"].values()))
        alln = subprocess.run(["sinfo", "-h", "-p", ",".join(parts), "-o", "%N"], capture_output=True, text=True).stdout
        hosts = set()
        for expr in alln.split():
            hosts |= set(subprocess.run(["scontrol", "show", "hostnames", expr], capture_output=True,
                                        text=True).stdout.split())
        excl = sorted(h for h in hosts if h not in sl["nodes"])
        if any("a09u32" in h for h in sl["nodes"]):
            raise RegressError("a09u32 must never be an extra-LB target")
        sb += [f"--partition={','.join(parts)}"] + ([f"--exclude={','.join(excl)}"] if excl else [])
        say(f"any of {sorted(sl['nodes'])}; excluded {len(excl)} other node(s) in {parts}: {excl}")
    elif sl.get("nodes"):
        if not args.node:
            raise RegressError(f"{g['system']}: pick a node with --node ({', '.join(sl['nodes'])})")
        node = args.node if args.node in sl["nodes"] else next((n for n in sl["nodes"] if n.endswith(args.node)), None)
        if not node:
            raise RegressError(f"node {args.node} not in {list(sl['nodes'])}")
        sb += [f"--partition={sl['nodes'][node]}", f"--nodelist={node}"]
    else:
        part = sl["partition"]
        if os.environ.get("REGRESS_GLX_SC36_2") == "1" and galaxy_node_regex(sl) != sl.get("node_regex"):
            part = "bh_sc36_2," + part  # the 120-c single Galaxies (same exception as galaxy_node_regex); listed first:
            # with it last, Slurm's estimate placed the job on a busy 110-a node an hour out while 120-c nodes were idle
        sb += [f"--partition={part}"]
        if args.node:  # pin a Galaxy node (still checked against node_regex here and in the job)
            node = args.node if args.node.startswith("bh-") else f"bh-glx-110-{args.node}"
            if not re.match(galaxy_node_regex(sl), node):
                raise RegressError(f"node {node} does not match {galaxy_node_regex(sl)}")
            sb += [f"--nodelist={node}"]
        excl = list(args.exclude.split(",")) if args.exclude else []
        excl += sl.get("exclude") or []  # standing excludes (full /tmp, coordinator notes)
        if any("ring_mapping" in s["needs"] for s in steps):
            excl += sl.get("exclude_for_ring") or []
        if excl:
            sb += [f"--exclude={','.join(sorted(set(excl)))}"]
    inodes = None
    if sl.get("min_data_inodes_free"):  # /data is shared; a job that cannot create files dies mid-run (job 172554)
        inodes = data_inodes_free()
        say(f"/data free inodes now: {inodes:,} (a submit needs >= {sl['min_data_inodes_free']:,})")
        if inodes < sl["min_data_inodes_free"] and args.go:
            raise RegressError("/data is short of free inodes: leave the steps NOT_RUN, do not retry in a loop")
    if sl.get("max_concurrent"):
        jobs = slurm_count_galaxy()
        cap = int(os.environ.get("REGRESS_GALAXY_CAP", sl["max_concurrent"]))  # raised only for a window the user names
        say(f"study Galaxy jobs now, running + pending, quad excepted ({len(jobs)}/{cap}): {jobs}")
        if len(jobs) >= cap and args.go:
            raise RegressError("at the Galaxy allocation cap: wait")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = os.path.join("/data/aagarwal/regress/stage", f"{stamp}-{g['id']}")
    os.makedirs(stage, exist_ok=True)
    os.makedirs("/data/aagarwal/regress/logs", exist_ok=True)
    for f in ("run_group.py", "manifest.yaml", "regress_junit_plugin.py", "regress.sbatch", "dit_ring_joint_perf.py"):
        shutil.copy2(os.path.join(HERE, f), stage)
    if args.manifest != DEFAULT_MANIFEST:
        shutil.copy2(args.manifest, os.path.join(stage, "manifest.yaml"))
    extra = []
    if args.routers:
        extra += ["--routers", args.routers]
    if args.passes:
        extra += ["--passes", str(args.passes)]
    if args.steps:
        extra += ["--steps", args.steps]
    if args.local_cache:
        extra += ["--local-cache"]
    exports = {"REGRESS_STAGE": stage, "REGRESS_GROUP": g["id"], "REGRESS_TREE": args.tree,
               "REGRESS_TREE_PATH": t["data_path"], "REGRESS_LABEL": args.label, "REGRESS_ARGS": " ".join(extra)}
    if inodes is not None:
        exports["REGRESS_DATA_INODES_AT_SUBMIT"] = str(inodes)
    sb.append("--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items() if "," not in v))
    if "," in exports["REGRESS_ARGS"]:
        with open(os.path.join(stage, "args.txt"), "w") as f:
            f.write(exports["REGRESS_ARGS"])
    sb.append(os.path.join(stage, "regress.sbatch"))
    say(f"{g['id']}: {len(steps)} steps, {per} min per router pass (CI timeouts), wall limit {fmt_time(minutes)}")
    print(" ".join(shlex.quote(x) for x in sb))
    if not args.go:
        r = subprocess.run(sb[:1] + ["--test-only"] + sb[1:], capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        say(f"dry run (sbatch --test-only) rc={r.returncode}: {out}")
        if r.returncode and "allocation failure" in out and any(a.startswith("--nodelist=") for a in sb):
            say("   (expected while the pinned node is allocated: --test-only only answers for a placeable node, "
                "as lb2_run.sh notes; the options themselves are accepted)")
            return 0
        return r.returncode
    r = subprocess.run(sb, capture_output=True, text=True)
    say(f"sbatch rc={r.returncode}: {(r.stdout + r.stderr).strip()}")
    return r.returncode


def cmd_collect(args):
    man = load_manifest(args.manifest)
    src = os.path.join(man["results"]["data"], args.label)
    dst = os.path.join(man["results"]["local"], args.label)
    if not os.path.isdir(src):
        raise RegressError(f"no {src}")
    os.makedirs(dst, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "--stats", src + "/", dst + "/"], capture_output=True, text=True)
    say(f"collect {src} -> {dst}: rc={r.returncode}; " +
        "; ".join(l.strip() for l in r.stdout.splitlines() if "files transferred" in l))
    return r.returncode


def iter_status(root, label):
    """status.json of every step result in a label (earlier attempts, <step>.attemptN/, and _-prefixed dirs excluded)."""
    for p in sorted(glob.glob(os.path.join(root, label, "*", "*", "pass*", "*", "status.json"))):
        if ".attempt" in p.split(os.sep)[-2] or os.path.relpath(p, os.path.join(root, label)).startswith("_"):
            continue
        try:
            with open(p) as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        st["_dir"] = os.path.dirname(p)
        yield st


def cmd_status(args):
    man = load_manifest(args.manifest)
    root = man["results"]["data"] if args.data else man["results"]["local"]
    rows = list(iter_status(root, args.label))
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    print(f"{'group':12s} {'router':6s} {'p':>1s} {'step':16s} {'status':7s} {'secs':>6s} {'node':18s} detail")
    for r in rows:
        det = []
        if r.get("tests"):
            det.append(" ".join(f"{a}={b}" for a, b in r["tests"].items() if b))
        if r.get("tt_fabric"):
            det.append(f"tt_fabric {r['tt_fabric']['executed']}/{r['tt_fabric']['failed']}f")
        if r.get("metrics") is not None:
            det.append(f"{r['metrics']} metrics")
        if r.get("reason"):
            det.append(r["reason"])
        print(f"{r['group']:12s} {r['router']:6s} {r['pass']:>1} {r['step']:16s} {r['status']:7s} "
              f"{r.get('duration_s') or 0:6.0f} {r.get('node', ''):18s} {'; '.join(det)}")
    return 0


def cmd_rescore(args):
    """Recompute PASS / FAIL of finished steps from their logs and XML under the current manifest (no device)."""
    man = load_manifest(args.manifest)
    root = man["results"]["data"] if args.data else man["results"]["local"]
    n = 0
    for st in iter_status(root, args.label):
        if args.group and st["group"] != args.group:
            continue
        if st.get("status") == "HANG" and not st.get("capped"):  # was the log still advancing when it was killed?
            log = os.path.join(st["_dir"], "log.txt")
            try:
                end = datetime.datetime.fromisoformat(st["end"]).timestamp()
                age = end - os.path.getmtime(log) - 25  # the kill takes up to ~25 s after the timeout
            except (OSError, ValueError, KeyError):
                age = None
            if age is not None and age < 120:
                sdir = st.pop("_dir")
                say(f"{st['group']}/{st['router']}/pass{st['pass']}/{st['step']}: HANG -> TIMEOUT (log written "
                    f"{max(0, int(age))} s before the timeout)")
                st.setdefault("rescored", []).append({"from": "HANG", "to": "TIMEOUT", "at": iso()})
                st["status"] = "TIMEOUT"
                write_status(sdir, st)
                n += 1
            continue
        if st.get("status") == "TIMEOUT":  # keep the verdict, refresh the case counts (from the log if no junit)
            g = get_group(man, st["group"])
            step = next((x for x in g.get("steps", []) if x["id"] == st["step"]), None)
            if step is not None:
                tree = st.get("tree") if st.get("tree") and os.path.isdir(st["tree"]) else tree_info(man, args.tree, "local")[1]
                sdir = st.pop("_dir")
                st.update(collect(resolve_step(man, step, tree), argparse.Namespace(home=tree), sdir, 0, copy=False))
                write_status(sdir, st)
            continue
        if st.get("status") not in ("PASS", "FAIL"):
            continue
        g = get_group(man, st["group"])
        step = next((x for x in g.get("steps", []) if x["id"] == st["step"]), None)
        if step is None:
            continue
        tree = st.get("tree") if st.get("tree") and os.path.isdir(st["tree"]) else tree_info(man, args.tree, "local")[1]
        rs = resolve_step(man, step, tree)
        sdir = st.pop("_dir")
        ctx = argparse.Namespace(home=tree)
        res = collect(rs, ctx, sdir, 0, copy=False)
        new = verdict_status(st.get("rc"), "", st.get("capped"), res, rs)
        if new != st["status"]:
            say(f"{st['group']}/{st['router']}/pass{st['pass']}/{st['step']}: {st['status']} -> {new}")
            st.setdefault("rescored", []).append({"from": st["status"], "to": new, "at": iso()})
            n += 1
        st.update(res, status=new)
        write_status(sdir, st)
    for st in iter_status(root, args.label):  # a known failure that also hangs or times out (known_fail_statuses)
        if (args.group and st["group"] != args.group) or st.get("status") not in ("HANG", "TIMEOUT"):
            continue
        g = get_group(man, st["group"])
        step = next((x for x in g.get("steps", []) if x["id"] == st["step"]), None)
        if step is None or not step.get("known_fail") or st["status"] not in (step.get("known_fail_statuses") or ["FAIL"]):
            continue
        sdir = st.pop("_dir")
        say(f"{st['group']}/{st['router']}/pass{st['pass']}/{st['step']}: {st['status']} -> KNOWN_FAIL")
        st.setdefault("rescored", []).append({"from": st["status"], "to": "KNOWN_FAIL", "at": iso()})
        st.update(status="KNOWN_FAIL", known_fail=step["known_fail"])
        write_status(sdir, st)
        n += 1
    say(f"rescore: {n} step(s) changed")
    return 0


# ------------------------------------------------------------------------------------------------- compare

def load_side(root, spec):
    """spec = LABEL[:ROUTER]; returns {(group, step): [status dicts over passes]}."""
    label, _, router = spec.partition(":")
    out = {}
    for st in iter_status(root, label):
        if router and st["router"] != router:
            continue
        out.setdefault((st["group"], st["step"]), []).append(st)
    return out


def tests_of(st):
    try:
        with open(os.path.join(st["_dir"], "tests.json")) as f:
            return json.load(f)["tests"]
    except (OSError, ValueError, KeyError):
        return {}


def metrics_of(st):
    p = os.path.join(st["_dir"], "metrics.csv")
    rows = {}
    if os.path.exists(p):
        seen = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                v = float(r.pop("value"))
                key = tuple(sorted(r.items()))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1:  # the same key twice in one run: keep them apart (occurrence index)
                    key += (("_n", str(seen[key])),)
                rows.setdefault(key, []).append(v)
    return rows


FABRIC_BW_KEYS = ("test_name", "ftype", "ntype", "topology", "num_devices", "num_links", "packet_size")
FABRIC_LAT_KEYS = ("test_name", "ftype", "ntype", "topology", "num_devices", "num_links", "payload_size")


def fabric_rows(st):
    """bandwidth (GB/s, higher is better) and latency (net_avg_ns, lower is better) summary rows of a step."""
    rows = {}
    for kind, pat, keys, col in (("bw", "bandwidth_summary_results_*.csv", FABRIC_BW_KEYS, "avg_bandwidth_gigabytes_per_s"),
                                 ("lat_ns", "latency_summary_results_*.csv", FABRIC_LAT_KEYS, "net_avg_ns")):
        for p in glob.glob(os.path.join(st["_dir"], "fabric", pat)):
            if "upload" in p:
                continue
            with open(p) as f:
                for r in csv.DictReader(f):
                    try:
                        v = float(r[col])
                    except (KeyError, ValueError):
                        continue
                    rows.setdefault((kind,) + tuple(r.get(k, "") for k in keys), []).append(v)
    return rows


def benchmark_rows(st):
    rows = {}
    for p in glob.glob(os.path.join(st["_dir"], "benchmark_data", "**", "*.json"), recursive=True):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        for m in d.get("measurements", []) if isinstance(d, dict) else []:
            if isinstance(m, dict) and isinstance(m.get("value"), (int, float)):
                key = (os.path.basename(p).split("_")[0], m.get("step_name", ""), m.get("name", ""))
                rows.setdefault(key, []).append(float(m["value"]))
    return rows


def spread(vals):
    return (max(vals) - min(vals)) / (sum(vals) / len(vals)) if len(vals) > 1 and sum(vals) else 0.0


def cmd_compare(args):
    man = load_manifest(args.manifest)
    root = man["results"]["data"] if args.data else man["results"]["local"]
    base, test = load_side(root, args.base), load_side(root, args.test)
    lines = [f"# Compare {args.base} (base) vs {args.test} (test)", "", f"generated {iso()} from {root}", ""]
    csv_rows = []
    keys = sorted(set(base) | set(test))
    if args.group:
        keys = [k for k in keys if k[0] == args.group]
    lines += ["## Steps", "", "| group | step | base | test | note |", "| --- | --- | --- | --- | --- |"]
    regress_cases, fixed_cases = [], []
    for k in keys:
        b, t = base.get(k, []), test.get(k, [])
        bs = ",".join(s["status"] for s in b) or "-"
        tss = ",".join(s["status"] for s in t) or "-"
        note = ""
        if b and t:
            bt, tt = {}, {}
            cov = lambda x: "PASS" if x == "PERF_WINDOW" else x  # perf windows are judged on the metric values
            for s in b:
                for n, c in tests_of(s).items():
                    bt[n] = "PASS" if bt.get(n) == "PASS" else cov(c["status"])
            for s in t:
                for n, c in tests_of(s).items():
                    tt[n] = "FAIL" if tt.get(n) == "FAIL" else cov(c["status"])
            reg = [n for n in tt if tt[n] == "FAIL" and bt.get(n) == "PASS"]
            fix = [n for n in tt if tt[n] == "PASS" and bt.get(n) == "FAIL"]
            only_b = [n for n in bt if n not in tt]
            only_t = [n for n in tt if n not in bt]
            regress_cases += [(k, n) for n in reg]
            fixed_cases += [(k, n) for n in fix]
            parts = []
            if reg:
                parts.append(f"**{len(reg)} PASS->FAIL**")
            if fix:
                parts.append(f"{len(fix)} FAIL->PASS")
            if only_b or only_t:
                parts.append(f"cases only in base {len(only_b)}, only in test {len(only_t)}")
            ok = ("PASS", "PERF_WINDOW")
            if any(s["status"] in ("KNOWN_FAIL", "ENV") for s in b) or any(s["status"] == "ENV" for s in t):
                parts.insert(0, "known-fail on base: not scored")
                reg_known = [n for n in reg]
                regress_cases[:] = [x for x in regress_cases if x[0] != k]
                parts = [p for p in parts if "PASS->FAIL" not in p] + ([f"({len(reg_known)} case changes ignored)"] if reg_known else [])
            elif any(s["status"] in ok for s in b) and all(s["status"] not in ok for s in t):
                parts.insert(0, "**step regressed**")
            note = "; ".join(parts)
        lines.append(f"| {k[0]} | {k[1]} | {bs} | {tss} | {note} |")
        csv_rows.append({"kind": "step", "group": k[0], "step": k[1], "key": "", "base": bs, "test": tss, "diff": "",
                         "noise": "", "verdict": note})
    if regress_cases:
        lines += ["", "## Test cases PASS on base, FAIL on test", ""]
        lines += [f"- {g}/{s}: `{n}`" for (g, s), n in regress_cases[:200]]
    # perf: metrics, fabric CSVs, benchmark data
    perf = []
    for k in keys:
        b, t = base.get(k, []), test.get(k, [])
        if not (b and t):
            continue
        for getter, kind in ((metrics_of, "metric"), (fabric_rows, "fabric"), (benchmark_rows, "benchmark")):
            bm, tm = {}, {}
            for s in b:
                for kk, v in getter(s).items():
                    bm.setdefault(kk, []).extend(v)
            for s in t:
                for kk, v in getter(s).items():
                    tm.setdefault(kk, []).extend(v)
            for kk in sorted(set(bm) & set(tm), key=str):
                bv, tv = bm[kk], tm[kk]
                bmean, tmean = sum(bv) / len(bv), sum(tv) / len(tv)
                if not bmean:
                    continue
                diff = tmean / bmean - 1
                noise = spread(bv)
                perf.append((k, kind, kk, bmean, tmean, diff, noise, len(bv), len(tv)))
    if perf:
        thr = args.threshold
        lines += ["", "## Perf rows (diff = test/base - 1; noise = base pass-to-pass spread; flagged beyond "
                      f"max(noise, {thr:.0%}))", "",
                  "| group/step | kind | key | base | test | diff | noise | n |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for (g, s), kind, kk, bm_, tm_, d, n, nb, nt in sorted(perf, key=lambda x: -abs(x[5])):
            flag = " **!**" if abs(d) > max(n, thr) else ""
            ks = " ".join(f"{a}={b}" for a, b in kk) if kind == "metric" else str(kk)
            lines.append(f"| {g}/{s} | {kind} | {ks} | {bm_:.4g} | {tm_:.4g} | {d:+.2%}{flag} | {n:.2%} | {nb}/{nt} |")
            csv_rows.append({"kind": kind, "group": g, "step": s, "key": ks, "base": bm_, "test": tm_, "diff": d,
                             "noise": n, "verdict": "flag" if flag else ""})
    text = "\n".join(lines) + "\n"
    out = args.out or os.path.join(root, f"compare_{args.base.replace(':', '-')}_vs_{args.test.replace(':', '-')}")
    with open(out + ".md", "w") as f:
        f.write(text)
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "group", "step", "key", "base", "test", "diff", "noise", "verdict"])
        w.writeheader()
        w.writerows(csv_rows)
    print(text)
    say(f"wrote {out}.md and .csv")
    return 1 if regress_cases else 0


def cmd_noise(args):
    """Pass-to-pass spread of one router's perf rows (metrics, fabric BW / latency, benchmark data) within a label."""
    man = load_manifest(args.manifest)
    root = man["results"]["data"] if args.data else man["results"]["local"]
    side = load_side(root, f"{args.label}:{args.router}")
    rows = []
    for (g, s_), sts in sorted(side.items()):
        if args.group and g != args.group:
            continue
        sts = sorted((x for x in sts if x.get("status") in ("PASS", "FAIL")), key=lambda x: x["pass"])
        for getter, kind in ((metrics_of, "metric"), (fabric_rows, "fabric"), (benchmark_rows, "benchmark")):
            per = {}
            for st in sts:
                for kk, v in getter(st).items():
                    per.setdefault(kk, {})[st["pass"]] = sum(v) / len(v)
            for kk, d in per.items():
                vals = [d[k] for k in sorted(d)]
                rows.append((g, s_, kind, kk, vals, spread(vals)))
    lines = [f"# Pass-to-pass noise: {args.label}, router {args.router}", "",
             "| group/step | kind | key | per pass | spread |", "| --- | --- | --- | --- | ---: |"]
    for g, s_, kind, kk, vals, sp in sorted(rows, key=lambda r: -r[5]):
        ks = " ".join(f"{a}={b}" for a, b in kk) if kind == "metric" else " ".join(str(x) for x in kk)
        lines.append(f"| {g}/{s_} | {kind} | {ks} | {' / '.join(f'{v:.4g}' for v in vals)} | {sp:.2%} |")
    if rows:
        sps = sorted(r[5] for r in rows if len(r[4]) > 1)
        if sps:
            lines += ["", f"{len(sps)} rows with >= 2 passes: median spread {sps[len(sps) // 2]:.2%}, max {sps[-1]:.2%}"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


# ------------------------------------------------------------------------------------------------- plan / resolve

def cmd_plan(args):
    man = load_manifest(args.manifest)
    levels = set(args.levels.split(",")) if args.levels else None
    systems = set(args.systems.split(",")) if args.systems else None
    routers_f = args.routers.split(",") if args.routers else None
    tname, tpath, _ = tree_info(man, args.tree, "local")
    node_h = 0.0
    for g in man["groups"]:
        if levels and g["level"] not in levels or systems and g["system"] not in systems:
            continue
        status = g.get("status", "ready")
        sysd = man["systems"].get(g["system"], {})
        routers = [r for r in g.get("routers", ["main"]) if not routers_f or r in routers_f]
        head = f"{g['id']:12s} {g['level']} {g['system']:11s} {g.get('kind', ''):8s} {status:12s}"
        if status != "ready" or not g.get("steps"):
            print(f"{head} {g.get('why', '')}")
            continue
        try:
            steps, per, total = group_minutes(man, g, tpath, routers, args.passes)
        except RegressError as e:
            print(f"{head} !!! {e}")
            continue
        print(f"{head} {len(steps)} steps, {per} min/router-pass (CI timeouts), {total} min for routers {routers}"
              f"{' passes ' + str(g.get('passes')) if g.get('kind') == 'perf' and g.get('passes') else ''}")
        for s in steps:
            src = f"{s['ci']['list']} :: {s['ci']['name']}" if s.get("ci") else "explicit"
            print(f"    {s['id']:16s} {s['timeout_min']:>4} min  needs {','.join(s['needs']) or '-':24s} {src}")
        if args.host == "a09u32" and sysd.get("broker"):
            b = sysd["broker"]
            est = sum(min(s["timeout_min"] * 30, s["timeout_min"] * 60) for s in steps)  # half the CI timeouts
            n = max(1, -(-est // b["budget_s"]))
            for r in routers:
                print(f"    broker x~{n} (repeat until 'ALL STEPS DONE'), timeout_sec {b['max_job_s']}, workspace {b['workspace']}:")
                print(f"      cd {tpath} && python3 {os.path.join(HERE, 'run_group.py')} run --group {g['id']} --tree {tname} "
                      f"--router {r} --label {args.label} --resume")
        elif sysd.get("slurm"):
            print(f"    slurm: python3 {os.path.join(HERE, 'run_group.py')} submit --group {g['id']} --tree {tname} "
                  f"--label {args.label}{' --routers ' + ','.join(routers) if routers_f else ''}"
                  f"{' --node <a07u28|a08u28|a04u24>' if (sysd['slurm'].get('nodes')) else ''}   (dry run; --go submits)")
            node_h += (total + 20) / 60
    if node_h:
        print(f"Slurm node-hours (upper bound, CI timeouts): {node_h:.1f}")
    return 0


def cmd_resolve(args):
    man = load_manifest(args.manifest)
    g = get_group(man, args.group)
    _, tpath, _ = tree_info(man, args.tree, "data" if args.data else "local")
    for s in g.get("steps", []):
        try:
            rs = resolve_step(man, s, tpath)
        except RegressError as e:
            print(f"--- {s['id']}: !!! {e}")
            continue
        print(f"--- {rs['id']} (timeout {rs['timeout_min']} min, auto_triage {rs['auto_triage']}, needs {rs['needs'] or '-'},"
              f" /mnt paths {mnt_paths(rs['cmd']) or '-'})")
        print(rs["cmd"].rstrip())
    return 0


# ------------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--levels")
    p.add_argument("--systems")
    p.add_argument("--routers")
    p.add_argument("--passes", type=int)
    p.add_argument("--tree", default="main")
    p.add_argument("--label", default="main")
    p.add_argument("--host", default="a09u32")

    p = sub.add_parser("resolve")
    p.add_argument("--group", required=True)
    p.add_argument("--tree", default="main")
    p.add_argument("--data", action="store_true")

    for name in ("run", "job"):
        p = sub.add_parser(name)
        p.add_argument("--group", required=True)
        p.add_argument("--tree", required=True)
        p.add_argument("--label", required=True)
        p.add_argument("--steps")
        p.add_argument("--out")
        p.add_argument("--force", action="store_true")
        p.add_argument("--any-node", action="store_true")
        p.add_argument("--stop-on-fail", action="store_true", help="CI semantics: stop a step at its first failing line")
        p.add_argument("--budget-s", type=int)
        if name == "run":
            p.add_argument("--router", default="main")
            p.add_argument("--pass", dest="pass_", type=int, default=1)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--rerun-failed", help="STEP[,STEP]: rerun each FAIL case of these steps (this --pass) on its own, as STEP.rerun")
            p.add_argument("--mode", choices=["broker", "slurm", "local"])
        else:
            p.add_argument("--routers")
            p.add_argument("--passes", type=int)
            p.add_argument("--scratch")
            p.add_argument("--keep-scratch", action="store_true")
            p.add_argument("--ignore-busy", action="store_true")
            p.add_argument("--no-reset", action="store_true")
            p.add_argument("--local-cache", action="store_true", help="JIT cache on node-local /tmp (removed at the end)")
            p.add_argument("--max-timeout-min", type=float, help="cap every step's timeout (an A/B's stall timeout)")

    p = sub.add_parser("stage")
    p.add_argument("--tree", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-only", action="store_true")

    p = sub.add_parser("submit")
    p.add_argument("--group", required=True)
    p.add_argument("--tree", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--routers")
    p.add_argument("--passes", type=int)
    p.add_argument("--steps")
    p.add_argument("--node")
    p.add_argument("--overflow", action="store_true", help="LoudBox groups: the overflow partition (bh_lb_single)")
    p.add_argument("--local-cache", action="store_true", help="the job's JIT cache on node-local /tmp")
    p.add_argument("--exclude")
    p.add_argument("--time-min", type=int)
    p.add_argument("--go", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("collect")
    p.add_argument("--label", required=True)

    p = sub.add_parser("status")
    p.add_argument("--label", required=True)
    p.add_argument("--group")
    p.add_argument("--data", action="store_true")

    p = sub.add_parser("rescore")
    p.add_argument("--label", required=True)
    p.add_argument("--group")
    p.add_argument("--tree", default="main")
    p.add_argument("--data", action="store_true")

    p = sub.add_parser("noise")
    p.add_argument("--label", required=True)
    p.add_argument("--router", default="main")
    p.add_argument("--group")
    p.add_argument("--data", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("compare")
    p.add_argument("--base", required=True, help="LABEL[:ROUTER]")
    p.add_argument("--test", required=True, help="LABEL[:ROUTER]")
    p.add_argument("--group")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--data", action="store_true")
    p.add_argument("--out")

    args = ap.parse_args(argv)
    fn = {"plan": cmd_plan, "resolve": cmd_resolve, "run": cmd_run, "job": cmd_job, "stage": cmd_stage,
          "submit": cmd_submit, "collect": cmd_collect, "status": cmd_status, "compare": cmd_compare,
          "rescore": cmd_rescore, "noise": cmd_noise}[args.cmd]
    try:
        return fn(args)
    except RegressError as e:
        say(f"!!! {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
