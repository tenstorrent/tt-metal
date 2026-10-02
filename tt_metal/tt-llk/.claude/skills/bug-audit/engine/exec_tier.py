#!/usr/bin/env python3
"""OPTIONAL execution tier: turn builds, analyzers and existing tests into per-batch leads for the hunters.

A static audit cannot catch what only a build flavour, a sanitizer or a test exposes. The post-fix miss analysis found
that for 11 of 29 misses, execution was the cheapest reliable catch. This tier runs the commands the USER chose at the
start of the audit (see SKILL.md, "Start of every audit"), parses their diagnostics, and writes
exec/signals/<batch>.json for every batch whose files they touch. The hunters then triage each signal as a lead.
Nothing here decides a bug: a warning or a failing test is a lead, and the normal verification still applies.

  exec_tier.py [--run DIR] configure [--build NAME=CMD ...] [--analyze NAME=CMD ...]
                                     [--test-cmd 'CMD with {tests}'] [--test-root DIR ...] [--max-tests 8]
                                     [--timeout SECONDS] [--devices IDS] [--reset-cmd 'tt-smi -r {devices}']
  exec_tier.py [--run DIR] run [--steps build,analyze,tests]
  exec_tier.py [--run DIR] status

Traps from earlier audits, handled here:
- A hung test WEDGES the device, and every later test then fails for no code reason. --reset-cmd runs before each
  test and after any timeout.
- On a shared machine, other people's cards must never be touched. --devices takes the cards the user confirmed
  (UMD chip ids or PCI BDFs, comma-separated, one kind only) and is required with --test-cmd or --reset-cmd. Every
  command runs with TT_VISIBLE_DEVICES set to them, and {devices} in the reset command expands to them, so the reset
  must name {devices} rather than a card number.
- A failing test is re-run once. Only a failure that reproduces becomes a signal; a pass on rerun is logged as flaky.
- A build from a different base commit makes unrelated tests fail. Build in the audited tree itself (the commands
  run there), from a clean build directory, never reusing another checkout's build.
Commands run with the audited tree as the working directory. They execute the repo's code, so run them only on a
machine the user has agreed to, and for tests only when the target hardware is available and the user said so.
{tree} in a command expands to the audited tree's path.
Each batch picks up to --max-tests tests (pick_tests); a test that several batches pick runs once, in its own
invocation ({tests} is that one path), and its signals go to every batch that picked it. One test per invocation
also means a runner's stop-at-first-failure flag (pytest -x) can only skip tests inside that test file, so leave it
out of --test-cmd.
Diagnostics understood: compiler, linker and clang-tidy "file:line[:col]: error|warning|note: msg" lines; sanitizer
"runtime error" lines and "#N 0x... in fn file:line" stack frames; and Python 'File "file", line N' frames. Test
selection ranks the tests under the test roots by how specifically they name the batch's files (pick_tests).
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spawn  # noqa: E402

from common import load, manifest, run_dir, save, state  # noqa: E402

out = run_dir()
st = state(out)
man = manifest(out)
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
EXEC = os.path.join(out, "exec")
DEVICES = ((st.get("execution") or {}).get("devices")) or []
BDF = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
os.makedirs(os.path.join(EXEC, "signals"), exist_ok=True)


def opts(flag):
    return [argv[i + 1] for i, x in enumerate(argv) if x == flag and i + 1 < len(argv)]


def pairs(flag):
    res = []
    for x in opts(flag):
        name, _, cmd = x.partition("=")
        res.append({"name": name.strip(), "cmd": cmd.strip()})
    return res


DIAG = re.compile(
    r"^(?P<file>[^\s:()\"']+\.[A-Za-z0-9_]+):(?P<line>\d+)(?::\d+)?:\s*(?P<sev>fatal error|error|warning|note|runtime error)\b:?\s*(?P<msg>.*)$"
)
FRAME = re.compile(
    r"#\d+\s+0x[0-9a-f]+\s+in\s+(?P<fn>\S+)\s+(?P<file>[^\s:]+):(?P<line>\d+)"
)
PYFRAME = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')
# identifiers too common to locate a link error's owner by (every distinctive token must appear in the file)
COMMON = {
    "std",
    "detail",
    "bool",
    "int",
    "char",
    "void",
    "const",
    "unsigned",
    "long",
    "short",
    "size_t",
    "handle",
    "pybind11",
    "nanobind",
    "ttnn",
    "tt",
    "tt_metal",
    "operator",
    "allocator",
    "basic_string",
    "vector",
    "__cxx11",
}
UNDEF = re.compile(r"undefined reference to [`'](?P<sym>[^'`]+)'")


def rel(path, tree):
    path = os.path.normpath(path)
    if os.path.isabs(path):
        return os.path.relpath(path, tree) if path.startswith(tree) else None
    return path


def parse(text, tree, tool, kind):
    sigs = []
    for ln in text.splitlines():
        m = DIAG.match(ln.strip())
        if m:
            f = rel(m.group("file"), tree)
            if f:
                sigs.append(
                    {
                        "kind": kind,
                        "tool": tool,
                        "file": f,
                        "line": int(m.group("line")),
                        "severity": m.group("sev"),
                        "message": m.group("msg")[:400],
                    }
                )
            continue
        for rx in (FRAME, PYFRAME):
            m = rx.search(ln)
            if m:
                f = rel(m.group("file"), tree)
                if f:
                    sigs.append(
                        {
                            "kind": kind,
                            "tool": tool,
                            "file": f,
                            "line": int(m.group("line")),
                            "severity": "stack-frame",
                            "message": ln.strip()[:400],
                        }
                    )
        m = UNDEF.search(ln)
        if m:
            sigs.append(
                {
                    "kind": kind,
                    "tool": tool,
                    "file": None,
                    "line": 0,
                    "severity": "link-error",
                    "message": ln.strip()[:400],
                    "symbol": m.group("sym"),
                }
            )
    return sigs


def run_cmd(name, cmd, tree, timeout):
    """Run one configured command through the shell.

    The commands are the ones the user typed at `configure` time and {tree} is the pinned worktree path, so this runs
    trusted input only: never build `cmd` or `tree` from repository or GitHub content. The command gets its own
    process group, and a timeout kills the whole group -- killing only the shell would leave a hung build or test
    running on the device while the reset and the rerun start.
    """
    cmd = cmd.replace("{tree}", tree).replace(
        "{devices}", " ".join(map(shlex.quote, DEVICES))
    )
    env = dict(os.environ, TT_VISIBLE_DEVICES=",".join(DEVICES)) if DEVICES else None
    logp = os.path.join(EXEC, f"{name}.log")
    t0 = time.time()
    with open(logp, "w") as fh:
        p = spawn.shell_popen(
            cmd,
            cwd=tree,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            rc = "timeout"
    return rc, open(logp, errors="replace").read(), round(time.time() - t0)


GENERIC_STEM = 50  # a stem found in more test files than this (common, utils, device) says little on its own


def grep_tests(args, roots, tree):
    """Test files under the roots that grep matches; -Z, because a path may hold spaces."""
    hits = set()
    for root in roots:
        out_ = spawn.run(
            "grep",
            ["-rlZ", "--include=*.py", "--include=*.cpp", *args, root],
            cwd=tree,
            capture_output=True,
            text=True,
        ).stdout
        hits |= {x for x in out_.split("\0") if x}
    return hits


def pick_tests(files, roots, n, tree):
    """The n tests that most specifically exercise a batch's files.

    A test that names a file itself (its file name with extension, as an #include does, or its Python module path)
    ranks first; then a test that names more of the batch's files; a bare-stem match last, and a stem that matches
    more than GENERIC_STEM test files counts only when nothing better exists. Ties break by path, so a pick is
    reproducible.
    """
    named, by_stem, generic = {}, {}, {}
    for f in files:
        exact = f.replace("/", ".")[:-3] if f.endswith(".py") else os.path.basename(f)
        for t in grep_tests(["-F", exact], roots, tree):
            named[t] = named.get(t, 0) + 1
        stem = os.path.splitext(os.path.basename(f))[0]
        hits = grep_tests(["-wF", stem], roots, tree)
        bucket = generic if len(hits) > GENERIC_STEM else by_stem
        for t in hits:
            bucket[t] = bucket.get(t, 0) + 1
    pool = set(named) | set(by_stem) or set(generic)
    score = lambda t: (-named.get(t, 0), -(by_stem.get(t, 0) + generic.get(t, 0)), t)
    return sorted(pool, key=score)[:n]


def file_to_batch():
    return {f: b for b, m in man.items() for f in m["files"]}


if argv[0] == "configure":
    ex = {
        "enabled": True,
        "build": pairs("--build"),
        "analyze": pairs("--analyze"),
        "tests": {
            "cmd": (opts("--test-cmd") or [None])[0],
            "roots": opts("--test-root"),
            "max_per_batch": int((opts("--max-tests") or ["8"])[0]),
        },
        "timeout": int((opts("--timeout") or ["7200"])[0]),
        "reset_cmd": (opts("--reset-cmd") or [None])[0],
        "devices": [
            d.strip() for d in (opts("--devices") or [""])[0].split(",") if d.strip()
        ],
    }
    if not (ex["build"] or ex["analyze"] or ex["tests"]["cmd"]):
        sys.exit(
            "nothing to configure: give at least one --build, --analyze or --test-cmd"
        )
    devs = ex["devices"]
    if (ex["tests"]["cmd"] or ex["reset_cmd"]) and not devs:
        sys.exit(
            "--devices is required with --test-cmd or --reset-cmd: name the cards the user confirmed, so no other "
            "card on the machine is opened or reset"
        )
    kinds = {"id" if d.isdigit() else "bdf" if BDF.match(d) else "bad" for d in devs}
    if "bad" in kinds or len(kinds) > 1:
        sys.exit(
            f"--devices {','.join(devs)}: give UMD chip ids or PCI BDFs (0000:0a:00.0), one kind, comma-separated"
        )
    if ex["reset_cmd"] and "{devices}" not in ex["reset_cmd"]:
        sys.exit(
            "--reset-cmd must name its cards as {devices} (e.g. 'tt-smi -r {devices}'), never a fixed card number"
        )
    tcmd = ex["tests"]["cmd"] or ""
    if re.search(r"(^|\s)(-x|--exitfirst|--maxfail(=|\s))", tcmd):
        print(
            "warning: --test-cmd stops at the first failure (-x/--exitfirst/--maxfail): the test functions after a "
            "failure in the same test file will not run; drop the flag to keep them"
        )
    st["execution"] = ex
    save(os.path.join(out, "state.json"), st)
    print(json.dumps(ex, indent=1))
elif argv[0] == "run":
    ex = st.get("execution") or {}
    if not ex.get("enabled"):
        sys.exit(
            "execution tier not enabled for this run (it is opt-in: configure it after asking the user)"
        )
    steps = (opts("--steps") or ["build,analyze,tests"])[0].split(",")
    # a run configured before --devices existed may carry a reset of a fixed card: never run it
    if (
        "tests" in steps
        and ((ex.get("tests") or {}).get("cmd") or ex.get("reset_cmd"))
        and not DEVICES
    ):
        sys.exit(
            "this run's execution tier names no --devices: re-run `configure` with the cards the user confirmed"
        )
    tree = st["root"]
    fmap = file_to_batch()
    sigs, runs = [], load(os.path.join(EXEC, "runs.json"), [])
    for step in ("build", "analyze"):
        if step not in steps:
            continue
        for c in ex.get(step, []):
            rc, text, secs = run_cmd(
                f"{step}-{c['name']}", c["cmd"], tree, ex["timeout"]
            )
            found = parse(text, tree, c["name"], step)
            runs.append(
                {
                    "step": step,
                    "name": c["name"],
                    "rc": rc,
                    "seconds": secs,
                    "signals": len(found),
                }
            )
            print(f"{step} {c['name']}: rc={rc} in {secs}s, {len(found)} diagnostics")
            sigs += found
    tc = ex.get("tests", {})
    if "tests" in steps and tc.get("cmd"):
        # Each batch picks its tests exactly as before; neighbouring batches mostly pick the same ones. A distinct
        # test runs ONCE, on its own, and its result goes to every batch that picked it, so every batch gets the
        # same evidence as running its own group, without re-running shared tests once per batch.
        # test -> the batches that picked it, in first-pick order (batch order, then rank)
        pickers = {}
        for b, m in sorted(man.items()):
            for t in pick_tests(
                m["files"],
                tc.get("roots") or ["tests"],
                tc.get("max_per_batch", 8),
                tree,
            ):
                pickers.setdefault(t, []).append(b)
        print(
            f"tests: {len(pickers)} distinct test files for {len({b for bs in pickers.values() for b in bs})} batches"
        )
        for i, (t, batches) in enumerate(pickers.items()):
            name = f"test-{i:04d}"
            if ex.get("reset_cmd"):
                run_cmd(f"reset-before-{name}", ex["reset_cmd"], tree, 600)
            # test paths are repository content: quote each one before it reaches the shell
            test_cmd = tc["cmd"].replace("{tests}", shlex.quote(t))
            rc, text, secs = run_cmd(name, test_cmd, tree, ex["timeout"])
            if rc == "timeout" and ex.get("reset_cmd"):
                run_cmd(
                    f"reset-after-timeout-{name}", ex["reset_cmd"], tree, 600
                )  # a hang wedges the device
            rec = {
                "step": "tests",
                "name": t,
                "log": name,
                "batches": batches,
                "rc": rc,
                "seconds": secs,
            }
            if rc not in (0,):
                if ex.get("reset_cmd"):
                    run_cmd(f"reset-before-rerun-{name}", ex["reset_cmd"], tree, 600)
                rc2, text2, _ = run_cmd(f"{name}-rerun", test_cmd, tree, ex["timeout"])
                if rc2 == 0:
                    runs.append({**rec, "signals": 0, "flaky": True})
                    print(
                        f"tests {t}: failed then passed on rerun, logged as flaky, not a signal"
                    )
                    continue
                # a failure that reproduced: judge the rerun, not the first run (whose output may be a timeout)
                rc, text = rc2, text2
                rec["rc"] = rc
            found = parse(text, tree, "tests", "test")
            if rc == "timeout" and not found:
                # it timed out on the first run AND the rerun: a reproducible hang, the lead this tier exists for
                found = [
                    {
                        "kind": "test",
                        "tool": "tests",
                        "line": 0,
                        "severity": "test-hang",
                        "message": f"test hung twice (timeout {ex['timeout']}s): {t[:300]}; see exec/{name}.log",
                    }
                ]
            if rc not in (0, "timeout") and not found:
                found = [
                    {
                        "kind": "test",
                        "tool": "tests",
                        "line": 0,
                        "severity": "test-failure",
                        "message": f"test failed (rc={rc}): {t[:300]}; see exec/{name}.log",
                    }
                ]
            for b in batches:
                for s in found:
                    s = {**s, "batch": b}
                    if s["severity"] in ("test-hang", "test-failure"):
                        # a whole-test lead points at the picking batch
                        s["file"] = man[b]["files"][0]
                    sigs.append(s)
            runs.append({**rec, "signals": len(found)})
            print(
                f"tests {t}: rc={rc} in {secs}s, {len(found)} signals, for {len(batches)} batches"
            )
    save(os.path.join(EXEC, "runs.json"), runs)
    per = {}
    for s in sigs:
        b = s.get("batch") or fmap.get(s.get("file"))
        if b is None and s.get("severity") == "link-error":
            # a link error names a symbol, not a file: attach it to batches whose files mention the symbol
            toks = {
                t
                for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s["symbol"])
                if t not in COMMON and len(t) > 2
            }
            for f, bb in fmap.items():
                try:
                    body = open(os.path.join(tree, f), errors="replace").read()
                except OSError:
                    continue
                if toks and all(t in body for t in toks):
                    per.setdefault(bb, []).append(s)
            continue
        if b:
            per.setdefault(b, []).append(s)
    for b, lst in per.items():
        path = os.path.join(EXEC, "signals", f"{b}.json")
        prev = load(path, [])
        seen = {
            (x.get("tool"), x.get("file"), x.get("line"), x.get("message"))
            for x in prev
        }
        prev += [
            x
            for x in lst
            if (x.get("tool"), x.get("file"), x.get("line"), x.get("message"))
            not in seen
        ]
        save(path, prev[:200])
    print(
        f"signals written for {len(per)} batches in {EXEC}/signals/ (the next waves hand them to the hunters)"
    )
elif argv[0] == "status":
    ex = st.get("execution")
    print("execution tier:", "enabled" if ex and ex.get("enabled") else "OFF (opt-in)")
    for r in load(os.path.join(EXEC, "runs.json"), []):
        print(
            f"  {r['step']:8s} {r['name']:30s} rc={r['rc']} {r['seconds']}s signals={r['signals']}"
        )
    n = len(
        [f for f in os.listdir(os.path.join(EXEC, "signals")) if f.endswith(".json")]
    )
    print(f"  {n} batches have signals")
else:
    sys.exit(__doc__)
