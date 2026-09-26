#!/usr/bin/env python3
"""OPTIONAL execution tier: turn builds, analyzers and existing tests into per-batch leads for the hunters.

A static audit cannot catch what only a build flavour, a sanitizer or a test exposes. The post-fix miss analysis found
that for 11 of 29 misses, execution was the cheapest reliable catch. This tier runs the commands the USER chose at the
start of the audit (see SKILL.md, "Start of every audit"), parses their diagnostics, and writes
exec/signals/<batch>.json for every batch whose files they touch. The hunters then triage each signal as a lead.
Nothing here decides a bug: a warning or a failing test is a lead, and the normal verification still applies.

  exec_tier.py [--run DIR] configure [--build NAME=CMD ...] [--analyze NAME=CMD ...]
                                     [--test-cmd 'CMD with {tests}'] [--test-root DIR ...] [--max-tests 8]
                                     [--timeout SECONDS] [--reset-cmd 'tt-smi -r 0']
  exec_tier.py [--run DIR] run [--steps build,analyze,tests]
  exec_tier.py [--run DIR] status

Traps from earlier audits, handled here:
- A hung test WEDGES the device, and every later test then fails for no code reason. --reset-cmd runs before each
  test group and after any timeout.
- A failing test is re-run once. Only a failure that reproduces becomes a signal; a pass on rerun is logged as flaky.
- A build from a different base commit makes unrelated tests fail. Build in the audited tree itself (the commands
  run there), from a clean build directory, never reusing another checkout's build.
Commands run with the audited tree as the working directory. They execute the repo's code, so run them only on a
machine the user has agreed to, and for tests only when the target hardware is available and the user said so.
{tree} in a command expands to the audited tree's path.
Diagnostics understood: compiler, linker and clang-tidy "file:line[:col]: error|warning|note: msg" lines; sanitizer
"runtime error" lines and "#N 0x... in fn file:line" stack frames; and Python 'File "file", line N' frames. Test
selection greps the test roots for each batch file's stem, or for its module path for Python files.
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, manifest, run_dir, save, state  # noqa: E402

out = run_dir()
st = state(out)
man = manifest(out)
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
EXEC = os.path.join(out, "exec")
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
    cmd = cmd.replace("{tree}", tree)
    logp = os.path.join(EXEC, f"{name}.log")
    t0 = time.time()
    with open(logp, "w") as fh:
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=tree,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, open(logp, errors="replace").read(), round(time.time() - t0)


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
    }
    if not (ex["build"] or ex["analyze"] or ex["tests"]["cmd"]):
        sys.exit(
            "nothing to configure: give at least one --build, --analyze or --test-cmd"
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
        for b, m in sorted(man.items()):
            wanted = set()
            for f in m["files"]:
                stem = os.path.splitext(os.path.basename(f))[0]
                pat = f.replace("/", ".")[:-3] if f.endswith(".py") else stem
                for root in tc.get("roots") or ["tests"]:
                    g = subprocess.run(
                        [
                            "grep",
                            "-rlw",
                            "--include=*.py",
                            "--include=*.cpp",
                            pat,
                            root,
                        ],
                        cwd=tree,
                        capture_output=True,
                        text=True,
                    ).stdout.split()
                    wanted |= set(g)
            chosen = sorted(wanted)[: tc.get("max_per_batch", 8)]
            if not chosen:
                continue
            if ex.get("reset_cmd"):
                run_cmd(f"reset-before-{b}", ex["reset_cmd"], tree, 600)
            test_cmd = tc["cmd"].replace("{tests}", " ".join(chosen))
            rc, text, secs = run_cmd(f"tests-{b}", test_cmd, tree, ex["timeout"])
            if rc == "timeout" and ex.get("reset_cmd"):
                run_cmd(
                    f"reset-after-timeout-{b}", ex["reset_cmd"], tree, 600
                )  # a hang wedges the device
            if rc not in (0,):
                if ex.get("reset_cmd"):
                    run_cmd(f"reset-before-rerun-{b}", ex["reset_cmd"], tree, 600)
                rc2, text2, _ = run_cmd(
                    f"tests-{b}-rerun", test_cmd, tree, ex["timeout"]
                )
                if rc2 == 0:
                    runs.append(
                        {
                            "step": "tests",
                            "name": b,
                            "rc": rc,
                            "seconds": secs,
                            "tests": chosen,
                            "signals": 0,
                            "flaky": True,
                        }
                    )
                    print(
                        f"tests {b}: failed then passed on rerun, logged as flaky, not a signal"
                    )
                    continue
            found = parse(text, tree, "tests", "test")
            if rc not in (0, "timeout") and not found:
                found = [
                    {
                        "kind": "test",
                        "tool": "tests",
                        "file": m["files"][0],
                        "line": 0,
                        "severity": "test-failure",
                        "message": f"selected tests failed (rc={rc}): {' '.join(chosen)[:300]}; see exec/tests-{b}.log",
                    }
                ]
            for s in found:
                s.setdefault("batch", b)
            runs.append(
                {
                    "step": "tests",
                    "name": b,
                    "rc": rc,
                    "seconds": secs,
                    "tests": chosen,
                    "signals": len(found),
                }
            )
            print(
                f"tests {b}: rc={rc} in {secs}s, {len(chosen)} test files, {len(found)} signals"
            )
            sigs += found
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
