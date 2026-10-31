"""Review probe helper for run.sh (not for main), standard library only.

  summary.py modules '<json>'                 check the modules input, print "func|perf" per module
  summary.py rerun <log> <junit.xml> <module> print what to rerun after a functional run, one per line: its failing
                                              ids (the first RERUN_MAX), or the module when the run crashed, or nothing
  summary.py summary <out dir> <fail>         print summary.json from the files run.sh left in <out dir>
                                              (env ARCH, REPS, SIDES, MODULES, <SIDE>_SHA)

summary.json: {"arch", "runner", "reps", "fail", "sides": {side: {"sha", "setup_rc"}},
 "modules": [{"func", "perf", "sides": {side: {
   "func": {"summary", "rc", "failed_ids", "crashed", "rerun_truncated",
            "rerun": {"summary", "rc", "failed_ids", "crashed"} or null} or null,
   "perf_build": {"summary", "rc"} or null, "perf": [{"rep", "summary", "rc"}]}}}]}
"summary" is pytest's last summary line ("16 passed, 22 skipped in 3.28s"), "rc" its exit code, null when the
step did not run. "failed_ids" lists the ids that failed or errored, null when the run left no junit xml.
"crashed": no junit xml and an rc other than 0, 4 (usage) or 5 (nothing collected), i.e. pytest died or was
killed; a crashed run is rerun as the whole module, a failed one as its failing ids (at most RERUN_MAX, the
first ones; "rerun_truncated" when there were more). The rerun's failed_ids still fail; a crash reproduces
when the rerun crashed too.
"""
import json
import os
import re
import socket
import sys
import xml.etree.ElementTree as ET

RERUN_MAX = 50
NOT_A_CRASH = (0, 4, 5)  # pytest rc: passed, usage error, no tests collected
MODULE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*\.py")
SUMMARY = re.compile(r"^=*\s*((?:\d+ \w+|no tests ran)(?:, \d+ \w+)* in [0-9.]+s(?: \([0-9:]+\))?)\s*=*$")


def modules(text):
    """[(func, perf)] of the modules input; ValueError when it is not a list of module pairs."""
    pairs = json.loads(text)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("modules must be a non-empty list")
    out = []
    for m in pairs:
        if not isinstance(m, dict) or set(m) - {"func", "perf"}:
            raise ValueError(f"bad module pair {m!r}")
        func, perf = m.get("func", ""), m.get("perf", "")
        for v in (func, perf):
            if not isinstance(v, str) or (v and not MODULE.fullmatch(v)):
                raise ValueError(f"bad module file {v!r}")
        if not func and not perf:
            raise ValueError("a module pair needs a func or a perf file")
        out.append((func, perf))
    return out


def failed_ids(xml_path, module):
    """Node ids of the testcases with a failure or error in a junit xml, in order; None without the xml.
    A testcase outside the module (a collection error) stands for the whole module."""
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return None
    dotted = module[:-3].replace("/", ".")
    ids = []
    for case in root.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        cls, name = case.get("classname", ""), case.get("name", "")
        if cls == dotted:
            nid = f"{module}::{name}"
        elif cls.startswith(dotted + "."):
            nid = f"{module}::{cls[len(dotted) + 1:].replace('.', '::')}::{name}"
        else:
            nid = module
        if nid not in ids:
            ids.append(nid)
    return ids


def step(log):
    """{"summary", "rc"} of a run.sh log ("rc=<n>" is its last line); None when the step did not run."""
    try:
        with open(log, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    rc = None
    if lines and lines[-1].startswith("rc="):
        rc = int(lines.pop()[3:])
    line = next((m.group(1) for m in map(SUMMARY.match, reversed(lines)) if m), None)
    return {"summary": line, "rc": rc}


def crashed(result, ids):
    """A run.sh step result without a junit xml (ids None) whose rc says pytest did not end normally."""
    return ids is None and result is not None and result["rc"] not in NOT_A_CRASH


def rerun_args(log, xml_path, module):
    """What run.sh reruns after a functional run: the first RERUN_MAX failing ids, the whole module
    when the run crashed, else nothing."""
    ids = failed_ids(xml_path, module)
    if ids is None:
        return [module] if crashed(step(log), ids) else []
    return ids[:RERUN_MAX]


def func_result(base, module):
    """The functional step at <base>.log/.xml and its rerun at <base>.rerun.log/.xml, or None."""
    f = step(f"{base}.log")
    if f is None:
        return None
    ids = failed_ids(f"{base}.xml", module)
    f.update(failed_ids=ids, crashed=crashed(f, ids), rerun_truncated=ids is not None and len(ids) > RERUN_MAX)
    rerun = step(f"{base}.rerun.log")
    if rerun is not None:
        rids = failed_ids(f"{base}.rerun.xml", module)
        rerun.update(failed_ids=rids, crashed=crashed(rerun, rids))
    f["rerun"] = rerun
    return f


def summary(out, fail):
    sides, reps = os.environ["SIDES"].split(), int(os.environ["REPS"])
    doc = {"arch": os.environ["ARCH"], "runner": socket.gethostname(), "reps": reps, "fail": fail,
           "sides": {s: {"sha": os.environ.get(f"{s.upper()}_SHA"),
                         "setup_rc": (step(f"{out}/setup_{s}.log") or {}).get("rc")} for s in sides},
           "modules": []}
    for i, (func, perf) in enumerate(modules(os.environ["MODULES"]), 1):
        per_side = {}
        for s in sides:
            per_side[s] = {"func": func_result(f"{out}/m{i}_func_{s}", func) if func else None,
                           "perf_build": step(f"{out}/m{i}_perf_build_{s}.log") if perf else None,
                           "perf": [{"rep": r, **(step(f"{out}/m{i}_perf_{s}_r{r}.log") or {"summary": None, "rc": None})}
                                    for r in range(1, reps + 1)] if perf else []}
        doc["modules"].append({"func": func, "perf": perf, "sides": per_side})
    return doc


def main(argv):
    if argv[:1] == ["modules"] and len(argv) == 2:
        try:
            pairs = modules(argv[1])
        except ValueError as e:
            sys.exit(f"modules: {e}")
        sys.stdout.write("".join(f"{f}|{p}\n" for f, p in pairs))
    elif argv[:1] == ["rerun"] and len(argv) == 4:
        sys.stdout.write("".join(f"{i}\n" for i in rerun_args(*argv[1:])))
    elif argv[:1] == ["summary"] and len(argv) == 3:
        print(json.dumps(summary(argv[1], int(argv[2])), indent=1))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
