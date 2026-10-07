#!/usr/bin/env python3
"""Helpers for fixer.sh: ledger bookkeeping, eligibility, dispatch mapping,
diff guards and PR-body rendering. Python 3.8 (host interpreter) — keep it
free of 3.9+ syntax.

The ledger (ledger.json) is the local half of the no-duplicate guarantee: one
record per regression SIGNATURE (workflow :: job-without-SKU :: test id), not
per run or tick. GitHub is the other half (branch name + hidden body marker),
checked by fixer.sh before anything is created.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import sys

import yaml

HOME = os.path.expanduser("~/.sdpa-fix")
LEDGER = os.path.join(HOME, "ledger.json")

# States in which a signature may be (re)attempted. proposed_dryrun is added
# in live mode so a dry-run proposal graduates to a real PR exactly once.
ATTEMPTABLE = {"tracking"}
# States that end when main goes green for the signature.
OPEN_STATES = {"tracking", "proposed_dryrun", "no_fix", "fix_pending", "pr_open", "ci_passed", "ci_failed",
               "awaiting_decision", "decided", "with_owner"}
# States the fix scan looks at: still failing, and no PR of ours on it.
SCAN_STATES = {"tracking", "no_fix", "proposed_dryrun", "fix_pending", "awaiting_decision"}
# States a re-appearing failure re-opens (a fresh regression after a fix).
REOPENABLE = {"resolved_on_main", "verified"}
# A fix landed (our merged PR, or a commit already on main). The signature is
# only done once a run that CONTAINS fix_sha reports on it: green → verified,
# still red → back to tracking. Runs older than the fix change nothing.
FIX_PENDING = {"merged", "fixed_upstream"}
FIXABLE_KINDS = {"code_regression", "perf_threshold"}
OWNED = {"sdpa_op", "k3_model"}


def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def today():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- ledger I/O
class Ledger(object):
    """Read-modify-write under an exclusive flock so a manual run and a cron
    tick can never interleave writes."""

    def __enter__(self):
        os.makedirs(HOME, exist_ok=True)
        self._lock = open(LEDGER + ".lock", "w")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        if os.path.exists(LEDGER):
            with open(LEDGER) as f:
                self.d = json.load(f)
        else:
            self.d = {"version": 1, "sigs": {}, "triaged": {}, "daily": {}}
        return self.d

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            tmp = LEDGER + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=2, sort_keys=True)
            os.replace(tmp, LEDGER)
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()
        return False


def norm_job(name):
    """'Prefix (Box, sku) / Leg name [sku]' -> 'Leg name'. SKU is dropped on
    purpose: the same test failing on two boxes is one regression."""
    leg = name.rsplit(" / ", 1)[-1]
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", leg).strip()


def signature(workflow, job, test):
    key = "%s::%s::%s" % (workflow, norm_job(job), test.strip())
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _contains(run_sha, reason):
    """True when run_sha already contains the commit named in reason."""
    import subprocess
    m = re.search(r"\b[0-9a-f]{7,40}\b", reason or "")
    repo = os.environ.get("TT_METAL_DIR")
    if not m or not repo or not run_sha:
        return False
    return subprocess.call(["git", "-C", repo, "merge-base", "--is-ancestor", m.group(0), run_sha],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


# ------------------------------------------------------------ triage update
def cmd_update(a):
    """Fold one analyzed run into the ledger. failures = triage output (may be
    empty for a green run); jobs = [{name, conclusion}] for the whole run, used
    to tell 'test passed' from 'job never ran'."""
    with open(a.triage) as f:
        failures = json.load(f).get("failures", [])
    with open(a.jobs) as f:
        jobs = json.load(f)
    job_concl = {}
    for j in jobs:
        n = norm_job(j["name"])
        # A leg on several SKUs: any failure wins, then success.
        prev = job_concl.get(n)
        if prev != "failure":
            job_concl[n] = j.get("conclusion") or "pending"

    run = {"id": a.run_id, "number": int(a.run_number), "sha": a.sha, "url": a.url}
    newly_green = []
    events = []
    with Ledger() as d:
        sigs = d["sigs"]
        seen = set()
        for f in failures:
            sig = signature(a.workflow, f["job"], f["test"])
            seen.add(sig)
            r = sigs.get(sig)
            if r is None:
                r = sigs[sig] = {
                    "workflow": a.workflow, "job": norm_job(f["job"]), "test": f["test"],
                    "state": "tracking", "streak": 0, "runs": [], "attempts": [],
                    "first_seen": run, "created": now_iso(),
                }
            if r["state"] in FIX_PENDING and _contains(a.sha, r.get("fix_sha") or r.get("reason", "")):
                # The run already includes the fix and still fails: that fix
                # did not cover it. Back in the queue.
                events.append({"type": "still_failing", "sig": sig, "test": r["test"], "pr": r.get("pr"),
                               "fix_sha": r.get("fix_sha", ""), "run": run})
                r["state"] = "tracking"
                r["streak"] = 0
            if r["state"] in REOPENABLE:
                r["state"] = "tracking"
                r["streak"] = 0
                r["first_seen"] = run
                r.pop("pr", None)
            if run["number"] > (r.get("last_seen") or {}).get("number", -1):
                r["streak"] = r.get("streak", 0) + 1
                r["runs"] = (r.get("runs", []) + [run["number"]])[-20:]
            r["last_seen"] = run
            r["sku_jobs"] = sorted(set(r.get("sku_jobs", []) + [f["job"]]))
            for k in ("kind", "owner", "error_key", "summary", "culprit_sha",
                      "culprit_evidence", "fixable", "group"):
                r[k] = f.get(k)
            r["updated"] = now_iso()

        # Signatures of this workflow absent from this run.
        for sig, r in sigs.items():
            if r["workflow"] != a.workflow or sig in seen:
                continue
            if run["number"] <= (r.get("last_seen") or {}).get("number", -1):
                continue
            concl = job_concl.get(r["job"])
            if concl == "success" and r["state"] in FIX_PENDING:
                r["streak"] = 0
                if _contains(a.sha, r.get("fix_sha") or r.get("reason", "")):
                    r["state"] = "verified"
                    r["verified"] = {"run": run, "at": now_iso()}
                    r["updated"] = now_iso()
                    events.append({"type": "verified", "sig": sig, "test": r["test"], "pr": r.get("pr"),
                                   "fix_sha": r.get("fix_sha", ""), "run": run})
                # green but the fix is not in this run yet: keep waiting
            elif concl == "success":
                r["streak"] = 0
                if r["state"] in OPEN_STATES:
                    newly_green.append({"sig": sig, "state": r["state"], "pr": r.get("pr"),
                                        "test": r["test"], "job": r["job"], "run": run})
                    r["state"] = "resolved_on_main"
                    r["resolved"] = {"run": run, "at": now_iso(), "prev_state": newly_green[-1]["state"]}
                    r["updated"] = now_iso()
            # failure without this test (infra, another test), cancelled,
            # skipped or not finished yet: no evidence either way.
        # Per-job bookkeeping: which jobs of the current run are triaged, and
        # the latest run in which each job finished (eligibility uses it to
        # tell "still failing" from "its job has not run again yet").
        t = d["triaged"].get(a.workflow) or {}
        ids = t.get("jobs", []) if t.get("run_id") == a.run_id else []
        new_ids = [x for x in (a.job_ids.split(",") if a.job_ids else []) if x]
        d["triaged"][a.workflow] = {"run_id": a.run_id, "number": run["number"], "at": now_iso(),
                                   "jobs": sorted(set(ids + new_ids))}
        jr = d.setdefault("job_runs", {}).setdefault(a.workflow, {})
        for n in job_concl:
            if job_concl[n] in ("success", "failure"):
                jr[n] = max(jr.get(n, 0), run["number"])
    json.dump({"newly_green": newly_green, "events": events, "seen": sorted(seen)}, sys.stdout)


def cmd_triaged(a):
    """Without --run-id: the last triaged run id. With it: JSON list of that
    run's already-triaged job ids, or "ALL" for a run triaged as a whole by
    the pre-per-job fixer."""
    with Ledger() as d:
        t = d["triaged"].get(a.workflow) or {}
        if not a.run_id:
            print(t.get("run_id", ""))
        elif t.get("run_id") != a.run_id:
            print("[]")
        elif "jobs" not in t:
            print('"ALL"')
        else:
            print(json.dumps(t["jobs"]))


# --------------------------------------------------------------- eligibility
def cmd_eligible(a):
    """Emit fixable groups, best first, honouring the daily cap."""
    attemptable = set(ATTEMPTABLE)
    if a.mode == "live":
        attemptable.add("proposed_dryrun")
    with Ledger() as d:
        used = d["daily"].get(today(), 0)
        latest = {w: (t or {}).get("number") for w, t in d["triaged"].items()}
        # A human picked an option ("decided"), or a manual run forces one
        # signature: these go first and bypass kind / streak / daily cap.
        forced = []
        for sig, r in d["sigs"].items():
            if (r["state"] == "decided" and r.get("decision_choice")) or \
               (a.only_sig and sig == a.only_sig and r["state"] in ("tracking", "no_fix", "proposed_dryrun", "decided")):
                forced.append({"workflow": r["workflow"], "group": r.get("group") or sig, "sigs": [sig],
                               "prio": [-1, 0], "forced": True})
        if a.only_sig:
            json.dump({"groups": [g for g in forced if a.only_sig in g["sigs"]], "daily_used": used,
                       "daily_room": max(0, a.max_per_day - used)}, sys.stdout)
            return
        groups = {}
        for sig, r in d["sigs"].items():
            if r["state"] not in attemptable:
                continue
            if r.get("kind") not in FIXABLE_KINDS or r.get("owner") not in OWNED or not r.get("fixable"):
                continue
            last = (r.get("last_seen") or {}).get("number")
            jr = d.get("job_runs", {}).get(r["workflow"], {}).get(r["job"])
            if jr is not None:
                if last is None or last < jr:
                    continue  # its job ran again since and this test passed
            elif last != latest.get(r["workflow"]):
                continue  # not failing in the latest analyzed run any more
            if r.get("streak", 0) < a.min_streak and not r.get("culprit_sha"):
                continue
            key = (r["workflow"], r.get("group") or sig)
            groups.setdefault(key, []).append(sig)
        out = []
        for (wf, grp), sl in groups.items():
            recs = [d["sigs"][s] for s in sl]
            out.append({
                "workflow": wf, "group": grp, "sigs": sorted(sl),
                "prio": (0 if any(r["kind"] == "code_regression" for r in recs) else 1,
                         -max(r.get("streak", 0) for r in recs)),
            })
        out.sort(key=lambda g: g["prio"])
        room = max(0, a.max_per_day - used)
        fsigs = set(x for g in forced for x in g["sigs"])
        out = [g for g in out if not (set(g["sigs"]) & fsigs)]
        json.dump({"groups": forced + out[:room], "daily_used": used, "daily_room": room}, sys.stdout)


def cmd_get(a):
    with Ledger() as d:
        json.dump({s: d["sigs"][s] for s in a.sigs}, sys.stdout)


def cmd_mark(a):
    """Record an attempt outcome for a set of signatures."""
    extra = json.loads(a.extra) if a.extra else {}
    with Ledger() as d:
        for s in a.sigs:
            r = d["sigs"][s]
            if a.state != "keep":
                r["state"] = a.state
            r["updated"] = now_iso()
            if a.attempt:
                r.setdefault("attempts", []).append(dict(extra, at=now_iso(), state=a.state))
            for k, v in extra.items():
                if k in ("pr", "proposal", "dispatched", "verdict_title", "reason", "fix_sha", "slack_ts", "fix_pr", "fix_author",
                         "decision", "decision_choice"):
                    r[k] = v
                elif k == "checked":
                    r.setdefault("checked", {}).update(v)
        if a.count_daily:
            d["daily"][today()] = d["daily"].get(today(), 0) + 1
            # keep two weeks of counters
            for k in sorted(d["daily"])[:-14]:
                del d["daily"][k]


FILE_RE = re.compile(r"[\w./-]+\.(?:py|sh|cpp|hpp|h|cc|yaml|yml|json|txt)\b")
GENERIC = {"conftest.py", "__init__.py", "utils.py", "common.py", "test.py", "run.sh"}


def _scan_paths(r, repo):
    """Repo files a fix for this failure would likely touch: the test file,
    plus files named in the test label / summary (basenames resolved with
    git ls-files)."""
    import subprocess
    paths, cands = [], []
    t = r.get("test", "")
    if "::" in t and "/" in t.split("::")[0]:
        paths.append(t.split("::")[0].strip())
    for txt in (t, r.get("summary") or "", r.get("error_key") or ""):
        cands += FILE_RE.findall(txt)
    files = None
    for c in cands:
        c = c.strip("./")
        if "/" in c and os.path.exists(os.path.join(repo, c)):
            paths.append(c)
            continue
        base = os.path.basename(c)
        if base in GENERIC or len(base) < 6:
            continue
        if files is None:
            try:
                files = subprocess.check_output(["git", "-C", repo, "ls-files"], text=True).splitlines()
            except Exception:
                files = []
        hits = [f for f in files if os.path.basename(f) == base]
        if 0 < len(hits) <= 3:
            paths += hits
    out = []
    for x in paths:
        if x not in out:
            out.append(x)
    return out[:6]


def cmd_scan_list(a):
    """Failures to scan for human-made fixes: still failing, in a scan state."""
    repo = os.environ.get("TT_METAL_DIR", ".")
    with Ledger() as d:
        out = []
        for sig, r in d["sigs"].items():
            if r["state"] not in SCAN_STATES:
                continue
            last = (r.get("last_seen") or {}).get("number")
            jr = d.get("job_runs", {}).get(r["workflow"], {}).get(r["job"])
            if jr is not None and (last is None or last < jr):
                continue  # passed since; nothing to look for
            fn = r["test"].split("::")[-1].split("[")[0]
            terms = []
            if re.match(r"^test_\w{6,}$", fn):
                terms.append(fn)
            paths = _scan_paths(r, repo)
            for pth in paths:
                b = os.path.basename(pth)
                if b not in terms and len(b) > 8 and b not in GENERIC:
                    terms.append(b)
            out.append({"sig": sig, "workflow": r["workflow"], "job": r["job"], "test": r["test"],
                        "state": r["state"], "summary": r.get("summary") or r.get("error_key") or "",
                        "first_sha": (r.get("first_seen") or {}).get("sha", ""),
                        "first_at": r.get("created", ""), "paths": paths, "terms": terms[:3],
                        "fix_pr": r.get("fix_pr"), "checked": r.get("checked", {})})
        json.dump(out, sys.stdout)


def cmd_list(a):
    with Ledger() as d:
        rows = []
        for s, r in sorted(d["sigs"].items(), key=lambda kv: kv[1].get("updated", "")):
            rows.append("%s  %-16s streak=%-2s %-15s %-8s %s :: %s%s" % (
                s, r["state"], r.get("streak", 0), r.get("kind"), r.get("owner"),
                r["workflow"].replace(".yaml", ""), r["test"][-90:],
                ("  -> " + r["pr"]["url"]) if r.get("pr") else ""))
        print("\n".join(rows) if rows else "(ledger empty)")


# ----------------------------------------------------------------- dispatch
def _blaze(job, repo):
    with open(os.path.join(repo, "tests/pipeline_reorg/blaze_models_prefill_tests.yaml")) as f:
        rows = yaml.safe_load(f)
    for r in rows:
        if isinstance(r, dict) and r.get("name", "").strip() == norm_job(job) and r.get("test_type"):
            return {"test-type": r["test_type"]}
    return None


def _bh_e2e(job, repo):
    with open(os.path.join(repo, "tests/pipeline_reorg/blackhole_e2e_tests.yaml")) as f:
        rows = yaml.safe_load(f)
    inputs = {}
    m = re.match(r"^[^(]*\((.+), ([a-z0-9_]+)\) / ", job)
    if m:
        inputs["system-type"] = m.group(1)
    with open(os.path.join(repo, ".github/workflows/blackhole-e2e-tests.yaml")) as f:
        wf = yaml.safe_load(f)
    win = (wf.get(True) or wf.get("on"))["workflow_dispatch"]["inputs"]
    for r in rows:
        if isinstance(r, dict) and str(r.get("name", "")).strip() == norm_job(job):
            if r.get("id") in (win["test-selection"].get("options") or []):
                inputs["test-selection"] = r["id"]
            elif r.get("model-name") in (win["model"].get("options") or []):
                inputs["model"] = r["model-name"]
            return inputs
    return inputs or None


def _sanity(job, repo):
    inputs = {k: "false" for k in (
        "run-sim", "run-t3000-sanity-tests", "run-fabric-sanity-tests", "run-ops-sanity-tests",
        "run-umd-sanity-tests", "run-ttsim-sanity-tests", "run-models-sanity-tests",
        "run-ttnn-sanity-tests", "run-blackhole-multi-card-sanity-tests", "build-inplace-wheel")}
    sku = (re.search(r"\[([a-z0-9_]+)\]\s*$", job) or [None, ""])[1]
    if "QB2 only" in job or sku == "bh_quietbox_2":
        inputs["run-blackhole-multi-card-sanity-tests"] = "true"
    else:
        inputs["run-ttnn-sanity-tests"] = "true"
    if sku.startswith("sim_"):
        inputs["run-sim"] = inputs["run-ttsim-sanity-tests"] = "true"
    inputs["run-wormhole"] = "true" if sku.startswith("wh_") else "false"
    inputs["run-blackhole"] = "true" if sku.startswith("bh_") else "false"
    return inputs


def _l2(job, repo):
    return {"run_wormhole": "true" if "wormhole" in job or "wh_" in job else "false",
            "run_blackhole": "true" if "blackhole" in job or "bh_" in job else "false",
            "run_triage_tests": "false"}


def _t3k(job, repo):
    return {"run-unit-tests": "false", "run-integration-tests": "false", "run-e2e-tests": "true"}


RESOLVERS = {
    "blaze-models-prefill-tests.yaml": _blaze,
    "blackhole-e2e-tests.yaml": _bh_e2e,
    "sanity-tests.yaml": _sanity,
    "sanity-tests-debug.yaml": _sanity,
    "tt-metal-l2-nightly.yaml": _l2,
    "t3000-tests.yaml": _t3k,
}


def cmd_dispatch(a):
    """Resolve the failing jobs of a group into a de-duplicated list of
    `gh workflow run` invocations (one per distinct input set)."""
    jobs = json.loads(a.jobs)
    res = RESOLVERS.get(a.workflow)
    plans, seen = [], set()
    for j in jobs:
        try:
            inputs = res(j, a.repo) if res else None
        except Exception as e:  # malformed matrix etc. — degrade to manual
            inputs = None
            sys.stderr.write("dispatch resolve failed for %r: %s\n" % (j, e))
        if inputs is None:
            plans.append({"workflow": a.workflow, "job": j, "inputs": None})
            continue
        key = json.dumps(inputs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        plans.append({"workflow": a.workflow, "job": j, "inputs": inputs})
    json.dump(plans, sys.stdout)


# -------------------------------------------------------------------- guard
SKIP_RE = re.compile(r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|pytest\.xfail\(|@skip_for_|\bskipif\b")
# Comment lines that log history (dates, run/job ids, "re-centred …"): git and
# the PR keep that, the code keeps only the durable rule.
HISTORY_RE = re.compile(r"^\s*(#|//|\*).*(\b20\d\d-\d\d-\d\d\b|\b(run|job)s? \d{8,}|re-?cent(e|r)?red|re-?baselined)", re.I)
THRESH_WORDS = re.compile(r"pcc|atol|rtol|toleran|threshold|band|margin|expected|_ns\b|lower|upper|perf|bound", re.I)
NUM = re.compile(r"\d+(\.\d+)?(e-?\d+)?")


def cmd_guard(a):
    with open(a.verdict) as f:
        v = json.load(f)
    with open(a.diff) as f:
        diff = f.read()
    reasons, files, added, deleted = [], [], 0, 0
    thresh_hits = []
    cur = None
    minus_nums = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            files.append(cur)
            continue
        if line.startswith("--- ") or line.startswith("diff --git") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
            body = line[1:]
            if SKIP_RE.search(body):
                reasons.append("adds a skip/xfail in %s: %s" % (cur, body.strip()[:120]))
            if HISTORY_RE.search(body):
                reasons.append("adds a history comment in %s: %s" % (cur, body.strip()[:120]))
            if cur and THRESH_WORDS.search(body) and NUM.search(body) and minus_nums:
                thresh_hits.append("%s: %s" % (cur, body.strip()[:140]))
        elif line.startswith("-"):
            deleted += 1
            body = line[1:]
            if THRESH_WORDS.search(body) and NUM.search(body):
                minus_nums.append(body)
        else:
            minus_nums = []
    if v.get("fixed") and not diff.strip():
        reasons.append("verdict says fixed but the diff is empty")
    if added + deleted > a.max_lines:
        reasons.append("diff too large: %d changed lines > %d" % (added + deleted, a.max_lines))
    for p in files:
        if p.startswith(".github/"):
            reasons.append("touches CI config: %s" % p)
    if "/dev/null" in diff and re.search(r"^deleted file mode", diff, re.M):
        reasons.append("deletes a file")
    threshold_detected = bool(thresh_hits)
    if threshold_detected and not v.get("threshold_change"):
        reasons.append("diff looks like a threshold change but verdict.threshold_change=false: "
                       + "; ".join(thresh_hits[:3]))
    json.dump({"ok": not reasons, "reasons": reasons, "files": files,
               "added": added, "deleted": deleted,
               "threshold_detected": threshold_detected, "threshold_hits": thresh_hits[:5]},
              sys.stdout)


# --------------------------------------------------------------- PR render
def _shield(text):
    """Escape one shields.io static-badge field (dash/underscore doubled)."""
    from urllib.parse import quote
    return quote(text.replace("-", "--").replace("_", "__"), safe="")


def _short_test(test):
    """'path/to/test_x.py::test_fn[blackhole-k3-torus]' -> 'test_fn[blackhole-k3-torus]'."""
    return test.split("::")[-1] if "::" in test else test


def cmd_render(a):
    """PR body. House style: short, straight, scannable. One section each for
    what regressed, why, what changed, how it is being validated, and what a
    reviewer must check; badges link the failing run and the validation runs.
    No history, no narrative: git and the linked runs hold the rest."""
    with open(a.meta) as f:
        m = json.load(f)
    v, recs, repo = m["verdict"], m["records"], m["repo"]
    gh = "https://github.com/" + repo
    from urllib.parse import quote
    L = ["> [!WARNING]",
         "> Automated fix, not run on hardware. Needs human review before merge."]
    if v.get("threshold_change"):
        L.append("> This PR changes a test threshold.")
    L += ["", "### Regression"]
    ls = recs[0].get("last_seen") or {}
    pipe = m.get("pipeline") or recs[0]["workflow"].replace(".yaml", "")
    label = "%s #%s" % (pipe, ls.get("number", "?"))
    L.append("[![%s](https://img.shields.io/badge/%s-failed-d73a49?logo=githubactions&logoColor=white)](%s)"
             % (label, _shield(label), ls.get("url", "")))
    L.append("")
    tests = sorted(set(_short_test(r["test"]) for r in recs))
    reg = v.get("regression") or "; ".join(sorted(set(r.get("summary") or r.get("error_key") or "" for r in recs)))
    L.append("%s: %s" % (", ".join("`%s`" % t for t in tests), reg))
    L += ["", "### Why", v.get("why") or v["root_cause"]]
    L += ["", "### Fix", v.get("fix") or v["change_summary"]]
    L += ["", "### Validation"]
    branch = m.get("branch") or ""
    for p in m.get("dispatch", []):
        wf = p["workflow"]
        if p.get("inputs") is None:
            L.append("- No dispatch mapping for job *%s*: run it manually." % p["job"])
            continue
        badge = "%s/actions/workflows/%s/badge.svg?branch=%s&event=workflow_dispatch" % (gh, wf, quote(branch, safe=""))
        target = p.get("run_url") or "%s/actions/workflows/%s?query=branch%%3A%s" % (gh, wf, quote(branch, safe=""))
        L.append("[![%s](%s)](%s) `%s` on this branch" % (wf.replace(".yaml", ""), badge, target, _fmt_inputs(p["inputs"])))
    L += ["", "### Review"]
    for item in v.get("reviewer_focus") or []:
        L.append("- [ ] " + item)
    if m.get("related_prs"):
        nums = [("#" + x.group(1)) for x in (re.search(r"/pull/(\d+)", r) for r in m["related_prs"]) if x]
        if nums:
            L.append("- [ ] Possibly related: " + ", ".join(nums))
    L.append("")
    for s_ in m["sigs"]:
        L.append("<!-- autofixsig%s -->" % s_)
    L += ["", "🤖 Generated with [Claude Code](https://claude.com/claude-code)"]
    sys.stdout.write("\n".join(L) + "\n")


def _fmt_inputs(inp):
    return " ".join("%s=%s" % (k, v) for k, v in sorted((inp or {}).items()))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd")
    u = sp.add_parser("update")
    for k in ("workflow", "run_id", "run_number", "sha", "url", "triage", "jobs"):
        u.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    u.add_argument("--job-ids", default="", help="comma-separated ids of the jobs this update covers")
    t = sp.add_parser("triaged"); t.add_argument("--workflow", required=True)
    t.add_argument("--run-id", default="")
    e = sp.add_parser("eligible")
    e.add_argument("--mode", required=True)
    e.add_argument("--min-streak", type=int, required=True)
    e.add_argument("--max-per-day", type=int, required=True)
    e.add_argument("--only-sig", default="")
    g = sp.add_parser("get"); g.add_argument("sigs", nargs="+")
    mk = sp.add_parser("mark")
    mk.add_argument("--state", required=True)
    mk.add_argument("--extra", default="")
    mk.add_argument("--attempt", action="store_true")
    mk.add_argument("--count-daily", action="store_true")
    mk.add_argument("sigs", nargs="+")
    sp.add_parser("list")
    sp.add_parser("scan-list")
    dp = sp.add_parser("dispatch")
    dp.add_argument("--workflow", required=True)
    dp.add_argument("--jobs", required=True, help="JSON list of full GH job names")
    dp.add_argument("--repo", required=True)
    gu = sp.add_parser("guard")
    gu.add_argument("--verdict", required=True)
    gu.add_argument("--diff", required=True)
    gu.add_argument("--max-lines", type=int, required=True)
    rd = sp.add_parser("render"); rd.add_argument("--meta", required=True)
    a = p.parse_args()
    {"update": cmd_update, "triaged": cmd_triaged, "eligible": cmd_eligible, "get": cmd_get,
     "mark": cmd_mark, "list": cmd_list, "dispatch": cmd_dispatch, "guard": cmd_guard,
     "render": cmd_render, "scan-list": cmd_scan_list}[a.cmd](a)


if __name__ == "__main__":
    main()
