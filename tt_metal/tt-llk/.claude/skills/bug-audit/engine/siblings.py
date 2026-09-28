#!/usr/bin/env python3
"""Turn the unfixed siblings named by mined deep reads into leads for an audit run to verify.

  siblings.py [--run DIR] from-deep DEEP.jsonl[=REPO] [DEEP.jsonl[=REPO] ...] [--include-unsure] [--batch 20]
  siblings.py [--run DIR] show

A deep read of a past fix (mining/deep-wave.js) lists the fix's siblings in the CURRENT tree, each with a status:
fixed, unfixed, unsure or not-applicable. An `unfixed` sibling is a lead: the analyst who read the fix says the same
defect is still present there. This is how the history finds bugs the hunt does not: a fix lands in one arch, dtype or
overload, and its copies elsewhere keep the bug. It needs the deep reads, not the pack.

`from-deep` merges the leads by location -- one lead per file:line, carrying every past case that names it -- and
writes them into the run as uncertain findings (source `history-sibling`, batches S-NNNN). `recheck.py queue` then
verifies them exactly like an audit's own unsettled candidates. A lead is never filed as it stands: only a verified one.
A location the analyst gave in words ("the trisc kernel entry") has no file:line; it keeps a unique key built from its
text, so two such leads never collide on one key.

REPO labels the source repo in each lead's text (default: the deep file's name). Re-running replaces the leads.
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, run_dir, save, state  # noqa: E402

out = run_dir()
st = state(out)
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)

PATH = re.compile(r"^\s*([^\s:{}()*]+\.[A-Za-z0-9]+)(?=[:\s(]|$)")
LINE = re.compile(r":(\d+)")
PLACEHOLDER_FIX = "Not written yet: a site-grounded fix is written once the lead is verified (suggested_fixes.json)."


def parse_location(loc):
    """(file, line) for a `path[:line[-line]]` location, or (None, 0) for one given in words."""
    m = PATH.match(loc)
    if not m:
        return None, 0
    ln = LINE.search(loc[m.end() :])
    return m.group(1), int(ln.group(1)) if ln else 0


def leads_from(paths, include_unsure):
    """{key: {"file", "line", "loc_raw", "cases": [...]}} merged by location."""
    wanted = {"unfixed", "unsure"} if include_unsure else {"unfixed"}
    leads, n_entries = {}, 0
    for spec in paths:
        path, _, repo = spec.partition("=")
        repo = repo or os.path.basename(path).split("_")[0]
        for line in open(path):
            if not line.strip():
                continue
            case = json.loads(line)
            for sib in case.get("siblings") or []:
                if sib.get("status") not in wanted:
                    continue
                n_entries += 1
                loc = (sib.get("location") or "").strip()
                file, ln = parse_location(loc)
                # an unlocated lead keeps its own text as its file, so it cannot share a key with another
                key = f"{file}:{ln}" if file else f"(unlocated) {loc}:0"
                lead = leads.setdefault(
                    key,
                    {
                        "file": file or f"(unlocated) {loc}",
                        "line": ln,
                        "loc_raw": loc,
                        "cases": [],
                    },
                )
                lead["cases"].append(
                    {
                        "repo": repo,
                        "case": case,
                        "why": sib.get("why", ""),
                        "status": sib["status"],
                    }
                )
    return leads, n_entries


def finding(lead, batch):
    first = lead["cases"][0]
    c = first["case"]
    reasons = [
        f"[history-mining analyst, from {x['repo']} case {x['case'].get('id')}] {x['why']}"
        for x in lead["cases"]
    ]
    return {
        "file": lead["file"],
        "line": lead["line"],
        "category": c.get("primary_class") or "history-sibling",
        "severity": "medium",  # a placeholder: re-rate the verified ones with severity-wave.js
        "summary": f"Possible unfixed copy of a past bug: {first['why']}",
        "failure_scenario": f"Past bug ({first['repo']}): {c.get('root_cause', '')} Trigger: {c.get('trigger', '')}",
        "evidence": f"Past fix ({c.get('fix_verdict', '?')}): {c.get('fix_summary', '')} Audit check: {c.get('audit_check', '')}",
        "suggested_fix": PLACEHOLDER_FIX,
        "source": "history-sibling",
        "batch": batch,
        "status": "uncertain",
        "reasons": reasons,
        "location": lead["loc_raw"],
    }


if argv[0] == "from-deep":
    size = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else 20
    paths = [a for a in argv[1:] if not a.startswith("--") and not a.isdigit()]
    if not paths:
        sys.exit("from-deep needs at least one deep-read JSONL")
    leads, n_entries = leads_from(paths, "--include-unsure" in argv)
    vdir = os.path.join(out, "verdicts")
    # a sweep can be its own run (no hunt): give it the layout the rest of the engine reads
    for sub in ("verdicts", "findings", "batches"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    if not os.path.exists(os.path.join(out, "batches", "manifest.json")):
        save(os.path.join(out, "batches", "manifest.json"), [])
    for fn in os.listdir(
        vdir
    ):  # re-running replaces the previous leads, never duplicates them
        if fn.startswith("S-") and fn.endswith(".json"):
            os.remove(os.path.join(vdir, fn))
    keys = sorted(leads)
    for n, i in enumerate(range(0, len(keys), size)):
        batch = f"S-{n:04d}"
        save(
            os.path.join(vdir, f"{batch}.json"),
            {
                "batch": batch,
                "wave": "siblings",
                "findings": [finding(leads[k], batch) for k in keys[i : i + size]],
            },
        )
    by_repo = defaultdict(int)
    for lead in leads.values():
        by_repo[lead["cases"][0]["repo"]] += 1
    located = sum(
        1 for lead in leads.values() if not lead["file"].startswith("(unlocated)")
    )
    print(
        f"{n_entries} sibling entries -> {len(leads)} leads ({located} at a file:line, {len(leads) - located} in words) "
        f"in {(len(keys) + size - 1) // size} batch(es); by first source: {dict(by_repo)}.\n"
        "Next: recheck.py queue --to-dir DIR [--max 330], then recheck-wave.js, recheck.py persist, consolidate.py."
    )
elif argv[0] == "show":
    vdir = os.path.join(out, "verdicts")
    n = sum(
        len(load(os.path.join(vdir, fn), {}).get("findings", []))
        for fn in sorted(os.listdir(vdir))
        if fn.startswith("S-")
    )
    print(f"{n} history-sibling lead(s) in {out}")
else:
    sys.exit(__doc__)
