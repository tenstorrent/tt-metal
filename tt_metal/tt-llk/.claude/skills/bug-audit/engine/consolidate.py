#!/usr/bin/env python3
"""Rebuild the derived reports of a run from verdicts/*.json, recheck.json and dispositions.json.

  consolidate.py [--run DIR]

Writes (all DERIVED — never annotate them by hand, the next run overwrites them):
  CONFIRMED.json / CONFIRMED.md   confirmed findings, deduplicated by file:line, with dispositions stamped in; a bug
                                  present at several sites (e.g. arch copies) is MERGED into one entry that carries
                                  every site's content; the merged-in rows stay listed, marked "merged into"
  UNCERTAIN.md                    findings no verifier could settle — surfaced, never filed, never dropped
  OPEN.md                         confirmed findings nothing has been done about yet: the work queue
  REFUTED.md                      refuted candidates, so later runs do not re-raise them
  COVERAGE.md                     findings per class and per priority, and the classes that produced nothing
Record outcomes with disposition.py, never in these files.
"""
import collections
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEV_ORDER, key_of, load, manifest, run_dir, save, state  # noqa: E402

ACTIVE = ("in_progress", "pr_open")
DONE = (
    "merged",
    "wont_fix",
    "not_a_bug",
    "duplicate",
    "already_filed",
    "fixed_upstream",
)
ATTENTION = ("pr_closed", "filed_closed")

out = run_dir()
st = state(out)
man = manifest(out)
rows = []
for fn in sorted(os.listdir(os.path.join(out, "verdicts"))):
    if fn.endswith(".json"):
        rows += load(os.path.join(out, "verdicts", fn), {}).get("findings", [])
# a recheck overrides the wave verdict of the candidate it re-examined
recheck = load(os.path.join(out, "recheck.json"), {})
for f in rows:
    rc = recheck.get(key_of(f))
    if rc and rc.get("outcome") in ("confirmed", "refuted", "uncertain"):
        f["status_wave"], f["status"] = f["status"], rc["outcome"]
        f["recheck"] = rc

by_status = collections.defaultdict(dict)
for f in rows:
    k = key_of(f)
    cur = by_status[f["status"]].get(k)
    if cur is None or f.get("votes", {}).get("confirmed", 0) > cur.get("votes", {}).get(
        "confirmed", 0
    ):
        by_status[f["status"]][k] = f
conf = list(by_status["confirmed"].values())
unc = {
    k: f
    for s in ("needs_recheck", "uncertain")
    for k, f in by_status[s].items()
    if k not in by_status["confirmed"]
}
unc = list(unc.values())
ref = [f for k, f in by_status["refuted"].items() if k not in by_status["confirmed"]]

# semantic dedup (dedup.py + dedup-wave.js): duplicates hang under their canonical finding, never silently dropped
dj = load(os.path.join(out, "dedup.json"), {"auto": {}, "clusters": []})
dup_of = dict(dj.get("auto", {}))
relation = {k: "same-defect" for k in dup_of}
for c in dj.get("clusters", []):
    for k in c["duplicates"]:
        dup_of[k] = c["canonical"]
        relation[k] = c["relation"]
confk = {key_of(f): f for f in conf}
for f in conf:
    k = key_of(f)
    if k in dup_of and dup_of[k] in confk and dup_of[k] != k:
        f["duplicate_of"], f["dup_relation"] = dup_of[k], relation[k]
        canon = confk[dup_of[k]]
        canon.setdefault("also_at", []).append(f"{k} ({relation[k]})")
        # MERGED, never dropped: the merged entry carries every site's own content, since per-arch copies can
        # differ in how they fail and how to fix them
        canon.setdefault("merged_sites", []).append(
            {
                "site": k,
                "relation": relation[k],
                "severity": f["severity"],
                "summary": f["summary"],
                "failure_scenario": f["failure_scenario"],
                "suggested_fix": f["suggested_fix"],
            }
        )
for (
    f
) in (
    conf
):  # a merged bug is as severe as its worst site: a HIGH copy must not hide behind a MEDIUM one
    for m in f.get("merged_sites", []):
        if SEV_ORDER.get(m["severity"], 3) < SEV_ORDER.get(f["severity"], 3):
            f.setdefault("severity_own", f["severity"])
            f["severity"] = m["severity"]
disp = load(os.path.join(out, "dispositions.json"), {})
for f in conf:
    d = disp.get(key_of(f))
    f["disposition"] = d or None
    if d and d.get("severity_override"):
        f["severity_audit"], f["severity"] = f["severity"], d["severity_override"]
srt = lambda fs: sorted(
    fs, key=lambda f: (SEV_ORDER.get(f["severity"], 3), f["file"], f["line"])
)  # noqa: E731
conf, unc, ref = srt(conf), srt(unc), srt(ref)
save(os.path.join(out, "CONFIRMED.json"), conf)

state_of = lambda f: (f.get("disposition") or {}).get("state")  # noqa: E731
open_f = [
    f
    for f in conf
    if (state_of(f) is None or state_of(f) in ATTENTION) and not f.get("duplicate_of")
]
act_f = [f for f in conf if state_of(f) in ACTIVE]
done_f = [f for f in conf if state_of(f) in DONE]
stamp = f"Regenerated {datetime.datetime.now():%Y-%m-%d %H:%M} from `{out}` · tree `{st['root']}` @ `{st['commit'][:11]}`"


def tag(f):
    d = f.get("disposition")
    if not d:
        return "open"
    return d.get("state", "open") + (f" #{d['pr']}" if d.get("pr") else "")


def table(fh, fs):
    fh.write(
        "| # | Sev | Status | Location | Class | Summary |\n|---|---|---|---|---|---|\n"
    )
    for i, f in enumerate(fs, 1):
        s = f["summary"].replace("|", "\\|")[:150]
        loc = f"`{key_of(f)}`" + "".join(
            f"<br>+ `{x.split(' (')[0]}`" for x in f.get("also_at", [])
        )  # every site, e.g. WH and BH
        fh.write(
            f"| {i} | {f['severity'].upper()} | {tag(f)} | {loc} | {f['category']} | {s} |\n"
        )


def detail(fh, fs):
    for i, f in enumerate(fs, 1):
        fh.write(
            f"### {i}. [{f['severity'].upper()}] `{key_of(f)}` — {f['category']}\n\n**{f['summary']}**\n\n"
        )
        fh.write(
            f"*Failure scenario:* {f['failure_scenario']}\n\n*Evidence:* {f['evidence']}\n\n"
        )
        fh.write(f"*Suggested fix:* {f['suggested_fix']}\n\n")
        if f.get("merged_sites"):
            fh.write(
                f"*Merged sites: this ONE bug is present at {1 + len(f['merged_sites'])} sites; every site needs its fix:*\n\n"
                f"- `{key_of(f)}` (this entry)\n"
            )
            for m in f["merged_sites"]:
                fh.write(
                    f"- `{m['site']}` ({m['relation']}, {m['severity']}): {m['summary']}\n"
                    f"  - fails: {m['failure_scenario']}\n  - fix: {m['suggested_fix']}\n"
                )
            fh.write("\n")
        if f.get("duplicate_of"):
            fh.write(
                f"*Merged into* `{f['duplicate_of']}` ({f['dup_relation']}): tracked and filed as one bug with it\n\n"
            )
        v = f.get("votes", {})
        fh.write(
            f"*Votes:* {v.get('confirmed', 0)} confirmed / {v.get('refuted', 0)} refuted / "
            f"{v.get('uncertain', 0)} uncertain / {v.get('died', 0)} died · source `{f.get('source', '?')}` · "
            f"batch `{f['batch']}`\n\n"
        )
        if f.get("reasons") and f["status"] != "confirmed":
            fh.write(
                "*Verifier reasons:*\n"
                + "".join(f"- {r[:400]}\n" for r in f["reasons"])
                + "\n"
            )
        d = f.get("disposition")
        if d:
            fh.write(
                f"*Disposition:* **{d.get('state', 'open')}** "
                + " · ".join(
                    x
                    for x in (
                        f"PR #{d['pr']}" if d.get("pr") else "",
                        f"issue #{d['issue']}" if d.get("issue") else "",
                        d.get("note", ""),
                    )
                    if x
                )
                + "\n\n"
            )
        fh.write("---\n\n")


c = collections.Counter(f["severity"] for f in conf)
with open(os.path.join(out, "CONFIRMED.md"), "w") as fh:
    fh.write(f"# {st['name']} — confirmed findings\n\n{stamp}\n\n")
    fh.write(
        f"**{len(conf)} confirmed** — {c['high']} high, {c['medium']} medium, {c['low']} low. "
        f"{len(unc)} uncertain (UNCERTAIN.md), {len(ref)} refuted (REFUTED.md).\n\n"
        f"Status: **{len(open_f)} open**, {len(act_f)} in flight, {len(done_f)} closed.\n\n"
    )
    table(fh, conf)
    fh.write("\n---\n\n## Detail\n\n")
    detail(fh, conf)
with open(os.path.join(out, "OPEN.md"), "w") as fh:
    fh.write(
        f"# {st['name']} — OPEN findings (nothing done about these yet)\n\n{stamp}\n\n"
    )
    fh.write(
        f"{len(open_f)} open of {len(conf)} confirmed · {len(act_f)} in flight · {len(done_f)} closed.\n\n"
    )
    table(fh, open_f)
    fh.write("\n---\n\n## Detail\n\n")
    detail(fh, open_f)
    att = [f for f in open_f if state_of(f) in ATTENTION]
    if att:
        fh.write(
            "## Needs attention: a PR closed unmerged, or the bug was reported and closed yet is still present\n\n"
        )
        table(fh, att)
        fh.write("\n")
    if act_f:
        fh.write("## In flight — do not start these\n\n")
        table(fh, act_f)
    if done_f:
        fh.write("\n## Closed — never re-raise\n\n")
        table(fh, done_f)
with open(os.path.join(out, "UNCERTAIN.md"), "w") as fh:
    fh.write(
        f"# {st['name']} — UNCERTAIN findings\n\n{stamp}\n\nNeither confirmed nor refuted. Each needs a closer "
        "look (a repro, the owner, or a deeper read) — they are surfaced, not filed, and never silently dropped.\n\n"
    )
    table(fh, unc)
    fh.write("\n---\n\n")
    detail(fh, unc)
with open(os.path.join(out, "REFUTED.md"), "w") as fh:
    fh.write(f"# {st['name']} — refuted candidates (do not re-raise)\n\n{stamp}\n\n")
    for f in ref:
        why = (f.get("reasons") or [""])[0][:300]
        fh.write(f"- `{key_of(f)}` [{f['category']}] {f['summary'][:200]}\n  - {why}\n")

cls = collections.Counter(f["category"] for f in conf)
cand_cls = collections.Counter(
    f["category"] for f in {key_of(x): x for x in rows}.values()
)
checked = collections.Counter()
for fn in os.listdir(os.path.join(out, "findings")):
    if fn.endswith(".json") and not fn.endswith(".history.jsonl"):
        h = load(os.path.join(out, "findings", fn), {})
        for c in set(h.get("classes_checked", [])):
            checked[c] += len(h.get("files_read", []))
src = collections.Counter(f.get("source", "?") for f in conf)
prio = collections.Counter(man[f["batch"]]["prio"] for f in conf if f["batch"] in man)
with open(os.path.join(out, "COVERAGE.md"), "w") as fh:
    fh.write(
        f"# {st['name']} — coverage\n\n{stamp}\n\n## Yield by class\n\nCandidates raised vs confirmed. "
        "A class with many candidates and a near-zero confirm rate points at a PROMPT defect, not at clean code; "
        "order later waves by measured yield.\n\n| Class | Candidates | Confirmed | Confirm rate |\n|---|---|---|---|\n"
    )
    for k, v in cand_cls.most_common():
        fh.write(f"| {k} | {v} | {cls.get(k, 0)} | {100 * cls.get(k, 0) / v:.0f}% |\n")
    fh.write(
        "\n## Class x file coverage (what the hunters say they swept)\n\n| Class | Files swept |\n|---|---|\n"
    )
    for k, v in checked.most_common():
        fh.write(f"| {k} | {v} |\n")
    nf = sum(len(m["files"]) for m in man.values())
    fh.write(
        f"\nOut of {nf} files in scope. A class far below that was not swept everywhere: its coverage is a CAP, "
        "and needs a targeted pass before the run is called exhaustive.\n"
    )
    fh.write(
        "\n## By what led to them\n\n"
        + "".join(f"- `{k}`: {v}\n" for k, v in src.most_common())
    )
    fh.write(
        "\n## By priority\n\n"
        + "".join(f"- {k}: {v}\n" for k, v in sorted(prio.items()))
    )
print(
    f"confirmed {len(conf)} ({dict(c)}), uncertain {len(unc)}, refuted {len(ref)}; "
    f"open {len(open_f)}, in flight {len(act_f)}, closed {len(done_f)}"
)
