#!/usr/bin/env python3
"""Aggregate a repo's mined triage / deep-read / review results into the numbers and raw material for its pack.

  synthesize.py --cases cases.jsonl --triage triage.jsonl [--deep deep.jsonl] [--reviews reviews.jsonl] \
                --classes a.md,b.md --out-prefix synth/<repo> [--path-filter 'tt_metal/tt-llk/*' ...] [--exclude holdout.jsonl]

Writes <prefix>-stats.json (every count) and <prefix>-draft.md, which has these sections:
  class weights   code-bug share per class, overall and by symptom
  hot spots       components and directories ranked by distinct bug cases, with their top classes
  fix quality     deep-read verdicts, and every incomplete, reverted or wrong-root-cause fix, with its evidence
  unfixed siblings candidate open bugs from the deep reads (verify before filing!)
  seeds           audit checks and seed patterns from deep reads, grouped by class
  reviewer checks what reviewers caught, grouped by class, with the reusable check
The draft is raw material. The pack is written from it by hand, public-safe, with no issue numbers.
Class ids are canonicalised against the class files (hyphen and case insensitive); ids the files do not define are
counted under their own name and listed as "new classes": candidates to add to a class file.
--path-filter keeps only cases whose fix touches a matching path (for example, the tt-llk subtree of tt-metal).
"""
import argparse
import collections
import fnmatch
import json
import os
import re

p = argparse.ArgumentParser()
p.add_argument("--cases", action="append", required=True)
p.add_argument("--triage", action="append", required=True)
p.add_argument("--deep", action="append", default=[])
p.add_argument("--reviews", action="append", default=[])
p.add_argument("--classes", required=True)
p.add_argument("--out-prefix", required=True)
p.add_argument("--path-filter", action="append", default=[])
p.add_argument("--exclude", action="append", default=[])
p.add_argument(
    "--label-filter",
    action="append",
    default=[],
    help="also keep cases carrying one of these labels",
)
a = p.parse_args()


def jl(paths):
    rows = {}
    for path in paths:
        if os.path.exists(path):
            for ln in open(path):
                if ln.strip():
                    x = json.loads(ln)
                    rows[str(x.get("id", x.get("number")))] = x
    return rows


known = {}
for f in a.classes.split(","):
    for m in re.finditer(r"^- `([a-z0-9-]+)`", open(f).read(), re.M):
        known[m.group(1).replace("-", "")] = m.group(1)
canon = lambda c: known.get(
    re.sub(r"[^a-z0-9]", "", c.lower()), c.lower()
)  # noqa: E731

cases, tri, deep = jl(a.cases), jl(a.triage), jl(a.deep)
excl = set().union(*[set(jl([f])) for f in a.exclude]) if a.exclude else set()


def files_of(c):
    return {f for fx in c.get("fix", []) for f in fx["files"]}


def keep(cid):
    if cid in excl or cid not in cases:
        return False
    if not a.path_filter and not a.label_filter:
        return True
    c = cases[cid]
    if a.label_filter and set(c.get("labels", [])) & set(a.label_filter):
        return True
    return any(fnmatch.fnmatch(f, g) for f in files_of(c) for g in a.path_filter)


ids = [i for i in tri if keep(i)]
# a deep read outranks triage: cases it judged not real never count as bugs
bugs = [
    i
    for i in ids
    if tri[i]["verdict"] == "code-bug" and deep.get(i, {}).get("is_real_bug") != "no"
]
tbugs = [i for i in ids if tri[i]["verdict"] == "test-bug"]
S = {
    "cases": len(ids),
    "verdicts": collections.Counter(tri[i]["verdict"] for i in ids),
    "not_bug_kinds": collections.Counter(
        tri[i]["not_bug_kind"] for i in ids if tri[i]["verdict"] == "not-a-code-bug"
    ),
}


def primary(i):
    d = deep.get(i)
    if d and d.get("is_real_bug") == "yes":
        return canon(d["primary_class"])
    cl = tri[i]["classes"]
    return canon(cl[0]) if cl else "unclassified"


cls = collections.Counter(primary(i) for i in bugs)
any_cls = collections.Counter(canon(c) for i in bugs for c in set(tri[i]["classes"]))
by_sym = collections.defaultdict(collections.Counter)
for i in bugs:
    by_sym[tri[i]["symptom"]][primary(i)] += 1
comp = collections.Counter(tri[i]["component"].lower() for i in bugs)
comp_cls = collections.defaultdict(collections.Counter)
for i in bugs:
    comp_cls[tri[i]["component"].lower()][primary(i)] += 1
dirs = collections.Counter()
for i in bugs:
    for d in {
        os.path.dirname(f)
        for f in files_of(cases[i])
        if not re.search(r"(^|/)tests?/", f)
    }:
        dirs[d] += 1
arch = collections.Counter(x for i in bugs for x in tri[i]["arch"])
trig = collections.Counter(t.lower() for i in bugs for t in tri[i]["triggers"])
new_cls = sorted(
    {c for c in any_cls if c not in known.values()}, key=lambda c: -any_cls[c]
)
dv = [deep[i] for i in ids if i in deep]
verdicts = collections.Counter(d["fix_verdict"] for d in dv)
S.update(
    classes_primary=cls,
    classes_any=any_cls,
    components=comp,
    dirs=dirs.most_common(80),
    arch=arch,
    triggers=trig.most_common(80),
    fix_verdicts=verdicts,
    new_classes=new_cls[:80],
    code_bugs=len(bugs),
    test_bugs=len(tbugs),
    deep_read=len(dv),
)

reviews = []
for path in a.reviews:
    if os.path.exists(path):
        reviews += [json.loads(x) for x in open(path) if x.strip()]
rv = [r for r in reviews if r.get("confidence") in ("high", "medium") and "pr" in r]
rv_cls = collections.Counter(canon(r["class"]) for r in rv)
S["review_classes"] = rv_cls
os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
json.dump(S, open(a.out_prefix + "-stats.json", "w"), indent=1, default=dict)

with open(a.out_prefix + "-draft.md", "w") as fh:
    w = fh.write
    w(
        f"# Synthesis draft: {os.path.basename(a.out_prefix)}\n\n{len(ids)} cases triaged: "
        + ", ".join(f"{k} {v}" for k, v in S["verdicts"].most_common())
        + f". Deep-read {len(dv)}. Review items {len(rv)}.\n\n"
    )
    w(
        "## Class weights (code bugs, primary class)\n\n| Class | Cases | Share | Also-tagged |\n|---|---|---|---|\n"
    )
    for c, n in cls.most_common():
        w(f"| {c} | {n} | {100 * n / max(len(bugs), 1):.1f}% | {any_cls[c]} |\n")
    w("\n## By symptom\n\n")
    for s, cc in sorted(by_sym.items(), key=lambda kv: -sum(kv[1].values())):
        w(
            f"- **{s}** ({sum(cc.values())}): "
            + ", ".join(f"{c} {n}" for c, n in cc.most_common(8))
            + "\n"
        )
    w("\n## Hot components\n\n")
    for c, n in comp.most_common(60):
        w(
            f"- **{c}** ({n}): "
            + ", ".join(f"{k} {v}" for k, v in comp_cls[c].most_common(5))
            + "\n"
        )
    w("\n## Hot directories (distinct bug fixes touching non-test code)\n\n")
    for d, n in dirs.most_common(60):
        w(f"- `{d}`: {n}\n")
    w("\n## Arch\n\n" + ", ".join(f"{k} {v}" for k, v in arch.most_common()) + "\n")
    w(
        "\n## Top triggers\n\n"
        + ", ".join(f"{k} ({v})" for k, v in trig.most_common(60))
        + "\n"
    )
    w(
        "\n## Classes not in the class files (candidates to add)\n\n"
        + ", ".join(f"{c} ({any_cls[c]})" for c in new_cls[:60])
        + "\n"
    )
    w(
        "\n## Fix quality (deep reads)\n\n"
        + ", ".join(f"{k} {v}" for k, v in verdicts.most_common())
        + "\n\n"
    )
    for d in sorted(dv, key=lambda d: d["fix_verdict"]):
        if d["fix_verdict"] in (
            "partial",
            "reverted",
            "wrong-root-cause",
            "workaround",
            "superseded",
            "reverted-then-relanded",
        ):
            w(
                f"- **{d['fix_verdict']}** [{canon(d['primary_class'])}] {d['component']}: {d['root_cause']}\n"
                f"  - fix: {d['fix_summary']}\n  - evidence: {d['fix_verdict_evidence']}\n"
            )
    w(
        "\n## Unfixed siblings reported by deep reads (CANDIDATES: verify before filing)\n\n"
    )
    for d in dv:
        for s in d["siblings"]:
            if s["status"] == "unfixed":
                w(
                    f"- `{s['location']}` [{canon(d['primary_class'])}] {s['why']}  (from {d['id']})\n"
                )
    w("\n## Seeds and audit checks, by class\n\n")
    by = collections.defaultdict(list)
    for d in dv:
        if d.get("is_real_bug") == "yes":
            by[canon(d["primary_class"])].append(d)
    for c, ds in sorted(by.items(), key=lambda kv: -len(kv[1])):
        w(f"### {c} ({len(ds)})\n\n")
        for d in ds:
            w(
                f"- check: {d['audit_check']}\n  - seed: {d['seed']}\n  - trigger: {d['trigger']}"
                + ("" if d.get("public_safe", True) else "  **[NOT public-safe]**")
                + "\n"
            )
        w("\n")
    w("## Reviewer checks, by class\n\n")
    rby = collections.defaultdict(list)
    for r in rv:
        rby[canon(r["class"])].append(r)
    for c, rs in sorted(rby.items(), key=lambda kv: -len(kv[1])):
        w(f"### {c} ({len(rs)})\n\n")
        for r in rs[:40]:
            w(f"- [{r['addressed']}] {r['defect']} → check: {r['reviewer_check']}\n")
        w("\n")
print(
    f"{a.out_prefix}: {len(ids)} cases, {len(bugs)} code bugs, {len(dv)} deep, {len(rv)} review items; "
    f"top classes: {', '.join(f'{c} {n}' for c, n in cls.most_common(8))}"
)
