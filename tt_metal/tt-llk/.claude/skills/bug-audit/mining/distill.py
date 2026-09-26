#!/usr/bin/env python3
"""Turn mined results into a repo pack: write per-class distillation inputs, then assemble the pack from the
distilled sections.

  distill.py inputs --deep a.jsonl [--deep b.jsonl] [--reviews r.jsonl] --stats synth/<x>-stats.json \
                    --classes a.md,b.md --out-dir distill/<pack> [--ids ids.txt] [--min 3]
  distill.py assemble --out-dir distill/<pack> --raw <distill workflow output> --stats synth/<x>-stats.json \
                      --pack packs/<pack>.md --title "<repo>" --provenance "<one line>"
  (writes packs/<pack>.md, the core, and packs/<pack>-detail-1.md, -2.md ..., each under 400 KB)

inputs: one JSON file per class with >= --min real bugs in the deep reads, holding each bug's root cause, trigger,
fix verdict and evidence, audit check, seed and unfixed siblings, plus that class's reviewer items. Classes below
--min are pooled into one "other" file. Prints the distill-wave args.
assemble: writes the pack. It opens with how to use it and a class-weight table computed from triage over EVERY
case (not only the deep reads), then one section per distilled class, ordered by weight.
"""
import argparse
import collections
import json
import os
import re
import sys

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["inputs", "assemble"])
p.add_argument("--deep", action="append", default=[])
p.add_argument("--reviews", action="append", default=[])
p.add_argument("--stats", required=True)
p.add_argument("--classes")
p.add_argument("--out-dir", required=True)
p.add_argument(
    "--ids",
    action="append",
    default=[],
    help="restrict deep cases to these ids (namespaced repo:id ok)",
)
p.add_argument("--min", type=int, default=3)
p.add_argument("--raw")
p.add_argument("--pack")
p.add_argument("--title")
p.add_argument("--provenance", default="")
p.add_argument(
    "--core-sections",
    type=int,
    default=30,
    help="classes that get a full core entry; the rest get one line",
)
p.add_argument("--core-checks", type=int, default=4)
p.add_argument("--core-seeds", type=int, default=3)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
stats = json.load(open(a.stats))


def canon_fn():
    known = {}
    for f in (a.classes or "").split(","):
        if f:
            for m in re.finditer(r"^- `([a-z0-9-]+)`", open(f).read(), re.M):
                known[m.group(1).replace("-", "")] = m.group(1)
    return lambda c: known.get(
        re.sub(r"[^a-z0-9]", "", (c or "").lower()), (c or "").lower()
    )


def split_parts(txt, limit=400_000):
    """Split at class-section boundaries into parts under `limit` bytes (repos cap committed files at 500 KB)."""
    secs = re.split(r"(?=\n## `)", txt)
    head, parts, cur = secs[0], [], ""
    for sec in secs[1:]:
        if cur and len((head + cur + sec).encode()) > limit:
            parts.append(head + cur)
            cur = ""
        cur += sec
    parts.append(head + cur)
    return [p.rstrip() + "\n" for p in parts]


if a.cmd == "inputs":
    canon = canon_fn()
    ids = set()
    for f in a.ids:
        ids |= set(open(f).read().split())
    by = collections.defaultdict(list)
    for path in a.deep:
        tag = os.path.basename(path).split("_")[0]
        for ln in open(path):
            d = json.loads(ln)
            if d.get("is_real_bug") != "yes" or (
                ids and d["id"] not in ids and f"{tag}:{d['id']}" not in ids
            ):
                continue
            by[canon(d["primary_class"])].append(
                {
                    "component": d["component"],
                    "root_cause": d["root_cause"],
                    "trigger": d["trigger"],
                    "symptom": d["symptom"],
                    "fix_verdict": d["fix_verdict"],
                    "fix_verdict_evidence": d["fix_verdict_evidence"][:900],
                    "audit_check": d["audit_check"],
                    "seed": d["seed"],
                    "public_safe": d.get("public_safe", True),
                    "unfixed_siblings": [
                        s for s in d["siblings"] if s["status"] in ("unfixed", "unsure")
                    ][:6],
                }
            )
    rv = collections.defaultdict(list)
    for path in a.reviews:
        for ln in open(path):
            r = json.loads(ln)
            if r.get("confidence") in ("high", "medium"):
                rv[canon(r["class"])].append(
                    {
                        "defect": r["defect"],
                        "check": r["reviewer_check"],
                        "path": r["path"],
                        "addressed": r["addressed"],
                    }
                )
    files = []
    other = []
    for c in sorted(set(by) | set(rv), key=lambda c: -len(by.get(c, []))):
        if len(by.get(c, [])) < a.min and len(rv.get(c, [])) < 2 * a.min:
            other += [{**x, "class": c} for x in by.get(c, [])]
            continue
        path = os.path.join(os.path.abspath(a.out_dir), f"class-{c}.json")
        json.dump(
            {"class": c, "bugs": by.get(c, []), "reviews": rv.get(c, [])[:80]},
            open(path, "w"),
        )
        files.append(path)
    if other:
        path = os.path.join(os.path.abspath(a.out_dir), "class-other.json")
        json.dump({"class": "other", "bugs": other, "reviews": []}, open(path, "w"))
        files.append(path)
    print(json.dumps({"files": files}))
    print(
        f"{len(files)} class inputs; {sum(len(v) for v in by.values())} bugs, {sum(len(v) for v in rv.values())} review items",
        file=sys.stderr,
    )
else:
    d = json.load(open(a.raw))
    r = d.get("result", d)
    if isinstance(r, str):
        r = json.loads(r)
    secs = {s["class"]: s for s in r.get("sections", [])}
    prim = collections.Counter(stats["classes_primary"])
    total = sum(prim.values()) or 1
    order = sorted(secs, key=lambda c: (c == "other", -prim.get(c, 0)))
    detail_name = os.path.basename(a.pack).replace(".md", "-detail-*.md")
    head = [f"# {a.title} — bug-audit knowledge pack", "", a.provenance, ""]
    out = head + [
        "**How to use:** check the classes in weight order, and grep this file for your batch's directories and file "
        "names to pick up the hot spots and seeds that apply. This pack only REORDERS the universal and domain class "
        f"lists. Every class in them is still hunted, including those that never appear here. The full per-class "
        f"material (every check and seed, how past fixes went incomplete, and candidate sites) is in `{detail_name}`: "
        f"grep it for your batch's paths and for the classes you are checking.",
        "",
        "## Class weights",
        "",
        f"Share of {total} triaged code bugs, by primary class (a bug can also carry other classes).",
        "",
        "| Class | Share | Bugs |",
        "|---|---|---|",
    ]
    for c, n in prim.most_common(30):
        out.append(f"| `{c}` | {100 * n / total:.1f}% | {n} |")
    out += [
        "",
        "## Hot areas",
        "",
        "Directories with the most distinct bug fixes to non-test code:",
        "",
    ]
    out += [
        "; ".join(f"`{dname}` ({n})" for dname, n in stats["dirs"][:30]),
        "",
        "## Classes, by weight",
        "",
    ]
    first_sentences = lambda t, k: " ".join(
        re.split(r"(?<=[.!?])\s+", t.strip())[:k]
    )  # noqa: E731
    for i, c in enumerate(order):
        s = secs[c]
        if i < a.core_sections:
            out += [
                f"### `{c}`: {s['title']}",
                "",
                first_sentences(s["what_goes_wrong"], 3),
                "",
            ]
            out += [f"- {x}" for x in s["checks"][: a.core_checks]]
            if s["seeds"]:
                out += ["- *Seeds:* " + " | ".join(s["seeds"][: a.core_seeds])]
            if s["hot_areas"]:
                out += ["- *Clusters in:* " + "; ".join(s["hot_areas"][:4])]
            out += [""]
        else:
            out += [
                f"- `{c}`: {s['title']}. {first_sentences(s['what_goes_wrong'], 1)}"
            ]
    det = head + [
        f"Full per-class material for `{os.path.basename(a.pack)}`. **Candidate sites** are places where earlier "
        "analysis suspected an unfixed copy of a fixed bug. They are leads to verify, never findings to "
        "report as-is. Line numbers drift, so re-find each site by its code before trusting it.",
        "",
    ]
    for c in order:
        s = secs[c]
        det += ["", f"## `{c}`: {s['title']}", "", s["what_goes_wrong"], ""]
        if s["checks"]:
            det += ["**Checks**", ""] + [f"- {x}" for x in s["checks"]] + [""]
        if s["seeds"]:
            det += (
                ["**Seeds** (code shapes to hunt for)", ""]
                + [f"- {x}" for x in s["seeds"]]
                + [""]
            )
        if s["hot_areas"]:
            det += ["**Where it clusters:** " + "; ".join(s["hot_areas"]), ""]
        if s["incomplete_fix_lessons"]:
            det += (
                ["**How fixes for this class went incomplete**", ""]
                + [f"- {x}" for x in s["incomplete_fix_lessons"]]
                + [""]
            )
        if s["candidate_sites"]:
            det += (
                ["**Candidate sites** (unverified leads)", ""]
                + [f"- {x}" for x in s["candidate_sites"]]
                + [""]
            )
    dtxt = "\n".join(det).rstrip() + "\n"
    parts = split_parts(dtxt)
    base = a.pack.replace(".md", "-detail")
    for old_part in [
        f
        for f in os.listdir(os.path.dirname(os.path.abspath(a.pack)))
        if f.startswith(os.path.basename(base))
    ]:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(a.pack)), old_part))
    for i, part in enumerate(parts, 1):
        open(f"{base}-{i}.md", "w").write(part)
    txt = "\n".join(out).rstrip() + "\n"
    leaks = re.findall(r"(?<![\w/])#\d{3,6}\b|\.sv\b|issues/\d+|pull/\d+", txt + dtxt)
    open(a.pack, "w").write(txt)
    print(
        f"wrote {a.pack}: {len(txt)} chars (+ detail {len(dtxt)} chars), {len(order)} class sections"
        + (
            f"; !! {len(leaks)} possible issue/PR/RTL references to scrub: {sorted(set(leaks))[:20]}"
            if leaks
            else ""
        )
    )
