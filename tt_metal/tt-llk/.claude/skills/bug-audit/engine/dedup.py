#!/usr/bin/env python3
"""Semantic dedup of confirmed findings: the same defect reported at several lines, files or per-arch copies.

  dedup.py [--run DIR] inputs [--variant REGEX ...]  # one judge input per group; prints dedup-wave.js args
  dedup.py [--run DIR] persist <raw>   # record the judges' clusters in dedup.json
  dedup.py [--run DIR] show

consolidate.py dedupes exact file:line only. Earlier large audits needed a judge pass on top: hunters report one
defect at a call site and at its definition, or in several architecture copies of the same file. Groups are
directories with every architecture or platform variant component normalised (Grayskull, Wormhole, Blackhole,
Quasar, tt-Nxx, and generic ones such as x86/arm64/riscv/cuda; add your repo's with --variant), so all copies meet
in one group. However many variants carry the bug, they MERGE into one entry that lists every site. Nothing is merged mechanically: two findings of one class a few lines apart are often separate defects (two
mis-bound arguments of one call), so the judge decides them too -- they share a directory, so they share its group.
All variant copies of one defect are clustered as "arch-copy": ONE merged issue, one fix site per variant. The copies stay listed and are
never dropped, because each still needs fixing (and per the cross-arch rule, each is judged against its own arch).
Run it after the hunt is complete, then consolidate.py again.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, load, run_dir, save  # noqa: E402

out = run_dir()
argv = sys.argv[1:]
if not argv:
    sys.exit(__doc__)
# Path components that name an architecture or platform VARIANT of the same code. Copies of a file that differ only
# in these components land in one dedup group, however many variants exist. Extend per repo with --variant REGEX
# (saved in the run's state).
DEFAULT_VARIANTS = [
    r"grayskull",
    r"wormhole_b0",
    r"wormhole",
    r"blackhole",
    r"quasar",
    r"tt-[0-9]xx",
    r"tt_llk_[a-z0-9_]+?",
    r"x86_64",
    r"x86",
    r"amd64",
    r"i[3-6]86",
    r"arm64",
    r"aarch64",
    r"armv[0-9a-z]+",
    r"arm",
    r"riscv(?:32|64)?",
    r"ppc64(?:le)?",
    r"s390x",
    r"cuda",
    r"hip",
    r"rocm",
    r"sm_?[0-9]{2,3}",
    r"avx(?:2|512[a-z]*)?",
    r"sse[0-9_]*",
    r"neon",
    r"sve2?",
    r"linux",
    r"darwin",
    r"macos",
    r"windows",
    r"win32",
]
st_ = load(os.path.join(out, "state.json"), {})
extra = [
    argv[i + 1] for i, x in enumerate(argv) if x == "--variant" and i + 1 < len(argv)
]
if extra:
    st_["variant_tokens"] = sorted(set(st_.get("variant_tokens", []) + extra))
    save(os.path.join(out, "state.json"), st_)
VARIANTS = st_.get("variant_tokens", []) + DEFAULT_VARIANTS
ARCH = re.compile(
    r"(?<=/)(" + "|".join(VARIANTS) + r")(?=/|$)|^(" + "|".join(VARIANTS) + r")(?=/)"
)
path = os.path.join(out, "dedup.json")


def confirmed():
    """Confirmed findings AFTER consolidation: CONFIRMED.json applies recheck outcomes on top of the wave verdicts
    (a finding promoted by recheck.py is still "uncertain" in its verdict file)."""
    conf = load(os.path.join(out, "CONFIRMED.json"))
    if conf is None:
        sys.exit("no CONFIRMED.json: run consolidate.py first")
    rows = {}
    for f in conf:
        if not f.get("duplicate_of"):
            rows.setdefault(key_of(f), f)
    return rows


if argv[0] == "inputs":
    rows = confirmed()
    # no mechanical pre-merge: proximity alone does not make two findings one defect (see the module docstring)
    auto, seen = {}, sorted(rows.values(), key=lambda f: (f["file"], f["line"]))
    groups = {}
    for f in seen:
        g = ARCH.sub("*", os.path.dirname(f["file"]))
        groups.setdefault(g, []).append(f)
    d = os.path.join(out, "dedup_inputs")
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, (g, fs) in enumerate(sorted(groups.items())):
        if len(fs) < 2:
            continue
        p = os.path.join(d, f"group-{i:04d}.json")
        json.dump(
            {
                "group": g,
                "findings": [
                    {
                        "key": key_of(f),
                        "file": f["file"],
                        "line": f["line"],
                        "category": f["category"],
                        "summary": f["summary"],
                        "failure_scenario": f["failure_scenario"],
                    }
                    for f in fs
                ],
            },
            open(p, "w"),
            indent=1,
        )
        paths.append(p)
    save(path, {"auto": auto, "clusters": load(path, {}).get("clusters", [])})
    json.dump({"inputs": paths}, sys.stdout)
    print(
        f"\n{len(rows)} confirmed, {len(auto)} merged mechanically, {len(paths)} groups to judge",
        file=sys.stderr,
    )
elif argv[0] == "persist":
    raw = load(argv[1])
    r = raw.get("result", raw)
    r = json.loads(r) if isinstance(r, str) else r
    dj = load(path, {"auto": {}, "clusters": []})
    dj["clusters"] = [
        c
        for res in r.get("results", [])
        for c in res.get("clusters", [])
        if c.get("duplicates")
    ]
    save(path, dj)
    print(
        f"{len(dj['clusters'])} clusters recorded ({sum(len(c['duplicates']) for c in dj['clusters'])} duplicates); "
        "run consolidate.py"
    )
elif argv[0] == "show":
    dj = load(path, {"auto": {}, "clusters": []})
    print(f"{len(dj['auto'])} mechanical merges")
    for c in dj["clusters"]:
        print(f"  [{c['relation']}] {c['canonical']} <- {', '.join(c['duplicates'])}")
else:
    sys.exit(__doc__)
