# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-reference audit over the fork's four documents. Run it after any doc edit.

This fork navigates by pointer: code carries `NOTES.md [gpt-26]`, notes carry `STATUS.md §6.67`,
and STATUS carries `§6.x` throughout. Those pointers are load-bearing -- §6.68's whole method is
re-reading old sections -- and nothing checks them, so they rot silently. The sweep that added this
found `[codec-22]` cited twice and defined nowhere, and it had been wrong long enough that the two
citations pointed at two DIFFERENT topics.

Checks four things:

  1. BROKEN POINTERS  -- a `[id]` used anywhere with no `### [id]` entry in tt/NOTES.md.
  2. ORPHAN ENTRIES   -- a NOTES entry nothing points at, i.e. prose the code lost its handle on.
  3. DANGLING §REFS   -- a `§6.x` with no matching STATUS heading.
  4. LONG COMMENTS    -- runs of 5+ comment lines in tt/ and tests/, which this fork moves to
                        NOTES.md by convention.

What it CANNOT check, and what actually bites: a pointer that resolves to a section since REVERSED.
[gpt-04] named §6.39 as "the current answer" long after §6.67 overturned it, and every check here
passed. Reversal is a judgement, so when you reverse something, grep for the section number and fix
the banners by hand.

    python tests/probes/xref_audit.py        # exits 1 if anything is broken
"""
import glob
import os
import re
import sys
from collections import defaultdict

V = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = glob.glob(f"{V}/tt/*.py") + glob.glob(f"{V}/tests/*.py") + glob.glob(f"{V}/scripts/*.py")
DOCS = [f"{V}/STATUS.md", f"{V}/ONBOARDING.md", f"{V}/tt/NOTES.md", f"{V}/tests/probes/README.md"]


def main():
    notes = open(f"{V}/tt/NOTES.md").read()
    defined = set(re.findall(r"^### \[([a-z]+-\d+)\]", notes, re.M))

    used = defaultdict(list)
    for f in CODE + DOCS:
        for m in re.finditer(r"\[([a-z]+-\d+)\]", open(f).read()):
            used[m.group(1)].append(os.path.relpath(f, V))

    bad = 0
    broken = {k: v for k, v in used.items() if k not in defined}
    print(f"NOTES entries defined : {len(defined)}")
    print(f"broken pointers       : {len(broken)}")
    for k, v in sorted(broken.items()):
        print(f"    [{k}] cited by {sorted(set(v))}")
        bad += 1

    orphans = sorted(defined - set(used))
    print(f"orphan NOTES entries  : {len(orphans)}   {orphans if orphans else ''}")
    bad += len(orphans)

    status = open(f"{V}/STATUS.md").read()
    heads = set(re.findall(r"^#+ +§?(\d+\.\d+)", status, re.M))
    # `§?` above is deliberate: STATUS headings are written `### 6.67`, references `§6.67`.
    refs = defaultdict(list)
    for f in CODE + DOCS:
        # REQUIRE the § or an explicit "STATUS.md " -- matching a bare decimal makes every
        # "2.42 ms" a false positive, which is how the first version of this reported 124 of them.
        for m in re.finditer(r"§(\d+\.\d+)|STATUS\.md (\d+\.\d+)", open(f).read()):
            refs[m.group(1) or m.group(2)].append(os.path.relpath(f, V))
    missing = {k: v for k, v in refs.items() if k not in heads}
    print(f"STATUS sections       : {len(heads)}")
    print(f"dangling section refs : {len(missing)}")
    for k, v in sorted(missing.items(), key=lambda x: [int(i) for i in x[0].split(".")]):
        print(f"    §{k} cited by {sorted(set(v))}")
        bad += 1

    print("long comment runs (>=5 lines), which belong in NOTES.md:")
    for f in sorted(glob.glob(f"{V}/tt/*.py") + glob.glob(f"{V}/tests/*.py")):
        lines = open(f).read().splitlines()
        run = 0
        for i, ln in enumerate(lines):
            run = run + 1 if ln.strip().startswith("#") else 0
            if run == 5:
                print(f"    {os.path.relpath(f, V)}:{i - 3}")

    print("\nOK" if not bad else f"\n{bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
