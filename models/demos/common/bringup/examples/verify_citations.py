# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Re-verify every load-bearing `path:line` citation used by a bring-up's code and logs.

Copy this into `<your package>/scripts/` and set `PKG` below. Each entry in `CITES` is
`(path_relative_to_repo_root, line_number, substring_that_must_be_on_that_line)`; the script reads
the file and reports the true line numbers of any needle whose claimed line is wrong.

Run:
    python <your package>/scripts/verify_citations.py

Exit 0 iff every citation verifies. **Extend `CITES` in every phase.** In the bring-up this came
from it caught five wrong line numbers in the recipe itself and five more in a survey's own first
draft; an unverified `path:line` is worth less than no citation, because it reads as authoritative.
Two failure modes it exists to catch: a cited file grows and every line below the edit shifts, and a
bare basename silently resolves to a *different* package's file of the same name.
"""

import os
import re
import sys

# repo root = four levels up from <package>/scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TT = "models/tt_transformers"
GO = "models/demos/gpt_oss_d_p"
M3 = "models/demos/minimax_m3"
DS = "models/demos/deepseek_v3_d_p"
CP = "models/demos/common/prefill"
CM = "models/common"
# Your package, relative to the repo root. The only line you must edit.
PKG = "models/demos/<your_package>"
LL = PKG
# P8: the ring-joint SDPA device op — cited by tt/attention/dense_sp.py and tt/attention/config.py.
RJ = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp"

# (file, line, substring that MUST appear on that line)

# ---------------------------------------------------------------------------------------------
# CITES — populate this as you write. One entry per path:line claim you make in code or docs:
#     (f"{PKG}/tt/mlp.py", 42, "def __call__"),
# The needle is a substring that must appear ON that line. When a file shifts, this tells you
# which citations moved instead of letting them rot into confident nonsense.
#
# In the bring-up this came from it reached 661 entries and caught wrong line numbers in the
# recipe itself as well as in agents' first drafts. An unverified path:line is worse than no
# citation, because it reads as authoritative.
# ---------------------------------------------------------------------------------------------
CITES = []

# DOCS — every markdown file whose `path:line` references should be range-checked.
# Include your recipe and your README: leaving the recipe out is how a stale citation survived
# a whole run in the original.
DOCS = [
    # f"{PKG}/README.md",
    # f"{PKG}/bringup_log/06_GATES.md",
]


# The package's **own Python docstrings** carry as many
# load-bearing `path:line` refs as the logs do, and none of them were checked. They are also where
# citation shadowing bites hardest: `tt/layer.py`, `tt/model.py` and `tt/embedding.py` now shadow
# gpt-oss files of the same basename, so a bare model.py line 211 in a docstring is genuinely ambiguous
# and pass 2's AMBIGUOUS handling (line must be in range for *every* candidate) is exactly the right
# check for it. Globbed rather than listed so a new file cannot be added without being scanned.
DOCS += sorted(
    os.path.relpath(str(path), ROOT)
    # P7 addition: `scripts/*.py`. The two golden-KV scripts carry as many load-bearing `path:line`
    # refs as any module and were the only Python in the package pass 2 could not see (Appendix F.7
    # says extend the verifier every phase). `verify_citations.py` itself is matched by the glob and
    # is harmless: its own citations are tuples, not backtick-quoted refs, so the regex skips them.
    for pattern in (
        "tt/*.py",
        "tt/*/*.py",
        "tests/*.py",
        "tests/unit/*.py",
        "utils/*.py",
        "scripts/*.py",
        "conftest.py",
    )
    for path in __import__("pathlib").Path(os.path.join(ROOT, PKG)).glob(pattern)
)
# package-relative shorthands used in the logs
DOC_PREFIXES = {
    "BRINGUP_RECIPE.md": f"{PKG}/BRINGUP_RECIPE.md",
    "00_MODEL_CARD.md": f"{PKG}/bringup_log/00_MODEL_CARD.md",
    "01_REFERENCE.md": f"{PKG}/bringup_log/01_REFERENCE.md",
    "02_SURVEY.md": f"{PKG}/bringup_log/02_SURVEY.md",
    "03_OUTLINE.md": f"{PKG}/bringup_log/03_OUTLINE.md",
    "04_CCL_PLAN.md": f"{PKG}/bringup_log/04_CCL_PLAN.md",
    "05_DECISIONS.md": f"{PKG}/bringup_log/05_DECISIONS.md",
    "06_GATES.md": f"{PKG}/bringup_log/06_GATES.md",
    "07_RISKS.md": f"{PKG}/bringup_log/07_RISKS.md",
}
_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cpp|hpp|md|json|textproto|yaml)):(\d+)(?:-(\d+))?`")

# Logs and the recipe write citations with the *same* one- and
# two-letter aliases this script defines above (GO/tt/ccl.py line 55, TT/tt/common.py line 489) — many of
# them. Every one of those resolved before this map existed, but only by falling through to the
# ambiguous-basename path, i.e. by luck: the day a second `ccl.py` is cited anywhere the ref flips
# to a failure that has nothing to do with the ref being wrong. Expanding the alias makes them
# LITERAL, which is what a `path:line` is supposed to be.
_ALIASES = {"TT/": TT, "GO/": GO, "M3/": M3, "DS/": DS, "CP/": CP, "CM/": CM, "LL/": LL}


# Logs also use abbreviated forms — a bare basename (common.py line 564, continuing an earlier
# full citation) or a partial path (gpt_oss_d_p/tt/config.py line 55). Resolving them instead of
# reporting them "unresolved" is what makes pass 2 cover the decision log and the gate ledger, where
# the shorthand is the norm. Ambiguous basenames are REPORTED, not silently dropped.
# P6 addition: the package's own root, so a package-relative ref (tt/config.py line 134,
# tests/unit/test_reference_model.py line 136 — the shorthand every file in this package uses for its
# own siblings) resolves LITERALLY instead of falling through to the ambiguous-basename path. Before
# this, tt/config.py line 134 was matched against `gpt_oss_d_p/tt/attention/config.py` (108 lines) and
# reported out of range — a false positive from citation shadowing, and it must be listed FIRST so
# a package-local file wins over a same-named file elsewhere in the tree.
_PARTIAL_PREFIXES = (
    f"{PKG}/",
    "models/demos/",
    "models/",
    "python_env/lib/python3.12/site-packages/",
    "",
)


def _basename_index():
    """basename -> the set of full repo paths that basename could mean, from CITES + the docs."""
    index = {}
    candidates = {path for path, _, _ in CITES}
    for doc in DOCS:
        full = os.path.join(ROOT, doc)
        if not os.path.isfile(full):
            continue
        for m in _REF.finditer(open(full, errors="replace").read()):
            candidates.add(m.group(1))
    for cand in candidates:
        if "/" in cand and os.path.isfile(os.path.join(ROOT, cand)):
            index.setdefault(os.path.basename(cand), set()).add(cand)
    return index


def _resolve(path, index):
    """Return (resolved_path, note). `note` is non-empty when the resolution was not literal."""
    path = DOC_PREFIXES.get(path, path)
    for alias, real in _ALIASES.items():
        if path.startswith(alias):
            expanded = real + "/" + path[len(alias) :]
            if os.path.isfile(os.path.join(ROOT, expanded)):
                return expanded, f"alias {alias} -> {expanded}"
    if os.path.isfile(os.path.join(ROOT, path)):
        return path, ""
    stripped = path.lstrip("./")
    for prefix in _PARTIAL_PREFIXES:
        cand = prefix + stripped
        if os.path.isfile(os.path.join(ROOT, cand)):
            return cand, f"partial path -> {cand}"
    hits = index.get(os.path.basename(path), set())
    if len(hits) == 1:
        only = next(iter(hits))
        return only, f"basename -> {only}"
    if len(hits) > 1:
        # A bare basename shared by several real files (model_config.py line 19) carries less
        # information than a full path. Rather than drop it, require the line to be IN RANGE for
        # EVERY candidate: then whichever file the author meant, the reference resolves.
        return sorted(hits), f"AMBIGUOUS basename, {len(hits)} candidates"
    return None, ""


def scan_docs():
    ok = bad = 0
    failures = []
    refs = set()
    index = _basename_index()
    for doc in DOCS:
        full = os.path.join(ROOT, doc)
        if not os.path.isfile(full):
            continue
        for m in _REF.finditer(open(full, errors="replace").read()):
            path, lo, hi = m.group(1), int(m.group(2)), m.group(3)
            resolved, note = _resolve(path, index)
            if resolved is None:
                bad += 1
                failures.append(f"DOC UNRESOLVED  {doc}: {path}:{lo}" + (f"  ({note})" if note else ""))
                continue
            hi_i = int(hi) if hi else lo
            for cand in resolved if isinstance(resolved, list) else [resolved]:
                refs.add((doc, cand, lo, hi_i))
    for doc, path, lo, hi in sorted(refs):
        target = os.path.join(ROOT, path)
        if not os.path.isfile(target):
            bad += 1
            failures.append(f"DOC UNRESOLVED  {doc}: {path}:{lo}")
            continue
        n = sum(1 for _ in open(target, errors="replace"))
        if hi > n:
            bad += 1
            failures.append(f"DOC OUT OF RANGE  {doc}: {path}:{lo}-{hi} (file has {n} lines)")
        else:
            ok += 1
    print(f"doc refs scanned  : {ok + bad}")
    print(f"  resolved        : {ok}")
    print(f"  unresolved      : {bad}")
    if failures:
        print("\nDOC FAILURES:")
        for f in failures:
            print("  " + f)
    return bad


def main():
    ok = bad = missing = 0
    failures = []
    for path, lineno, needle in CITES:
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            missing += 1
            failures.append(f"MISSING FILE  {path}")
            continue
        with open(full, errors="replace") as f:
            lines = f.readlines()
        if lineno > len(lines):
            bad += 1
            failures.append(f"OUT OF RANGE  {path}:{lineno} (file has {len(lines)} lines)")
            continue
        line = lines[lineno - 1]
        if needle in line:
            ok += 1
        else:
            bad += 1
            # look for the needle nearby to report the true line
            near = [i + 1 for i, l in enumerate(lines) if needle in l]
            failures.append(
                f"MISMATCH      {path}:{lineno} expected {needle!r}\n"
                f"                got: {line.rstrip()!r}\n"
                f"                needle actually on lines: {near[:6]}"
            )
    print(f"citations checked : {len(CITES)}")
    print(f"  verified        : {ok}")
    print(f"  mismatched      : {bad}")
    print(f"  missing files   : {missing}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
    doc_bad = scan_docs()
    return 0 if bad == 0 and missing == 0 and doc_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
