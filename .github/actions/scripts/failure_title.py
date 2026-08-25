#!/usr/bin/env python3
"""Turn a list of failed GitHub job names into a Jira ticket title.

Reads job names on stdin, one per line, and writes one title. A job name is a
path -- "build-test-publish (Ubuntu 22.04) / release-demo-tests / ... /
Gemma-4-31B e2e tests [bh_quietbox_2]" -- so the second segment names the suite
and the leaf names the thing that broke. Both beat "4 job(s)" in a title.

    Release v0.63.0 — 9 demo tests failed: Gemma-4-31B, GPT-OSS 120B +6 more
"""

from __future__ import annotations

import re
import sys

_RUNNER_TAG = re.compile(r"\s*\[[^\]]*\]$")
_TRAILING_PARENS = re.compile(r"\s*\([^)]*\)\s*$")
# "Gemma-4-31B e2e tests", "TT-DiT Wan2.2 multihost quad e2e tests (BH QuietBox 2)"
_E2E = re.compile(r"^(.*?)\s+(?:multihost\s+quad\s+)?e2e tests(?:\s*\([^)]*\))?$")

# A name longer than this crowds out everything else: one real job is
# "Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4)".
MAX_NAME = 30
MAX_NAMES = 2
JIRA_SUMMARY_MAX = 255


def family(job):
    """The suite a job belongs to, or "" when the job is not nested."""
    parts = job.split(" / ")
    if len(parts) < 2:
        return ""
    # "create-docker-release-image (release-models)" -> "create docker release image"
    fam = _TRAILING_PARENS.sub("", parts[1])
    return re.sub(r"^release-", "", fam).replace("-", " ").strip()


def name(job):
    """The failing thing, as a model-ish name.

    "Gemma-4-31B e2e tests [bh_p150]"            -> Gemma-4-31B
    "TT-DiT Wan2.2 e2e tests (BH QuietBox 2)"    -> TT-DiT Wan2.2   (paren = runner)
    "Demo Test with Perf Metrics (DeepSeek V3)"  -> unchanged       (paren = the model)

    A trailing parenthetical is dropped only when it follows "e2e tests", which
    is where the runner qualifier lives. Elsewhere it carries the model name.
    """
    leaf = _RUNNER_TAG.sub("", job.split(" / ")[-1]).strip()
    m = _E2E.match(leaf)
    return (m.group(1) if m else leaf).strip()


def _distinct(values):
    out = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


def _named(jobs):
    """Up to MAX_NAMES short names, plus a count of everything left over."""
    all_names = _distinct(name(j) for j in jobs)
    shown = [n for n in all_names if len(n) <= MAX_NAME][:MAX_NAMES]
    rest = len(all_names) - len(shown)
    if not shown:
        return ""
    return ", ".join(shown) + (f" +{rest} more" if rest else "")


def title(jobs, ref):
    n = len(jobs)
    fams = _distinct(family(j) for j in jobs)

    if len(fams) == 1 and n:
        what = fams[0]
        if n == 1 and what.endswith("s"):
            what = what[:-1]  # "1 demo test failed", not "1 demo tests failed"
        detail = _named(jobs)
        head = f"Release {ref} — {n} {what} failed"
        summary = f"{head}: {detail}" if detail else head
    else:
        # Mixed suites, or nothing nested: the suites are the useful detail,
        # not one arbitrary leaf. Count each so the shape is visible.
        counts = {}
        for j in jobs:
            f = family(j) or name(j)
            counts[f] = counts.get(f, 0) + 1
        bits = [f"{f} ({c})" if c > 1 else f for f, c in counts.items()]
        summary = f"Release {ref} — {n} jobs failed: " + ", ".join(bits)

    if len(summary) > JIRA_SUMMARY_MAX:
        summary = summary[: JIRA_SUMMARY_MAX - 3] + "..."
    return summary


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    jobs = [line.strip() for line in sys.stdin if line.strip()]
    if not jobs:
        sys.exit("error: no job names on stdin")
    print(title(jobs, ref))


if __name__ == "__main__":
    main()
