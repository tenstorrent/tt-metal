#!/usr/bin/env python3
"""
Query agent JSON findings for specific (script, pattern, cohort) combos.

Use this to drill down from "scripts X had N UNEXPECTED outcomes due to EYY"
to the actual file_keys / test_functions / hosts / job URLs, and optionally
the error text from each file.

Examples:

  # All Cohort A files where check_noc_status.py hit E06
  python3 tools/triage/query_findings.py --script check_noc_status.py --pattern E06

  # All files (any cohort) where dump_callstacks.py hit E01 — show the GitHub job URL too
  python3 tools/triage/query_findings.py --script dump_callstacks.py --pattern E01 --cohort any --urls

  # Show the actual error lines from each matching file
  python3 tools/triage/query_findings.py --script check_noc_status.py --pattern E06 --grep

  # All files where any triage-bug pattern fired
  python3 tools/triage/query_findings.py --pattern-category triage_bug
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINDINGS_DIR = REPO_ROOT / "triage_agent_findings"
SPLIT_DIR = REPO_ROOT / "triage_outputs_split"

# Substring used by --grep to find each pattern's signature inside the raw
# triage output. Keep roughly in sync with PATTERN_META regexes in
# consolidate_report.py but deliberately coarser so fragments match even when
# Rich wrapping injects whitespace.
PATTERN_GREP = {
    "E01": r"Errno 24",
    "E02": r"os-release",
    "E03": r"fabric_erisc_router",
    "E04": r"Was halted by triage",
    "E05": r"unsafe access at address",
    "E06": r"Cannot find global variable noc_mode",
    "E07": r"SyntaxWarning",
    "E08": r"is not halted",
    "E09": r"Failed to halt",
    "E10": r"Skipping: '(?:tensix|active_eth|idle_eth)'",
    "E11": r"Data mismatch in section",
    "E12": r"PC was not in range",
    "E13": r"Core is in reset",
    "E14": r"has timed out after",
    "E16": r"Mismatched state:",
    "E17": r"Unknown motherboard",
    "E23": r"Traceback \(most recent call last\)",
    # E25 grep should match a table row where Prev Op Name contains Matmul
    # (4 pipe-separated cells before a Matmul name).
    "E25": r"│\s*\d+\s*│[^│]*│[^│]*│\s*\S+\s*│[^│]*[Mm]atmul",
}


def load_findings(cohort):
    """cohort: 'A', 'B', or 'any'."""
    entries = []
    patterns = []
    if cohort in ("A", "any"):
        patterns.append("cohort_a_batch_*.json")
    if cohort in ("B", "any"):
        patterns.append("cohort_b_batch_*.json")
    for pat in patterns:
        for f in sorted(FINDINGS_DIR.glob(pat)):
            cohort_tag = "A" if "cohort_a_" in f.name else "B"
            for e in json.load(open(f)):
                e["_cohort"] = cohort_tag
                entries.append(e)
    return entries


def matches(entry, args):
    # Collect (script, pattern_id) tuples from known_patterns_found.
    hits = entry.get("known_patterns_found") or []
    for hit in hits:
        sn = hit.get("script", "")
        pid = hit.get("pattern_id", "")
        if args.script and sn != args.script:
            continue
        if args.pattern and pid != args.pattern:
            continue
        if args.pattern_category:
            # Agent output doesn't embed category, so we re-derive from the ID.
            cat = _category_for(pid)
            if cat != args.pattern_category:
                continue
        return hit
    return None


# Category map mirrors PATTERN_META in consolidate_report.py.
_CATEGORY = {
    "E01": "triage_bug",
    "E02": "triage_bug",
    "E03": "triage_bug",
    "E04": "environment",
    "E05": "environment",
    "E06": "environment",
    "E07": "triage_bug",
    "E08": "environment",
    "E09": "environment",
    "E10": "environment",
    "E11": "diagnostic",
    "E12": "informational",
    "E13": "environment",
    "E14": "environment",
    "E15": "environment",
    "E16": "diagnostic",
    "E17": "environment",
    "E18": "informational",
    "E19": "init_failure",
    "E20": "init_failure",
    "E21": "init_failure",
    "E22": "init_failure",
    "E23": "triage_bug",
    "E24": "environment",
    "E25": "diagnostic",
}


def _category_for(pid):
    return _CATEGORY.get(pid, "unknown")


def grep_context(file_key, pattern_regex, max_matches=3, context=2):
    """Return the first `max_matches` matches for `pattern_regex` inside the
    split triage file, with `context` lines around each."""
    fpath = SPLIT_DIR / f"{file_key}.txt"
    if not fpath.exists():
        return "<file missing>"
    try:
        # Use ripgrep via subprocess; fall back to Python regex if rg isn't there.
        result = subprocess.run(
            ["rg", "-n", "-C", str(context), "--max-count", str(max_matches), pattern_regex, str(fpath)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode in (0, 1):
            return result.stdout.strip() or "<no matches>"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Python fallback
    lines = fpath.read_text().splitlines()
    out, hits = [], 0
    rx = re.compile(pattern_regex)
    for i, line in enumerate(lines):
        if rx.search(line) and hits < max_matches:
            lo = max(0, i - context)
            hi = min(len(lines), i + context + 1)
            out.append("\n".join(f"{j+1}: {lines[j]}" for j in range(lo, hi)))
            hits += 1
    return "\n--\n".join(out) or "<no matches>"


def github_url(entry):
    """Reconstruct the GitHub Actions job URL from the job_id."""
    job_id = entry.get("job_id") or entry.get("original_job_id")
    if not job_id:
        return None
    # Format used elsewhere: https://github.com/tenstorrent/tt-metal/actions/runs/{run_id}/job/{job_id}
    # run_id is only in the split index. We include just the job URL (enough to open it).
    return f"https://github.com/tenstorrent/tt-metal/actions/jobs/{job_id}"


def main():
    ap = argparse.ArgumentParser(description="Query tt-triage agent findings.")
    ap.add_argument("--script", help="Filter by triage script filename (e.g. check_noc_status.py)")
    ap.add_argument("--pattern", help="Filter by pattern ID (e.g. E06)")
    ap.add_argument(
        "--pattern-category",
        choices=["triage_bug", "environment", "diagnostic", "informational", "init_failure"],
        help="Filter by pattern category",
    )
    ap.add_argument("--cohort", default="A", choices=["A", "B", "any"], help="Which cohort to search (default: A)")
    ap.add_argument("--urls", action="store_true", help="Print the GitHub Actions job URL for each match")
    ap.add_argument(
        "--grep", action="store_true", help="Print the matching error lines from each file (with 2 lines of context)"
    )
    ap.add_argument("--grep-max", type=int, default=3, help="Max grep matches per file when --grep (default: 3)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after this many files (0 = no limit)")
    args = ap.parse_args()

    if not (args.script or args.pattern or args.pattern_category):
        ap.error("must pass at least one of --script / --pattern / --pattern-category")

    entries = load_findings(args.cohort)
    matched = []
    for e in entries:
        hit = matches(e, args)
        if hit:
            matched.append((e, hit))

    if args.limit:
        matched = matched[: args.limit]

    print(
        f"# Matched {len(matched)} file(s)  "
        f"[script={args.script or '*'}, pattern={args.pattern or '*'}, "
        f"category={args.pattern_category or '*'}, cohort={args.cohort}]"
    )

    for e, hit in matched:
        fk = e.get("file_key", "?")
        tf = e.get("test_function", "?")
        host = e.get("host_name", "?")
        arch = e.get("arch", "?")
        print()
        print(
            f"{fk}  [{e['_cohort']}]  {tf}  host={host}  arch={arch}  "
            f"pattern={hit.get('pattern_id')} script={hit.get('script')} count={hit.get('count')}"
        )
        if args.urls:
            url = github_url(e)
            if url:
                print(f"  {url}")
        if args.grep:
            pid = hit.get("pattern_id", "")
            regex = PATTERN_GREP.get(pid, pid)
            print("  ---")
            for line in grep_context(fk, regex, max_matches=args.grep_max).splitlines():
                print(f"  {line}")
            print("  ---")


if __name__ == "__main__":
    main()
