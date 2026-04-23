#!/usr/bin/env python3
"""
Consolidate analysis agent JSON findings into the 5 weekly output files.

Reads triage_agent_findings/*.json (emitted by the analysis agents in
Step 4 of the playbook), combines with split-index metadata, and writes
the week's CSVs + report + week-over-week trends (when previous-week
CSVs exist).

Inputs:
  - triage_outputs_split/index.json            (per-file metadata)
  - triage_agent_findings/cohort_{a,b}_batch_*.json  (agent JSON output)
  - triage_{script_reliability,error_patterns}_{PREV_YYYYMMDD}.csv  (trends, optional)

Outputs (written to repo root):
  - triage_script_reliability_{YYYYMMDD}.csv
  - triage_error_patterns_{YYYYMMDD}.csv
  - triage_per_test_breakdown_{YYYYMMDD}.csv
  - triage_new_errors_{YYYYMMDD}.csv
  - triage_weekly_report_{YYYYMMDD}.md

Env vars (optional):
  TRIAGE_YYYYMMDD     - output date suffix (default: today)
  TRIAGE_PREV_YYYYMMDD - previous week's suffix for trend comparison
                         (default: today - 7 days)
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "triage_outputs_split"
FINDINGS_DIR = REPO_ROOT / "triage_agent_findings"
OUTPUT_DIR = REPO_ROOT

# All 20 scripts in execution order
ALL_SCRIPTS = [
    "dump_configuration.py",
    "check_arc.py",
    "check_cb_inactive.py",
    "check_eth_status.py",
    "check_noc_locations.py",
    "device_info.py",
    "device_telemetry.py",
    "dump_running_operations.py",
    "check_binary_integrity.py",
    "check_core_magic.py",
    "check_noc_status.py",
    "dump_aggregated_callstacks.py",
    "dump_callstacks.py",
    "dump_fast_dispatch.py",
    "dump_lightweight_asserts.py",
    "dump_watcher_ringbuffer.py",
    "firmware_versions.py",
    "system_info.py",
    "check_broken_components.py",
    "dump_risc_debug_signals.py",
]

# Pattern metadata (name, category, is_triage_bug). Keep in sync with
# agent_analysis_instructions.md Section 5.
PATTERN_META = {
    "E01": ("FD Exhaustion (Errno 24)", "triage_bug", True),
    "E02": ("FD Exhaustion Crashes system_info/Rich", "triage_bug", True),
    "E03": ("Missing Fabric ERISC Router ELF", "triage_bug", True),
    "E04": ("Cores Broken During Triage (known HW halt/resume limitation)", "environment", False),
    "E05": ("Unsafe ARC Memory Access", "environment", False),
    "E06": ("Missing noc_mode DWARF Variable", "environment", False),
    "E07": ("ttexalens SyntaxWarning", "triage_bug", True),
    "E08": ("Core Not Halted (skip)", "environment", False),
    "E09": ("Failed to Halt Core", "environment", False),
    "E10": ("No Cores Available", "environment", False),
    "E11": ("Binary Integrity Mismatch", "diagnostic", False),
    "E12": ("PC Not in ELF Range", "informational", False),
    "E13": ("Core Is In Reset", "environment", False),
    "E14": ("Test Action Timeout", "environment", False),
    "E15": ("No Triage Section", "environment", False),
    "E16": ("NOC Transaction Mismatch", "diagnostic", False),
    "E17": ("Unknown Motherboard Warning", "environment", False),
    "E18": ("N/A Kernel Name in Callstacks", "informational", False),
    "E19": ("Triage Init: Inspector RPC Unavailable", "init_failure", True),
    "E20": ("Triage Init: FW Init Failure", "init_failure", False),
    "E21": ("Triage Init: Module Not Found", "init_failure", True),
    "E22": ("Triage Init: Device Broken by Previous Triage", "init_failure", True),
    "E23": ("Triage Mid-Script Crash", "triage_bug", True),
    "E24": ("Triage Output Truncated", "environment", False),
    "E25": ("Likely di/dt — MatMul preceded hang", "diagnostic", False),
}


def load_all_findings():
    """Merge all agent JSON findings, tagged by cohort via filename."""
    cohort_a, cohort_b = [], []
    for f in sorted(FINDINGS_DIR.glob("cohort_a_batch_*.json")):
        cohort_a.extend(json.load(open(f)))
    for f in sorted(FINDINGS_DIR.glob("cohort_b_batch_*.json")):
        cohort_b.extend(json.load(open(f)))
    return cohort_a, cohort_b


def script_status(entry, script_name):
    """Returns the status string for script_name, or None if not present."""
    val = entry.get("scripts", {}).get(script_name)
    if isinstance(val, dict):
        return val.get("status")
    return val


def get_patterns(entry):
    """Return {pattern_id: count} extracted from entry.known_patterns_found."""
    out = Counter()
    for p in entry.get("known_patterns_found", []):
        pid = p.get("pattern_id")
        cnt = p.get("count", 1)
        if pid:
            out[pid] += cnt
    for pid, cnt in entry.get("patterns", {}).items():
        if pid not in out:
            out[pid] = cnt
    return out


def build_script_reliability(results, cohort_tag):
    rows = []
    total_files = len(results)

    # Synthetic _triage_init row tracking init failures at the triage level.
    init_failures = sum(1 for r in results if r.get("init_failure"))
    init_reasons = Counter()
    for r in results:
        if r.get("init_failure"):
            for pid in get_patterns(r):
                init_reasons[pid] += 1
    rows.append(
        {
            "cohort": cohort_tag,
            "script_name": "_triage_init",
            "total_runs": total_files,
            "pass_count": total_files - init_failures,
            "pass_pct": round((total_files - init_failures) / total_files * 100, 1) if total_files else 0,
            "expected_count": 0,
            "expected_pct": 0.0,
            "unexpected_count": init_failures,
            "unexpected_pct": round(init_failures / total_files * 100, 1) if total_files else 0,
            "top_unexpected_errors": "; ".join(pid for pid, _ in init_reasons.most_common(3)),
        }
    )

    for script_name in ALL_SCRIPTS:
        counts = Counter()
        unexpected_patterns = Counter()
        for r in results:
            if r.get("init_failure"):
                continue
            status = script_status(r, script_name)
            if status:
                counts[status] += 1
            if status == "UNEXPECTED":
                for p in r.get("known_patterns_found", []):
                    if p.get("script") == script_name:
                        unexpected_patterns[p.get("pattern_id", "?")] += 1

        total_present = counts["PASS"] + counts["EXPECTED"] + counts["UNEXPECTED"]
        rows.append(
            {
                "cohort": cohort_tag,
                "script_name": script_name,
                "total_runs": total_present,
                "pass_count": counts["PASS"],
                "pass_pct": round(counts["PASS"] / total_present * 100, 1) if total_present else 0,
                "expected_count": counts["EXPECTED"],
                "expected_pct": round(counts["EXPECTED"] / total_present * 100, 1) if total_present else 0,
                "unexpected_count": counts["UNEXPECTED"],
                "unexpected_pct": round(counts["UNEXPECTED"] / total_present * 100, 1) if total_present else 0,
                "top_unexpected_errors": "; ".join(pid for pid, _ in unexpected_patterns.most_common(3)),
            }
        )
    return rows


def build_error_patterns(results, cohort_tag):
    total = len(results)
    pattern_jobs = defaultdict(set)
    pattern_total = Counter()
    for r in results:
        for pid, cnt in get_patterns(r).items():
            pattern_jobs[pid].add(r.get("job_id", r.get("file_key")))
            pattern_total[pid] += cnt
    rows = []
    for pid in sorted(pattern_jobs.keys()):
        name, category, is_bug = PATTERN_META.get(pid, (pid, "unknown", False))
        jobs = len(pattern_jobs[pid])
        rows.append(
            {
                "cohort": cohort_tag,
                "pattern_id": pid,
                "pattern_name": name,
                "category": category,
                "is_triage_bug": is_bug,
                "jobs_affected": jobs,
                "jobs_pct": round(jobs / total * 100, 1) if total else 0,
                "total_occurrences": pattern_total[pid],
                "avg_per_job": round(pattern_total[pid] / jobs, 1) if jobs else 0,
            }
        )
    return rows


def build_per_test(results, cohort_tag):
    test_arch = {}
    test_count = Counter()
    test_script_stats = defaultdict(lambda: defaultdict(Counter))
    test_script_patterns = defaultdict(lambda: defaultdict(Counter))
    for r in results:
        if r.get("init_failure"):
            continue
        tf = r.get("test_function", "unknown")
        test_arch[tf] = r.get("arch", "unknown")
        test_count[tf] += 1
        for sn in ALL_SCRIPTS:
            status = script_status(r, sn)
            if status:
                test_script_stats[tf][sn][status] += 1
        for p in r.get("known_patterns_found", []):
            script = p.get("script", "")
            pid = p.get("pattern_id", "")
            if script and pid:
                test_script_patterns[tf][script][pid] += 1

    rows = []
    for tf in sorted(test_count.keys(), key=lambda x: -test_count[x]):
        for sn in ALL_SCRIPTS:
            c = test_script_stats[tf][sn]
            unexp = c.get("UNEXPECTED", 0)
            exp = c.get("EXPECTED", 0)
            if unexp > 0 or exp > 0:
                dominant = ""
                if test_script_patterns[tf][sn]:
                    dominant = test_script_patterns[tf][sn].most_common(1)[0][0]
                rows.append(
                    {
                        "cohort": cohort_tag,
                        "test_function": tf,
                        "arch": test_arch.get(tf, "unknown"),
                        "total_jobs": test_count[tf],
                        "script_name": sn,
                        "pass_count": c.get("PASS", 0),
                        "expected_count": exp,
                        "unexpected_count": unexp,
                        "dominant_pattern": dominant,
                    }
                )
    return rows


def build_new_errors(results, cohort_tag):
    grouped = defaultdict(
        lambda: {
            "script_name": "",
            "error_text": "",
            "suggested_name": "",
            "suggested_regex": "",
            "jobs": set(),
            "file_keys": set(),
        }
    )
    for r in results:
        job_id = r.get("job_id", "")
        fk = r.get("file_key", "")
        for e in r.get("new_errors", []):
            key = e.get("suggested_name", "UNKNOWN")
            g = grouped[key]
            g["script_name"] = e.get("script", g["script_name"])
            g["error_text"] = (e.get("error_text") or "")[:500] or g["error_text"]
            g["suggested_name"] = key
            g["suggested_regex"] = e.get("suggested_regex", g["suggested_regex"])
            g["jobs"].add(job_id)
            g["file_keys"].add(fk)
    rows = []
    for _, g in sorted(grouped.items(), key=lambda x: -len(x[1]["jobs"])):
        rows.append(
            {
                "cohort": cohort_tag,
                "script_name": g["script_name"],
                "error_text": g["error_text"],
                "suggested_name": g["suggested_name"],
                "suggested_regex": g["suggested_regex"],
                "jobs_affected": len(g["jobs"]),
                "file_keys": ";".join(sorted(g["file_keys"])),
            }
        )
    return rows


_RISC_RE = re.compile(r"\b(subordinate_erisc\d?|active_erisc\d?|idle_erisc\d?|erisc\d?|brisc|trisc\d?|ncrisc)\b")


def e12_risc_breakdown(cohort):
    """Per-risc count of 'PC was not in range' occurrences across a cohort.

    Parses the callstack tables (dump_callstacks.py + dump_aggregated_callstacks.py)
    from each file's raw triage output. Rows are separated by the `├─…┤` line
    drawn by Rich; within each row we look for the risc column value and count
    the E12 message. erisc hits are informational (context switch to base
    firmware, expected); tensix-core hits (brisc/trisc/ncrisc) indicate a
    tooling/ELF gap and are the interesting ones.
    """
    overall = Counter()
    for entry in cohort:
        fk = entry.get("file_key") or ""
        fpath = SPLIT_DIR / f"{fk}.txt"
        if not fpath.exists():
            continue
        text = fpath.read_text()
        for section_name in ("dump_callstacks.py", "dump_aggregated_callstacks.py"):
            m = re.search(rf"{section_name}:\n(.+?)(?=\n\w+\.py:|\Z)", text, re.DOTALL)
            if not m:
                continue
            rows = re.split(r"^├[─┼]+┤\s*$", m.group(1), flags=re.MULTILINE)
            for row in rows:
                if "PC was not in range" not in row:
                    continue
                rm = _RISC_RE.search(row)
                risc = rm.group(1) if rm else "unknown"
                overall[risc] += row.count("PC was not in range")
    return overall


def colored_delta(d, good_if="down"):
    """Format a delta as a colored markdown math-span.

    good_if="down": lower is better (error rates, UNEXPECTED%, jobs%). + is red, - is green.
    good_if="up":   higher is better (PASS%, reliability). + is green, - is red.
    0 is rendered plain.
    """
    if d == 0:
        return "+0"
    is_good = (d < 0 and good_if == "down") or (d > 0 and good_if == "up")
    color = "green" if is_good else "red"
    sign = "+" if d > 0 else ""
    return rf"$\color{{{color}}}{{{sign}{d}}}$"


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def load_prev_csv(path):
    if not path.exists():
        return None
    return list(csv.DictReader(open(path)))


def build_report(
    cohort_a, cohort_b, rel_rows, pat_rows, test_rows, new_err_rows, prev_rel, prev_pat, yyyymmdd, week_start, week_end
):
    n_total_a = len(cohort_a)
    n_total_b = len(cohort_b)
    n_bh_a = sum(1 for r in cohort_a if r.get("arch") == "blackhole")
    n_wh_a = sum(1 for r in cohort_a if r.get("arch") in ("wormhole_b0", "wormhole"))
    n_init_fail = sum(1 for r in cohort_a if r.get("init_failure"))
    n_aborted = sum(1 for r in cohort_a if r.get("triage_outcome") == "ABORTED")
    # When an agent doesn't emit triage_outcome, treat "no init failure, not
    # aborted" as COMPLETED.
    n_completed = sum(
        1
        for r in cohort_a
        if not r.get("init_failure") and r.get("triage_outcome") not in ("ABORTED", "FAILED_TO_START")
    )

    rel_a = [r for r in rel_rows if r["cohort"] == "A" and r["script_name"] != "_triage_init"]
    rel_b = [r for r in rel_rows if r["cohort"] == "B" and r["script_name"] != "_triage_init"]
    pat_a = [r for r in pat_rows if r["cohort"] == "A"]

    out = []
    out.append(f"# tt-triage Weekly Analysis Report")
    out.append(f"## Week of {week_start} to {week_end}")
    out.append("")
    out.append("### Executive Summary")
    out.append(f"- Total hang jobs: {n_total_a}")
    out.append(f"- Jobs with triage output: {n_total_a - n_init_fail}")
    out.append(
        f"- Init failures (no triage output): {n_init_fail} ({round(n_init_fail/n_total_a*100, 1) if n_total_a else 0}%)"
    )
    out.append(f"- Triage completed: {n_completed} ({round(n_completed/n_total_a*100, 1) if n_total_a else 0}%)")
    out.append(
        f"- Aborted runs (triage truncated mid-execution): {n_aborted} ({round(n_aborted/n_total_a*100, 1) if n_total_a else 0}%)"
    )
    out.append(f"- Architecture breakdown: {n_bh_a} Blackhole, {n_wh_a} Wormhole")
    out.append(f"- Cohort B sampled (last run of multi-run jobs): {n_total_b}")
    out.append("")

    # Why triage didn't complete — one row per ABORTED / FAILED_TO_START file.
    incomplete = [
        r for r in cohort_a if r.get("init_failure") or r.get("triage_outcome") in ("ABORTED", "FAILED_TO_START")
    ]
    if incomplete:
        out.append("### Why Triage Did Not Complete (Cohort A)")
        out.append("")
        out.append("| file_key | test_function | Outcome | Last script run | Reason |")
        out.append("|----------|---------------|---------|-----------------|--------|")
        reason_labels = {
            "E14": "E14 — CI action timeout",
            "E15": "E15 — no triage section",
            "E19": "E19 — Inspector RPC unavailable",
            "E20": "E20 — FW init failure",
            "E21": "E21 — ttexalens module not found",
            "E22": "E22 — device broken by previous triage",
            "E23": "E23 — triage crashed mid-script",
            "E24": "E24 — output truncated",
        }
        for r in incomplete:
            # Last script that actually produced output (not ABSENT).
            last = None
            for sn in ALL_SCRIPTS:
                s = script_status(r, sn)
                if s and s != "ABSENT":
                    last = sn
            outcome = "FAILED_TO_START" if r.get("init_failure") else r.get("triage_outcome", "ABORTED")
            reasons = []
            for p in r.get("known_patterns_found", []):
                pid = p.get("pattern_id")
                if pid in reason_labels:
                    reasons.append(reason_labels[pid])
            reason_str = "; ".join(dict.fromkeys(reasons)) or "unknown"
            out.append(
                f"| {r.get('file_key', '?')} | {r.get('test_function', '?')} | "
                f"{outcome} | {last or '—'} | {reason_str} |"
            )
        out.append("")

    # Script Reliability — Cohort A
    out.append("### Script Reliability — Cohort A (First Runs on Fresh Device)")
    out.append("")
    out.append("| Script | Total | PASS | EXPECTED | UNEXPECTED | Top Errors |")
    out.append("|--------|-------|------|----------|------------|------------|")
    for r in rel_a:
        out.append(
            f"| {r['script_name']} | {r['total_runs']} | "
            f"{r['pass_count']} ({r['pass_pct']}%) | "
            f"{r['expected_count']} ({r['expected_pct']}%) | "
            f"{r['unexpected_count']} ({r['unexpected_pct']}%) | "
            f"{r['top_unexpected_errors'] or '—'} |"
        )
    out.append("")
    out.append("**Scripts with >10% UNEXPECTED rate (Cohort A):**")
    for r in rel_a:
        if r["unexpected_pct"] > 10:
            out.append(
                f"- **{r['script_name']}**: {r['unexpected_pct']}% UNEXPECTED — {r['top_unexpected_errors'] or '—'}"
            )
    out.append("")

    # Cohort A vs B
    out.append("### Cohort A vs B (degradation on contaminated device)")
    out.append("")
    out.append("| Script | A UNEXPECTED% | B UNEXPECTED% | Δ pp |")
    out.append("|--------|--------------:|--------------:|-----:|")
    b_map = {r["script_name"]: r for r in rel_b}
    for r in rel_a:
        b = b_map.get(r["script_name"])
        if b:
            delta = round(b["unexpected_pct"] - r["unexpected_pct"], 1)
            out.append(
                f"| {r['script_name']} | {r['unexpected_pct']}% | {b['unexpected_pct']}% | {colored_delta(delta)} |"
            )
    out.append("")

    # Error patterns
    out.append("### Error Patterns (Cohort A)")
    out.append("")
    out.append("#### Triage Bugs (Actionable)")
    out.append("")
    out.append("| Pattern | Jobs | % | Avg/Job |")
    out.append("|---------|-----:|--:|--------:|")
    for r in sorted(pat_a, key=lambda x: -x["jobs_affected"]):
        if r["is_triage_bug"] or r["category"] == "triage_bug":
            out.append(
                f"| {r['pattern_id']}: {r['pattern_name']} | {r['jobs_affected']} | {r['jobs_pct']}% | {r['avg_per_job']} |"
            )
    out.append("")
    out.append("#### Environment Issues")
    out.append("")
    out.append("| Pattern | Jobs | % |")
    out.append("|---------|-----:|--:|")
    for r in sorted(pat_a, key=lambda x: -x["jobs_affected"]):
        if r["category"] == "environment":
            out.append(f"| {r['pattern_id']}: {r['pattern_name']} | {r['jobs_affected']} | {r['jobs_pct']}% |")
    out.append("")
    out.append("#### Diagnostic Findings (Triage Working Correctly)")
    out.append("")
    out.append("| Pattern | Jobs | % |")
    out.append("|---------|-----:|--:|")
    for r in sorted(pat_a, key=lambda x: -x["jobs_affected"]):
        if r["category"] == "diagnostic":
            out.append(f"| {r['pattern_id']}: {r['pattern_name']} | {r['jobs_affected']} | {r['jobs_pct']}% |")
    out.append("")

    # E25 throttle-level breakdown: how many of the MatMul-preceded hangs had
    # TT_MM_THROTTLE_PERF set vs unset? Throttle set + hang = mitigation didn't
    # prevent it; throttle unset + hang = unmitigated di/dt risk.
    e25_details = []
    for r in cohort_a:
        for p in r.get("known_patterns_found", []):
            if p.get("pattern_id") == "E25":
                e25_details.append(p.get("details", "") or "")
    if e25_details:
        throttle_dist = Counter()
        for d in e25_details:
            m = re.search(r"throttle\s*=\s*([^\s,]+)", d, re.IGNORECASE)
            throttle_dist[m.group(1) if m else "unknown"] += 1
        unset = throttle_dist.get("unset", 0) + throttle_dist.get("UNSET", 0)
        set_count = sum(v for k, v in throttle_dist.items() if k not in ("unset", "UNSET", "unknown"))
        levels = ", ".join(f"{k}={v}" for k, v in sorted(throttle_dist.items()))
        out.append(
            f"> **E25 throttle breakdown**: of {len(e25_details)} MatMul-preceded hangs in Cohort A, "
            f"{unset} had `TT_MM_THROTTLE_PERF` unset, {set_count} had it set "
            f"(level distribution: {levels}). Unset-and-hung suggests unmitigated di/dt risk; "
            f"set-and-still-hung suggests the throttle level was insufficient."
        )
        out.append("")

    # Informational patterns (E12, E18) — not shown above by category.
    info_rows = [r for r in pat_a if r["category"] == "informational"]
    if info_rows:
        out.append("#### Informational")
        out.append("")
        out.append("| Pattern | Jobs | % |")
        out.append("|---------|-----:|--:|")
        for r in sorted(info_rows, key=lambda x: -x["jobs_affected"]):
            out.append(f"| {r['pattern_id']}: {r['pattern_name']} | {r['jobs_affected']} | {r['jobs_pct']}% |")
        out.append("")

        # E12 per-risc breakdown — separates expected (erisc context switch)
        # from the interesting tensix-core tooling gap.
        if any(r["pattern_id"] == "E12" for r in info_rows):
            e12 = e12_risc_breakdown(cohort_a)
            if e12:
                erisc_count = sum(v for k, v in e12.items() if "erisc" in k)
                tensix_count = sum(v for k, v in e12.items() if k == "brisc" or k == "ncrisc" or k.startswith("trisc"))
                lines = ", ".join(f"{k}={v}" for k, v in e12.most_common())
                out.append(
                    f"> **E12 per-risc breakdown (Cohort A)**: {lines}. "
                    f"Erisc total: {erisc_count} (informational — PC context-switch to base ERISC firmware, expected on erisc). "
                    f"Tensix-core total: {tensix_count} (brisc/trisc/ncrisc — indicates tooling/ELF resolution gap on worker cores, worth investigating)."
                )
                out.append("")

    # Top tests by UNEXPECTED
    out.append("### Top Tests by UNEXPECTED Failure Count (Cohort A)")
    out.append("")
    out.append("| Test | Jobs | Worst Scripts |")
    out.append("|------|-----:|---------------|")
    test_rows_a = [r for r in test_rows if r["cohort"] == "A"]
    test_unexp = Counter()
    test_worst = defaultdict(list)
    test_jobs = {}
    for r in test_rows_a:
        if r["unexpected_count"] > 0:
            test_unexp[r["test_function"]] += r["unexpected_count"]
            test_worst[r["test_function"]].append((r["script_name"], r["unexpected_count"]))
        test_jobs[r["test_function"]] = r["total_jobs"]
    for tf, _ in test_unexp.most_common(10):
        worst = sorted(test_worst[tf], key=lambda x: -x[1])[:3]
        worst_str = ", ".join(f"{s}({c})" for s, c in worst)
        out.append(f"| {tf} | {test_jobs.get(tf, '?')} | {worst_str} |")
    out.append("")

    # New errors
    new_err_a = [r for r in new_err_rows if r["cohort"] == "A" and r["jobs_affected"] > 0]
    if new_err_a:
        out.append("### New Errors Discovered (Cohort A)")
        out.append("")
        for e in sorted(new_err_a, key=lambda x: -x["jobs_affected"]):
            out.append(f"#### {e['suggested_name']}")
            out.append(f"- **Script**: `{e['script_name']}`")
            out.append(f"- **Jobs affected**: {e['jobs_affected']}")
            et = e["error_text"][:300].replace("\n", " ").replace("|", "\\|")
            out.append(f"- **Error text**: `{et}`")
            out.append(f"- **Suggested regex**: `{e['suggested_regex']}`")
            if e["jobs_affected"] >= 3:
                out.append(f"- **Recommended action**: Add to catalog")
            out.append("")
    else:
        out.append("### New Errors Discovered")
        out.append("")
        out.append("*None detected this week.*")
        out.append("")

    # Week-over-Week Trends
    if prev_rel and prev_pat:
        out.append("### Week-over-Week Trends (Cohort A)")
        out.append("")

        prev_rel_a = [r for r in prev_rel if r.get("cohort") == "A" and r.get("script_name") != "_triage_init"]
        prev_rel_map = {r["script_name"]: r for r in prev_rel_a}

        out.append("#### Script UNEXPECTED% Trends")
        out.append("")
        out.append("| Script | Last Week | This Week | Δ pp |")
        out.append("|--------|----------:|----------:|-----:|")
        for r in rel_a:
            prev = prev_rel_map.get(r["script_name"])
            if prev:
                p_un = float(prev.get("unexpected_pct", 0) or 0)
                t_un = r["unexpected_pct"]
                delta = round(t_un - p_un, 1)
                out.append(f"| {r['script_name']} | {p_un}% | {t_un}% | {colored_delta(delta)} |")
        out.append("")

        prev_pat_a = [r for r in prev_pat if r.get("cohort") == "A"]
        prev_pat_map = {r["pattern_id"]: r for r in prev_pat_a}

        out.append("#### Error Pattern Trends (jobs%)")
        out.append("")
        out.append("| Pattern | Last Week | This Week | Δ pp |")
        out.append("|---------|----------:|----------:|-----:|")
        all_pids = sorted(set(list(prev_pat_map.keys()) + [r["pattern_id"] for r in pat_a]))
        for pid in all_pids:
            prev = prev_pat_map.get(pid)
            this = next((r for r in pat_a if r["pattern_id"] == pid), None)
            p_pct = float(prev.get("jobs_pct", 0) or 0) if prev else 0.0
            t_pct = this["jobs_pct"] if this else 0.0
            delta = round(t_pct - p_pct, 1)
            name = PATTERN_META.get(pid, (pid, "?", False))[0]
            out.append(f"| {pid}: {name} | {p_pct}% | {t_pct}% | {colored_delta(delta)} |")
        out.append("")

    # Recommendations derived from top triage-bug patterns present this week
    out.append("### Recommendations (Priority Order)")
    out.append("")
    pat_a_map = {r["pattern_id"]: r for r in pat_a}
    recs = {
        "E01": "**Fix FD leak in callstack_provider** — close ELF handles after each core",
        "E03": "**Cache fabric ERISC router ELFs** — ensure idle_erisc/subordinate_idle_erisc ELFs are written to cache",
        "E07": "**Fix ttexalens SyntaxWarning** — trivial escape-sequence fix",
        "E06": "**Add noc_mode to ERISC DWARF** — or infer NOC from other data",
    }
    # E04 (cores broken during triage) is NOT listed — it's a known HW halt/resume
    # limitation, not a triage bug. Tracked via category=environment instead.
    rec_idx = 1
    for pid in ("E01", "E03", "E07", "E06"):
        p = pat_a_map.get(pid)
        if p and p["jobs_affected"] > 0:
            out.append(
                f"{rec_idx}. {recs[pid]} — affects {p['jobs_affected']}/{n_total_a} ({p['jobs_pct']}%) of first runs."
            )
            rec_idx += 1
    out.append("")

    out.append("### Appendix: Data Files")
    out.append(f"- [Script Reliability](triage_script_reliability_{yyyymmdd}.csv)")
    out.append(f"- [Error Patterns](triage_error_patterns_{yyyymmdd}.csv)")
    out.append(f"- [Per-Test Breakdown](triage_per_test_breakdown_{yyyymmdd}.csv)")
    out.append(f"- [New Errors](triage_new_errors_{yyyymmdd}.csv)")
    out.append(f"- [Script Drill-down](triage_script_drilldown_{yyyymmdd}.md)")

    return "\n".join(out) + "\n"


def main():
    yyyymmdd = os.environ.get("TRIAGE_YYYYMMDD") or datetime.now().strftime("%Y%m%d")
    prev_yyyymmdd = os.environ.get("TRIAGE_PREV_YYYYMMDD") or (
        (datetime.strptime(yyyymmdd, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
    )
    week_end = datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    week_start = datetime.strptime(prev_yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")

    cohort_a, cohort_b = load_all_findings()
    print(f"Loaded Cohort A: {len(cohort_a)}, Cohort B: {len(cohort_b)}")

    # Backfill missing metadata from the split index (defensive — agents may
    # omit test_function/host_name/arch if not in their prompt).
    split_index = json.load(open(SPLIT_DIR / "index.json"))
    for entry in cohort_a + cohort_b:
        fk = entry.get("file_key")
        if fk and fk in split_index:
            for k in ("test_function", "host_name", "run_number", "total_runs", "original_job_id"):
                if not entry.get(k):
                    entry[k] = split_index[fk].get(k)
            if not entry.get("job_id"):
                entry["job_id"] = split_index[fk].get("original_job_id")

    rel_rows = build_script_reliability(cohort_a, "A") + build_script_reliability(cohort_b, "B")
    pat_rows = build_error_patterns(cohort_a, "A") + build_error_patterns(cohort_b, "B")
    test_rows = build_per_test(cohort_a, "A") + build_per_test(cohort_b, "B")
    new_err_rows = build_new_errors(cohort_a, "A") + build_new_errors(cohort_b, "B")

    print("Writing CSVs...")
    write_csv(
        OUTPUT_DIR / f"triage_script_reliability_{yyyymmdd}.csv",
        rel_rows,
        [
            "cohort",
            "script_name",
            "total_runs",
            "pass_count",
            "pass_pct",
            "expected_count",
            "expected_pct",
            "unexpected_count",
            "unexpected_pct",
            "top_unexpected_errors",
        ],
    )
    write_csv(
        OUTPUT_DIR / f"triage_error_patterns_{yyyymmdd}.csv",
        pat_rows,
        [
            "cohort",
            "pattern_id",
            "pattern_name",
            "category",
            "is_triage_bug",
            "jobs_affected",
            "jobs_pct",
            "total_occurrences",
            "avg_per_job",
        ],
    )
    write_csv(
        OUTPUT_DIR / f"triage_per_test_breakdown_{yyyymmdd}.csv",
        test_rows,
        [
            "cohort",
            "test_function",
            "arch",
            "total_jobs",
            "script_name",
            "pass_count",
            "expected_count",
            "unexpected_count",
            "dominant_pattern",
        ],
    )
    write_csv(
        OUTPUT_DIR / f"triage_new_errors_{yyyymmdd}.csv",
        new_err_rows,
        ["cohort", "script_name", "error_text", "suggested_name", "suggested_regex", "jobs_affected", "file_keys"],
    )

    prev_rel = load_prev_csv(OUTPUT_DIR / f"triage_script_reliability_{prev_yyyymmdd}.csv")
    prev_pat = load_prev_csv(OUTPUT_DIR / f"triage_error_patterns_{prev_yyyymmdd}.csv")

    report = build_report(
        cohort_a,
        cohort_b,
        rel_rows,
        pat_rows,
        test_rows,
        new_err_rows,
        prev_rel,
        prev_pat,
        yyyymmdd,
        week_start,
        week_end,
    )
    report_path = OUTPUT_DIR / f"triage_weekly_report_{yyyymmdd}.md"
    report_path.write_text(report)
    print(f"  wrote {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
