#!/usr/bin/env python3
"""
Produce a per-script drill-down of error types.

For each triage script, produces a table showing:
- How many jobs hit each specific error pattern
- Broken down by PASS / EXPECTED reason / UNEXPECTED reason / UNEXPECTED reason
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta

SPLIT_DIR = Path(__file__).resolve().parents[2] / "triage_outputs_split"
OUTPUT_DIR = Path(__file__).resolve().parents[2]
YYYYMMDD = os.environ.get("TRIAGE_YYYYMMDD") or datetime.now().strftime("%Y%m%d")
PREV_YYYYMMDD = os.environ.get("TRIAGE_PREV_YYYYMMDD") or (
    (datetime.strptime(YYYYMMDD, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
)

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


def parse_script_sections(text):
    sections = {}
    script_pattern = re.compile(r"^(\w+\.py):$", re.MULTILINE)
    matches = list(script_pattern.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]
    return sections


def is_fast_dispatch_enabled(sections):
    """Inspect dump_configuration output to determine fast-dispatch mode.

    Returns True if fast dispatch is enabled, False if slow dispatch, None if
    the rtOption row wasn't found (assume fast — current default).
    """
    cfg = sections.get("dump_configuration.py", "")
    m = re.search(r"\bfast_dispatch\s*│\s*(true|false)", cfg, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    # Fallback: TT_METAL_SLOW_DISPATCH_MODE env var sets slow dispatch when truthy.
    env_m = re.search(r"TT_METAL_SLOW_DISPATCH_MODE\s*│\s*(\S+)", cfg)
    if env_m and env_m.group(1).lower() not in ("", "0", "false", "none"):
        return False
    return None


def get_mm_throttle(sections):
    """Read TT_MM_THROTTLE_PERF from dump_configuration; returns value or 'unset'."""
    cfg = sections.get("dump_configuration.py", "")
    m = re.search(r"TT_MM_THROTTLE_PERF\s*│\s*(\S+)", cfg)
    return m.group(1) if m else "unset"


def classify_script_detailed(script_name, section_text, fast_dispatch=None):
    """Classify with a specific reason string for the drill-down.

    `fast_dispatch` is True / False / None (unknown). Used for scripts whose
    classification depends on dispatch mode (e.g. dump_running_operations).
    """
    s = section_text.strip()

    # UNEXPECTED: traceback/OSError in section
    if "Traceback (most recent call last)" in s or "OSError: [Errno 24]" in s:
        if re.search(r"rich.*unicode|rich.*cell", s, re.IGNORECASE):
            return "UNEXPECTED", "Errno 24 cascade → Rich library can't load unicode data"
        if re.search(r"/etc/os-release", s):
            return "UNEXPECTED", "Errno 24 cascade → can't open /etc/os-release"
        return "UNEXPECTED", "Unhandled exception/traceback"

    # --- Per-script classification with specific reasons ---

    if script_name == "check_noc_status.py":
        reasons = []
        if re.search(r"Cannot find global variable noc_mode", s):
            reasons.append("UNEXPECTED: Missing noc_mode DWARF variable (E06)")
        if re.search(r"Skipping:.*is not halted", s):
            reasons.append("UNEXPECTED: Cores not halted, skipped (E08)")
        if re.search(r"Skipping: '(?:tensix|active_eth|idle_eth)'", s):
            reasons.append("UNEXPECTED: No cores available (E10)")
        if re.search(r"unsafe access", s):
            reasons.append("UNEXPECTED: Unsafe ARC access (E05)")
        if reasons:
            return "UNEXPECTED", "; ".join(reasons)
        if re.search(r"Mismatched state:", s):
            return "EXPECTED", "NOC transaction mismatch detected (E16)"
        return "PASS", "PASS"

    if script_name == "check_binary_integrity.py":
        if re.search(r"Data mismatch in section \.text", s):
            return "EXPECTED", "Binary corruption detected on device (E11)"
        if re.search(r"Skipping:.*is not halted", s):
            return "UNEXPECTED", "Cores not halted, skipped (E08)"
        if re.search(r"Skipping: '(?:tensix|idle_eth)'", s):
            return "UNEXPECTED", "No cores available (E10)"
        if re.search(r"ELF file.*does not exist", s):
            return "UNEXPECTED", "ELF file missing (E03)"
        return "PASS", "PASS"

    if script_name == "check_arc.py":
        if re.search(r"unsafe access at address", s):
            return "UNEXPECTED", "Unsafe ARC access on remote WH device (E05)"
        return "PASS", "PASS"

    if script_name == "check_broken_components.py":
        has_triage_broke = re.search(r"Was halted by triage but is no longer halted", s)
        has_errno = re.search(r"OSError.*Errno 24|Traceback", s)
        if has_errno:
            return "UNEXPECTED", "Errno 24 cascade → Rich library can't load unicode data"
        if has_triage_broke:
            return "EXPECTED", "Cores broken during triage halt/resume (E04)"
        return "PASS", "PASS"

    if script_name == "check_eth_status.py":
        if re.search(r"retrain count|port.*down|no heartbeat", s, re.IGNORECASE):
            return "EXPECTED", "Ethernet link issue detected"
        return "PASS", "PASS"

    if script_name in ("dump_callstacks.py", "dump_aggregated_callstacks.py"):
        reasons = []
        if re.search(r"\[Errno 24\]", s):
            reasons.append("FD exhaustion — Errno 24 (E01)")
        if re.search(r"fabric_erisc_router.*does not exist", s):
            reasons.append("Missing fabric ERISC router ELF (E03)")
        if re.search(r"Failed to halt", s):
            reasons.append("Failed to halt core (E09)")
        if reasons:
            return "UNEXPECTED", "; ".join(reasons)
        return "PASS", "PASS"

    if script_name == "dump_fast_dispatch.py":
        if re.search(r"Failed to halt|Failed to read symbol", s):
            return "UNEXPECTED", "Failed to halt/read dispatch core symbols (E09)"
        return "PASS", "PASS"

    if script_name == "system_info.py":
        if re.search(r"\[Errno 24\]|OSError", s):
            return "UNEXPECTED", "Errno 24 cascade → can't open /etc/os-release"
        return "PASS", "PASS"

    if script_name == "device_info.py":
        if re.search(r"unsafe access at address", s):
            return "UNEXPECTED", "Unsafe ARC access in Postcode column (E05)"
        return "PASS", "PASS"

    if script_name == "dump_running_operations.py":
        # E25 wins over the slow-dispatch N/A case — more specific.
        # Table layout: │ Op Id │ Op Name │ Op Params │ Prev Op Id │ Prev Op Name │ …
        # Match Matmul only when it appears in the 5th cell (Prev Op Name),
        # not the 2nd cell (current Op Name).
        if re.search(r"│\s*\d+\s*│[^│]*│[^│]*│\s*\S+\s*│[^│]*[Mm]atmul", s):
            return "EXPECTED", "MatMul preceded hang — likely di/dt (E25)"
        if re.search(r"│\s+\d+\s+│\s+N/A\s+│\s+N/A\s+│", s):
            if fast_dispatch is False:
                return "EXPECTED", "Op Name N/A — slow dispatch (op tracking not yet supported)"
            return "UNEXPECTED", "Current Op Name/Params N/A — metadata resolution failed (fast dispatch)"
        return "PASS", "PASS"

    if script_name == "dump_lightweight_asserts.py":
        reasons = []
        if re.search(r"\[Errno 24\]", s):
            reasons.append("FD exhaustion — Errno 24 (E01)")
        if re.search(r"does not exist", s):
            reasons.append("Missing ELF file (E03)")
        if reasons:
            return "UNEXPECTED", "; ".join(reasons)
        return "PASS", "PASS"

    if script_name in (
        "check_cb_inactive.py",
        "check_noc_locations.py",
        "check_core_magic.py",
        "dump_watcher_ringbuffer.py",
    ):
        if re.search(r"Skipping: '(?:tensix|active_eth|idle_eth)'", s):
            return "UNEXPECTED", "No cores available — Skipping tensix/eth (E10)"
        return "PASS", "PASS"

    return "PASS", "PASS"


def detect_arch(text):
    chunk = text[:10000].lower()
    if "blackhole" in chunk:
        return "blackhole"
    if "wormhole" in chunk:
        return "wormhole_b0"
    return "unknown"


def analyze_cohort(file_keys):
    """Classify per script for a set of file_keys.

    Returns (script_reasons, script_status, total_files, reason_files):
      - script_reasons[script][label] = count
      - script_status[script][status] = count
      - reason_files[script][label] = [file_key, ...] (ordered by appearance)
    """
    script_reasons = defaultdict(lambda: Counter())
    script_status = defaultdict(lambda: Counter())
    reason_files = defaultdict(lambda: defaultdict(list))
    total_files = 0
    for file_key in file_keys:
        fpath = SPLIT_DIR / f"{file_key}.txt"
        if not fpath.exists():
            continue
        text = fpath.read_text()
        total_files += 1
        if text.strip() == "[NO TRIAGE SECTION FOUND]":
            for sn in ALL_SCRIPTS:
                script_reasons[sn]["ABSENT: No triage section (E15)"] += 1
                script_status[sn]["ABSENT"] += 1
                reason_files[sn]["ABSENT: No triage section (E15)"].append(file_key)
            continue
        sections = parse_script_sections(text)
        fast_dispatch = is_fast_dispatch_enabled(sections)
        for script_name in ALL_SCRIPTS:
            if script_name in sections:
                status, reason = classify_script_detailed(
                    script_name, sections[script_name], fast_dispatch=fast_dispatch
                )
                label = f"{status}: {reason}"
                script_reasons[script_name][label] += 1
                script_status[script_name][status] += 1
                reason_files[script_name][label].append(file_key)
            else:
                label = "ABSENT: Script not present"
                script_reasons[script_name][label] += 1
                script_status[script_name]["ABSENT"] += 1
                reason_files[script_name][label].append(file_key)
    return script_reasons, script_status, total_files, reason_files


def render_cohort_section(cohort_label, subtitle, script_reasons, script_status, total_files):
    """Markdown section body for one cohort."""
    md = f"## {cohort_label} — {total_files} {subtitle}\n\n"
    md += "For each script, a breakdown of outcomes by specific error type. "
    md += "Only non-PASS outcomes are shown in the detail tables.\n\n---\n\n"

    for script_name in ALL_SCRIPTS:
        st = script_status[script_name]
        total_present = st["PASS"] + st["EXPECTED"] + st["UNEXPECTED"]
        if total_present == 0:
            continue

        pass_pct = round(st["PASS"] / total_present * 100, 1)
        expected_pct = round(st["EXPECTED"] / total_present * 100, 1)
        unexpected_pct = round(st["UNEXPECTED"] / total_present * 100, 1)

        md += f"### `{script_name}`\n\n"
        md += f"**Summary**: {total_present} runs | "
        md += f"PASS: {st['PASS']} ({pass_pct}%) | "
        md += f"EXPECTED: {st['EXPECTED']} ({expected_pct}%) | "
        md += f"UNEXPECTED: {st['UNEXPECTED']} ({unexpected_pct}%)"
        if st["ABSENT"] > 0:
            md += f" | ABSENT: {st['ABSENT']}"
        md += "\n\n"

        non_pass = [
            (reason, count) for reason, count in script_reasons[script_name].items() if not reason.startswith("PASS")
        ]
        non_pass.sort(key=lambda x: -x[1])
        if non_pass:
            md += "| Status | Reason | Count | % of runs |\n"
            md += "|--------|--------|------:|----------:|\n"
            for reason, count in non_pass:
                status_tag = reason.split(":")[0]
                reason_text = reason.split(": ", 1)[1] if ": " in reason else reason
                pct = round(count / total_present * 100, 1)
                md += f"| {status_tag} | {reason_text} | {count} | {pct}% |\n"
            md += "\n"
        else:
            md += "*All runs passed — no issues detected.*\n\n"
        md += "---\n\n"
    return md


def colored_delta(d, good_if="down"):
    """Format a delta as a colored markdown math-span.

    See consolidate_report.colored_delta for semantics.
    """
    if d == 0:
        return "+0"
    is_good = (d < 0 and good_if == "down") or (d > 0 and good_if == "up")
    color = "green" if is_good else "red"
    sign = "+" if d > 0 else ""
    return rf"$\color{{{color}}}{{{sign}{d}}}$"


def load_prev_drilldown(prev_path):
    """Return {(cohort, script): {reason: (count, pct)}}. Empty if missing.

    Last week's CSV may be missing the `cohort` column — in that case treat
    all rows as Cohort A (the only one that existed before the split).
    """
    if not prev_path.exists():
        return {}
    out = defaultdict(dict)
    reader = csv.DictReader(open(prev_path))
    for row in reader:
        cohort = row.get("cohort") or "A"
        key = (cohort, row["script_name"])
        status = row.get("status", "")
        reason = row.get("reason", "")
        label = f"{status}: {reason}"
        try:
            cnt = int(row.get("count") or 0)
            pct = float(row.get("pct_of_runs") or 0)
        except ValueError:
            continue
        out[key][label] = (cnt, pct)
    return out


def render_wow_section(cohort_tag, script_reasons, script_status, prev_drilldown):
    """Render a per-script week-over-week comparison section.

    For each script, shows PASS/EXPECTED/UNEXPECTED% delta vs. previous week,
    and for each non-PASS reason that appeared in either week, the count/pct
    delta with a REGRESSION/IMPROVEMENT/NEW/CLEARED tag.
    """
    md = f"## Week-over-Week — Cohort {cohort_tag}\n\n"
    md += f"Comparison against `triage_script_drilldown_{PREV_YYYYMMDD}.csv`. "
    md += "Threshold: |Δ pp| > 5 = REGRESSION/IMPROVEMENT; appeared from 0 = NEW; cleared to 0 = CLEARED.\n\n"

    any_script_rendered = False
    for script_name in ALL_SCRIPTS:
        st = script_status[script_name]
        total_present = st["PASS"] + st["EXPECTED"] + st["UNEXPECTED"]
        prev = prev_drilldown.get((cohort_tag, script_name), {})
        if total_present == 0 and not prev:
            continue

        # Recompute this-week % per status from script_status
        this_pass_pct = round(st["PASS"] / total_present * 100, 1) if total_present else 0.0
        this_exp_pct = round(st["EXPECTED"] / total_present * 100, 1) if total_present else 0.0
        this_unexp_pct = round(st["UNEXPECTED"] / total_present * 100, 1) if total_present else 0.0

        # Sum prev-week per status from its reasons
        prev_status_pct = Counter()
        for label, (_, pct) in prev.items():
            status_tag = label.split(":")[0]
            prev_status_pct[status_tag] += pct

        # Collect the union of reason-labels (skip the PASS/PASS entry — tracked at status level above)
        labels = set(l for l in script_reasons[script_name].keys() if not l.startswith("PASS"))
        labels |= set(l for l in prev.keys() if not l.startswith("PASS"))
        if not labels and total_present == 0:
            continue

        any_script_rendered = True
        md += f"### `{script_name}`\n\n"

        # Status-level summary
        md += "| Status | Last Week % | This Week % | Δ pp |\n"
        md += "|--------|------------:|------------:|-----:|\n"
        # PASS up is good (green). UNEXPECTED up is bad (red). EXPECTED up
        # generally means the script surfaced more known findings — neutral,
        # so we still color it by "down is good" for consistency with error
        # rates (more findings = more issues showing up).
        status_direction = {"PASS": "up", "EXPECTED": "down", "UNEXPECTED": "down"}
        for status_tag, this_pct in (
            ("PASS", this_pass_pct),
            ("EXPECTED", this_exp_pct),
            ("UNEXPECTED", this_unexp_pct),
        ):
            p_pct = round(prev_status_pct.get(status_tag, 0.0), 1)
            delta = round(this_pct - p_pct, 1)
            md += f"| {status_tag} | {p_pct}% | {this_pct}% | {colored_delta(delta, good_if=status_direction[status_tag])} |\n"
        md += "\n"

        # Reason-level deltas (non-PASS only). Sign + color already convey the
        # trend (regression/improvement/stable), and Last/This Week columns
        # make NEW/CLEARED obvious, so we drop the Trend column.
        if labels:
            md += "| Status | Reason | Last Week | This Week | Δ pp |\n"
            md += "|--------|--------|----------:|----------:|-----:|\n"
            rows = []
            for label in labels:
                this_count = script_reasons[script_name].get(label, 0)
                this_pct = round(this_count / total_present * 100, 1) if total_present else 0.0
                _, prev_pct = prev.get(label, (0, 0.0))
                delta = round(this_pct - prev_pct, 1)
                status_tag = label.split(":")[0]
                reason_text = label.split(": ", 1)[1] if ": " in label else label
                rows.append((status_tag, reason_text, prev_pct, this_pct, delta, abs(delta)))
            rows.sort(key=lambda r: -r[5])
            for status_tag, reason_text, p, t, d, _ in rows:
                md += f"| {status_tag} | {reason_text} | {p}% | {t}% | {colored_delta(d)} |\n"
            md += "\n"
        md += "---\n\n"

    if not any_script_rendered:
        md += "*No comparable data — previous-week drilldown CSV missing or empty.*\n\n"
    return md


def csv_rows_for_cohort(cohort_tag, script_reasons, script_status, reason_files):
    rows = []
    for script_name in ALL_SCRIPTS:
        st = script_status[script_name]
        total_present = st["PASS"] + st["EXPECTED"] + st["UNEXPECTED"]
        for reason, count in sorted(script_reasons[script_name].items(), key=lambda x: -x[1]):
            status_tag = reason.split(":")[0]
            reason_text = reason.split(": ", 1)[1] if ": " in reason else reason
            files = reason_files[script_name].get(reason, [])
            rows.append(
                {
                    "cohort": cohort_tag,
                    "script_name": script_name,
                    "status": status_tag,
                    "reason": reason_text,
                    "count": count,
                    "pct_of_runs": round(count / total_present * 100, 1) if total_present else 0,
                    "file_keys": ";".join(files),
                }
            )
    return rows


def main():
    index = json.load(open(SPLIT_DIR / "index.json"))
    cohort_a_keys = sorted([k for k, v in index.items() if v.get("run_number") == 1])
    cohort_b_keys = sorted([k for k, v in index.items() if v.get("run_number", 1) > 1])

    # Cohort A is always analyzed fully. Cohort B: the dispatch step may have
    # sampled or not per the <200-runs rule in the playbook — here we simply
    # classify whichever split files are present on disk.
    a_reasons, a_status, a_total, a_files = analyze_cohort(cohort_a_keys)
    b_reasons, b_status, b_total, b_files = analyze_cohort(cohort_b_keys)

    # --- Markdown ---
    report = "# tt-triage Script Reliability Drill-Down\n\n"
    report += render_cohort_section(
        "Cohort A (First Runs on Fresh Device)",
        "jobs analyzed",
        a_reasons,
        a_status,
        a_total,
    )
    if b_total > 0:
        report += render_cohort_section(
            "Cohort B (Subsequent Runs on Contaminated Device)",
            "runs analyzed",
            b_reasons,
            b_status,
            b_total,
        )

    # Week-over-week — only emit if previous-week drilldown CSV exists
    prev_drilldown = load_prev_drilldown(OUTPUT_DIR / f"triage_script_drilldown_{PREV_YYYYMMDD}.csv")
    if prev_drilldown:
        report += render_wow_section("A", a_reasons, a_status, prev_drilldown)
        if b_total > 0 and any(k[0] == "B" for k in prev_drilldown):
            report += render_wow_section("B", b_reasons, b_status, prev_drilldown)

    # --- CSV ---
    csv_rows = csv_rows_for_cohort("A", a_reasons, a_status, a_files)
    if b_total > 0:
        csv_rows.extend(csv_rows_for_cohort("B", b_reasons, b_status, b_files))

    csv_path = OUTPUT_DIR / f"triage_script_drilldown_{YYYYMMDD}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["cohort", "script_name", "status", "reason", "count", "pct_of_runs", "file_keys"]
        )
        w.writeheader()
        w.writerows(csv_rows)

    report_path = OUTPUT_DIR / f"triage_script_drilldown_{YYYYMMDD}.md"
    report_path.write_text(report)

    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Cohort A: {a_total} files | Cohort B: {b_total} files")

    # Per-cohort summary
    for tag, st_map in (("A", a_status), ("B", b_status)):
        print(f"\n=== Cohort {tag}: scripts with non-PASS outcomes ===")
        reasons_map = a_reasons if tag == "A" else b_reasons
        for script_name in ALL_SCRIPTS:
            st = st_map[script_name]
            issues = st["EXPECTED"] + st["UNEXPECTED"]
            if issues > 0:
                total_present = st["PASS"] + st["EXPECTED"] + st["UNEXPECTED"]
                print(f"\n  {script_name} ({issues}/{total_present} non-PASS):")
                for reason, count in sorted(reasons_map[script_name].items(), key=lambda x: -x[1]):
                    if not reason.startswith("PASS"):
                        print(f"    {count:4d}  {reason}")


if __name__ == "__main__":
    main()
