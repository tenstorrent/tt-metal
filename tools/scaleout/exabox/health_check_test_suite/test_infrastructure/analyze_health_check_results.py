#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Translate a fabric health-check diag_report.json into CSVs for Superset.

The diag tool emits a structured diag_report.json (phases -> checks + per-chip
telemetry). This writes two CSVs per host per run, joined on run_id
({hostname}:{slurm_job_id}):

    health_check_{hostname}_{slurm_job_id}.runs.csv    1 row, machine rollup
    health_check_{hostname}_{slurm_job_id}.checks.csv  one row per check

The caller (run_health_check.py) passes an already-normalized report so the
verdict reflects post-reset state. Standalone, the CLI applies that same
normalize first.

If no report is produced (timeout/kill) or it can't be parsed, a 1-row runs.csv
with overall_status=ERROR is written (no checks file) so a run is never dropped.

Output is validated against the Pydantic models in utils/health_check_models.py
when available. Schema: docs/exabox-fabric-health-check.md. stdlib-only.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# check ip/phase -> dashboard category. "other" is the catch-all.
IP_TO_CATEGORY = {
    "pcie": "pcie",
    "gddr": "gddr",
    "asic": "asic",
    "fw": "firmware",
    "thermal": "thermal",
    "board": "board",
    "eth": "eth",
}
# eth deployment tests live in the "tests" phase with ip="other". Route them to
# the eth category by name so they don't land in stress_test.
CHECK_CATEGORY = {
    "post_reset_state_stable": "reset",
    "eth_bandwidth": "eth",
    "eth_link_up": "eth",
}
STRESS_PHASE = "tests"

# Benign fleet-wide WARNs: tracked but kept out of the actionable rollups.
ACKNOWLEDGED_CHECKS = {"cpld_fw_old"}

# Infra/capture steps (not hardware). A FAIL/WARN here is a tooling hiccup, not a
# verdict, so it is re-labelled EXCLUDED: kept visible, counts as nothing.
EXCLUDED_CHECKS = {"snapshot_capture"}

# checks whose details list offending BDFs - keep more text for triage.
DETAIL_RICH = {"pcie_gen", "gddr_speed", "pcie_lane_width", "physical_vs_fw_location", "asic_location_per_ubb"}

# numeric forensic checks: dropped from checks_fact, folded into runs rollups.
GDDR_INFO_PREFIX = "gddr_info_"

SEVERITY = {"PASS": 0, "SKIP": 0, "EXCLUDED": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": 3, "ERROR": 3}
COVERED = {"PASS", "WARN", "FAIL"}

RUNS_COLS = [
    "schema_version",
    "run_id",
    "date",
    "timestamp",
    "hostname",
    "slurm_job_id",
    "row",
    "rack",
    "slot",
    "overall_status",
    "discard",
    "discard_reason",
    "prev_status",
    "is_regression",
    "fail_streak",
    "tier",
    "tool_version",
    "tt_smi_version",
    "tt_kmd_version",
    "board_rev",
    "fw_bundle_version",
    "num_chips",
    "total_duration_s",
    "checks_total",
    "checks_pass",
    "checks_warn",
    "checks_fail",
    "checks_skip",
    "checks_covered",
    "pct_covered",
    "checks_warn_actionable",
    "top_fail_category",
    "reset_count",
    "reset_stable",
    "gddr_uncorr_total",
    "pcie_downgraded",
    "eth_links_down",
    "max_asic_temp_c",
    "max_gddr_temp_c",
    "min_aiclk_mhz",
    "telemetry_available",
    "eth_retrain_total",
    "eth_crc_total",
    "eth_uncorr_cw_total",
    "jira_ticket",
]

# checks_fact is normalized: check fields + run_id (+ date). Run-level attributes
# live once on runs.csv and join back on run_id.
CHECKS_COLS = [
    "schema_version",
    "run_id",
    "date",
    "category",
    "phase",
    "phase_kind",
    "check_name",
    "status",
    "severity",
    "is_pass",
    "is_warn",
    "is_fail",
    "is_skip",
    "is_covered",
    "acknowledged",
    "testcases_passed",
    "testcases_failed",
    "executed",
    "details_short",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_host(hostname: str):
    """(row, rack, slot) from the two BH-Galaxy naming schemes.

    bh-glx-110-c03u08 -> ("110", "c03", "08"), bh-glx-b02u08 -> ("", "b02", "08").
    Unparseable hosts get rack="unparsed" so they surface instead of blanking out.
    """
    m = re.match(r"bh-glx-(\d+)-([a-z]\d+)u(\d+)", hostname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"bh-glx-([a-z]\d+)u(\d+)", hostname)
    if m:
        return "", m.group(1), m.group(2)
    return "", "unparsed", ""


def phase_kind(phase_name: str) -> str:
    return "post_reset" if phase_name.startswith("snapshot_after_") else "primary"


def category_for(phase: str, check: dict) -> str:
    name = check.get("name", "")
    if name in CHECK_CATEGORY:
        return CHECK_CATEGORY[name]
    if phase == "reset_loop":
        return "reset"
    if phase == STRESS_PHASE:
        return "stress_test"
    return IP_TO_CATEGORY.get(check.get("ip", ""), "other")


def iter_checks(report: dict, primary_only: bool = True):
    for pname, ph in report.get("phases", {}).items():
        if primary_only and phase_kind(pname) != "primary":
            continue
        for c in ph.get("checks", []):
            yield pname, ph.get("status", ""), c


# fw reads back as all-0xFF when the device can't be queried. Treat those values
# as unknown rather than a real firmware version.
FW_SENTINELS = {"255.255.255.255", "0.0.0.0"}


def _find_check(report: dict, name: str, primary_only: bool = True):
    for _p, _s, c in iter_checks(report, primary_only=primary_only):
        if c.get("name") == name:
            return c
    return None


def _hexint(v):
    try:
        return int(str(v), 16)
    except (ValueError, TypeError):
        return None


def _max_asic_temp(report: dict):
    c = _find_check(report, "asic_thermal_precheck")
    if not c:
        return ""
    temps = (c.get("data", {}) or {}).get("all_temps_c", {}) or {}
    vals = [float(v) for v in temps.values() if v not in ("", None)]
    return round(max(vals), 1) if vals else ""


def _gddr_temp_and_uncorr(report: dict):
    max_t, uncorr = "", ""
    ct = _find_check(report, "gddr_info_max_gddr_temp")
    if ct:
        vals = [_hexint(v) for v in (ct.get("data", {}).get("per_chip", {}) or {}).values()]
        vals = [v for v in vals if v is not None]
        max_t = max(vals) if vals else ""
    cu = _find_check(report, "gddr_info_gddr_uncorr_errs")
    if cu:
        vals = [_hexint(v) for v in (cu.get("data", {}).get("per_chip", {}) or {}).values()]
        uncorr = sum(v for v in vals if v is not None)
    return max_t, uncorr


def _reset_info(report: dict):
    ph = report.get("phases", {}).get("reset_loop")
    if not ph:
        return "", ""
    n = len([c for c in ph.get("checks", []) if c.get("name", "").startswith("reset_")])
    return n, int(ph.get("status") != "FAIL")


def _pcie_downgraded(report: dict):
    c = _find_check(report, "pcie_gen")
    if not c:
        return ""
    return int(bool((c.get("data", {}) or {}).get("under_expected") or []))


def _eth_links_down(report: dict):
    c = _find_check(report, "eth_links_up")
    if not c or c.get("status") == "SKIP":
        return ""
    m = re.search(r"(\d+)\s*/\s*(\d+)", c.get("details", "") or "")
    return int(m.group(2)) - int(m.group(1)) if m else ""


def _testcases(check: dict):
    m = re.search(r"passed=(\d+)\s+failed=(\d+)", check.get("details", "") or "")
    return (int(m.group(1)), int(m.group(2))) if m else ("", "")


def machine_meta(report, hostname, job_id, jira_ticket, ts, versions=None):
    versions = versions or {}
    # fw can be missing from the primary snapshot when a run fails early. Fall
    # back to a post-reset snapshot, then the console log, ignoring 0xFF reads.
    fw = (
        _find_check(report, "fw_bundle_version_consistent")
        or _find_check(report, "fw_bundle_version_consistent", primary_only=False)
        or {}
    )
    log_fw = versions.get("fw_bundle_version", "")
    if log_fw in FW_SENTINELS:
        log_fw = ""
    report_fw = fw.get("data", {}).get("value", "")
    if report_fw in FW_SENTINELS:
        report_fw = ""
    fw_bundle = str(report_fw or log_fw)
    enum = _find_check(report, "pcie_enum_count") or {}
    num_chips = enum.get("data", {}).get("actual", "")
    overall = report.get("overall_status", "") or "UNKNOWN"
    row, rack, slot = parse_host(hostname)
    dry = bool(report.get("dry_run"))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{hostname}:{job_id}",
        "date": ts[:10],
        "timestamp": ts,
        "hostname": hostname,
        "row": row,
        "rack": rack,
        "slot": slot,
        "slurm_job_id": job_id,
        "overall_status": overall,
        "discard": int(dry),
        "discard_reason": "dry_run" if dry else "",
        "tier": report.get("tier", ""),
        "tool_version": report.get("tool_version", ""),
        "tt_smi_version": report.get("tt_smi_version", "") or versions.get("tt_smi_version", ""),
        "tt_kmd_version": versions.get("tt_kmd_version", ""),
        "board_rev": report.get("detected_board_rev", ""),
        "fw_bundle_version": fw_bundle,
        "num_chips": num_chips,
        "total_duration_s": report.get("total_duration_s", ""),
        "jira_ticket": jira_ticket,
    }


def checks_rows(report: dict, meta: dict):
    rows = []
    for pname, _ps, c in iter_checks(report):
        name = c.get("name", "")
        if name.startswith(GDDR_INFO_PREFIX):
            continue
        st = c.get("status", "")
        # a failed capture/precondition step is not a hardware verdict: mark it
        # EXCLUDED so it never counts as a fail (see EXCLUDED_CHECKS).
        excluded = name in EXCLUDED_CHECKS and st in ("FAIL", "WARN")
        if excluded:
            st = "EXCLUDED"
        cat = category_for(pname, c)
        tp, tf = _testcases(c)
        ran = isinstance(tp, int) and (tp + tf) > 0
        executed = 0 if (cat == "stress_test" and not ran) else 1
        cap = 400 if name in DETAIL_RICH else 140
        rows.append(
            {
                "schema_version": meta["schema_version"],
                "run_id": meta["run_id"],
                "date": meta["date"],
                "category": cat,
                "phase": pname,
                "phase_kind": phase_kind(pname),
                "check_name": name,
                "status": st,
                "severity": SEVERITY.get(st, 3),
                "is_pass": int(st == "PASS"),
                "is_warn": int(st == "WARN"),
                "is_fail": int(st == "FAIL"),
                "is_skip": int(st == "SKIP"),
                "is_covered": int(st in COVERED and executed == 1),
                "acknowledged": int(name in ACKNOWLEDGED_CHECKS or excluded),
                "testcases_passed": tp,
                "testcases_failed": tf,
                "executed": executed,
                "details_short": (c.get("details", "") or "").replace("\n", " | ")[:cap],
            }
        )
    return rows


def runs_row(report, meta, checks, telemetry=None):
    tel = telemetry or {}
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    fail_cat, warn_cat = {}, {}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
        # rollups skip acknowledged checks so top_fail_category reflects real hw.
        if c["status"] == "FAIL" and not c.get("acknowledged"):
            fail_cat[c["category"]] = fail_cat.get(c["category"], 0) + 1
        elif c["status"] == "WARN" and not c.get("acknowledged"):
            warn_cat[c["category"]] = warn_cat.get(c["category"], 0) + 1
    if fail_cat:
        top = max(fail_cat, key=fail_cat.get)
    elif warn_cat:
        top = max(warn_cat, key=warn_cat.get)
    else:
        top = ""
    total = len(checks)
    covered = sum(c["is_covered"] for c in checks)
    reset_count, reset_stable = _reset_info(report)
    max_gddr_t, gddr_uncorr = _gddr_temp_and_uncorr(report)

    # Effective verdict = worst non-excluded, non-acknowledged check. The raw
    # report overall_status also trips on pre-reset and EXCLUDED steps, which
    # aren't real verdicts. Mirrors the runner's has_actionable_failure() gate.
    if not checks:
        overall_status = meta["overall_status"]
    elif any(c["is_fail"] and not c.get("acknowledged") for c in checks):
        overall_status = "FAIL"
    elif any(c["is_warn"] and not c.get("acknowledged") for c in checks):
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    return {
        **{
            k: meta[k]
            for k in (
                "schema_version",
                "run_id",
                "date",
                "timestamp",
                "hostname",
                "row",
                "rack",
                "slot",
                "slurm_job_id",
                "discard",
                "discard_reason",
                "tier",
                "tool_version",
                "tt_smi_version",
                "tt_kmd_version",
                "board_rev",
                "fw_bundle_version",
                "num_chips",
                "total_duration_s",
                "jira_ticket",
            )
        },
        "overall_status": overall_status,
        "prev_status": "",
        "is_regression": "",
        "fail_streak": "",
        "checks_total": total,
        "checks_pass": counts["PASS"],
        "checks_warn": counts["WARN"],
        "checks_fail": counts["FAIL"],
        "checks_skip": counts["SKIP"],
        "checks_covered": covered,
        "pct_covered": round(100 * covered / total, 1) if total else "",
        "checks_warn_actionable": sum(warn_cat.values()),
        "top_fail_category": top,
        "reset_count": reset_count,
        "reset_stable": reset_stable,
        "gddr_uncorr_total": gddr_uncorr,
        "pcie_downgraded": _pcie_downgraded(report),
        "eth_links_down": _eth_links_down(report),
        "max_asic_temp_c": _max_asic_temp(report),
        "max_gddr_temp_c": max_gddr_t,
        "min_aiclk_mhz": tel.get("min_aiclk_mhz", ""),
        "telemetry_available": int(bool(tel.get("available"))),
        "eth_retrain_total": tel.get("eth_retrain_total", ""),
        "eth_crc_total": tel.get("eth_crc_total", ""),
        "eth_uncorr_cw_total": tel.get("eth_uncorr_cw_total", ""),
    }


def error_runs_row(hostname, job_id, ts, reason, jira_ticket=""):
    row, rack, slot = parse_host(hostname)
    r = {c: "" for c in RUNS_COLS}
    r.update(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": f"{hostname}:{job_id}",
            "date": ts[:10],
            "timestamp": ts,
            "hostname": hostname,
            "row": row,
            "rack": rack,
            "slot": slot,
            "slurm_job_id": job_id,
            "overall_status": "ERROR",
            "discard": 0,
            "top_fail_category": "run",
            "checks_total": 0,
            "checks_pass": 0,
            "checks_warn": 0,
            "checks_fail": 0,
            "checks_skip": 0,
            "checks_covered": 0,
            "checks_warn_actionable": 0,
            "telemetry_available": 0,
            "jira_ticket": jira_ticket,
        }
    )
    return r


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _validate(runs, checks):
    """Validate emitted rows against the Pydantic schema (health_check_models)
    when importable. Best-effort: logs drift, never blocks CSV production."""
    try:
        from utils.health_check_models import RunRecord, CheckRecord
    except Exception:
        return
    bad = 0
    for r in runs:
        try:
            RunRecord(**{c: r.get(c, "") for c in RUNS_COLS})
        except Exception as exc:
            bad += 1
            print(f"WARNING: runs row failed schema validation: " f"{str(exc).splitlines()[0]}", file=sys.stderr)
    for c in checks:
        try:
            CheckRecord(**{k: c.get(k, "") for k in CHECKS_COLS})
        except Exception as exc:
            bad += 1
            print(f"WARNING: checks row failed schema validation: " f"{str(exc).splitlines()[0]}", file=sys.stderr)
    if bad == 0:
        print("schema validation: OK")


def _normalize(report):
    """Apply the post-reset verdict normalization (single source of truth in
    report.py). Standalone-CLI only: in prod the runner passes a normalized
    report."""
    try:
        from utils.report import normalize_health_report
    except Exception as exc:
        print(
            f"WARNING: could not import utils.report.normalize_health_report ({exc}); " "using report as-is",
            file=sys.stderr,
        )
        return report
    return normalize_health_report(report)


def analyze(
    report,
    csv_output_dir,
    hostname,
    slurm_job_id,
    jira_ticket="",
    versions=None,
    telemetry=None,
    discard=None,
    discard_reason=None,
):
    """Translate a normalized diag_report.json dict into runs.csv (+ checks.csv).

    report must already be normalized by the caller so the verdict reflects
    post-reset state. A missing/empty report (None) yields a lone ERROR runs.csv.
    Returns the list of CSV paths written.

    A truthy ``discard`` (e.g. run_health_check.py --exclude) forces discard=1 with
    ``discard_reason`` so a full end-to-end dev run still uploads but stays out of
    the fleet dashboards; it takes precedence over the report's own dry_run flag.
    """
    os.makedirs(csv_output_dir, exist_ok=True)
    ts = now_iso()
    base = os.path.join(csv_output_dir, f"health_check_{hostname}_{slurm_job_id}")
    runs_path, checks_path = base + ".runs.csv", base + ".checks.csv"

    def _apply_exclusion(row):
        if discard:
            row["discard"] = 1
            row["discard_reason"] = discard_reason or row.get("discard_reason") or "excluded"
        return row

    if not report:
        print("WARNING: no diag_report.json - writing fail-closed ERROR runs row")
        row = _apply_exclusion(error_runs_row(hostname, slurm_job_id, ts, "no diag_report.json", jira_ticket))
        _validate([row], [])
        _write_csv(runs_path, RUNS_COLS, [row])
        return [runs_path]

    meta = machine_meta(report, hostname, slurm_job_id, jira_ticket, ts, versions)
    checks = checks_rows(report, meta)
    runs = _apply_exclusion(runs_row(report, meta, checks, telemetry))
    _validate([runs], checks)
    _write_csv(runs_path, RUNS_COLS, [runs])
    _write_csv(checks_path, CHECKS_COLS, checks)
    print(f"CSV written: {runs_path} (1 run), {checks_path} ({len(checks)} checks)")
    return [runs_path, checks_path]


def main():
    p = argparse.ArgumentParser(description="Analyze diag_report.json to CSV")
    p.add_argument("report_file", help="Path to diag_report.json")
    p.add_argument("--csv", required=True, help="Output directory for the CSV files")
    p.add_argument("--hostname", required=True)
    p.add_argument("--slurm-job-id", required=True)
    p.add_argument("--jira-ticket", default="")
    p.add_argument(
        "--discard",
        action="store_true",
        help="Mark the run as discarded (discard=1) so it is excluded from fleet dashboards.",
    )
    p.add_argument(
        "--discard-reason",
        default="",
        help="Reason recorded in discard_reason when --discard is set.",
    )
    args = p.parse_args()

    report = None
    if os.path.isfile(args.report_file):
        try:
            with open(args.report_file) as f:
                report = json.load(f)
        except (ValueError, OSError) as exc:
            print(f"WARNING: could not parse {args.report_file}: {exc}", file=sys.stderr)
    if report:
        report = _normalize(report)

    analyze(
        report=report,
        csv_output_dir=args.csv,
        hostname=args.hostname,
        slurm_job_id=args.slurm_job_id,
        jira_ticket=args.jira_ticket,
        discard=1 if args.discard else None,
        discard_reason=args.discard_reason or None,
    )


if __name__ == "__main__":
    main()
