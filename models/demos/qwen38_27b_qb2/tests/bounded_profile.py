# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded profile contracts; no device imports or model-quality claims."""

import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.layer_profile_report import DURATIONS, analyze, write_report

# Every geometry runs in its own process. User priority changed Oct 8 UTC:
# 32K ISL first, 16K second; 128K/256K remain active secondary tuning targets.
CASES = ((32768, 16), (32768, 32), (16384, 16), (16384, 32), (131072, 16), (262016, 8))
VARIANTS = ("native", "single_step")
SCOPE = (
    "Warm eager two-layer diagnostic with real weights and synthetic populated caches; "
    "not natural prompt activations, model accuracy, traced TPOT or a complete P0 pass"
)


def check_artifact_budget(root, *, maximum_total=4 * 1024**3, maximum_file=1024**3, minimum_free=16 * 1024**3):
    """Bound capture/export growth independently of the process memory cap."""
    root = Path(root)
    if shutil.disk_usage(root).free < minimum_free:
        raise RuntimeError("Profile filesystem has insufficient free space")
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            if size > maximum_file:
                raise RuntimeError(f"Profile file exceeds byte budget: {path}")
            total += size
    if total > maximum_total:
        raise RuntimeError("Profile artifacts exceed total byte budget")
    return total


def collect(root, length, batch, variant):
    """Tracy status alone is insufficient; require test, cleanup and all ranks."""
    if (length, batch) not in CASES or variant not in VARIANTS:
        raise ValueError("Unplanned bounded profile case")
    root = Path(root)
    receipt_path = root / "profile.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("state") != "completed"
        or receipt.get("passed") is not True
        or receipt.get("cleanup_completed") is not True
        or receipt.get("scope") != SCOPE
        or receipt.get("recurrence") != variant
        or receipt.get("prefill_calls") != 0
        or receipt.get("decode_calls") != 3
        or len(receipt.get("output_hashes", [])) != 3
        or len(set(receipt["output_hashes"])) != 1
    ):
        raise ValueError("Missing complete, repeatable and clean bounded-profile receipt")
    suites = ET.parse(root / "hardware.xml").getroot().findall(".//testsuite")
    if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
        int(s.get(field, 0)) for s in suites for field in ("failures", "errors", "skipped")
    ):
        raise ValueError("Bounded hardware test did not pass")
    reports = list((root / "tracy").rglob("ops_perf_results*.csv"))
    if len(reports) != 1:
        raise ValueError("Require exactly one bounded per-op CSV")
    if list((root / "tracy").rglob("profile_log_device.csv")):
        raise ValueError("Unexpected unbounded raw-device dump")
    compact = list((root / "tracy").rglob("cpp_device_perf_report.csv"))
    if not compact or any(not p.stat().st_size for p in compact):
        raise ValueError("Missing compact device timing report")
    with reports[0].open(newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"OP CODE", "OP TYPE", "DEVICE ID", "GLOBAL CALL COUNT", *DURATIONS.values()}.issubset(
            rows.fieldnames or []
        ):
            raise ValueError("Missing required timing columns")
        report = analyze(rows, receipt, expected_cases=[(length, batch)])
    if not report["measurements_complete"]:
        raise ValueError("Incomplete bounded device timings")
    # Every applicable RISC interval must exist. Proven data-movement-only
    # programs have no compute interval. Intervals include waits and overlap.
    if any(
        row.get(f"missing_{name}_rows")
        for row in report["device_totals"]
        for name in ("reader_ns", "writer_ns", "compute_ns")
    ):
        raise ValueError("Missing wait-inclusive per-RISC timings")
    required_stages = {f"P0_S{length}_B{batch}_L{i}_decode_forward" for i in (0, 3)}
    for device in receipt["device_ids"]:
        stages = {r["stage"] for r in report["inclusive_stages"] if r["device"] == device}
        if not required_stages.issubset(stages):
            raise ValueError("Missing a profiled decoder layer")
    report.update(scope=SCOPE, recurrence=variant, synthetic_caches=True, prefill_calls=0)
    report["sources"] = {
        p.name: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (receipt_path, reports[0])
    }
    write_report(report, root / "analysis")
    return report
