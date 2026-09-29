# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for the fabric system health-check result schema.
"""

from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

# 0/1 integer flag and bounded severity, kept as ints because the dashboard SUM()s
# them directly in SQL.
Flag = Annotated[int, Field(ge=0, le=1)]
Severity = Annotated[int, Field(ge=0, le=3)]


class OverallStatus(str, Enum):
    """Run-level verdict. PASS/WARN/FAIL come from the diag tool; ERROR/UNKNOWN are
    injected by the run wrapper when a report is missing or unparseable."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class CheckStatus(str, Enum):
    """Per-check status. PASS/WARN/FAIL/SKIP come from the diag tool. EXCLUDED is
    injected by the analyzer for infrastructure/capture steps (e.g.
    ``snapshot_capture``) whose FAIL/WARN is a tooling/precondition hiccup, not a
    hardware verdict: it stays visible but drives no failure and carries no
    ``is_*`` flag (all zero)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    EXCLUDED = "EXCLUDED"


class Category(str, Enum):
    """Low-cardinality subsystem the dashboard aggregates over. ``other`` is the
    documented catch-all and should stay empty in practice."""

    pcie = "pcie"
    gddr = "gddr"
    asic = "asic"
    firmware = "firmware"
    thermal = "thermal"
    board = "board"
    eth = "eth"
    reset = "reset"
    stress_test = "stress_test"
    other = "other"


class PhaseKind(str, Enum):
    """Primary phases vs the per-reset re-snapshots the diag tool splices in on
    instability (segregated so an unstable machine doesn't double-count)."""

    primary = "primary"
    post_reset = "post_reset"


def _blanks_to_none(values):
    """CSV serializes missing values as empty strings; treat those as ``None`` so
    optional numeric/enum fields validate. Applied before field validation."""
    if isinstance(values, dict):
        return {k: (None if v == "" else v) for k, v in values.items()}
    return values


class CheckRecord(BaseModel):
    """One diagnostic check within a run (``checks.csv``). ~28 rows per run, all
    sharing the same ``run_id``. Check-level fields only."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(description="Schema version; bump on any breaking change.")
    run_id: str = Field(description="Join key to runs.csv: '{hostname}:{slurm_job_id}'.")
    date: Date = Field(description="Run date (kept for time partitioning/filtering).")

    category: Category = Field(description="Subsystem this check belongs to.")
    phase: str = Field(description="snapshot / reset_loop / tests / snapshot_after_reset_N.")
    phase_kind: PhaseKind = Field(description="primary vs post_reset re-snapshot.")
    check_name: str = Field(description="Check name, e.g. pcie_enum_count, gddr_speed, eth_link_up.")

    status: CheckStatus = Field(description="PASS / WARN / FAIL / SKIP.")
    severity: Severity = Field(description="0 PASS/SKIP, 1 WARN, 2 FAIL, 3 UNKNOWN.")

    is_pass: Flag = Field(description="1 if status is PASS (pre-computed for SUM aggregation).")
    is_warn: Flag = Field(description="1 if status is WARN.")
    is_fail: Flag = Field(description="1 if status is FAIL.")
    is_skip: Flag = Field(description="1 if status is SKIP.")
    is_covered: Flag = Field(description="1 if the check actually ran; a stress test that executed 0 cases is 0.")
    acknowledged: Flag = Field(
        description="1 for known-benign fleet-wide WARNs (e.g. cpld_fw_old), excluded from actionable rollups."
    )

    testcases_passed: Optional[int] = Field(None, description="gtest sub-cases passed; None for non-test checks.")
    testcases_failed: Optional[int] = Field(None, description="gtest sub-cases failed; None for non-test checks.")
    executed: Flag = Field(description="0 if a stress test executed 0 cases (check-level fail-closed).")

    details_short: Optional[str] = Field(None, description="Human-readable check detail, newlines flattened.")

    @model_validator(mode="before")
    @classmethod
    def _pre(cls, values):
        return _blanks_to_none(values)

    @model_validator(mode="after")
    def _consistency(self):
        # the is_* flags must agree with status. EXCLUDED is deliberately flagless
        # (all is_* = 0): it is neither a pass nor a failure signal.
        expected = {
            CheckStatus.PASS: "is_pass",
            CheckStatus.WARN: "is_warn",
            CheckStatus.FAIL: "is_fail",
            CheckStatus.SKIP: "is_skip",
        }.get(self.status)
        if expected is not None and getattr(self, expected) != 1:
            raise ValueError(f"{expected}=1 expected for status={self.status.value}")
        if self.status is CheckStatus.EXCLUDED and (self.is_pass or self.is_warn or self.is_fail or self.is_skip):
            raise ValueError("EXCLUDED must have all is_* flags = 0")
        return self


class RunRecord(BaseModel):
    """Machine-level rollup for a single health-check run (``runs.csv``). Exactly
    one row per run — including a fail-closed row with overall_status=ERROR when a
    run produced no report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(description="Schema version; bump on any breaking change.")
    run_id: str = Field(description="Primary key: '{hostname}:{slurm_job_id}'.")
    date: Date = Field(description="UTC run date (partition key).")
    timestamp: datetime = Field(description="Run/analysis timestamp with timezone.")
    hostname: str = Field(description="Node name, e.g. bh-glx-b02u02.")
    slurm_job_id: str = Field(description="SLURM job id.")

    # spatial dims parsed from hostname (rack/row clustering). rack='unparsed' if the
    # hostname matches neither known scheme.
    row: Optional[str] = Field(None, description="Rack row (empty for schemes without a row segment).")
    rack: Optional[str] = Field(None, description="Rack; 'unparsed' if hostname is unrecognized.")
    slot: Optional[str] = Field(None, description="Slot within the rack.")

    overall_status: OverallStatus = Field(description="Run verdict: PASS/WARN/FAIL/ERROR/UNKNOWN.")

    # exclude non-real runs (dry-run / maintenance / manual) from fleet stats.
    discard: Flag = Field(description="1 = exclude this run from fleet statistics.")
    discard_reason: Optional[str] = Field(
        None, description="Reason when discard=1 (dry_run/maintenance/manual_test/reboot_pending/…)."
    )

    # history: computed by the Data team across the day-series; empty on a lone run.
    prev_status: Optional[OverallStatus] = Field(None, description="This host's previous run status (Data team).")
    is_regression: Optional[Flag] = Field(None, description="Severity increased vs previous run (Data team).")
    fail_streak: Optional[int] = Field(None, description="Consecutive FAIL/ERROR runs (Data team).")

    tier: Optional[str] = Field(None, description="light / medium / deploy.")
    tool_version: Optional[str] = Field(None, description="diag tool version.")
    tt_smi_version: Optional[str] = Field(None, description="tt-smi version.")
    tt_kmd_version: Optional[str] = Field(None, description="tt-kmd version (from run wrapper; may be empty).")
    board_rev: Optional[str] = Field(None, description="Board revision, e.g. RevA/B, RevC (thresholds differ).")
    fw_bundle_version: Optional[str] = Field(None, description="Firmware bundle version.")
    num_chips: Optional[int] = Field(None, description="Chips enumerated on the node.")
    total_duration_s: Optional[float] = Field(None, description="Wall-clock duration of the run in seconds.")

    # coverage / quality
    checks_total: int = Field(description="Number of checks.")
    checks_pass: int = Field(description="Checks with PASS.")
    checks_warn: int = Field(description="Checks with WARN.")
    checks_fail: int = Field(description="Checks with FAIL.")
    checks_skip: int = Field(description="Checks with SKIP.")
    checks_covered: int = Field(description="Checks that actually ran (PASS/WARN/FAIL, executed).")
    pct_covered: Optional[float] = Field(None, description="100 * checks_covered / checks_total.")
    checks_warn_actionable: int = Field(description="WARNs excluding acknowledged fleet-wide benign ones.")
    top_fail_category: Optional[str] = Field(
        None,
        description="Worst category (FAIL first, then actionable WARN); 'run' for ERROR; empty if only benign WARNs.",
    )

    # subsystem rollups
    reset_count: Optional[int] = Field(None, description="Reset iterations in the reset loop.")
    reset_stable: Optional[Flag] = Field(None, description="1 if the reset loop did not FAIL.")
    gddr_uncorr_total: Optional[int] = Field(None, description="Sum of GDDR uncorrectable errors across chips.")
    pcie_downgraded: Optional[Flag] = Field(None, description="1 if any chip is below its expected PCIe gen.")
    eth_links_down: Optional[int] = Field(None, description="Chips with a down non-QSFP Ethernet link.")

    # thermal correlation (from the report)
    max_asic_temp_c: Optional[float] = Field(None, description="Hottest ASIC temperature.")
    max_gddr_temp_c: Optional[float] = Field(None, description="Hottest GDDR temperature.")

    # telemetry correlation (tt-telemetry; empty unless the endpoint was reachable)
    min_aiclk_mhz: Optional[float] = Field(None, description="Lowest AI clock (tt-telemetry).")
    telemetry_available: Flag = Field(description="1 if tt-telemetry was reachable for this run.")
    eth_retrain_total: Optional[int] = Field(None, description="Per-machine Ethernet retrains (tt-telemetry).")
    eth_crc_total: Optional[int] = Field(None, description="Per-machine Ethernet CRC errors (tt-telemetry).")
    eth_uncorr_cw_total: Optional[int] = Field(
        None, description="Per-machine Ethernet uncorrectable codewords (tt-telemetry)."
    )

    jira_ticket: Optional[str] = Field(None, description="JIRA key if a ticket was filed for this run.")

    @model_validator(mode="before")
    @classmethod
    def _pre(cls, values):
        return _blanks_to_none(values)


class RunResult(BaseModel):
    """Logical unit for one run: the rollup plus its checks. Convenience container
    for producers that validate a full run before writing the two CSVs; the CSVs
    themselves are flat (this is not serialized as-is)."""

    run: RunRecord
    checks: List[CheckRecord] = []

    @model_validator(mode="after")
    def _checks_belong_to_run(self):
        for c in self.checks:
            if c.run_id != self.run.run_id:
                raise ValueError(f"check run_id {c.run_id!r} != run {self.run.run_id!r}")
        # (run_id, check_name, phase) is the checks upsert key — must be unique.
        keys = [(c.run_id, c.check_name, c.phase) for c in self.checks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (run_id, check_name, phase) in checks")
        return self
