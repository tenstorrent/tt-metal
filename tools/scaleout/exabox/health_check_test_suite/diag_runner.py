#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole Galaxy system diagnostic tool (DRAFT).

Orchestrates: tt-smi snapshot validation, direct FW-telemetry-table reads,
reset stability loop, and tt-metal deployment-test gtest invocation.
Emits a single JSON pass/fail report.

Run via run_diag.sh which sets TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants — sourced from tt-system-firmware and tt-smi (see comments).
# ─────────────────────────────────────────────────────────────────────────────

# GDDR train+BIST bitmask layout — tt-system-firmware telemetry.h:103-118.
# Exposed by tt-smi -s as smbus_telem.DDR_STATUS (legacy WH name; on BH this is
# really TAG_GDDR_STATUS, tag 22). All 8 channels packed into one u32.
#   bit [2*ch    ] = train_done   (ch=0..7)
#   bit [2*ch + 1] = error
#   bit [16 + 2*ch    ] = bist_done
#   bit [16 + 2*ch + 1] = bist_failed
# Fully-trained + BIST-passed = 0x55555555.

# BH Galaxy 6U bus-byte high-nibble -> tray number.
# tt-smi/tt_smi/constants.py:BH_UBB_BUS_IDS = {1: 0x00, 2: 0x40, 3: 0xC0, 4: 0x80}
# Note non-monotonic: 0xC -> 3, 0x8 -> 4.
UBB_NIBBLE_TO_TRAY = {0x0: 1, 0x4: 2, 0xC: 3, 0x8: 4}
TRAY_TO_UBB_NIBBLE = {v: k for k, v in UBB_NIBBLE_TO_TRAY.items()}


def expected_bdf(tray: int, asic_loc: int) -> str:
    """Reconstruct the expected PCIe BDF for a given (tray, asic_loc) physical slot."""
    return f"0000:{TRAY_TO_UBB_NIBBLE[tray]:x}{asic_loc:x}:00.0"


def parse_bdf_physical(bdf: str) -> tuple[int, int] | None:
    """Parse a BDF '0000:XY:00.0' into (tray, asic_loc). None if unparseable."""
    m = re.match(r"^[0-9a-f]+:([0-9a-f])([0-9a-f]):", bdf or "", re.IGNORECASE)
    if not m:
        return None
    tray = UBB_NIBBLE_TO_TRAY.get(int(m.group(1), 16))
    if tray is None:
        return None
    return tray, int(m.group(2), 16)


def format_bdf(bdf: str) -> str:
    """'0000:46:00.0' -> '0000:46:00.0 (UBB2/U6)'. Falls back to raw bdf if unparseable."""
    pos = parse_bdf_physical(bdf)
    if pos is None:
        return bdf
    tray, asic = pos
    return f"{bdf} (UBB{tray}/U{asic})"


EXPECTED_CHIP_COUNT = 32

# Per-ASIC eth port classification on a BH Galaxy UBB.
# Keys are ASIC_LOCATION (1..8 within a UBB). Each value is the port-index
# bitmask for its connection type. Ports are ETH0..ETH11 (bits 0..11).
# Effective per-chip mask = bitmask & ENABLED_ETH (some ports are muxed off
# per chip — the table is the wiring intent, ENABLED_ETH is what FW actually
# brings up). User-confirmed mapping (outlogix / Adam C's org).
ETH_QSFP_BY_ASIC: dict[int, int] = {
    1: 0xC0F,  # ETH0-3, ETH10-11
    2: 0xC03,  # ETH0-1, ETH10-11
    3: 0xC03,  # ETH0-1, ETH10-11
    4: 0xC03,  # ETH0-1, ETH10-11
    5: 0xC0C,  # ETH2-3, ETH10-11
    6: 0xC00,  # ETH10-11
    7: 0xC00,  # ETH10-11
    8: 0xC00,  # ETH10-11
}
ETH_INTERNAL_BY_ASIC: dict[int, int] = {
    1: 0x3F0,  # ETH4-9
    2: 0x3FC,  # ETH2-9
    3: 0x3FC,  # ETH2-9
    4: 0x38C,  # ETH2-3, ETH7-9
    5: 0x073,  # ETH0-1, ETH4-6
    6: 0x07F,  # ETH0-6
    7: 0x07F,  # ETH0-6
    8: 0x00F,  # ETH0-3
}
# ExaMAX (passive + retimer combined — treated as one category per spec).
ETH_EXAMAX_BY_ASIC: dict[int, int] = {
    1: 0x000,
    2: 0x000,
    3: 0x000,
    4: 0x070,  # ETH4-6 (retimer)
    5: 0x380,  # ETH7-9 (passive)
    6: 0x380,  # ETH7-9 (passive)
    7: 0x380,  # ETH7-9 (passive)
    8: 0x3F0,  # ETH4-6 (retimer) + ETH7-9 (passive)
}

# BH Galaxy board ID prefixes → hardware revision.
# board_id source: tt-smi -s `device_info[].board_info.board_id` (hex string).
# Spec: BH Galaxy Revisions Comparison Table (Confluence SYS-4055).
# RevA/B silicon: GDDR runs at 14G, host-PCIe at Gen4.
# RevC silicon:   GDDR runs at 16G, host-PCIe at Gen5.
# Mixing the wrong expected values on the wrong rev is unsafe (RevC commands
# on RevA/B hardware can brick the board), so we gate the rev-dependent checks
# below and FAIL if the rev can't be unambiguously detected.
BOARD_ID_PREFIX_REVAB = "00000471"
BOARD_ID_PREFIX_REVC = "00000473"
BOARD_REV_EXPECTATIONS = {
    "RevA/B": {"gddr_speed": "14G", "pcie_gen": 4, "board_id_prefix": BOARD_ID_PREFIX_REVAB},
    "RevC": {"gddr_speed": "16G", "pcie_gen": 5, "board_id_prefix": BOARD_ID_PREFIX_REVC},
}

# Default provisioned tt-smi path (metal team deploy hosts; absent on debug units).
DEFAULT_TT_SMI_PROVISIONED = "/opt/tt_metal_infra/provisioning/provisioning_repos/tt-smi"

# tt-metal deployment test binary candidate paths relative to repo root.
# build_metal.sh defaults to build_Release/; raw cmake invocations often use build/.
DEPLOYMENT_BIN_CANDIDATES = [
    "build_Release/test/tt_metal/unit_tests_deployment",
    "build/test/tt_metal/unit_tests_deployment",
    "build_Debug/test/tt_metal/unit_tests_deployment",
    "build_RelWithDebInfo/test/tt_metal/unit_tests_deployment",
]

# Reset cadence per SYS-4365: first reset is -r, then stick to -glx_reset.
RESET_PLAN = {
    "light": ["-r"],
    "medium": ["-r", "-glx_reset"],
    "deploy": ["-r", "-glx_reset", "-glx_reset"],
}

# tt-smi prints this banner before attempting `-r` on Galaxy units when the CPLD
# FW is older than v1.16. The PCIe-level reset path is unreliable on those CPLDs
# (chips re-enumerate but UMD's post-reset register reads return 0xffffffff).
# Workaround per the banner: use `-glx_reset` instead.
CPLD_OLD_BANNER_RE = re.compile(r"CPLD FW v1\.16 or higher is required to use tt-smi -r", re.IGNORECASE)

# gtest filter -> test name table. Names match deployment_tests_dram_glx (outlogix).
# eth_bandwidth uses a trailing wildcard so it picks up BandwidthBidir automatically
TESTS = {
    "eth_link_up": "*TensixDeploymentEthernetLinkUp",
    "eth_bandwidth": "*TensixDeploymentEthernetBandwidth*",
    "gddr_fast": "*DramDeployment_PersistentOptimalWorkersAllDramBanks",
    "gddr_full": "*DramDeployment_*",
}

TIER_TESTS = {
    # ETH tests are temporarily disabled; they will be re-enabled once the ETH
    # tests are updated.
    "light": [],  # "eth_link_up"
    "medium": ["gddr_fast"],  # "eth_link_up", "eth_bandwidth"
    "deploy": ["gddr_full"],  # "eth_link_up", "eth_bandwidth"
}

# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    details: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ip: str = "other"  # pcie / gddr / eth / fw / asic / thermal / other — used for console grouping
    console_visible: bool = True  # False → kept in JSON but hidden from console summary


@dataclass
class Phase:
    name: str
    status: str = SKIP
    duration_s: float = 0.0
    checks: list[Check] = field(default_factory=list)
    error: str = ""

    def add(self, c: Check) -> None:
        self.checks.append(c)

    def rollup(self) -> None:
        if any(c.status == FAIL for c in self.checks):
            self.status = FAIL
        elif any(c.status == WARN for c in self.checks):
            self.status = WARN
        elif self.checks:
            self.status = PASS


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list[str], dry_run: bool = False, **kw) -> subprocess.CompletedProcess:
    """Run a command. In dry_run, print and return a stubbed CompletedProcess."""
    log(f"$ {' '.join(cmd)}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, **kw)


def log(msg: str) -> None:
    print(f"[diag] {msg}", file=sys.stderr, flush=True)


def resolve_tt_smi(override: str | None) -> str:
    """Pick tt-smi binary. Priority: override > provisioned > PATH."""
    if override:
        p = Path(override)
        # Accept either a binary or a repo path containing bin/tt-smi
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        cand = p / "bin" / "tt-smi"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
        raise SystemExit(f"--tt-smi-path={override} not executable / not found")
    prov = Path(DEFAULT_TT_SMI_PROVISIONED) / "bin" / "tt-smi"
    if prov.is_file() and os.access(prov, os.X_OK):
        return str(prov)
    found = shutil.which("tt-smi")
    if not found:
        raise SystemExit("tt-smi not found on PATH; pass --tt-smi-path")
    return found


def detect_tt_smi_version(tt_smi: str) -> tuple[int, ...] | None:
    """Parse `tt-smi -v` -> (major, minor, patch). None if unparseable.

    Both 4.1.2 and 5.1.1 emit a single line with just the version string."""
    try:
        cp = subprocess.run([tt_smi, "-v"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (cp.stdout or cp.stderr).strip().splitlines() if (cp.stdout or cp.stderr) else []
    line = lines[0] if lines else ""
    # Extract first dotted-int sequence in case future tt-smi prefixes with "tt-smi "
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", line)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — snapshot validation
# ─────────────────────────────────────────────────────────────────────────────


def detect_board_rev(devs: list[dict]) -> tuple[str | None, dict[str, list[str]]]:
    """Classify chips by board ID prefix → revision.

    Returns (rev_label, per_rev_bdfs).
    rev_label is 'RevA/B', 'RevC', or None when chips are mixed or any chip
    has an unrecognised board_id (callers should treat None as indeterminate
    and surface a FAIL on the board_rev check).
    per_rev_bdfs maps each observed bucket label (including 'unknown:<prefix>')
    to its list of BDFs, for diagnostic output.
    """
    per_rev: dict[str, list[str]] = defaultdict(list)
    for d in devs:
        bid = (d.get("board_info") or {}).get("board_id") or ""
        bdf = (d.get("board_info") or {}).get("bus_id", "?")
        if bid.startswith(BOARD_ID_PREFIX_REVAB):
            per_rev["RevA/B"].append(bdf)
        elif bid.startswith(BOARD_ID_PREFIX_REVC):
            per_rev["RevC"].append(bdf)
        else:
            per_rev[f"unknown:{bid[:8] or '<empty>'}"].append(bdf)
    buckets = list(per_rev.keys())
    if len(buckets) == 1 and buckets[0] in BOARD_REV_EXPECTATIONS:
        return buckets[0], dict(per_rev)
    return None, dict(per_rev)


def take_snapshot(tt_smi: str, out_path: Path) -> dict:
    """tt-smi -f <out_path>. Always executes — read-only operation.

    Use --input-snapshot for fully-offline runs."""
    log(f"$ {tt_smi} -f {out_path}")
    cp = subprocess.run([tt_smi, "-f", str(out_path)], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"tt-smi -f failed (rc={cp.returncode}): {cp.stderr or cp.stdout}")
    if not out_path.exists():
        raise RuntimeError(f"snapshot file not written: {out_path}")
    with out_path.open() as f:
        return json.load(f)


def validate_snapshot(snap: dict, phase: Phase) -> str | None:
    """Run all snapshot checks. Returns the detected board revision
    ('RevA/B', 'RevC') or None if indeterminate, so the caller can record it
    on the report.
    """
    devs = snap.get("device_info") or []

    # 1. Enumeration count — on short count, identify which physical slots didn't
    # show up (set diff against the full 32-BDF Galaxy 6U topology). Helps ops
    # localise the missing chip(s) without manual cross-referencing.
    n = len(devs)
    actual_bdfs = sorted((d.get("board_info") or {}).get("bus_id", "") for d in devs)
    all_expected_bdfs = sorted(expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9))
    missing_bdfs_enum = sorted(set(all_expected_bdfs) - set(actual_bdfs))
    enum_details = f"{n}/{EXPECTED_CHIP_COUNT} chips enumerated"
    if missing_bdfs_enum:
        enum_details += "; missing: " + ", ".join(format_bdf(b) for b in missing_bdfs_enum)
    phase.add(
        Check(
            name="pcie_enum_count",
            status=PASS if n == EXPECTED_CHIP_COUNT else FAIL,
            details=enum_details,
            data={
                "actual": n,
                "expected": EXPECTED_CHIP_COUNT,
                "missing_bdfs": missing_bdfs_enum,
                "actual_bdfs": actual_bdfs,
            },
            ip="pcie",
        )
    )
    if n == 0:
        return None  # nothing else we can check

    # 1b. Board revision — gates GDDR speed + PCIe gen expectations below.
    # RevA/B (board_id 00000471xxxxxxxx): 14G GDDR, Gen4 PCIe.
    # RevC   (board_id 00000473xxxxxxxx): 16G GDDR, Gen5 PCIe.
    # Mixed-rev or unknown-prefix board_ids => FAIL here, downstream
    # rev-dependent checks => SKIP. Spec: Confluence SYS-4055.
    rev, per_rev_bdfs = detect_board_rev(devs)
    rev_dist = {k: len(v) for k, v in per_rev_bdfs.items()}
    if rev is not None:
        prefix = BOARD_REV_EXPECTATIONS[rev]["board_id_prefix"]
        phase.add(
            Check(
                name="board_rev",
                status=PASS,
                details=f"Rev: {rev} ({rev_dist[rev]}/{n} chips, board_id prefix {prefix})",
                data={
                    "rev": rev,
                    "distribution": rev_dist,
                    "per_rev_bdfs": per_rev_bdfs,
                    "expectations": BOARD_REV_EXPECTATIONS[rev],
                },
                ip="board",
            )
        )
    else:
        phase.add(
            Check(
                name="board_rev",
                status=FAIL,
                details=f"indeterminate board revision: {rev_dist}",
                data={"rev": None, "distribution": rev_dist, "per_rev_bdfs": per_rev_bdfs},
                ip="board",
            )
        )

    # 1c. BDF -> physical location mapping. Visible topology readout for the
    # enumerated chips, so any BDF appearing in subsequent failure details can
    # be correlated to its UBB/U position without referring to docs.
    by_ubb_bdfs: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for d in devs:
        bdf = (d.get("board_info") or {}).get("bus_id") or ""
        pos = parse_bdf_physical(bdf)
        if pos is None:
            continue
        tray, asic = pos
        by_ubb_bdfs[tray].append((bdf, asic))
    topo_lines: list[str] = []
    for tray in sorted(by_ubb_bdfs):
        nibble = TRAY_TO_UBB_NIBBLE.get(tray, 0)
        chips = sorted(by_ubb_bdfs[tray], key=lambda x: x[1])
        rendered = "  ".join(f"{bdf}=U{a}" for bdf, a in chips)
        topo_lines.append(f"UBB{tray} (bus {nibble:X}X): {rendered}")
    phase.add(
        Check(
            name="bdf_topology",
            status=PASS,
            details=f"{n} chips mapped\n" + "\n".join(topo_lines),
            data={"per_ubb": {str(t): [bdf for bdf, _ in v] for t, v in by_ubb_bdfs.items()}},
            ip="board",
        )
    )

    # 2. dram_status
    dram_ok = [bool(d.get("board_info", {}).get("dram_status")) for d in devs]
    bad = [d["board_info"]["bus_id"] for d, ok in zip(devs, dram_ok) if not ok]
    phase.add(
        Check(
            name="dram_status",
            status=PASS if all(dram_ok) else FAIL,
            details=f"{sum(dram_ok)}/{n} dram_status=True"
            + (f"; failed: [{', '.join(format_bdf(b) for b in bad)}]" if bad else ""),
            data={"failed_chips": bad},
            ip="gddr",
        )
    )

    # 3. ENABLED_GDDR uniform == 0xff
    gddr_masks = [d.get("smbus_telem", {}).get("ENABLED_GDDR") for d in devs]
    gddr_bad = [(d["board_info"]["bus_id"], m) for d, m in zip(devs, gddr_masks) if m != "0xff"]
    phase.add(
        Check(
            name="enabled_gddr_full",
            status=PASS if not gddr_bad else FAIL,
            details=f"{n - len(gddr_bad)}/{n} chips have ENABLED_GDDR=0xff"
            + (f"; partial: [{', '.join(f'{format_bdf(b)}={m}' for b, m in gddr_bad[:5])}]" if gddr_bad else ""),
            data={"partial_chips": gddr_bad},
            ip="gddr",
        )
    )

    # 4. HARVESTING_STATE: 0x0 or 0x1 valid; other values are a failure.
    # Hidden from console (kept in JSON for forensics).
    HARV_OK = ("0x0", "0", "0x1", "1")
    harv = [d.get("smbus_telem", {}).get("HARVESTING_STATE") for d in devs]
    harv_bad = [(d["board_info"]["bus_id"], h) for d, h in zip(devs, harv) if h not in (None, *HARV_OK)]
    phase.add(
        Check(
            name="harvesting_state",
            status=PASS if not harv_bad else FAIL,
            details=f"{n - len(harv_bad)}/{n} chips have HARVESTING_STATE in {{0,1}}",
            data={"harvested_chips": harv_bad, "distribution": dict(Counter(harv))},
            ip="asic",
            console_visible=False,
        )
    )

    # 5. ASIC_LOCATION per UBB completeness (1..8 each)
    by_ubb = defaultdict(set)
    for d in devs:
        bus = d["board_info"]["bus_id"].split(":")[1]
        tray = UBB_NIBBLE_TO_TRAY.get(int(bus[0], 16))
        if tray is None:
            continue
        try:
            loc = int(d["smbus_telem"]["ASIC_LOCATION"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        by_ubb[tray].add(loc)
    expected = set(range(1, 9))
    missing = {tray: sorted(expected - locs) for tray, locs in by_ubb.items() if expected - locs}
    missing_trays = sorted(set(range(1, 5)) - set(by_ubb.keys()))
    asic_loc_ok = not missing and not missing_trays
    # Build expected-BDF list for every missing chip — entire-tray-missing means all 8 asics.
    missing_bdfs: list[str] = []
    for tray in missing_trays:
        missing_bdfs.extend(expected_bdf(tray, a) for a in range(1, 9))
    for tray, asics in missing.items():
        missing_bdfs.extend(expected_bdf(tray, a) for a in asics)
    details = []
    if missing_trays:
        details.append(f"missing UBBs: {missing_trays}")
    if missing:
        details.append(f"missing ASICs: {missing}")
    if missing_bdfs:
        details.append(f"expected BDFs: {missing_bdfs}")
    phase.add(
        Check(
            name="asic_location_per_ubb",
            status=PASS if asic_loc_ok else FAIL,
            details="all UBBs complete" if asic_loc_ok else "; ".join(details),
            data={
                "per_ubb": {t: sorted(s) for t, s in by_ubb.items()},
                "missing": missing,
                "missing_trays": missing_trays,
                "missing_bdfs": missing_bdfs,
            },
            ip="asic",
        )
    )

    # 6. Physical BDF ↔ firmware ASIC_LOCATION agreement
    mismatches = []
    for d in devs:
        bus = d["board_info"]["bus_id"].split(":")[1]
        phys_asic = int(bus[1], 16)
        try:
            fw_asic = int(d["smbus_telem"]["ASIC_LOCATION"], 16)
        except (KeyError, TypeError, ValueError):
            continue
        if phys_asic != fw_asic:
            mismatches.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": phys_asic,
                    "fw_asic_location": fw_asic,
                }
            )

    def _fmt_mismatch(m: dict) -> str:
        return f"{format_bdf(m['bdf'])} phys=U{m['physical_asic']} " f"fw=U{m['fw_asic_location']}"

    phase.add(
        Check(
            name="physical_vs_fw_location",
            status=PASS if not mismatches else FAIL,
            details=f"{n - len(mismatches)}/{n} chips match"
            + (f"; mismatches: [{'; '.join(_fmt_mismatch(m) for m in mismatches[:3])}]" if mismatches else ""),
            data={"mismatches": mismatches},
            ip="asic",
        )
    )

    # 7. Firmware versions consistent across chips (warn-only per ticket).
    # gddr_fw is the M-RISC (GDDR memory controller) firmware — 5.1+ only.
    # dm_app_fw / dm_bl_fw don't apply on Galaxy; always report SKIP (kept in JSON for completeness).
    FW_FIELDS_CHECKED = ("fw_bundle_version", "cm_fw", "eth_fw", "gddr_fw")
    FW_FIELDS_NA_ON_GALAXY = ("dm_app_fw", "dm_bl_fw")
    for fw_field in FW_FIELDS_CHECKED:
        vals = [d.get("firmwares", {}).get(fw_field) for d in devs]
        # If the field is absent on every chip, gate as SKIP (older tt-smi).
        if all(v is None for v in vals):
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=SKIP,
                    details=f"{fw_field} not present in snapshot (requires tt-smi 5.1+)",
                    data={},
                    ip="fw",
                )
            )
            continue
        unique = Counter(vals)
        if len(unique) == 1:
            ((only_val, _),) = unique.most_common(1)
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=PASS,
                    details=f"all chips: {only_val}",
                    data={"value": only_val},
                    ip="fw",
                )
            )
        else:
            phase.add(
                Check(
                    name=f"{fw_field}_consistent",
                    status=WARN,
                    details=f"mismatch across chips: {dict(unique)}",
                    data={"distribution": dict(unique)},
                    ip="fw",
                )
            )
    for fw_field in FW_FIELDS_NA_ON_GALAXY:
        vals = [d.get("firmwares", {}).get(fw_field) for d in devs]
        unique = Counter(vals)
        phase.add(
            Check(
                name=f"{fw_field}_consistent",
                status=SKIP,
                details="not applicable to Galaxy",
                data={"distribution": dict(unique)},
                ip="fw",
                console_visible=False,
            )
        )

    # 8. GDDR per-channel training + BIST decoded from smbus_telem.DDR_STATUS.
    # Layout in header comment at top of file.
    train_failures: list[dict] = []
    bist_failures: list[dict] = []
    for d in devs:
        bdf = d["board_info"]["bus_id"]
        status_str = d.get("smbus_telem", {}).get("DDR_STATUS")
        if status_str is None:
            train_failures.append({"bdf": bdf, "reason": "DDR_STATUS missing"})
            continue
        try:
            status = int(status_str, 16)
        except (ValueError, TypeError):
            train_failures.append({"bdf": bdf, "reason": f"DDR_STATUS unparseable: {status_str!r}"})
            continue
        for ch in range(8):
            train_done = (status >> (ch * 2)) & 1
            err = (status >> (ch * 2 + 1)) & 1
            bist_done = (status >> (16 + ch * 2)) & 1
            bist_fail = (status >> (17 + ch * 2)) & 1
            if not train_done or err:
                train_failures.append({"bdf": bdf, "channel": ch, "train_done": train_done, "error": err})
            if not bist_done or bist_fail:
                bist_failures.append({"bdf": bdf, "channel": ch, "bist_done": bist_done, "bist_failed": bist_fail})

    def _fmt_chan_fail(f: dict) -> str:
        bdf = format_bdf(f.get("bdf", "?"))
        if "reason" in f:
            return f"{bdf}: {f['reason']}"
        return (
            f"{bdf} ch={f.get('channel','?')} " f"train_done={f.get('train_done','?')} err={f.get('error','?')}"
            if "train_done" in f
            else f"{bdf} ch={f.get('channel','?')} "
            f"bist_done={f.get('bist_done','?')} bist_failed={f.get('bist_failed','?')}"
        )

    total_channels = n * 8
    phase.add(
        Check(
            name="gddr_training_per_channel",
            status=PASS if not train_failures else FAIL,
            details=f"{total_channels - len(train_failures)}/{total_channels} channels trained"
            + (
                f"; first failures: [{'; '.join(_fmt_chan_fail(f) for f in train_failures[:3])}]"
                if train_failures
                else ""
            ),
            data={"failures": train_failures},
            ip="gddr",
        )
    )
    phase.add(
        Check(
            name="gddr_bist_per_channel",
            status=PASS if not bist_failures else FAIL,
            details=f"{total_channels - len(bist_failures)}/{total_channels} channels BIST passed"
            + (
                f"; first failures: [{'; '.join(_fmt_chan_fail(f) for f in bist_failures[:3])}]"
                if bist_failures
                else ""
            ),
            data={"failures": bist_failures},
            ip="gddr",
        )
    )

    # 9. PCIe lane width — U6 chips (the x8 host-PCIe-attached ASIC on each UBB)
    # should be at x8; all other ASICs should be x1. Lane mismatch is a hard fail.
    #
    # Uses BDF-derived physical position rather than FW ASIC_LOCATION so this
    # check is robust to FW telemetry bugs (e.g., multiple chips wrongly
    # reporting the same ASIC_LOCATION). Any such telemetry issue is surfaced
    # by physical_vs_fw_location; this check evaluates against physical reality.
    U6_ASIC = 0x6

    def _phys_asic_loc(d: dict) -> int | None:
        pos = parse_bdf_physical((d.get("board_info") or {}).get("bus_id", ""))
        return pos[1] if pos is not None else None

    def _is_u6(d: dict) -> bool:
        return _phys_asic_loc(d) == U6_ASIC

    lane_mismatches = []
    for d in devs:
        loc = _phys_asic_loc(d)
        width = str(d["board_info"].get("pcie_width"))
        expected_width = "8" if loc == U6_ASIC else "1"
        if width != expected_width:
            lane_mismatches.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": loc,
                    "actual_width": width,
                    "expected_width": expected_width,
                }
            )

    def _fmt_lane_mm(m: dict) -> str:
        return f"{format_bdf(m['bdf'])} " f"x{m['actual_width']} (expected x{m['expected_width']})"

    phase.add(
        Check(
            name="pcie_lane_width",
            status=PASS if not lane_mismatches else FAIL,
            details=(
                f"{n - len(lane_mismatches)}/{n} chips at expected lane width "
                f"(x8 on physical U6, x1 elsewhere)"
                + (
                    f"; mismatches: [{'; '.join(_fmt_lane_mm(m) for m in lane_mismatches[:3])}]"
                    if lane_mismatches
                    else ""
                )
            ),
            data={"mismatches": lane_mismatches},
            ip="pcie",
        )
    )

    # 10. PCIe gen — U6 chips should all be at the rev's expected gen
    # (Gen4 on RevA/B, Gen5 on RevC). Known bug: chips occasionally train down
    # to Gen1; flag as WARN so it surfaces without failing the run. If the
    # board rev is indeterminate we SKIP — no safe baseline to check against.
    # Selects U6 chips by BDF-derived physical position (not FW ASIC_LOCATION).
    u6_chips = [d for d in devs if _is_u6(d)]
    per_chip_gen = [(d["board_info"]["bus_id"], d["board_info"].get("pcie_speed")) for d in u6_chips]
    if rev is None:
        phase.add(
            Check(
                name="pcie_gen",
                status=SKIP,
                details="board_rev indeterminate; cannot pick expected gen (see board_rev)",
                data={"per_chip": dict(per_chip_gen), "expected": None, "rev": None},
                ip="pcie",
            )
        )
    else:
        expected_gen = BOARD_REV_EXPECTATIONS[rev]["pcie_gen"]

        # WARN only on chips trained *below* the rev's expected gen (the known
        # down-train bug). Above-expected is silicon-out-of-spec but not a
        # regression — note it in details, leave status unaffected.
        def _gen_marker(g) -> str:
            if not isinstance(g, int):
                return "(?)"
            if g < expected_gen:
                return "(!)"
            if g > expected_gen:
                return "(^)"
            return ""

        under = [(bdf, g) for bdf, g in per_chip_gen if isinstance(g, int) and g < expected_gen]
        over = [(bdf, g) for bdf, g in per_chip_gen if isinstance(g, int) and g > expected_gen]
        listing = ", ".join(f"{format_bdf(bdf)}=Gen{g}{_gen_marker(g)}" for bdf, g in per_chip_gen)
        bits = [f"U6 chips (expected Gen{expected_gen} on {rev}): {listing}"]
        if under:
            bits.append(f"below: {', '.join(format_bdf(bdf) for bdf, _ in under)}")
        if over:
            bits.append(f"above-spec: {', '.join(format_bdf(bdf) for bdf, _ in over)}")
        phase.add(
            Check(
                name="pcie_gen",
                status=WARN if under else PASS,
                details="; ".join(bits),
                data={
                    "per_chip": dict(per_chip_gen),
                    "under_expected": under,
                    "above_expected": over,
                    "expected": expected_gen,
                    "rev": rev,
                },
                ip="pcie",
            )
        )

    # 11. GDDR speed — RevA/B chips expected at 14G; RevC chips expected at 16G.
    # Rev-indeterminate => SKIP (don't FAIL with a wrong baseline).
    speeds = Counter(d["board_info"].get("dram_speed") for d in devs)
    if rev is None:
        phase.add(
            Check(
                name="gddr_speed",
                status=SKIP,
                details="board_rev indeterminate; cannot pick expected speed (see board_rev)",
                data={"distribution": dict(speeds), "expected": None, "rev": None},
                ip="gddr",
            )
        )
    else:
        expected_speed = BOARD_REV_EXPECTATIONS[rev]["gddr_speed"]
        gddr_speed_bad = [
            (d["board_info"]["bus_id"], d["board_info"].get("dram_speed"))
            for d in devs
            if d["board_info"].get("dram_speed") != expected_speed
        ]
        phase.add(
            Check(
                name="gddr_speed",
                status=PASS if not gddr_speed_bad else FAIL,
                details=(
                    f"{n - len(gddr_speed_bad)}/{n} chips at {expected_speed} ({rev})"
                    + (
                        f"; off-speed: [{', '.join(f'{format_bdf(b)}={s}' for b, s in gddr_speed_bad[:3])}]"
                        if gddr_speed_bad
                        else ""
                    )
                ),
                data={
                    "distribution": dict(speeds),
                    "off_speed": gddr_speed_bad,
                    "expected": expected_speed,
                    "rev": rev,
                },
                ip="gddr",
            )
        )

    # 12. ETH speed — no eth speed field in tt-smi 4.1.2 or 5.1.1 schema.
    phase.add(
        Check(
            name="eth_speed",
            status=SKIP,
            details="no eth speed field in tt-smi snapshot schema",
            data={},
            ip="eth",
        )
    )

    # 12b. ETH links up — verify internal (non-QSFP) ports report as live.
    # Expected per chip = (Internal | ExaMAX) & ENABLED_ETH. Compare to
    # ETH_LIVE_STATUS masked to the same non-QSFP set.
    #
    # The port-topology tables (ETH_INTERNAL_BY_ASIC, ETH_EXAMAX_BY_ASIC) are
    # indexed by *physical* ASIC_LOCATION (the chip's slot on the UBB, fixed
    # by wiring). We derive that from the BDF rather than FW telemetry so the
    # check stays correct when FW ASIC_LOCATION is misreported — that's
    # separately surfaced by physical_vs_fw_location.
    #
    # ETH_LIVE_STATUS is only populated by FW bundle >= 19.9. On older FW
    # it stays 0x0 regardless of actual link state, so we SKIP rather than
    # WARN to avoid misattributing a firmware capability gap to a real link
    # issue. On unknown/newer FW we keep the original WARN-on-all-zero path.
    ETH_LIVE_MIN_FW = (19, 9)
    eth_live_supported = True
    detected_fw = None
    for d in devs:
        ver = d.get("firmwares", {}).get("fw_bundle_version")
        if ver:
            detected_fw = ver
            try:
                major, minor = (int(p) for p in ver.split(".")[:2])
                eth_live_supported = (major, minor) >= ETH_LIVE_MIN_FW
            except (ValueError, IndexError):
                pass
            break

    live_status_vals: list[int | None] = []
    eth_link_failures: list[dict] = []
    for d in devs:
        loc = _phys_asic_loc(d)
        live_str = d.get("smbus_telem", {}).get("ETH_LIVE_STATUS")
        try:
            live = int(live_str, 16) if live_str is not None else None
        except (ValueError, TypeError):
            live = None
        live_status_vals.append(live)
        if loc is None or live is None or loc not in ETH_INTERNAL_BY_ASIC:
            continue
        try:
            enabled = int(d["smbus_telem"]["ENABLED_ETH"], 16)
        except (KeyError, ValueError, TypeError):
            continue
        non_qsfp = (ETH_INTERNAL_BY_ASIC[loc] | ETH_EXAMAX_BY_ASIC[loc]) & enabled
        actual = live & non_qsfp
        missing = non_qsfp & ~actual
        if missing:
            down_ports = [b for b in range(12) if missing & (1 << b)]
            eth_link_failures.append(
                {
                    "bdf": d["board_info"]["bus_id"],
                    "physical_asic": loc,
                    "down_ports": down_ports,
                    "expected_mask": f"0x{non_qsfp:03x}",
                    "actual_mask": f"0x{actual:03x}",
                }
            )
    all_zero = all(v == 0 for v in live_status_vals if v is not None)
    any_present = any(v is not None for v in live_status_vals)
    if not any_present:
        phase.add(
            Check(
                name="eth_links_up", status=SKIP, details="ETH_LIVE_STATUS not present in snapshot", data={}, ip="eth"
            )
        )
    elif all_zero and not eth_live_supported:
        phase.add(
            Check(
                name="eth_links_up",
                status=SKIP,
                details=f"ETH_LIVE_STATUS requires FW bundle >= 19.9 "
                f"(detected: {detected_fw}) — firmware does not "
                f"populate this field; cannot validate links",
                data={"detected_fw": detected_fw},
                ip="eth",
            )
        )
    elif all_zero:
        phase.add(
            Check(
                name="eth_links_up",
                status=WARN,
                details=f"ETH_LIVE_STATUS=0x0 on all {len(devs)} chips — "
                f"FW {detected_fw} should populate this field; "
                f"check firmware/tt-smi",
                data={"detected_fw": detected_fw},
                ip="eth",
            )
        )
    elif eth_link_failures:
        first = eth_link_failures[0]
        phase.add(
            Check(
                name="eth_links_up",
                status=FAIL,
                details=(
                    f"{len(devs) - len(eth_link_failures)}/{len(devs)} chips OK; "
                    f"first failure: {format_bdf(first['bdf'])} "
                    f"down_ports={first['down_ports']}"
                ),
                data={"failures": eth_link_failures},
                ip="eth",
            )
        )
    else:
        phase.add(
            Check(
                name="eth_links_up",
                status=PASS,
                details=f"{len(devs)}/{len(devs)} chips: all non-QSFP links up",
                data={},
                ip="eth",
            )
        )

    # 13. GDDR thermal + error counters (5.1+ only). Store-only — surfaces values
    # but never alerts; useful for forensics / regression baselines.
    GDDR_INFO_FIELDS = (
        "MAX_GDDR_TEMP",
        "GDDR_0_1_TEMP",
        "GDDR_2_3_TEMP",
        "GDDR_4_5_TEMP",
        "GDDR_6_7_TEMP",
        "GDDR_0_1_CORR_ERRS",
        "GDDR_2_3_CORR_ERRS",
        "GDDR_4_5_CORR_ERRS",
        "GDDR_6_7_CORR_ERRS",
        "GDDR_UNCORR_ERRS",
    )
    for tag in GDDR_INFO_FIELDS:
        vals = {d["board_info"]["bus_id"]: d.get("smbus_telem", {}).get(tag) for d in devs}
        present = [v for v in vals.values() if v is not None]
        if not present:
            phase.add(
                Check(
                    name=f"gddr_info_{tag.lower()}",
                    status=SKIP,
                    details=f"{tag} not present (requires tt-smi 5.1+)",
                    data={},
                    ip="gddr",
                    console_visible=False,
                )
            )
            continue
        # Store every value verbatim; no thresholds. Hidden from console — too noisy.
        phase.add(
            Check(
                name=f"gddr_info_{tag.lower()}",
                status=PASS,
                details=f"{tag}: {len(present)} chips reporting",
                data={"per_chip": vals},
                ip="gddr",
                console_visible=False,
            )
        )

    # 14. ASIC thermal pre-check — kept in JSON only.
    temps = []
    for d in devs:
        t = d.get("telemetry", {}).get("asic_temperature")
        if t is not None:
            try:
                temps.append((d["board_info"]["bus_id"], float(t)))
            except (ValueError, TypeError):
                pass
    if temps:
        hottest_bdf, hottest_t = max(temps, key=lambda x: x[1])
        thm_limit = float(devs[0].get("limits", {}).get("thm_limit", 110))
        threshold = thm_limit - 20  # idle should be well below throttle
        status = PASS if hottest_t < threshold else WARN
        phase.add(
            Check(
                name="asic_thermal_precheck",
                status=status,
                details=f"hottest chip {hottest_bdf} @ {hottest_t:.1f}°C (limit {thm_limit}°C)",
                data={
                    "hottest_chip": hottest_bdf,
                    "hottest_temp_c": hottest_t,
                    "thm_limit_c": thm_limit,
                    "all_temps_c": dict(temps),
                },
                ip="thermal",
                console_visible=False,
            )
        )

    return rev


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — reset stability loop
# ─────────────────────────────────────────────────────────────────────────────

_USE_INPLACE_PROGRESS = sys.stdout.isatty()


def _emit_running(name: str) -> None:
    """Print 'running...' line. If stdout is a TTY we'll overwrite it with \\r;
    otherwise just print on its own line so the captured log stays readable."""
    if _USE_INPLACE_PROGRESS:
        sys.stdout.write(f"  {name:30} running...")
    else:
        sys.stdout.write(f"  {name:30} running...\n")
    sys.stdout.flush()


def _emit_result(name: str, status: str, suffix: str = "") -> None:
    """Print final result. On TTY, \\r overwrites the running... line."""
    prefix = "\r" if _USE_INPLACE_PROGRESS else ""
    extra = f"  {suffix}" if suffix else ""
    sys.stdout.write(f"{prefix}  {name:30} {status:5}{extra}\n")
    sys.stdout.flush()


def reset_loop(
    tt_smi: str,
    plan: list[str],
    phase: Phase,
    dry_run: bool,
    snapshot_out: Path | None = None,
    post_reset_phases: list | None = None,
) -> None:
    """Per SYS-4365: first -r, then stick to -glx_reset for subsequent iterations.

    If `snapshot_out` and `post_reset_phases` are provided, a fresh snapshot is
    taken + validated after each reset iteration (including any CPLD auto-recover),
    and appended to `post_reset_phases` as (name, Phase) pairs for the caller to
    splice into the report.
    """

    def _post_reset_snapshot(label: str) -> None:
        if dry_run or snapshot_out is None or post_reset_phases is None:
            return
        sp = Phase(name=f"snapshot_after_{label}")
        t0 = time.time()
        try:
            snap = take_snapshot(tt_smi, snapshot_out)
            validate_snapshot(snap, sp)
        except Exception as e:
            sp.error = repr(e)
            sp.add(Check(name="snapshot_capture", status=FAIL, details=repr(e), ip="other"))
        sp.duration_s = time.time() - t0
        sp.rollup()
        post_reset_phases.append((sp.name, sp))
        print_phase_summary(sp.name, asdict(sp))

    for i, flag in enumerate(plan):
        check_name = f"reset_{i+1}_{flag.lstrip('-')}"
        _emit_running(check_name)
        t0 = time.time()
        cp = run([tt_smi, flag], dry_run=dry_run, capture_output=True, text=True)
        dt = time.time() - t0
        # Quick post-reset enum check
        if dry_run:
            post_count = EXPECTED_CHIP_COUNT
        else:
            try:
                lspci = subprocess.run(
                    ["bash", "-c", "lspci -d 1e52: | wc -l"], capture_output=True, text=True, check=True
                )
                post_count = int(lspci.stdout.strip())
            except Exception:
                post_count = -1
        status = PASS if (dry_run or (cp.returncode == 0 and post_count == EXPECTED_CHIP_COUNT)) else FAIL
        _emit_result(check_name, status, suffix=f"({dt:.1f}s, post_pcie={post_count})")
        phase.add(
            Check(
                name=check_name,
                status=status,
                details=f"flag={flag} rc={cp.returncode if not dry_run else '(dry)'} "
                f"dur={dt:.1f}s post_pcie={post_count}",
                data={
                    "flag": flag,
                    "rc": cp.returncode,
                    "duration_s": dt,
                    "post_pcie_count": post_count,
                    "stdout_tail": cp.stdout[-2000:] if cp.stdout else "",
                    "stderr_tail": cp.stderr[-2000:] if cp.stderr else "",
                },
                ip="other",
            )
        )
        cpld_old = flag == "-r" and not dry_run and cp.stdout and CPLD_OLD_BANNER_RE.search(cp.stdout)
        if cpld_old:
            phase.add(
                Check(
                    name="cpld_fw_old",
                    status=WARN,
                    details="CPLD FW < v1.16 detected — `tt-smi -r` unreliable on this system; "
                    "use `tt-smi -glx_reset` instead and request a CPLD update from the sysadmin",
                    ip="other",
                )
            )
        # Auto-recover: -r on CPLD < v1.16 doesn't just fail, it leaves chips in a
        # bad state (UMD reads 0xffffffff on subsequent calls) until -glx_reset is
        # run. Recover so we don't strand the system.
        if cpld_old and status == FAIL:
            _emit_running("cpld_auto_recover")
            t0 = time.time()
            rcp = run([tt_smi, "-glx_reset"], dry_run=False, capture_output=True, text=True)
            rdt = time.time() - t0
            try:
                lspci = subprocess.run(
                    ["bash", "-c", "lspci -d 1e52: | wc -l"], capture_output=True, text=True, check=True
                )
                rpost = int(lspci.stdout.strip())
            except Exception:
                rpost = -1
            recovered = rcp.returncode == 0 and rpost == EXPECTED_CHIP_COUNT
            rstatus = PASS if recovered else FAIL
            _emit_result("cpld_auto_recover", rstatus, suffix=f"({rdt:.1f}s, post_pcie={rpost})")
            phase.add(
                Check(
                    name="cpld_auto_recover",
                    status=rstatus,
                    details=(
                        f"ran `tt-smi -glx_reset` after -r failed; "
                        f"rc={rcp.returncode} post_pcie={rpost} "
                        f"{'recovered' if recovered else 'still broken — manual intervention needed'}"
                    ),
                    data={
                        "rc": rcp.returncode,
                        "duration_s": rdt,
                        "post_pcie_count": rpost,
                        "stdout_tail": rcp.stdout[-2000:] if rcp.stdout else "",
                        "stderr_tail": rcp.stderr[-2000:] if rcp.stderr else "",
                    },
                    ip="other",
                )
            )
            # If recovery worked, refresh post_count so the enumeration-break
            # bailout below doesn't trip on the pre-recovery state.
            if recovered:
                post_count = rpost
                # Downgrade the original -r FAIL to WARN now that we know it's
                # CPLD-attributable and the unit is back up. Hard FAIL is reserved
                # for -r failures we *can't* explain or recover from.
                for c in phase.checks:
                    if c.name == check_name and c.status == FAIL:
                        c.status = WARN
                        c.details += " (downgraded to WARN: CPLD-attributable, auto-recovered)"
                        break
        # Post-reset snapshot: validate the unit's state after this iteration
        # (including any auto-recovery). Skipped on dry-run / offline.
        _post_reset_snapshot(f"r{i+1}_{flag.lstrip('-')}")
        # Bail if a reset broke enumeration — no point chasing the rest
        if not dry_run and post_count != EXPECTED_CHIP_COUNT:
            log(f"Reset {i+1} broke enumeration ({post_count}/{EXPECTED_CHIP_COUNT}); stopping reset loop")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — gtest invocations
# ─────────────────────────────────────────────────────────────────────────────

# gtest progress markers — used for per-testcase pass/fail accounting.
# Matches only on lines with a "Suite.Test" pattern, so summary lines like
# "[  FAILED  ] 1 test, listed below:" don't get counted as testcases.
GTEST_RUN_RE = re.compile(r"\[\s*RUN\s*\]\s+(\S+\.\S+)")
GTEST_OK_RE = re.compile(r"\[\s*OK\s*\]\s+(\S+\.\S+)")
GTEST_FAIL_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\S+\.\S+)")


# Console summary helpers
IP_ORDER = ("board", "pcie", "gddr", "eth", "asic", "fw", "thermal", "other")


def print_phase_summary(phase_name: str, phase_dict: dict) -> None:
    """Print one phase's grouped per-IP summary to stdout (flushed).

    Multi-line check details (containing '\\n') render with continuation lines
    indented under the first detail line — used by bdf_topology to print its
    per-UBB mapping table without breaking the columnar layout.
    """
    lines = [f"  {phase_name:14} {phase_dict['status']:5} ({phase_dict['duration_s']:.1f}s)"]
    visible = [c for c in phase_dict["checks"] if c.get("console_visible", True)]
    by_ip: dict[str, list] = defaultdict(list)
    for c in visible:
        by_ip[c.get("ip", "other")].append(c)
    for ip in IP_ORDER:
        checks = by_ip.get(ip, [])
        if not checks:
            continue
        for i, c in enumerate(checks):
            head = f"{ip}:" if i == 0 else ""
            detail_lines = (c["details"] or "").split("\n")
            lines.append(f"    {head:7} {c['name']:32} {c['status']:5}  {detail_lines[0]}")
            for extra in detail_lines[1:]:
                lines.append(f"      {extra}")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def run_tests(tt_metal: Path, tier: str, phase: Phase, dry_run: bool, logs_dir: Path) -> None:
    binary = next(
        (tt_metal / c for c in DEPLOYMENT_BIN_CANDIDATES if (tt_metal / c).is_file()),
        None,
    )
    if binary is None:
        phase.add(
            Check(
                name="deployment_binary_present",
                status=FAIL,
                details=f"none of {DEPLOYMENT_BIN_CANDIDATES} found under {tt_metal}",
                ip="other",
            )
        )
        return

    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal)
    env["TT_METAL_RUNTIME_ROOT"] = str(tt_metal)  # newer name post-rename
    # LD_LIBRARY_PATH: match the build dir of the binary we picked
    env.setdefault("LD_LIBRARY_PATH", str(binary.parent.parent.parent / "lib"))

    logs_dir.mkdir(parents=True, exist_ok=True)

    for name in TIER_TESTS[tier]:
        filt = TESTS[name]
        # `gddr_fast` tier hint: use DRAM_TEST_FAST=1
        test_env = env.copy()
        if name == "gddr_fast":
            test_env["DRAM_TEST_FAST"] = "1"
        env_overrides = {k: test_env[k] for k in ("DRAM_TEST_FAST",) if k in test_env}
        cmd = [str(binary), f"--gtest_filter={filt}"]
        log_path = logs_dir / f"{name}.log"
        prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        cmdline = f"{prefix + ' ' if prefix else ''}{' '.join(cmd)}"
        log(f"--- test '{name}' ---")
        log(f"  cmd:  {cmdline}")
        log(f"  log:  {log_path}")

        if dry_run:
            print(f"  {name:30} (dry-run)")
            phase.add(
                Check(
                    name=name,
                    status=PASS,
                    ip="other",
                    details=f"(dry) filter={filt} log={log_path}",
                    data={
                        "filter": filt,
                        "command": cmdline,
                        "rc": 0,
                        "duration_s": 0.0,
                        "env_overrides": env_overrides,
                        "log_file": str(log_path),
                        "testcases": {"passed": 0, "failed": 0, "failures": []},
                    },
                )
            )
            continue

        _emit_running(name)

        t0 = time.time()
        passed = 0
        failed = 0
        failures: list[str] = []
        in_flight: set[str] = set()  # testcases that have started but not yet ended
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=test_env,
                cwd=str(tt_metal),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                # All gtest output goes to the log file only — keep the console clean.
                logf.write(line)
                # Per-testcase accounting (dedup against double-counting from gtest's
                # end-of-run summary by tracking which tests are in-flight).
                if m := GTEST_RUN_RE.search(line):
                    in_flight.add(m.group(1))
                elif m := GTEST_OK_RE.search(line):
                    if m.group(1) in in_flight:
                        in_flight.discard(m.group(1))
                        passed += 1
                elif m := GTEST_FAIL_RE.search(line):
                    if m.group(1) in in_flight:
                        in_flight.discard(m.group(1))
                        failed += 1
                        failures.append(m.group(1))
            rc = proc.wait()
        dt = time.time() - t0

        status = PASS if rc == 0 and failed == 0 else FAIL
        suffix = f"({dt:.1f}s)"
        if failed:
            suffix += f"  passed={passed} failed={failed}"
        _emit_result(name, status, suffix=suffix)
        details = f"filter={filt} rc={rc} dur={dt:.1f}s " f"passed={passed} failed={failed} log={log_path}"
        phase.add(
            Check(
                name=name,
                status=status,
                details=details,
                data={
                    "filter": filt,
                    "command": cmdline,
                    "rc": rc,
                    "duration_s": dt,
                    "env_overrides": env_overrides,
                    "log_file": str(log_path),
                    "testcases": {"passed": passed, "failed": failed, "failures": failures},
                },
                ip="other",
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="SYS-4365 diag tool")
    ap.add_argument("--tier", choices=list(TIER_TESTS), required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print intended subprocess calls; skip destructive steps.")
    ap.add_argument("--skip-reset", action="store_true", help="Skip phase 3 (reset loop) entirely.")
    ap.add_argument("--skip-tests", action="store_true", help="Skip phase 4 (gtest deployment binary) entirely.")
    ap.add_argument(
        "--input-snapshot", type=Path, help="Use pre-captured tt-smi snapshot JSON instead of live tt-smi call."
    )
    ap.add_argument(
        "--tt-smi-path", help=f"tt-smi binary or repo path. Default: {DEFAULT_TT_SMI_PROVISIONED} else PATH."
    )
    ap.add_argument(
        "--tt-metal-path",
        type=Path,
        default=Path(os.environ.get("TT_METAL_HOME") or Path(__file__).resolve().parents[4]),
        help="tt-metal repo root (contains build/test/tt_metal/unit_tests_deployment).",
    )
    ap.add_argument("--output", type=Path, default=Path("diag_report.json"))
    ap.add_argument("--snapshot-out", type=Path, default=Path("/tmp/diag_snapshot.json"))
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    report = {
        "tool_version": "0.3.0-draft",
        "tier": args.tier,
        "host": socket.gethostname(),
        "started_utc": started.isoformat(),
        "dry_run": args.dry_run,
        "tt_smi_version": None,
        "detected_board_rev": None,
        "phases": {},
    }

    # Phase 1: snapshot
    snap_phase = Phase(name="snapshot")
    t0 = time.time()
    try:
        if args.input_snapshot:
            log(f"using input snapshot: {args.input_snapshot}")
            snap = json.loads(args.input_snapshot.read_text())
        else:
            tt_smi = resolve_tt_smi(args.tt_smi_path)
            version = detect_tt_smi_version(tt_smi)
            report["tt_smi_version"] = ".".join(str(x) for x in version) if version else None
            log(f"using tt-smi: {tt_smi} (version={report['tt_smi_version']})")
            snap = take_snapshot(tt_smi, args.snapshot_out)
        report["detected_board_rev"] = validate_snapshot(snap, snap_phase)
    except Exception as e:
        snap_phase.error = repr(e)
        snap_phase.add(Check(name="snapshot_capture", status=FAIL, details=repr(e)))
    snap_phase.duration_s = time.time() - t0
    snap_phase.rollup()
    report["phases"]["snapshot"] = asdict(snap_phase)
    print_phase_summary("snapshot", report["phases"]["snapshot"])

    # Phase 2: reset loop + per-reset snapshot revalidations
    reset_phase = Phase(name="reset_loop")
    post_reset_phases: list = []
    # Only revalidate via fresh snapshots when running live (not --input-snapshot).
    snap_out_for_resets = None if args.input_snapshot else args.snapshot_out
    t0 = time.time()
    if args.skip_reset:
        reset_phase.add(Check(name="reset_loop", status=SKIP, details="--skip-reset"))
    else:
        try:
            tt_smi = resolve_tt_smi(args.tt_smi_path)
            reset_loop(
                tt_smi,
                RESET_PLAN[args.tier],
                reset_phase,
                args.dry_run,
                snapshot_out=snap_out_for_resets,
                post_reset_phases=post_reset_phases,
            )
        except Exception as e:
            reset_phase.error = repr(e)
            reset_phase.add(Check(name="reset_loop", status=FAIL, details=repr(e)))
    reset_phase.duration_s = time.time() - t0
    reset_phase.rollup()
    report["phases"]["reset_loop"] = asdict(reset_phase)
    # reset_loop prints its own per-reset lines as they run; rollup line only.
    print(f"  {'reset_loop':14} {reset_phase.status:5} ({reset_phase.duration_s:.1f}s)", flush=True)
    # Dedupe per-reset snapshots: if all match initial, keep only initial and
    # record a single summary check. If any differ, splice only the differing
    # phases into the report (the matching ones are noise).
    if post_reset_phases:

        def _check_sig(checks: list) -> tuple:
            # Compare on (name, status) only. `details` and `data` carry
            # telemetry (temps, voltages, error counters) that fluctuates
            # between snapshots without being a status concern. Status
            # changes are the signal — a chip dropping out, GDDR un-training,
            # FW versions going inconsistent — those all flip a check's status.
            return tuple(sorted((c.get("name", ""), c.get("status", "")) for c in checks))

        initial_sig = _check_sig(report["phases"]["snapshot"]["checks"])
        matched, differed = [], []
        for name, p in post_reset_phases:
            (matched if _check_sig([asdict(c) for c in p.checks]) == initial_sig else differed).append((name, p))
        for name, p in differed:
            report["phases"][name] = asdict(p)
        summary_status = PASS if not differed else WARN
        summary_details = f"{len(matched)}/{len(post_reset_phases)} post-reset snapshots matched initial" + (
            f"; {len(differed)} differed (see {', '.join(n for n,_ in differed)})" if differed else ""
        )
        report["phases"]["snapshot"]["checks"].append(
            asdict(
                Check(
                    name="post_reset_state_stable",
                    status=summary_status,
                    details=summary_details,
                    ip="other",
                )
            )
        )
        # Re-roll the initial snapshot phase status since we just appended a check.
        statuses = [c["status"] for c in report["phases"]["snapshot"]["checks"]]
        if FAIL in statuses:
            report["phases"]["snapshot"]["status"] = FAIL
        elif WARN in statuses:
            report["phases"]["snapshot"]["status"] = WARN
        elif statuses:
            report["phases"]["snapshot"]["status"] = PASS

    # Phase 3: gtest
    test_phase = Phase(name="tests")
    t0 = time.time()
    if args.skip_tests:
        test_phase.add(Check(name="tests", status=SKIP, details="--skip-tests"))
    else:
        try:
            logs_dir = args.output.resolve().parent / "logs"
            run_tests(args.tt_metal_path, args.tier, test_phase, args.dry_run, logs_dir)
        except Exception as e:
            test_phase.error = repr(e)
            test_phase.add(Check(name="tests", status=FAIL, details=repr(e)))
    test_phase.duration_s = time.time() - t0
    test_phase.rollup()
    report["phases"]["tests"] = asdict(test_phase)
    # tests phase prints its own per-test lines as they run — no inline summary.
    print(f"  {'tests':14} {test_phase.status:5} ({test_phase.duration_s:.1f}s)", flush=True)

    ended = datetime.now(timezone.utc)
    report["ended_utc"] = ended.isoformat()
    report["total_duration_s"] = (ended - started).total_seconds()
    statuses = [p["status"] for p in report["phases"].values()]
    if FAIL in statuses:
        report["overall_status"] = FAIL
    elif WARN in statuses:
        report["overall_status"] = WARN
    else:
        report["overall_status"] = PASS

    args.output.write_text(json.dumps(report, indent=2))
    log(f"wrote report: {args.output} (overall={report['overall_status']})")

    # Separator delineates the live per-phase output above from the final
    # rollup below, so the OVERALL summary block is visually distinct.
    print("  " + "─" * 70)
    print(f"  OVERALL        {report['overall_status']}")
    # Per-phase rollup. For non-PASS phases, list the offending checks (ip:name=status).
    for phase_name, p in report["phases"].items():
        status = p["status"]
        if status == PASS:
            print(f"    {phase_name:11} {status}")
            continue
        bad = [c for c in p["checks"] if c["status"] in (WARN, FAIL)]
        if bad:
            items = ", ".join(f"{c.get('ip','other')}:{c['name']}={c['status']}" for c in bad)
            print(f"    {phase_name:11} {status} — {items}")
        else:
            print(f"    {phase_name:11} {status}")

    return 0 if report["overall_status"] != FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
