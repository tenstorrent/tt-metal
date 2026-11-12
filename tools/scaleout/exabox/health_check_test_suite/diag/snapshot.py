# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 1 inputs: the tt-smi snapshot and the BMC FRU inventory.

The checks run over the snapshot are in snapshot_checks.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .common import log, run
from .model import PASS, SKIP, Check, Phase

# Default provisioned tt-smi path (metal team deploy hosts; absent on debug units).
DEFAULT_TT_SMI_PROVISIONED = "/opt/tt_metal_infra/provisioning/provisioning_repos/tt-smi"


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


def parse_fru_print(text: str) -> list[dict]:
    """Parse `ipmitool fru print` output into one {field: value} dict per FRU
    device. Output is `Key : Value` lines; a `FRU Device Description` line
    starts a new device block. Splitting on the first colon keeps values that
    themselves contain colons (e.g. mfg date timestamps) intact.

    Unpopulated fields ("---" placeholders on Galaxy BMCs) are dropped, which
    also disposes of most of the repeated `* Extra` keys; a repeated key with a
    real value keeps the last occurrence."""
    devices: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "FRU Device Description":
            cur = {"description": value}
            devices.append(cur)
        elif cur is not None and key and value and value != "---":
            cur[key] = value
    return devices


def collect_fru_info(phase: Phase, dry_run: bool) -> None:
    """Capture the BMC FRU inventory (`ipmitool fru print`) into the report.

    Store-only forensics — board/product part numbers, serials, and mfg dates
    for correlating failures with specific hardware units. Never degrades the
    run status: PASS when captured, SKIP when ipmitool / sudo / the BMC is
    unavailable (e.g. offline --input-snapshot dev runs).
    """
    cmd = ["ipmitool", "fru", "print"]
    if os.geteuid() != 0:
        # -n: fail immediately rather than hang on a password prompt.
        cmd = ["sudo", "-n"] + cmd
    try:
        cp = run(cmd, dry_run=dry_run, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        phase.add(Check(name="host_fru_info", status=SKIP, details=f"ipmitool unavailable: {e!r}", ip="other"))
        return
    if dry_run:
        phase.add(Check(name="host_fru_info", status=SKIP, details=f"(dry) {' '.join(cmd)}", ip="other"))
        return
    if cp.returncode != 0:
        first_err = ((cp.stderr or cp.stdout or "").strip().splitlines() or ["no output"])[0]
        phase.add(
            Check(
                name="host_fru_info",
                status=SKIP,
                details=f"ipmitool fru print rc={cp.returncode}: {first_err}",
                ip="other",
            )
        )
        return
    devices = parse_fru_print(cp.stdout)
    if not devices:
        phase.add(Check(name="host_fru_info", status=SKIP, details="no FRU devices parsed from output", ip="other"))
        return
    # UBB board serials are the tray serials used for hardware-swap tracking —
    # surface them in the details; full field set is in data.devices.
    ubb_serials = [
        f"{d['description'].split(' ')[0]}={d['Board Serial']}"
        for d in devices
        if d.get("description", "").startswith("UBB") and d.get("Board Serial")
    ]
    detail_tail = "; ".join(ubb_serials)
    if not detail_tail:
        first = devices[0]
        detail_tail = ", ".join(
            f"{k}={first[k]}" for k in ("Product Name", "Product Serial", "Board Serial") if first.get(k)
        )
    phase.add(
        Check(
            name="host_fru_info",
            status=PASS,
            details=f"{len(devices)} FRU device(s) captured" + (f"; {detail_tail}" if detail_tail else ""),
            data={"devices": devices},
            ip="other",
        )
    )
