#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def run_tt_smi_snapshot(tt_smi_cmd: str = "tt-smi") -> Dict[str, Any]:
    """
    Run 'tt-smi -s' and return parsed JSON.
    Requires tt-smi to be installed and accessible in PATH.

    tt-smi supports JSON snapshots via -s/--snapshot.  :contentReference[oaicite:1]{index=1}
    """
    try:
        proc = subprocess.run(
            [tt_smi_cmd, "-s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(f"'{tt_smi_cmd}' not found in PATH. Is tt-smi installed?")

    if proc.returncode != 0:
        raise RuntimeError(
            f"tt-smi failed (rc={proc.returncode}). stderr:\n{proc.stderr.strip()}"
        )

    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("tt-smi returned empty stdout.")

    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        # Helpful debug: show first/last part
        snippet = out[:500] + ("\n...\n" if len(out) > 1000 else "\n") + out[-500:]
        raise RuntimeError(f"Failed to parse tt-smi JSON: {e}\nOutput snippet:\n{snippet}")


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        # Common formatting in tools: " 123.4"
        try:
            return float(s)
        except ValueError:
            return None
    return None


def extract_device_powers(snapshot: Dict[str, Any]) -> List[Tuple[str, Optional[float]]]:
    """
    Attempt to extract per-device power readings from tt-smi snapshot JSON.

    Because snapshot schema can evolve, this function uses a few heuristics:
    - Look for a list under common keys (e.g., "device_info", "devices")
    - For each device, look for nested "telemetry" dict with "power"
    - Fallback: search for "power" within device dict if telemetry not found

    Returns list of (device_label, power_watts_or_units) where device_label is best-effort.
    """
    # Find the device list
    device_list = None
    for k in ("device_info", "devices", "deviceInfos", "device_info_list"):
        v = snapshot.get(k)
        if isinstance(v, list):
            device_list = v
            break

    if device_list is None:
        # Some builds wrap it differently: snapshot["log"]["device_info"]
        log_obj = snapshot.get("log") or snapshot.get("data") or snapshot.get("result")
        if isinstance(log_obj, dict):
            v = log_obj.get("device_info")
            if isinstance(v, list):
                device_list = v

    if device_list is None:
        raise RuntimeError("Could not locate device list in snapshot JSON (no device_info/devices found).")

    results: List[Tuple[str, Optional[float]]] = []

    for idx, dev in enumerate(device_list):
        if not isinstance(dev, dict):
            results.append((f"dev{idx}", None))
            continue

        # Best-effort label
        board_type = None
        board_id = None
        name = None

        # Common nesting: dev["board_info"] contains ids
        board_info = dev.get("board_info") if isinstance(dev.get("board_info"), dict) else dev
        if isinstance(board_info, dict):
            board_type = board_info.get("board_type") or board_info.get("device_series")
            board_id = board_info.get("board_id") or board_info.get("serial") or board_info.get("id")

        name = dev.get("name") or dev.get("device_name") or dev.get("arch") or dev.get("chip") or None

        label_parts = []
        if name:
            label_parts.append(str(name))
        if board_type:
            label_parts.append(str(board_type))
        if board_id:
            label_parts.append(str(board_id))
        label = " / ".join(label_parts) if label_parts else f"dev{idx}"

        # Primary expected path: dev["telemetry"]["power"]
        power_val = None
        telemetry = dev.get("telemetry")
        if isinstance(telemetry, dict) and "power" in telemetry:
            power_val = _try_float(telemetry.get("power"))

        # Fallback 1: sometimes telemetry nested differently
        if power_val is None:
            telem = dev.get("device_telemetry") or dev.get("chip_telemetry")
            if isinstance(telem, dict) and "power" in telem:
                power_val = _try_float(telem.get("power"))

        # Fallback 2: search shallow keys
        if power_val is None and "power" in dev:
            power_val = _try_float(dev.get("power"))

        results.append((label, power_val))

    return results


def write_csv_header(fp, device_labels: List[str]) -> None:
    cols = ["timestamp"] + [f"power_{i}" for i in range(len(device_labels))]
    fp.write(",".join(cols) + "\n")
    fp.flush()


def write_csv_row(fp, ts: str, powers: List[Optional[float]]) -> None:
    def fmt(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.3f}"
    fp.write(",".join([ts] + [fmt(p) for p in powers]) + "\n")
    fp.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="Periodic TT power polling via tt-smi snapshot JSON.")
    ap.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0).")
    ap.add_argument("--count", type=int, default=0, help="Number of samples (0 = run forever).")
    ap.add_argument("--tt-smi", dest="tt_smi_cmd", default="tt-smi", help="tt-smi executable name/path.")
    ap.add_argument("--csv", type=str, default="", help="Optional CSV output file path.")
    args = ap.parse_args()

    csv_fp = None
    device_labels_cached: Optional[List[str]] = None

    if args.csv:
        out_path = Path(args.csv).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fp = out_path.open("w", encoding="utf-8")

    n = 0
    while True:
        ts = datetime.now().isoformat(timespec="seconds")

        try:
            snap = run_tt_smi_snapshot(args.tt_smi_cmd)
            dev_powers = extract_device_powers(snap)
        except Exception as e:
            print(f"[{ts}] ERROR: {e}", file=sys.stderr)
            # Keep going unless user wants strict behavior
            time.sleep(args.interval)
            n += 1
            if args.count and n >= args.count:
                break
            continue

        labels = [lp[0] for lp in dev_powers]
        powers = [lp[1] for lp in dev_powers]

        # Print a readable line to stdout
        parts = []
        for i, (lab, p) in enumerate(dev_powers):
            p_str = "N/A" if p is None else f"{p:.3f}"
            parts.append(f"[{i}] {p_str} ({lab})")
        print(f"[{ts}] " + " | ".join(parts))

        # CSV output
        if csv_fp is not None:
            if device_labels_cached is None:
                device_labels_cached = labels
                write_csv_header(csv_fp, device_labels_cached)
            write_csv_row(csv_fp, ts, powers)

        n += 1
        if args.count and n >= args.count:
            break
        time.sleep(args.interval)

    if csv_fp is not None:
        csv_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
