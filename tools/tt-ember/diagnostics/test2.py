#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from tt_umd import (
    ARCH,
    SmBusArcTelemetryReader,
    TelemetryTag,
    TopologyDiscovery,
    wormhole,
)

# In tt-smi backend, power is derived from SMBUS "TDP" field:
# - WH: int(hex) & 0xFFFF
# - BH: int(hex) & 0xFFFF
def _tdp_raw_to_watts(tdp_raw: int) -> float:
    return float(int(tdp_raw) & 0xFFFF)


def _read_tdp_raw(dev) -> Tuple[Optional[int], str]:
    """
    Returns (tdp_raw, status).
    status is one of: ok, TDP_not_available, no_TDP_tag, read_error:<...>, unknown_arch
    """
    try:
        arch = dev.get_arch()  # TTDevice API on your system
    except Exception as e:
        return None, f"get_arch_error:{e.__class__.__name__}"

    # Match tt-smi logic:
    if arch == ARCH.WORMHOLE_B0:
        reader = SmBusArcTelemetryReader(dev)
        tags = wormhole.TelemetryTag
    elif arch == ARCH.BLACKHOLE:
        # Some builds use dev.get_arc_telemetry_reader() for BH
        if not hasattr(dev, "get_arc_telemetry_reader"):
            return None, "no_arc_reader"
        reader = dev.get_arc_telemetry_reader()
        tags = TelemetryTag
    else:
        return None, "unknown_arch"

    # Find TDP tag id
    tdp_key = None
    for t in tags:
        if t.name == "TDP":
            tdp_key = int(t.value)
            break
    if tdp_key is None:
        return None, "no_TDP_tag"

    # If API supports availability, honor it (tt-smi does this)
    if hasattr(reader, "is_entry_available"):
        try:
            if not reader.is_entry_available(tdp_key):
                return None, "TDP_not_available"
        except Exception as e:
            # don't hard-fail, try reading anyway
            pass

    try:
        return int(reader.read_entry(tdp_key)), "ok"
    except Exception as e:
        return None, f"read_error:{e.__class__.__name__}"


def discover_devices() -> Dict[int, object]:
    """
    Your environment:
      TopologyDiscovery.discover() -> (ClusterDescriptor, dict[int, TTDevice])
    """
    res = TopologyDiscovery.discover()
    if not isinstance(res, tuple) or len(res) != 2:
        raise RuntimeError(f"Unexpected discover() return type/shape: {type(res)}")

    cd, devs = res
    if not isinstance(devs, dict):
        raise RuntimeError(f"Unexpected devices container: {type(devs)}")

    # Keys are device ids (ints). Values are TTDevice.
    return devs


def main() -> int:
    ap = argparse.ArgumentParser(description="Periodic TT power (TDP) poller via tt_umd (no tt-smi).")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    ap.add_argument("--count", type=int, default=0, help="0=run forever")
    ap.add_argument("--show-reason", action="store_true", help="append reason when value is N/A")
    args = ap.parse_args()

    devs = discover_devices()
    ids = sorted(devs.keys())
    print(f"Mode: UMD tuple (devices discovered: {len(ids)})")
    print("Device IDs:", ids)

    n = 0
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = []
        for dev_id in ids:
            raw, status = _read_tdp_raw(devs[dev_id])
            if raw is None:
                if args.show_reason:
                    parts.append(f"dev{dev_id}=N/A({status})")
                else:
                    parts.append(f"dev{dev_id}=N/A")
            else:
                parts.append(f"dev{dev_id}={_tdp_raw_to_watts(raw):.1f}W")

        print(f"[{ts}] " + " | ".join(parts))

        n += 1
        if args.count and n >= args.count:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
