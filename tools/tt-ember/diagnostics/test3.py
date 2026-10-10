#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import argparse
import time
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Force headless backend

import matplotlib.pyplot as plt

from tt_umd import (
    ARCH,
    SmBusArcTelemetryReader,
    TelemetryTag,
    TopologyDiscovery,
    wormhole,
)

# -----------------------------
# Discovery
# -----------------------------

def discover_devices() -> Dict[int, object]:
    res = TopologyDiscovery.discover()
    if not isinstance(res, tuple) or len(res) != 2:
        raise RuntimeError("Unexpected TopologyDiscovery return.")
    _cd, devs = res
    return devs


# -----------------------------
# Telemetry helpers
# -----------------------------

def _get_reader_and_tags(dev):
    arch = dev.get_arch()
    if arch == ARCH.WORMHOLE_B0:
        return SmBusArcTelemetryReader(dev), wormhole.TelemetryTag
    elif arch == ARCH.BLACKHOLE:
        return dev.get_arc_telemetry_reader(), TelemetryTag
    else:
        raise RuntimeError(f"Unsupported arch: {arch}")


def _find_tag_id(tags, name: str) -> Optional[int]:
    for t in tags:
        if t.name == name:
            return int(t.value)
    return None


def _read_entry(dev, tag_name: str) -> Optional[int]:
    reader, tags = _get_reader_and_tags(dev)
    tag_id = _find_tag_id(tags, tag_name)
    if tag_id is None:
        return None
    try:
        return int(reader.read_entry(tag_id))
    except Exception:
        return None


# -----------------------------
# Conversions
# -----------------------------

def raw_tdp_to_watts(raw: int) -> float:
    return float(raw & 0xFFFF)


def raw_vcore_to_volts(raw: int) -> float:
    return float(raw) / 1000.0


def raw_tdc_to_current(raw: int) -> float:
    return float(raw & 0xFFFF)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--out", type=str, default="telemetry.png")
    args = ap.parse_args()

    devs = discover_devices()
    dev_ids = sorted(devs.keys())
    if not dev_ids:
        raise RuntimeError("No devices discovered.")

    print("Devices:", dev_ids)

    maxlen = int(args.window / args.interval) + 5

    t_buf = deque(maxlen=maxlen)
    v_buf = {i: deque(maxlen=maxlen) for i in dev_ids}
    c_buf = {i: deque(maxlen=maxlen) for i in dev_ids}
    p_buf = {i: deque(maxlen=maxlen) for i in dev_ids}

    t0 = time.time()

    fig, (ax_v, ax_c, ax_p) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    while True:
        now = time.time()
        t = now - t0
        t_buf.append(t)

        for dev_id in dev_ids:
            dev = devs[dev_id]

            rv = _read_entry(dev, "VCORE")
            rc = _read_entry(dev, "TDC")
            rp = _read_entry(dev, "TDP")

            v_buf[dev_id].append(raw_vcore_to_volts(rv) if rv else float("nan"))
            c_buf[dev_id].append(raw_tdc_to_current(rc) if rc else float("nan"))
            p_buf[dev_id].append(raw_tdp_to_watts(rp) if rp else float("nan"))

        # Clear axes
        ax_v.cla()
        ax_c.cla()
        ax_p.cla()

        for dev_id in dev_ids:
            ax_v.plot(t_buf, v_buf[dev_id], label=f"dev{dev_id}")
            ax_c.plot(t_buf, c_buf[dev_id], label=f"dev{dev_id}")
            ax_p.plot(t_buf, p_buf[dev_id], label=f"dev{dev_id}")

        ax_v.set_ylabel("Voltage (V)")
        ax_c.set_ylabel("Current")
        ax_p.set_ylabel("Power (W)")
        ax_p.set_xlabel("Time (s)")

        ax_v.grid(True)
        ax_c.grid(True)
        ax_p.grid(True)

        ax_v.legend(loc="upper right", ncol=min(4, len(dev_ids)))
        ax_c.legend(loc="upper right", ncol=min(4, len(dev_ids)))
        ax_p.legend(loc="upper right", ncol=min(4, len(dev_ids)))

        plt.tight_layout()
        fig.savefig(args.out, dpi=150)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {args.out}")

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
