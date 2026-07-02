# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone IPMI power poller.

Runs in a separate process from the perf workload. Every cycle it takes ONE
bulk `ipmitool sensor` snapshot (cheap relative to N separate reads) and writes
a timestamped CSV row. Designed to be correlated against the compute window by
Unix time.

The IPMI transport on this BMC is slow (~3-10 s per read), so the achievable
sampling rate is ~0.1-0.3 Hz regardless of the requested interval. We log the
*actual* sample time (Unix epoch, float seconds) so downstream correlation is
exact, not assumed.

Usage:
    python ipmi_power_poller.py <output_csv> [interval_s]

    interval_s: target seconds between samples (default 1.0; effective rate is
                transport-bound, so this is a floor, not a guarantee).

Columns:
    epoch            float Unix seconds at the moment the snapshot was taken
    iso              human-readable ISO time
    read_latency_s   how long the ipmitool call took (diagnostic)
    Power_Total_W    whole-system total
    Power_UBB0..3_W  4 chip baseboards (8 Blackhole chips each)
    Power_CPU_W      host EPYC
    Power_Memory_W   DDR5
    Power_FAN_W      cooling

Stop with SIGTERM/SIGINT; it flushes and exits cleanly.
"""

import math
import signal
import subprocess
import sys
import time

SENSORS = [
    "Power_Total",
    "Power_UBB0",
    "Power_UBB1",
    "Power_UBB2",
    "Power_UBB3",
    "Power_CPU",
    "Power_Memory",
    "Power_FAN",
]

_running = True


def _stop(signum, frame):
    global _running
    _running = False


def _read_snapshot():
    """One bulk `ipmitool sensor` call. Returns {sensor: float_watts}."""
    out = subprocess.run(
        ["sudo", "-n", "ipmitool", "sensor"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    vals = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0]
        if name in SENSORS:
            try:
                vals[name] = float(parts[1])
            except (ValueError, IndexError):
                vals[name] = float("nan")
    return vals


def main():
    if len(sys.argv) < 2:
        print("usage: python ipmi_power_poller.py <output_csv> [interval_s]", file=sys.stderr)
        sys.exit(2)
    out_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    with open(out_path, "w", buffering=1) as f:  # line-buffered
        header = ["epoch", "iso", "read_latency_s"] + [f"{s}_W" for s in SENSORS]
        f.write(",".join(header) + "\n")

        while _running:
            t0 = time.time()
            try:
                vals = _read_snapshot()
            except Exception as e:  # noqa: BLE001
                vals = {}
                sys.stderr.write(f"read error: {e}\n")
            t1 = time.time()
            # Stamp the sample at the MIDPOINT of the read (best estimate of when
            # the BMC value was valid) -- matters because reads take seconds.
            mid = (t0 + t1) / 2.0
            iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mid))
            row = [f"{mid:.3f}", iso, f"{t1 - t0:.3f}"]
            for s in SENSORS:
                v = vals.get(s, float("nan"))
                row.append("NA" if math.isnan(v) else f"{v:.1f}")
            f.write(",".join(row) + "\n")

            # Sleep the remainder of the interval (reads slower than interval -> no sleep)
            remain = interval - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
