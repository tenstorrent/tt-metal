# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Board telemetry for the prefill harness (tests/perf/prefill.py): AICLK, power, current, vcore and
ASIC temperature of one Tenstorrent card, read from the kernel driver's sysfs files.

    /sys/class/tenstorrent/tenstorrent!<N>/tt_aiclk        MHz
    /sys/class/hwmon/hwmonX (name "blackhole")             power1_input (uW), curr1_input (mA),
                                                           in0_input (mV), temp1_input (mC)

These are plain file reads (the driver serves the firmware telemetry table); they do not open the
device, so they work while a benchmark process holds the card. `Sampler` runs the reader in a
separate process (`python -m ...prefill_telemetry --out f.csv --period-ms 100`) so the sampling
loop does not share the benchmark's interpreter.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ("t", "aiclk_mhz", "power_w", "current_a", "vcore_v", "asic_temp_c")


def _card_index() -> int:
    vis = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(vis) if vis.isdigit() else 0


def paths(card: int | None = None) -> dict[str, Path]:
    card = _card_index() if card is None else card
    tt = Path(f"/sys/class/tenstorrent/tenstorrent!{card}")
    hw = None
    hw_dir = tt / "device" / "hwmon"
    if hw_dir.exists():
        subs = sorted(hw_dir.iterdir())
        hw = subs[0] if subs else None
    if hw is None:  # fall back to the first hwmon named blackhole / wormhole
        for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            if (h / "name").read_text().strip() in ("blackhole", "wormhole"):
                hw = h
                break
    if hw is None:
        raise FileNotFoundError("no Tenstorrent hwmon directory found")
    return {
        "aiclk_mhz": tt / "tt_aiclk",
        "power_w": hw / "power1_input",
        "current_a": hw / "curr1_input",
        "vcore_v": hw / "in0_input",
        "asic_temp_c": hw / "temp1_input",
        "power_max_w": hw / "power1_max",
        "current_max_a": hw / "curr1_max",
        "fw_bundle": tt / "tt_fw_bundle_ver",
        "card_type": tt / "tt_card_type",
        "pcie_width": tt / "device" / "current_link_width",
        "pcie_speed": tt / "device" / "current_link_speed",
    }


_SCALE = {"aiclk_mhz": 1.0, "power_w": 1e-6, "current_a": 1e-3, "vcore_v": 1e-3, "asic_temp_c": 1e-3}


def read(p: dict[str, Path] | None = None) -> dict:
    p = paths() if p is None else p
    out = {"t": time.time()}
    for k, s in _SCALE.items():
        out[k] = float(p[k].read_text().strip()) * s
    return out


def identity(p: dict[str, Path] | None = None) -> dict:
    p = paths() if p is None else p
    return {
        "power_max_w": float(p["power_max_w"].read_text()) * 1e-6,
        "current_max_a": float(p["current_max_a"].read_text()) * 1e-3,
        "fw_bundle": p["fw_bundle"].read_text().strip(),
        "card_type": p["card_type"].read_text().strip(),
        "pcie_link_width": "x" + p["pcie_width"].read_text().strip(),
        "pcie_link_speed": p["pcie_speed"].read_text().strip(),
    }


class Sampler:
    """Start / stop a sampler subprocess writing FIELDS rows to `out` (CSV)."""

    def __init__(self, out: Path, period_ms: float = 100.0):
        self.out = Path(out)
        self.period_ms = period_ms
        self.proc = None

    def __enter__(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", __spec__.name if __spec__ else "prefill_telemetry", "--out", str(self.out)]
            + ["--period-ms", str(self.period_ms)]
        )
        deadline = time.time() + 10
        while time.time() < deadline and not (self.out.exists() and self.out.stat().st_size > 0):
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=10)

    def rows(self) -> list[dict]:
        if not self.out.exists():
            return []
        lines = self.out.read_text().strip().splitlines()[1:]
        return [dict(zip(FIELDS, map(float, ln.split(",")))) for ln in lines if ln.count(",") == len(FIELDS) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period-ms", type=float, default=100.0)
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    p = paths()
    with open(a.out, "w", buffering=1) as f:
        f.write(",".join(FIELDS) + "\n")
        while not stop:
            r = read(p)
            f.write(",".join(f"{r[k]:.6f}" if k == "t" else f"{r[k]:.4f}" for k in FIELDS) + "\n")
            time.sleep(a.period_ms / 1e3)


if __name__ == "__main__":
    main()
