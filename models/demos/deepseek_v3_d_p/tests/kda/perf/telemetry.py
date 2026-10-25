# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole clock and power telemetry for the KDA perf tools (reads sysfs and hwmon only; never sets anything).

Under back-to-back load every chip runs into its power limit and the firmware lowers its AI clock, so a sustained
wall measures energy per call as much as time. Telemetry samples each chip's AI clock (MHz, the kernel driver's
tt_aiclk), power (W) and temperature (C, the blackhole hwmon) from a thread. The firmware refreshes them only every
~25-100 ms, so attribute power to one op only over a sustained loop of that op.
"""

from __future__ import annotations

import glob
import statistics
import threading
import time

_PERIOD_S = 0.01


def _read_text(path: str) -> str | None:
    try:
        with open(path) as file:
            return file.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read_text(path)
    try:
        return None if text is None else int(text)
    except ValueError:
        return None


def _scaled(value: int | None, unit: float) -> float | None:
    return None if value is None else value / unit


class Telemetry(threading.Thread):
    """Samples every Blackhole chip's AI clock (MHz), power (W) and temperature (C) until stopped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.clocks = sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk"))
        self.hwmons = [
            path for path in sorted(glob.glob("/sys/class/hwmon/hwmon*")) if _read_text(f"{path}/name") == "blackhole"
        ]
        self.samples: list[tuple[float, list, list, list]] = []
        self._done = threading.Event()

    @property
    def available(self) -> bool:
        return bool(self.clocks and self.hwmons)

    def run(self) -> None:
        while not self._done.is_set():
            self.samples.append(
                (
                    time.perf_counter(),
                    [_read_int(path) for path in self.clocks],
                    [_scaled(_read_int(f"{path}/power1_input"), 1e6) for path in self.hwmons],
                    [_scaled(_read_int(f"{path}/temp1_input"), 1e3) for path in self.hwmons],
                )
            )
            time.sleep(_PERIOD_S)

    def stop(self) -> None:
        self._done.set()
        self.join()

    def window(self, start: float, end: float) -> dict:
        """Clock, power and temperature statistics of the samples taken between perf_counter() ``start`` and ``end``."""
        rows = [row for row in self.samples if start <= row[0] <= end]
        if not rows:
            return {}
        clocks = [value for row in rows for value in row[1] if value is not None]
        powers = [sum(value for value in row[2] if value is not None) for row in rows]
        return {
            "samples": len(rows),
            "aiclk_mean_mhz": statistics.fmean(clocks),
            "aiclk_min_mhz": min(clocks),
            "aiclk_per_chip_mean_mhz": [
                statistics.fmean(row[1][chip] for row in rows) for chip in range(len(self.clocks))
            ],
            "power_total_mean_w": statistics.fmean(powers),
            "power_chip_max_w": max(value for row in rows for value in row[2] if value is not None),
            "temp_max_c": max(value for row in rows for value in row[3] if value is not None),
        }
