#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# tt-telemetry-probe — periodic AICLK / TDP / TDC readout for CI logs
#
# Samples clock and power telemetry on a timer and prints one line per sample
# to stdout, so a long-running test shows its clock and power behaviour live in
# the CI console instead of only at the end.
#
# Two modes:
#   wrapper    tt_telemetry_probe.py [options] -- <command> [args...]
#              Probes while <command> runs, stops when it exits, and exits
#              with the command's exit code.
#   standalone tt_telemetry_probe.py [options]
#              Probes until SIGINT/SIGTERM.  Meant for `... &` plus a trap.
#
# Backends (--backend, default auto: sysfs when present, else tt-smi):
#   sysfs   Reads the attributes tt-kmd exports per device.  AICLK comes from
#           /sys/class/tenstorrent/tenstorrent!N/tt_aiclk (MHz); TDP and TDC
#           come from that device's hwmon node (power1_input in microwatts,
#           curr1_input in milliamps).  No subprocess and no device handle, so
#           it does not contend with the workload under test.
#   tt-smi  Shells out to `tt-smi -s` and parses the JSON snapshot.  Fallback
#           for hosts whose tt-kmd predates the tt_aiclk attribute.
#
# A blocked telemetry read cannot wedge the caller: sampling runs on a daemon
# thread under a timeout, and the probe reports the timeout and keeps going.
#
# Examples:
#   tt_telemetry_probe.py --interval 30 -- pytest tests/foo.py
#   tt_telemetry_probe.py --interval 60 --per-device --backend sysfs

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# (label, unit) in report order.
_FIELDS = (("AICLK", "MHz"), ("TDP", "W"), ("TDC", "A"))

_PREFIX = "[tt-telemetry-probe]"

_SYS = Path("/sys")
_TT_CLASS = _SYS / "class/tenstorrent"


# sysfs backend


def _contained(path: Path, base: Path) -> Path | None:
    """Resolve *path* and return it only if it stays under *base*."""
    resolved = path.resolve()
    return resolved if resolved.is_relative_to(base) else None


def _read_int(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.read_text().strip())
    except (IOError, OSError, ValueError):
        return None


class SysfsDevice:
    """Telemetry reader backed by the tt-kmd sysfs attributes."""

    backend_name = "sysfs"

    def __init__(self, index: int, label: str, aiclk: Path, power: Path | None, current: Path | None):
        self.index = index
        self.label = label
        self._aiclk = aiclk
        self._power = power
        self._current = current

    def read(self) -> dict[str, Any]:
        power_uw = _read_int(self._power)
        current_ma = _read_int(self._current)
        return {
            "index": self.index,
            "label": self.label,
            "AICLK": _read_int(self._aiclk),
            "TDP": None if power_uw is None else power_uw / 1e6,
            "TDC": None if current_ma is None else current_ma / 1e3,
        }


def discover_sysfs_devices() -> list[SysfsDevice]:
    """Return one reader per Tenstorrent device exposing tt_aiclk, ordered by device index."""
    if not _TT_CLASS.is_dir():
        return []

    devices = []
    for entry in sorted(_TT_CLASS.iterdir(), key=lambda p: p.name):
        # Nodes are named "tenstorrent!N", where N indexes /dev/tenstorrent/N.
        _, _, suffix = entry.name.partition("!")
        if not suffix.isdigit():
            continue

        aiclk = _contained(entry / "tt_aiclk", _SYS)
        if aiclk is None or not aiclk.is_file():
            continue

        pci_dir = _contained(entry / "device", _SYS)
        power = current = None
        label = entry.name
        if pci_dir is not None and pci_dir.is_dir():
            label = pci_dir.name  # PCI BDF, e.g. 0000:01:00.0
            hwmon_root = pci_dir / "hwmon"
            if hwmon_root.is_dir():
                for hwmon in sorted(hwmon_root.iterdir(), key=lambda p: p.name):
                    power = _contained(hwmon / "power1_input", _SYS)
                    current = _contained(hwmon / "curr1_input", _SYS)
                    break

        devices.append(SysfsDevice(int(suffix), label, aiclk, power, current))

    return devices


def sample_sysfs(devices: list[SysfsDevice]) -> tuple[str, list[dict[str, Any]]]:
    import socket

    return socket.gethostname(), [device.read() for device in devices]


# tt-smi backend

# Field label -> (raw smbus_telem key, decoded telemetry key).
_TT_SMI_KEYS = {"AICLK": ("AICLK", "aiclk"), "TDP": ("TDP", "power"), "TDC": ("TDC", "current")}


def _to_int(value: Any) -> int | None:
    """Coerce a tt-smi field to an int, or None if it is absent/unparseable.

    Raw smbus_telem fields are hex strings ("0x320"); decoded telemetry fields
    are space-padded decimal strings (" 800").
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.lower().startswith("0x"):
            # Wormhole packs [31:16]=limit, [15:0]=current; Blackhole reports
            # limits in separate fields and fits the value in the low half.
            return int(text, 16) & 0xFFFF
        return int(float(text))
    except ValueError:
        return None


def parse_snapshot(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse a `tt-smi -s` dump into (hostname, per-device readings)."""
    start = stdout.find("{")
    if start < 0:
        raise ValueError("no JSON object found in tt-smi output")
    doc, _ = json.JSONDecoder().raw_decode(stdout[start:])

    hostname = str(doc.get("host_info", {}).get("Hostname", "?"))

    devices = []
    for index, device in enumerate(doc.get("device_info") or []):
        smbus = device.get("smbus_telem") or {}
        decoded = device.get("telemetry") or {}
        board = device.get("board_info") or {}

        reading: dict[str, Any] = {"index": index, "label": str(board.get("bus_id", "?"))}
        for label, (raw_key, decoded_key) in _TT_SMI_KEYS.items():
            value = _to_int(smbus.get(raw_key))
            if value is None:
                value = _to_int(decoded.get(decoded_key))
            reading[label] = value
        devices.append(reading)

    return hostname, devices


def sample_tt_smi(timeout_s: float) -> tuple[str, list[dict[str, Any]]]:
    result = subprocess.run(
        ["tt-smi", "-s", "--snapshot_no_tty"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tt-smi exited {result.returncode}: {result.stderr.strip()[:200]}")
    return parse_snapshot(result.stdout)


# Reporting


def format_sample(hostname: str, devices: list[dict[str, Any]], backend: str, per_device: bool) -> str:
    """Render one sample: a min/avg/max summary line, optionally per device."""
    stamp = time.strftime("%H:%M:%S")
    parts = [f"{_PREFIX} {stamp} host={hostname} backend={backend} devices={len(devices)}"]

    for label, unit in _FIELDS:
        values = [d[label] for d in devices if d[label] is not None]
        if not values:
            parts.append(f"{label}(n/a)")
            continue
        stat = f"{label}({unit}) min={min(values):.0f} avg={sum(values) / len(values):.0f} max={max(values):.0f}"
        if label != "AICLK":
            stat += f" sum={sum(values):.0f}"
        parts.append(stat)

    lines = ["  ".join(parts)]
    if per_device:
        for d in devices:
            fields = "  ".join(
                f"{label}={'n/a' if d[label] is None else format(d[label], '.0f')}{unit}" for label, unit in _FIELDS
            )
            lines.append(f"{_PREFIX}   dev{d['index']:02d} {d['label']}  {fields}")
    return "\n".join(lines)


def _emit(text: str) -> None:
    """Write a whole sample in one call so it does not interleave mid-line."""
    print(text, flush=True)


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run *fn* on a daemon thread, raising TimeoutError if it overruns.

    A degraded multi-chip host can leave a telemetry read blocked in the
    kernel; abandoning the thread keeps the probe (and process exit) alive.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"telemetry read did not return within {timeout_s:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class Probe(threading.Thread):
    """Background thread that samples telemetry until stopped."""

    def __init__(self, sampler: Callable[[], tuple[str, list[dict[str, Any]]]], backend: str, args: Any):
        super().__init__(daemon=True)
        self._sampler = sampler
        self._backend = backend
        self._interval_s = args.interval
        self._timeout_s = args.timeout
        self._max_failures = args.max_failures
        self._per_device = args.per_device
        # Not _stop: that name is a private method on threading.Thread.
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()
        self.join(timeout=self._timeout_s + 5.0)

    def run(self) -> None:
        failures = 0
        while not self._stopping.is_set():
            try:
                hostname, devices = _call_with_timeout(self._sampler, self._timeout_s)
                _emit(format_sample(hostname, devices, self._backend, self._per_device))
                failures = 0
            except Exception as exc:
                failures += 1
                _emit(f"{_PREFIX} WARNING: sample failed ({failures}/{self._max_failures}): {exc}")
                if failures >= self._max_failures:
                    _emit(f"{_PREFIX} giving up after {failures} consecutive failures; the test is unaffected.")
                    return
            self._stopping.wait(self._interval_s)


def build_sampler(backend: str, timeout_s: float) -> tuple[Callable[[], tuple[str, list[dict[str, Any]]]], str]:
    """Resolve *backend* to a sampler callable and the name it settled on."""
    if backend in ("auto", "sysfs"):
        devices = discover_sysfs_devices()
        if devices:
            return (lambda: sample_sysfs(devices)), "sysfs"
        if backend == "sysfs":
            raise SystemExit(f"{_PREFIX} ERROR: --backend sysfs requested but no tt_aiclk attributes under {_TT_CLASS}")
        _emit(f"{_PREFIX} no tt-kmd sysfs telemetry under {_TT_CLASS}; falling back to tt-smi")

    return (lambda: sample_tt_smi(timeout_s)), "tt-smi"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="tt_telemetry_probe.py",
        description="Print AICLK/TDP/TDC on a timer, optionally around a command.",
        usage="%(prog)s [options] [-- <command> [args...]]",
    )

    def _positive_float(value: str) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
        return parsed

    parser.add_argument("--interval", type=_positive_float, default=30.0, help="Seconds between samples (default: 30)")
    parser.add_argument(
        "--timeout", type=_positive_float, default=60.0, help="Seconds to allow one sample (default: 60)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sysfs", "tt-smi"],
        default="auto",
        help="Telemetry source (default: auto, which prefers sysfs and falls back to tt-smi)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Stop probing after this many consecutive failed samples (default: 5)",
    )
    parser.add_argument(
        "--per-device",
        action="store_true",
        help="Also print one line per device; the default is a min/avg/max summary only",
    )

    if "--" in argv:
        separator = argv.index("--")
        return parser.parse_args(argv[:separator]), argv[separator + 1 :]
    return parser.parse_args(argv), []


def main(argv: list[str] | None = None) -> int:
    args, command = parse_args(sys.argv[1:] if argv is None else argv)

    sampler, backend = build_sampler(args.backend, args.timeout)
    probe = Probe(sampler, backend, args)
    probe.start()

    if not command:
        done = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda _signum, _frame: done.set())
        done.wait()
        probe.stop()
        return 0

    try:
        exit_code = subprocess.call(command)
    except FileNotFoundError:
        _emit(f"{_PREFIX} ERROR: command not found: {command[0]}")
        exit_code = 127
    except KeyboardInterrupt:
        exit_code = 130
    probe.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
