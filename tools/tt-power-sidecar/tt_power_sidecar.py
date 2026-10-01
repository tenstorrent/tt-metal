#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# tt-power-sidecar — live power measurement sidecar for Tenstorrent hardware
#
# Wraps any command and polls Wormhole/Blackhole power telemetry in the
# background.  Produces a JSON report with per-device energy (J), average /
# peak / min power (W), sample count, and wall-clock duration.
#
# Backends (tried in order):
#   1. sysfs hwmon  — reads /sys/class/hwmon/hwmonN/power1_input (microwatts)
#                     exposed by the tt-kmd kernel driver.  Zero dependencies.
#   2. pyluwen      — Python bindings to Tenstorrent firmware telemetry.
#                     Optional; used only when sysfs is unavailable.
#
# Usage:
#   tt-power-sidecar [options] -- <command> [args...]
#
# Examples:
#   tt-power-sidecar -- pytest tests/ops/test_matmul.py
#   tt-power-sidecar --interval 50 -o power.json -- ./build/test_add
#   tt-power-sidecar --devices 0,1 -v -- sleep 10

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Sysfs hwmon backend

_HWMON_BASE = Path("/sys/class/hwmon")

# hwmon names registered by tt-kmd per architecture.
_TT_HWMON_NAMES = frozenset({"wormhole", "blackhole"})


def _safe_hwmon_path(path: Path | str) -> Path | None:
    """Return *path* resolved to a real path if safe to read, else None.

    Checks: (1) original path is under _HWMON_BASE, (2) resolved path stays
    within /sys (kernel-managed, not user-writable).
    """
    p = Path(path)
    real = p.resolve()
    try:
        p.relative_to(_HWMON_BASE)
    except ValueError:
        _eprint(
            "[tt-power-sidecar] WARNING: device discovery skipped %s — "
            "path is outside hwmon base %s" % (path, _HWMON_BASE)
        )
        return None
    if not real.is_relative_to(Path("/sys")):
        _eprint(
            "[tt-power-sidecar] WARNING: device discovery skipped %s — "
            "resolved path %s is outside /sys" % (path, real)
        )
        return None
    return real


def _discover_sysfs_devices_inner() -> dict[int, Path]:
    """Return {device_index: hwmon_path} for Tenstorrent hwmon entries.

    Inner implementation; callers use ``_discover_sysfs_devices()`` (adds timeout).
    """
    devices_raw: list[tuple[str, Path]] = []  # (sort_key, power_file)
    if not _HWMON_BASE.is_dir():
        return {}

    for entry in sorted(p.name for p in _HWMON_BASE.iterdir()):
        hwmon_dir = _HWMON_BASE / entry
        name_file = _safe_hwmon_path(hwmon_dir / "name")
        power_file = _safe_hwmon_path(hwmon_dir / "power1_input")

        if name_file is None or power_file is None:
            continue
        if not name_file.is_file() or not power_file.is_file():
            continue

        try:
            name = name_file.read_text().strip().lower()
        except (IOError, OSError):
            continue

        if not (name in _TT_HWMON_NAMES or "tenstorrent" in name or name.startswith("tt")):
            continue

        # Try to get PCI bus address for stable ordering (matches /dev/tenstorrent/{N}).
        sort_key = entry  # fallback: hwmon dir name
        try:
            device_link = (hwmon_dir / "device").resolve()
            sort_key = device_link.name  # e.g. "0000:03:00.0"
        except (OSError, ValueError):
            pass  # device symlink absent or unresolvable; fall back to hwmon dir name sort key

        devices_raw.append((sort_key, power_file))

    devices_raw.sort(key=lambda t: t[0])
    return {idx: path for idx, (_, path) in enumerate(devices_raw)}


def _discover_sysfs_devices(timeout_s: float = 15) -> dict[int, Path]:
    """Return {device_index: hwmon_path} for Tenstorrent hwmon entries.

    Runs the scan on a daemon thread with a timeout.  On multi-chip systems
    (T3000, galaxy) a prior workload can leave the ARC RESPONSE_Q degraded,
    causing the kernel hwmon driver's ``show`` callback to block indefinitely.
    If the scan hangs, we return empty and let the wrapped command run
    without power monitoring.
    """
    result: dict[int, Path] = {}

    def _scan() -> None:
        found = _discover_sysfs_devices_inner()
        result.update(found)

    t = threading.Thread(target=_scan, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        _eprint(
            "[tt-power-sidecar] WARNING: sysfs device discovery did not complete "
            "within %ds — a hwmon sysfs read is likely blocked in the kernel "
            "(ARC RESPONSE_Q hung after a prior workload). "
            "Proceeding without sysfs devices." % timeout_s
        )
        return {}

    return result


def _read_sysfs_power_uw(path: Path) -> int | None:
    """Read instantaneous power in microwatts from a sysfs power1_input file.

    *path* must be a pre-validated resolved path from ``_safe_hwmon_path``.
    """
    try:
        return int(Path(path).read_text().strip())
    except (IOError, OSError, ValueError):
        return None


# pyluwen backend (optional fallback)

# Minimum interval between pyluwen ARC telemetry reads.  Polling faster than
# 1/s risks saturating the ARC response queue on multi-chip systems, causing
# "Timeout waiting for Ethernet core service remote IO request" in fabric ops.
# sysfs reads are unaffected (kernel-memory, no firmware involvement).
_PYLUWEN_MIN_INTERVAL_S = 1.0

_pyluwen = None  # lazy import


def _try_import_pyluwen() -> Any:
    global _pyluwen
    if _pyluwen is not None:
        return _pyluwen

    try:
        import pyluwen

        _pyluwen = pyluwen
    except ImportError:
        _pyluwen = False  # sentinel: tried and failed
    return _pyluwen


def _discover_pyluwen_devices(timeout_s: float = 10) -> dict[int, Any]:
    """Return {device_index: chip_object} via pyluwen, or empty dict.

    Runs detect_chips() in a daemon thread with a hard timeout.  On degraded
    multi-chip systems, detect_chips() holds the GIL and blocks — the daemon
    thread is abandoned if it does not complete within timeout_s.
    """
    pl = _try_import_pyluwen()
    if not pl:
        return {}

    result: dict[int, Any] = {}

    def _detect() -> None:
        try:
            chips = pl.detect_chips()
            for idx, chip in enumerate(chips):
                try:
                    local = chip.as_wh() is not None or chip.as_bh() is not None
                except Exception:
                    local = False
                if local:
                    result[idx] = chip
        except Exception:
            pass  # detect_chips() failed (pyluwen unavailable or ARC unresponsive); _detect returns empty result

    t = threading.Thread(target=_detect, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        _eprint(
            "[tt-power-sidecar] WARNING: pyluwen detect_chips() timed out after %ds "
            "(ARC RESPONSE_Q busy after a prior workload). "
            "Proceeding without pyluwen devices." % timeout_s
        )
        return {}

    # All-remote systems (T3000, galaxy): nothing pyluwen can add.
    if not result:
        return {}

    return result


def _read_pyluwen_power_w(chip: Any) -> float | None:
    """Read instantaneous power in watts from a pyluwen chip object.

    ``tdp`` register: packed u32, bits[15:0] = current power (W).
    Blackhole fallback: P = (tdc & 0xFFFF) * vcore_mV / 1000.
    """
    try:
        # Wormhole path
        wh = chip.as_wh()
        if wh is not None:
            telemetry = wh.get_telemetry()
            if hasattr(telemetry, "tdp"):
                return float(telemetry.tdp & 0xFFFF)
    except Exception:
        # Fall through to Blackhole path.
        pass

    try:
        # Blackhole path
        bh = chip.as_bh()
        if bh is not None:
            telemetry = bh.get_telemetry()
            if hasattr(telemetry, "tdp"):
                return float(telemetry.tdp & 0xFFFF)
            # Fallback: P = I * V
            if hasattr(telemetry, "tdc") and hasattr(telemetry, "vcore"):
                return float(telemetry.tdc & 0xFFFF) * float(telemetry.vcore) / 1000.0
    except Exception:
        pass  # Blackhole telemetry read failed; caller treats None as a dropped sample

    return None


# Unified device abstraction


@dataclass
class SysfsDevice:
    """Power reader backed by sysfs hwmon."""

    index: int
    path: Path
    backend_name: ClassVar[str] = "sysfs"

    def read_power_w(self) -> float | None:
        uw = _read_sysfs_power_uw(self.path)
        if uw is None:
            return None
        return uw / 1e6  # microwatts -> watts


@dataclass
class PyluwenDevice:
    """Power reader backed by pyluwen.

    Reads are throttled to 1/s to avoid flooding ARC response queues.
    Faster poll ticks return None (skipped); the energy integral over
    actual samples remains correct.
    """

    index: int
    chip: Any
    backend_name: ClassVar[str] = "pyluwen"
    # Initialised to -MIN_INTERVAL so the very first read is always allowed.
    _last_read_ts: float = field(default=-_PYLUWEN_MIN_INTERVAL_S, init=False, repr=False)

    def read_power_w(self) -> float | None:
        now = time.monotonic()
        if now - self._last_read_ts < _PYLUWEN_MIN_INTERVAL_S:
            return None  # throttled — skip this poll tick
        self._last_read_ts = now
        return _read_pyluwen_power_w(self.chip)


def detect_devices(
    requested_indices: set[int] | None = None,
    backend: str = "auto",
) -> list[SysfsDevice | PyluwenDevice]:
    """Detect Tenstorrent devices.  Returns a list of device reader objects.

    *backend*:
    - ``"auto"``: sysfs first, pyluwen for chips not covered by sysfs (e.g.
      the remote chip on N300).  Suppresses pyluwen if 2+ remote chips would
      be needed (T3000/galaxy) to avoid ARC RESPONSE_Q desync.
    - ``"sysfs"``: sysfs hwmon only.  Safe for all systems (kernel-memory
      reads, no ARC traffic).  N300 reports only the local PCIe chip.
    - ``"pyluwen"``: firmware telemetry only.  Unsafe on multi-chip systems.

    *requested_indices*: if given, filters to those device indices only.
    """
    if backend == "sysfs":
        sysfs = _discover_sysfs_devices()
        return [
            SysfsDevice(idx, path)
            for idx, path in sorted(sysfs.items())
            if requested_indices is None or idx in requested_indices
        ]

    if backend == "pyluwen":
        pl_devs = _discover_pyluwen_devices()
        return [
            PyluwenDevice(idx, chip)
            for idx, chip in sorted(pl_devs.items())
            if requested_indices is None or idx in requested_indices
        ]

    # auto: sysfs takes priority; pyluwen supplements for remote-only chips.
    sysfs = _discover_sysfs_devices()

    # 2+ sysfs chips means any pyluwen-only chips are remote, and the guard
    # below discards systems with >1 remote chip.  Skip pyluwen entirely.
    if len(sysfs) >= 2:
        return [
            SysfsDevice(idx, path)
            for idx, path in sorted(sysfs.items())
            if requested_indices is None or idx in requested_indices
        ]

    pl_devs = _discover_pyluwen_devices()

    merged: dict[int, SysfsDevice | PyluwenDevice] = {}
    for idx, path in sorted(sysfs.items()):
        merged[idx] = SysfsDevice(idx, path)
    for idx, chip in sorted(pl_devs.items()):
        if idx not in merged:
            merged[idx] = PyluwenDevice(idx, chip)

    # Guard: >1 remote pyluwen chip desynchronises ARC RESPONSE_Q regardless
    # of polling rate (concurrent cross-chip access, not frequency).  Fall back
    # to sysfs-only.
    remote_devs = [d for d in merged.values() if isinstance(d, PyluwenDevice)]
    if len(remote_devs) > 1:
        _eprint(
            "[tt-power-sidecar] WARNING: %d remote (ethernet-only) chips detected. "
            "Polling multiple remote chips via pyluwen desynchronises ARC RESPONSE_Q "
            "regardless of polling rate (RESPONSE_Q out of sync / Timeout waiting for "
            "Ethernet core service remote IO request). "
            "Falling back to sysfs-only for this system — remote chips will not be "
            "measured. Use --backend pyluwen explicitly to override (not recommended)." % len(remote_devs)
        )
        merged = {idx: dev for idx, dev in merged.items() if not isinstance(dev, PyluwenDevice)}

    return [merged[idx] for idx in sorted(merged) if requested_indices is None or idx in requested_indices]


# Polling thread


@dataclass
class PowerPoller:
    """Background thread that collects power samples from a set of devices."""

    devices: list[SysfsDevice | PyluwenDevice]
    interval_s: float
    verbose: bool = False
    # Populated in __post_init__; listed here so the type is visible.
    samples: dict[int, list[tuple[float, float]]] = field(default_factory=dict, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for dev in self.devices:
            self.samples[dev.index] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 3 + 1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.monotonic()
            for dev in self.devices:
                watts = dev.read_power_w()
                if watts is not None:
                    self.samples[dev.index].append((ts, watts))
                    if self.verbose:
                        _eprint("[tt-power-sidecar] dev %d  %.2f W  (%s)" % (dev.index, watts, dev.backend_name))
            self._stop.wait(self.interval_s)


# Report generation


def compute_report(
    command: list[str],
    exit_code: int,
    wall_start: float,
    wall_end: float,
    poll_interval_ms: int,
    devices: list[SysfsDevice | PyluwenDevice],
    poller: PowerPoller,
) -> dict[str, Any]:
    """Build the JSON-serialisable report dict."""
    duration_s = wall_end - wall_start

    device_reports: dict[str, Any] = {}
    for dev in devices:
        # Defensive None filter — poller already drops them, but guard aggregates.
        samples = [(ts, w) for ts, w in poller.samples.get(dev.index, []) if w is not None]
        n = len(samples)
        if n == 0:
            device_reports[str(dev.index)] = {
                "energy_J": 0.0,
                "energy_Wh": 0.0,
                "avg_power_W": 0.0,
                "peak_power_W": 0.0,
                "min_power_W": 0.0,
                "sample_count": 0,
                "backend": dev.backend_name,
            }
            continue

        powers = [s[1] for s in samples]
        peak = max(powers)
        minimum = min(powers)

        # Trapezoidal integration.
        energy = 0.0
        for i in range(1, n):
            dt = samples[i][0] - samples[i - 1][0]
            avg_p = (samples[i][1] + samples[i - 1][1]) / 2.0
            energy += avg_p * dt

        # Use sampling span (not wall-clock duration) to avoid avg < min when
        # the command outlasts the sampling window.
        sampling_span = samples[-1][0] - samples[0][0] if n > 1 else 0.0
        avg_power = energy / sampling_span if sampling_span > 0 else powers[0]

        device_reports[str(dev.index)] = {
            "energy_J": round(energy, 3),
            "energy_Wh": round(energy / 3600.0, 6),
            "avg_power_W": round(avg_power, 3),
            "peak_power_W": round(peak, 3),
            "min_power_W": round(minimum, 3),
            "sample_count": n,
            "backend": dev.backend_name,
        }

    return {
        "command": list(command),
        "exit_code": exit_code,
        "duration_s": round(duration_s, 3),
        "poll_interval_ms": poll_interval_ms,
        "devices": device_reports,
    }


def print_summary(report: dict[str, Any], file: Any = None) -> None:
    """Print a human-readable summary to *file* (default: stderr)."""
    if file is None:
        file = sys.stderr

    _eprint("", file=file)
    _eprint("=" * 60, file=file)
    _eprint("  tt-power-sidecar  —  measurement summary", file=file)
    _eprint("=" * 60, file=file)
    _eprint("  Command   : %s" % " ".join(report["command"]), file=file)
    _eprint("  Exit code : %d" % report["exit_code"], file=file)
    _eprint("  Duration  : %.2f s" % report["duration_s"], file=file)
    _eprint("  Interval  : %d ms" % report["poll_interval_ms"], file=file)
    _eprint("-" * 60, file=file)

    for dev_idx in sorted(report["devices"].keys(), key=int):
        d = report["devices"][dev_idx]
        _eprint("  Device %s (%s):" % (dev_idx, d["backend"]), file=file)
        _eprint("    Energy     : %.3f J  (%.6f Wh)" % (d["energy_J"], d["energy_Wh"]), file=file)
        _eprint("    Avg power  : %.3f W" % d["avg_power_W"], file=file)
        _eprint("    Peak power : %.3f W" % d["peak_power_W"], file=file)
        _eprint("    Min power  : %.3f W" % d["min_power_W"], file=file)
        _eprint("    Samples    : %d" % d["sample_count"], file=file)

    if not report["devices"]:
        _eprint("  (no devices detected — report contains timing only)", file=file)

    _eprint("=" * 60, file=file)
    _eprint("", file=file)


# Helpers


def _eprint(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


# CLI


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace | None, list[str] | None]:
    parser = argparse.ArgumentParser(
        prog="tt-power-sidecar",
        description="Measure Tenstorrent device power while running a command.",
        usage="%(prog)s [options] -- <command> [args...]",
    )

    def _positive_int(value: str) -> int:
        v = int(value)
        if v <= 0:
            raise argparse.ArgumentTypeError("--interval must be a positive integer, got %d" % v)
        return v

    parser.add_argument(
        "--interval",
        type=_positive_int,
        default=100,
        help=(
            "Poll interval in milliseconds, must be > 0 (default: 100).  "
            "sysfs devices honour this exactly.  pyluwen devices are always "
            "throttled to at most 1 read/s regardless of this setting — "
            "faster polling would flood ARC response queues."
        ),
    )
    parser.add_argument(
        "--out",
        "-o",
        default="power_report.json",
        help="Output JSON file path (default: power_report.json)",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated device indices to monitor (default: all detected)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sysfs", "pyluwen"],
        default="auto",
        help=(
            "Power measurement backend (default: auto).  "
            "'auto' tries sysfs first, falls back to pyluwen.  "
            "'sysfs' reads /sys/class/hwmon only — safe for multi-chip systems "
            "such as T3000.  "
            "'pyluwen' uses firmware telemetry — DO NOT use on T3000, or "
            "other multi-chip systems: pyluwen polls ARC via Ethernet and will "
            "corrupt ARC response queues, causing fabric test timeouts."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each power sample to stderr as it arrives",
    )

    # Split on '--' manually (argparse doesn't handle this natively).
    raw = argv if argv is not None else sys.argv[1:]

    try:
        sep = raw.index("--")
    except ValueError:
        parser.error("Missing command separator.  Usage: tt-power-sidecar [options] -- <command>")
        return None, None  # unreachable, but keeps linters happy

    sidecar_args = raw[:sep]
    cmd_args = raw[sep + 1 :]

    if not cmd_args:
        parser.error("No command given after '--'.")
        return None, None

    args = parser.parse_args(sidecar_args)
    return args, cmd_args


def main(argv: list[str] | None = None) -> int:
    args, cmd_args = parse_args(argv)

    requested = None
    if args.devices is not None:
        try:
            requested = set(int(x.strip()) for x in args.devices.split(","))
        except ValueError:
            _eprint("[tt-power-sidecar] ERROR: --devices must be comma-separated integers")
            return 2

    interval_s = args.interval / 1000.0

    devices = detect_devices(requested, backend=args.backend)
    if devices:
        _eprint(
            "[tt-power-sidecar] Detected %d device(s): %s"
            % (len(devices), ", ".join("%d (%s)" % (d.index, d.backend_name) for d in devices))
        )
        pyluwen_devs = [d for d in devices if d.backend_name == "pyluwen"]
        if pyluwen_devs and interval_s < _PYLUWEN_MIN_INTERVAL_S:
            _eprint(
                "[tt-power-sidecar] NOTE: %d pyluwen device(s) will be polled "
                "at most 1/s (throttled from %dms) to avoid flooding ARC queues." % (len(pyluwen_devs), args.interval)
            )
    else:
        _eprint(
            "[tt-power-sidecar] WARNING: No Tenstorrent devices detected.  "
            "Power data will be empty.  The command will still run."
        )

    poller = PowerPoller(devices, interval_s, verbose=args.verbose)
    poller.start()

    wall_start = time.monotonic()
    try:
        exit_code = subprocess.call(cmd_args)
    except FileNotFoundError:
        _eprint("[tt-power-sidecar] ERROR: command not found: %s" % cmd_args[0])
        poller.stop()
        return 127
    except KeyboardInterrupt:
        _eprint("[tt-power-sidecar] Interrupted.")
        poller.stop()
        return 130
    wall_end = time.monotonic()

    poller.stop()

    report = compute_report(
        cmd_args,
        exit_code,
        wall_start,
        wall_end,
        args.interval,
        devices,
        poller,
    )

    out_path = Path(args.out)
    try:
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        _eprint("[tt-power-sidecar] Report written to %s" % out_path)
    except (IOError, OSError) as exc:
        _eprint("[tt-power-sidecar] ERROR: could not write report: %s" % exc)

    print_summary(report)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
