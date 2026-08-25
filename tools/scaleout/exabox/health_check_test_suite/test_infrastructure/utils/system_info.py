# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect TT firmware / driver version info from the host."""

from __future__ import annotations

import subprocess
from pathlib import Path


def collect_version_info() -> dict[str, str]:
    """Collect TT firmware and driver version information.

    Discovery flow:
      tt_smi  : run `tt-smi --version` CLI
      tt_kmd  : read /sys/module/tenstorrent/version (sysfs, fast)
                -> fallback: `modinfo -F version tenstorrent` (works if module is
                   installed but not loaded)
      fw_bundle: scan /sys/class/tenstorrent/tenstorrent!*/tt_fw_bundle_ver
                 (first device wins; directory absent = driver not loaded)

    All values default to "N/A" if unavailable.
    """
    versions = {"tt_smi": "N/A", "tt_kmd": "N/A", "fw_bundle": "N/A"}

    try:
        result = subprocess.run(["tt-smi", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            versions["tt_smi"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    kmod_path = Path("/sys/module/tenstorrent/version")
    try:
        versions["tt_kmd"] = kmod_path.read_text().strip()
    except OSError:
        try:
            result = subprocess.run(
                ["modinfo", "-F", "version", "tenstorrent"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                versions["tt_kmd"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    tt_class_dir = Path("/sys/class/tenstorrent")
    if tt_class_dir.is_dir():
        for dev_path in sorted(tt_class_dir.glob("tenstorrent!*")):
            fw_file = dev_path / "tt_fw_bundle_ver"
            if fw_file.is_file():
                try:
                    versions["fw_bundle"] = fw_file.read_text().strip()
                    break
                except OSError:
                    continue

    return versions
