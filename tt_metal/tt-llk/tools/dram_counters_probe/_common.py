# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Helpers of dram_validate.py: the card busy check and device selection; nothing here touches a device register."""

import os
import sys
from pathlib import Path


def card_busy(device_path):
    """True when another process has the device node open."""
    target = os.path.realpath(device_path)
    me = str(os.getpid())
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        fd_dir = os.path.join("/proc", pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    return True
            except OSError:
                continue
    return False


def visible_devices_value(device_path):
    """TT_VISIBLE_DEVICES value that leaves only this device node: its index among the sorted /dev/tenstorrent nodes."""
    number = int(os.path.basename(os.path.realpath(device_path)))
    nodes = sorted(int(n) for n in os.listdir("/dev/tenstorrent") if n.isdigit())
    if number not in nodes:
        raise SystemExit(f"{device_path} is not a /dev/tenstorrent device node")
    return str(nodes.index(number))


def executable(path):
    """Absolute path of an existing executable file, or exit."""
    resolved = Path(path).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SystemExit(f"not an executable file: {path}")
    return str(resolved)


def fail(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)
