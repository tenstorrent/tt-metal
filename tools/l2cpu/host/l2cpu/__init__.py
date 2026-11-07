# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""l2cpu: host side of the Blackhole L2CPU (x280) bring-up component.

    from l2cpu import L2cpuHw, L2cpuCtl, make_backend
    ctl = L2cpuCtl(L2cpuHw(make_backend("umd"), tile=0, guard=True), region_pa)
    ctl.start(open("fw.bin", "rb").read())             # or start_tiles([ctl0, ctl1, ...], image): one release
"""
from . import layout
from .ctl import L2cpuCtl, L2cpuCtlError, start_tiles
from .hw import ClockGuard, L2cpuHw, TtnnClusterBackend, UmdBackend, make_backend
from .monitor import L2cpuMonitor

__all__ = [
    "layout",
    "L2cpuCtl",
    "L2cpuCtlError",
    "start_tiles",
    "ClockGuard",
    "L2cpuHw",
    "TtnnClusterBackend",
    "UmdBackend",
    "make_backend",
    "L2cpuMonitor",
]
