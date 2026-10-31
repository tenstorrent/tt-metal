# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Host monitor thread for the sampling firmware: the bring-up component's L2cpuMonitor (heartbeats, firmware error
word, 100 ms period, 2 s stall, dump + os._exit(17)) plus this example's check of the link block's wait status word
(the Tensix wait op writes 0xDEAD0000 | req there when it hits its bound).

    mon = SamplingMonitor(fw).start(); ...; summary = mon.stop()
exit_on_fail=False: record the failure only (the caller recovers, e.g. --retry-on-timeout). On exit the device is
not closed (queued wait ops drain on their own); the next run must start with a chip reset.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

from l2cpu.monitor import L2cpuMonitor  # noqa: E402
from tensix import ops as _link  # noqa: E402
from l2cpu_ops import LINK_BLOCK_OFF as _LINK  # noqa: E402


class SamplingMonitor(L2cpuMonitor):
    def __init__(self, fw, interval=0.1, stall_s=2.0, exit_code=17, log=None, on_fail=None, exit_on_fail=True):
        """fw: the sampling firmware session (host helper with .hw and .base = region base)."""
        ctl = fw.ctl
        ws_pa = fw.base + _LINK + _link.LINK_OFF_WAIT_STATUS

        def wait_status():
            ws = fw.hw.pa_read32(ws_pa)
            return f"wait op status word 0x{ws:08x} (0xDEAD....: done_seq wait hit its bound)" if ws else None

        super().__init__(
            ctl,
            interval=interval,
            stall_s=stall_s,
            exit_code=exit_code if exit_on_fail else None,
            log=log,
            on_fail=on_fail,
            extra_checks=(wait_status,),
        )
