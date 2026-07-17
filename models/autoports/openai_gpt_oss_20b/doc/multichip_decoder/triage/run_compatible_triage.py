#!/usr/bin/env python3
"""Run repo tt-triage with a local tt-exalens/tt-umd compatibility shim.

tt-exalens 0.3.21 passes writable ``memoryview`` objects to the five-argument
``tt_umd.TTDevice.noc_read`` overload, while the installed tt-umd 0.9.5 binding
accepts only ``bytearray`` for that overload.  Copy through a bytearray so live
read-only triage can collect core state without modifying either installation.
"""

from __future__ import annotations

import pathlib
import sys

from ttexalens.umd_device import UmdDevice

_original_read_from_device_reg = UmdDevice._UmdDevice__read_from_device_reg


def _compatible_read_from_device_reg(self, coord, address, buffer, dma_threshold):
    if isinstance(buffer, memoryview):
        compatible_buffer = bytearray(len(buffer))
        _original_read_from_device_reg(self, coord, address, compatible_buffer, dma_threshold)
        buffer[:] = compatible_buffer
        return
    _original_read_from_device_reg(self, coord, address, buffer, dma_threshold)


UmdDevice._UmdDevice__read_from_device_reg = _compatible_read_from_device_reg

repo_root = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(repo_root / "tools" / "triage"))

from triage import main  # noqa: E402

if __name__ == "__main__":
    main()
