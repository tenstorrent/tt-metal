# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reads l1 regions and registers for router cores through ttexalens.
"""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Callable

from tt_metal.fabric.debug.visualizer.capture.reads import RouterPlan
from tt_metal.fabric.debug.visualizer.capture.snapshot import CaptureError, OpenedRouter

# The ttexalens Device method that says a chip is of the manifest's run.arch.
_ARCH_CHECKS = {"wormhole_b0": "is_wormhole", "blackhole": "is_blackhole"}


class ExalensDevice:
    """A snapshot Device that reads through a ttexalens context."""

    def __init__(self, context, coordinate: Callable, read_from_device: Callable, version: str):
        self.context = context
        self.coordinate = coordinate  # ttexalens's OnChipCoordinate
        self.read_from_device = read_from_device
        self.version = version

    def open_router(self, router: RouterPlan, arch: str) -> OpenedRouter:
        chip = self.context.device_by_unique_id.get(router.asic_id)
        if chip is None:
            raise CaptureError(f"ttexalens has no chip with unique id 0x{router.asic_id:016x}")
        if arch not in _ARCH_CHECKS:
            raise CaptureError(f"capture can't check that a chip is {arch}")
        if not getattr(chip, _ARCH_CHECKS[arch])():
            raise CaptureError(f"ttexalens's chip {chip.id} isn't {arch}, the manifest's run.arch")
        return OpenedRouter(handle=self._find_core(chip, router), ttexalens_id=chip.id)

    def _find_core(self, chip, router: RouterPlan):
        """
        The router's core on `chip`: its virtual_core as a translated coordinate or, failing that, eth core number
        eth_chan, whichever ttexalens puts at the router's logical_core.
        """
        attempts = (
            (f"translated {router.virtual_core}", lambda: self.coordinate(*router.virtual_core, "translated", chip)),
            (f"eth core {router.eth_chan}", lambda: chip.get_block_locations("eth")[router.eth_chan]),
        )
        found = []
        for description, find in attempts:
            try:
                core = find()
                logical = tuple(core.to("logical-eth"))
            except Exception as error:
                found.append(f"{description} raises {type(error).__name__}: {error}")
                continue
            if logical == router.logical_core:
                if found:
                    print(f"{router.path}: {found[0]}; reading {description}", file=sys.stderr)
                return core
            found.append(f"{description} is logical {logical}")
        raise CaptureError(f"no core is at logical {router.logical_core}: {'; '.join(found)}")

    def read(self, opened: OpenedRouter, address: int, size: int) -> bytes:
        try:
            return self.read_from_device(
                opened.handle, address, device_id=opened.ttexalens_id, num_bytes=size, context=self.context
            )
        except Exception as error:
            raise CaptureError(
                f"reading {size} bytes at {address:#x} raises {type(error).__name__}: {error}"
            ) from error


def open_exalens() -> ExalensDevice:
    """
    Starts ttexalens with its defaults: NOC1, failing over to NOC0, and safe mode, which refuses reads outside the
    blocks ttexalens knows but not a wrong address inside one.
    """
    from ttexalens.coordinate import OnChipCoordinate
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.tt_exalens_lib import read_from_device

    return ExalensDevice(init_ttexalens(), OnChipCoordinate, read_from_device, importlib.metadata.version("tt-exalens"))
