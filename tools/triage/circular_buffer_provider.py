#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    circular_buffer_provider [--include-cb-content]

Options:
    --include-cb-content   Also read each CB's whole FIFO, as hex.

Description:
    Data provider for the circular buffers of running Tensix cores, one per CB of the program running
    on the core. get_circular_buffers(location) reads them, and get_cached_circular_buffers(location)
    reads them once and returns the same list after. With --include-cb-content, each is a
    CircularBufferDataWithContent, which adds the CB's whole FIFO as hex.

Owner:
    onenezicTT
"""

from dataclasses import dataclass
from functools import cache

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from triage import ScriptConfig, hex_serializer, run_script, triage_field, triage_singleton
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import RiscLocation
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device, read_words_from_device

script_config = ScriptConfig(
    data_provider=True,
    depends=["dispatcher_data"],
)

# (NUM_CIRCULAR_BUFFERS, tiles_received register, tiles_acked register); CB i uses overlay stream i.
Arch = tuple[int, int, int]
WORMHOLE: Arch = (32, 4, 3)
BLACKHOLE: Arch = (64, 10, 8)
STREAM_REG_BASE = 0xFFB40000
STREAM_REG_STRIDE = 0x1000
L1_ALIGNMENT = 16  # global CB counters count L1_ALIGNMENT-byte units


@dataclass
class CircularBufferData:
    cb: int = triage_field("CB")
    kind: str = triage_field("Kind")  # local, global sender or global receiver
    address: int = triage_field("Address", hex_serializer)
    size: int = triage_field("Size")
    page_size: int = triage_field("Page Size")
    size_pages: int = triage_field("Size (pages)")
    pushed: int = triage_field("Pushed")
    popped: int = triage_field("Popped")
    occupancy: int = triage_field("Occupancy")


@dataclass
class CircularBufferDataWithContent(CircularBufferData):
    content: str | None = triage_field("Content")  # whole FIFO as hex, None for a global sender


class CircularBufferProvider:
    def __init__(self, dispatcher_data: DispatcherData, include_content: bool):
        self.dispatcher_data = dispatcher_data
        self.include_content = include_content

    @cache
    def get_cached_circular_buffers(self, location: OnChipCoordinate) -> list[CircularBufferData]:
        return self.get_circular_buffers(location)

    def get_circular_buffers(self, location: OnChipCoordinate) -> list[CircularBufferData]:
        arch = WORMHOLE if location.device.is_wormhole() else BLACKHOLE if location.device.is_blackhole() else None
        if arch is None:
            return []
        core = self.dispatcher_data.get_cached_core_data(RiscLocation(location, None, "brisc"))
        if core.go_message == "DONE" or core.mailboxes is None:
            return []
        kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
        if int(kernel_config.enables) == 0:
            return []
        cbs = self._read_local_cbs(location, kernel_config, core.kernel_config_base, arch)
        return cbs + self._read_remote_cbs(location, kernel_config, core.kernel_config_base, arch)

    def _read_local_cbs(
        self, location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch
    ) -> list[CircularBufferData]:
        num_cbs, received_reg, acked_reg = arch
        mask = int(kernel_config.local_cb_mask)
        base = config_base + int(kernel_config.local_cb_offset)
        cbs: list[CircularBufferData] = []
        for cb in range(num_cbs):
            if not mask >> cb & 1:
                continue
            address, size, pages, page_size = read_words_from_device(location, base + 16 * cb, word_count=4)
            stream = STREAM_REG_BASE + STREAM_REG_STRIDE * cb
            pushed = read_word_from_device(location, stream + 4 * received_reg)
            popped = read_word_from_device(location, stream + 4 * acked_reg)
            occupancy = (pushed - popped) & 0xFFFF
            row = (cb, "local", address, size, page_size, pages, pushed, popped, occupancy)
            if self.include_content:
                content = read_from_device(location, address, num_bytes=size).hex()
                cbs.append(CircularBufferDataWithContent(*row, content))
            else:
                cbs.append(CircularBufferData(*row))
        return cbs

    def _read_remote_cbs(
        self, location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch
    ) -> list[CircularBufferData]:
        num_cbs = arch[0]
        base = config_base + int(kernel_config.remote_cb_offset)
        cbs: list[CircularBufferData] = []
        for cb in range(int(kernel_config.min_remote_cb_start_index), num_cbs):
            # Remote configs are laid out from the last CB index down.
            config_addr, page_size = read_words_from_device(location, base + 8 * (num_cbs - 1 - cb), word_count=2)
            if config_addr == 0:
                continue
            is_sender, num_receivers, address, size, _, _, counters, _ = read_words_from_device(
                location, config_addr, word_count=8
            )
            # A sender has a counter pair per receiver, a receiver only its own; keep the one furthest behind.
            slots = num_receivers if is_sender else 1
            sent_addrs = [counters + 2 * L1_ALIGNMENT * i for i in range(slots)]
            pairs = [
                (read_word_from_device(location, a), read_word_from_device(location, a + L1_ALIGNMENT))  # sent, acked
                for a in sent_addrs
            ]
            sent, acked = max(pairs, key=lambda pair: (pair[0] - pair[1]) & 0xFFFFFFFF)
            units = page_size // L1_ALIGNMENT  # counter units per page
            pushed, popped = sent // units, acked // units
            occupancy = ((sent - acked) & 0xFFFFFFFF) // units
            kind = "global sender" if is_sender else "global receiver"
            pages = size // page_size
            row = (cb, kind, address, size, page_size, pages, pushed, popped, occupancy)
            if self.include_content:
                content = None if is_sender else read_from_device(location, address, num_bytes=size).hex()
                cbs.append(CircularBufferDataWithContent(*row, content))
            else:
                cbs.append(CircularBufferData(*row))
        return cbs


@triage_singleton
def run(args, context: Context) -> CircularBufferProvider:
    return CircularBufferProvider(get_dispatcher_data(args, context), bool(args["--include-cb-content"]))


if __name__ == "__main__":
    run_script()
