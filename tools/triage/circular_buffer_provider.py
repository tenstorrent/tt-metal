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
    Data provider for the circular buffers of running Tensix cores. get_circular_buffers(location)
    reads a core's CBs once and caches them, one per CB of the program running on the core.

Owner:
    onenezicTT
"""

import threading
from dataclasses import dataclass

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
    content: str | None = triage_field("Content", verbose=2)  # whole FIFO as hex, only with --include-cb-content


def read_local_cbs(
    location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch, dump_content: bool
) -> list[CircularBufferData]:
    num_cbs, received_reg, acked_reg = arch
    mask = int(kernel_config.local_cb_mask)
    base = config_base + int(kernel_config.local_cb_offset)
    cbs = []
    for cb in range(num_cbs):
        if not mask >> cb & 1:
            continue
        address, size, pages, page_size = read_words_from_device(location, base + 16 * cb, word_count=4)
        stream = STREAM_REG_BASE + STREAM_REG_STRIDE * cb
        pushed = read_word_from_device(location, stream + 4 * received_reg)
        popped = read_word_from_device(location, stream + 4 * acked_reg)
        occupancy = (pushed - popped) & 0xFFFF
        content = read_from_device(location, address, num_bytes=size).hex() if dump_content else None
        cbs.append(CircularBufferData(cb, "local", address, size, page_size, pages, pushed, popped, occupancy, content))
    return cbs


def read_remote_cbs(
    location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch, dump_content: bool
) -> list[CircularBufferData]:
    num_cbs = arch[0]
    base = config_base + int(kernel_config.remote_cb_offset)
    cbs = []
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
        sent_addrs = [counters + 2 * L1_ALIGNMENT * i for i in range(slots)]  # pages_acked is one L1_ALIGNMENT later
        pairs = [
            (read_word_from_device(location, a), read_word_from_device(location, a + L1_ALIGNMENT)) for a in sent_addrs
        ]
        sent, acked = max(pairs, key=lambda pair: (pair[0] - pair[1]) & 0xFFFFFFFF)
        units = page_size // L1_ALIGNMENT  # counter units per page
        pushed, popped = sent // units, acked // units
        occupancy = ((sent - acked) & 0xFFFFFFFF) // units
        kind = "global sender" if is_sender else "global receiver"
        pages = size // page_size
        content = read_from_device(location, address, num_bytes=size).hex() if dump_content and not is_sender else None
        cbs.append(CircularBufferData(cb, kind, address, size, page_size, pages, pushed, popped, occupancy, content))
    return cbs


def read_core(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, dump_content: bool
) -> list[CircularBufferData]:
    arch = WORMHOLE if location.device.is_wormhole() else BLACKHOLE if location.device.is_blackhole() else None
    if arch is None:
        return []
    core = dispatcher_data.get_cached_core_data(RiscLocation(location, None, "brisc"))
    if core.go_message == "DONE" or core.mailboxes is None:
        return []
    kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
    if int(kernel_config.enables) == 0:
        return []
    cbs = read_local_cbs(location, kernel_config, core.kernel_config_base, arch, dump_content)
    return cbs + read_remote_cbs(location, kernel_config, core.kernel_config_base, arch, dump_content)


class CircularBufferProvider:
    def __init__(self, dispatcher_data: DispatcherData, dump_content: bool):
        self.dispatcher_data = dispatcher_data
        self.dump_content = dump_content
        self._cache: dict[OnChipCoordinate, list[CircularBufferData]] = {}
        self._lock = threading.Lock()

    def get_circular_buffers(self, location: OnChipCoordinate) -> list[CircularBufferData]:
        with self._lock:
            if location not in self._cache:
                self._cache[location] = read_core(location, self.dispatcher_data, self.dump_content)
            return self._cache[location]


@triage_singleton
def run(args, context: Context) -> CircularBufferProvider:
    return CircularBufferProvider(get_dispatcher_data(args, context), bool(args["--include-cb-content"]))


if __name__ == "__main__":
    run_script()
