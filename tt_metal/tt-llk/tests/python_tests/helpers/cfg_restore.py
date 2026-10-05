# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tensix hardware config state recording and restoring."""

import json
import os
import sys

from ttexalens.tt_exalens_lib import check_context, convert_coordinate, get_tensix_state

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .device_io import write_words_to_device
from .logger import logger

TENSIX_CFG_BASE = 0xFFEF0000
_CFG_STATE_SIZE = {
    ChipArchitecture.BLACKHOLE: 56,
    ChipArchitecture.WORMHOLE: 47,
}

# Stuff not to touch.
_BOOT_OWNED = {
    ChipArchitecture.BLACKHOLE: set(),
    ChipArchitecture.WORMHOLE: {158, 159, 160, 161},
}


def _word_core_addr(addr32: int, state: int, arch: ChipArchitecture) -> int:
    """CFG address of `addr32` in the given shadow state."""
    return TENSIX_CFG_BASE + (addr32 + state * _CFG_STATE_SIZE[arch] * 4) * 4


def _get_risc_debug(location: str, arch: ChipArchitecture, device_id: int, context):
    context = context or check_context()
    coordinate = convert_coordinate(location, device_id, context)
    block = coordinate.device.get_block(coordinate)
    return block.get_risc_debug(
        "trisc0", neo_id=0 if arch == ChipArchitecture.QUASAR else None
    )


def snapshot_cfg(location: str, items, *, device_id: int = 0, context=None) -> dict:
    """Read the given (state, addr32) words and return a {(state, addr32): value} map."""
    arch = get_chip_architecture()
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    snap = {}
    with risc_debug.ensure_private_memory_access():
        for state, addr32 in items:
            snap[(state, addr32)] = risc_debug.read_memory(
                _word_core_addr(addr32, state, arch)
            )
    return snap


def thread_items(arch: ChipArchitecture) -> list:
    """Every kernel-owned (0, addr32) word. Restore only targets shadow state 0."""
    excluded = _BOOT_OWNED.get(arch, set())
    words = [a for a in range(0, _CFG_STATE_SIZE[arch] * 4) if a not in excluded]
    return [(0, a) for a in words]


# L1 layout trisc.cpp expects
_INKERNEL_RESTORE_BASE = 0x15000
_INKERNEL_RESTORE_MAGIC = 0x43464731  # 'CFG1'


def write_inkernel_restore(
    location: str, entries, *, device_id: int = 0, context=None
) -> int:
    """Write a restore plan to L1 0x15000, which trisc.cpp will then apply.
    Layout is [space, addr32, v0, v1, v2, mask] and depending on space:
      RESTORE_SPACE_CONFIG:         Shared config space, v0 is masked
      RESTORE_SPACE_THREADCONFIG:   ThreadConfig, each thread sets its own
      RESTORE_SPACE_ADC_CH1X:       address_counters channel1-X. v0 is the unpacker
                                    value and v1 is the packer value.
    addr32/mask are ignored by spaces that don't use them.
    """
    words = [_INKERNEL_RESTORE_MAGIC, len(entries)]
    for space, addr32, v0, v1, v2, mask in entries:
        words += [space, addr32, v0, v1, v2, mask]
    write_words_to_device(
        location, _INKERNEL_RESTORE_BASE, words, device_id=device_id, context=context
    )
    return len(entries)


_THREAD_CONFIG_BASE_WORDS = {
    ChipArchitecture.BLACKHOLE: 3 * _CFG_STATE_SIZE[ChipArchitecture.BLACKHOLE] * 4,
    ChipArchitecture.WORMHOLE: 3 * _CFG_STATE_SIZE[ChipArchitecture.WORMHOLE] * 4,
}
_THD_STATE_SIZE = {
    ChipArchitecture.BLACKHOLE: 68,
    ChipArchitecture.WORMHOLE: 57,
}

_THREAD_CFG_IDS = (0, 1, 2)  # THREAD_0_CFG/1/2 == UNPACK/MATH/PACK

# Addr mod fields from cfg_defines.h. TODO: Wormhole
_ADDR_MOD_ADDR32_BH = sorted(
    set(range(12, 20))
    | set(range(20, 28))
    | set(range(28, 36))
    | set(range(37, 41))
    | set(range(47, 55))
)


def _thread_config_addr(thread: int, local_idx: int, arch: ChipArchitecture) -> int:
    """Address of ThreadConfig[thread][local_idx]."""
    return (
        TENSIX_CFG_BASE
        + (_THREAD_CONFIG_BASE_WORDS[arch] + thread * _THD_STATE_SIZE[arch] + local_idx)
        * 4
    )


def snapshot_addr_mod(location: str, *, device_id: int = 0, context=None) -> dict:
    """Read addr mods. TODO: Wormhole"""
    arch = get_chip_architecture()
    risc_debug = _get_risc_debug(location, arch, device_id, context)
    out = {}
    with risc_debug.ensure_private_memory_access():
        for thread in _THREAD_CFG_IDS:
            for addr32 in _ADDR_MOD_ADDR32_BH:
                out[(thread, addr32)] = risc_debug.read_memory(
                    _thread_config_addr(thread, addr32, arch)
                )
    return out


def snapshot_adc_ch1x(location: str, *, device_id: int = 0, context=None) -> dict:
    """Read address_counters' channel1-X residue.
    {un,}packer_addr_counter_init() deliberately skip this field.
    """
    state = get_tensix_state(location, device_id=device_id, context=context)
    return {
        "unpacker": state.address_counters["adcs0_unpacker0_channel1_x_counter"],
        "packer": state.address_counters["adcs2_packers_channel1_x_counter"],
    }


def maybe_restore_cfg_from_env(location: str, *, device_id: int = 0, context=None):
    """Restore / snapshot CFG based on env vars. Does nothing unless one is set.

    LLK_CFG_RESTORE=<path>  JSON -> write_inkernel_restore.
    LLK_CFG_SNAPSHOT=<path> Dump every kernel-owned word to <path> as JSON
    """
    arch = get_chip_architecture()

    restore_path = os.environ.get("LLK_CFG_RESTORE")
    if restore_path:
        with open(restore_path) as f:
            rplan = json.load(f)
        rentries = [tuple(e) for e in rplan["entries"]]
        nr = write_inkernel_restore(
            location, rentries, device_id=device_id, context=context
        )
        msg = f"cfg_restore: entries={nr} written to L1 0x{_INKERNEL_RESTORE_BASE:X}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)

    snap_path = os.environ.get("LLK_CFG_SNAPSHOT")
    if snap_path:
        items = thread_items(arch)
        snap = snapshot_cfg(location, items, device_id=device_id, context=context)
        with open(snap_path, "w") as f:
            json.dump([[s, a, v] for (s, a), v in snap.items()], f)
        msg = f"cfg_restore: words={len(snap)} written to {snap_path}"
        print(msg, file=sys.stderr, flush=True)
        logger.warning(msg)
        return None

    return None
