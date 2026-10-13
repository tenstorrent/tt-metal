# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from tt_llk_perf.headers import bank_tables
from tt_llk_perf.metrics import quasar_l1_client_label

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .device_io import read_words_from_device, write_words_to_device
from .test_config import TestConfig

COUNTER_SLOT_COUNT = TestConfig._PERF_COUNTERS_CONFIG_WORDS

_IS_QUASAR = get_chip_architecture() == ChipArchitecture.QUASAR

# Order matches llk::perf::Bank (perf_counters/types.h) and the config-word bank field. Quasar has
# no L1 counter bank, so slot 3 carries the l1_client event CSR there (counter_sel = subport*8 + event).
COUNTER_BANK_NAMES = {
    0: "INSTRN_THREAD",
    1: "FPU",
    2: "TDMA_UNPACK",
    3: "L1_CLIENT" if _IS_QUASAR else "L1",
    4: "TDMA_PACK",
}

_TABLE_BANK_TO_HARNESS = {
    "INSTRN": "INSTRN_THREAD",
    "FPU": "FPU",
    "TDMA_UNPACK": "TDMA_UNPACK",
    "TDMA_PACK": "TDMA_PACK",
    "L1": "L1",
}

# parents[2] is tests/: the device-side header whose layout this module mirrors.
LLK_COUNTERS_HEADER = (
    Path(__file__).resolve().parents[2] / "helpers" / "include" / "counters.h"
)

_C_LITERAL = r"0[xX][0-9a-fA-F]+|\d+"


def _parse_perf_cfg(text: str) -> dict:
    """PERF_CFG_* constants from counters.h: a literal or literal << literal, u/U suffixes allowed."""
    cfg = {}
    for name, expr in re.findall(r"PERF_CFG_([A-Z0-9_]+)\s*=\s*([^;]+);", text):
        expr = re.sub(rf"({_C_LITERAL})[uUlL]+", r"\1", expr).strip()
        m = re.fullmatch(rf"({_C_LITERAL})(?:\s*<<\s*({_C_LITERAL}))?", expr)
        if m is None:
            raise RuntimeError(
                f"PERF_CFG_{name} in counters.h is not a plain literal or shift: {expr!r}"
            )
        cfg[name] = int(m.group(1), 0) << int(m.group(2) or "0", 0)
    return cfg


def _parse_uint(text: str, name: str) -> int:
    """One `constexpr std::uint32_t <name> = <literal>;` from counters.h."""
    m = re.search(rf"{name}\s*=\s*({_C_LITERAL})[uUlL]*\s*;", text)
    if m is None:
        raise RuntimeError(f"{name} not found in counters.h")
    return int(m.group(1), 0)


# Config word layout, parsed from the device-side header so the two cannot drift apart.
_HEADER_TEXT = LLK_COUNTERS_HEADER.read_text()
_PERF_CFG = _parse_perf_cfg(_HEADER_TEXT)
PERF_CFG_VALID_BIT = _PERF_CFG["VALID_BIT"]
PERF_CFG_L1_MUX_SHIFT = _PERF_CFG["L1_MUX_SHIFT"]
PERF_CFG_L1_MUX_MASK = _PERF_CFG["L1_MUX_MASK"]
PERF_CFG_COUNTER_SHIFT = _PERF_CFG["COUNTER_SHIFT"]
PERF_CFG_COUNTER_MASK = _PERF_CFG["COUNTER_MASK"]
PERF_CFG_BANK_MASK = _PERF_CFG["BANK_MASK"]


def _load_counter_names(arch) -> dict:
    """{bank: {select: name}} from the tt-llk tables; L1 is keyed by (select, mux), empty on Quasar."""
    banks = {bank: {} for bank in _TABLE_BANK_TO_HARNESS.values()}
    for table_bank, entries in bank_tables(arch).items():
        bank = _TABLE_BANK_TO_HARNESS[table_bank]
        for entry in entries:
            key = (entry.select, entry.l1_mux) if bank == "L1" else entry.select
            banks[bank][key] = entry.name
    return banks


COUNTER_NAMES = _load_counter_names(get_chip_architecture())


def _zone_config_addr(zone: int) -> int:
    """Config is shared across all zones (single buffer at base addr)."""
    _ = zone
    return TestConfig.PERF_COUNTERS_CONFIG_ADDR


def _zone_data_addr(zone: int) -> int:
    """Per-zone data block start: bank cycles (5 words) then counter counts."""
    return (
        TestConfig.PERF_COUNTERS_ZONES_BASE + zone * TestConfig.PERF_COUNTERS_ZONE_SIZE
    )


def _zone_sync_ctrl_addr(zone: int) -> int:
    """Sync word lives at end of per-zone data block."""
    return _zone_data_addr(zone) + TestConfig._PERF_COUNTERS_ZONE_DATA_BYTES


# Lightweight sync: SYNC_ZONE_COMPLETE marker (matches counters.h)
_SYNC_ZONE_COMPLETE = _parse_uint(_HEADER_TEXT, "SYNC_ZONE_COMPLETE")
# The device stores this when the counter select never took effect; the count is unknown, not zero.
_COUNTER_SELECT_MISSED = _parse_uint(_HEADER_TEXT, "COUNTER_SELECT_MISSED")


def _read_zone_counters(location: str, zone: int, zone_name: str) -> list[dict]:
    """
    Read performance counter results for a single zone from L1.

    Returns list of result dicts, or empty list if zone was not used.
    """
    sync_addr = _zone_sync_ctrl_addr(zone)
    sync_ctrl = read_words_from_device(location=location, addr=sync_addr, word_count=1)
    if not sync_ctrl:
        return []

    sync_word = sync_ctrl[0]
    logger.debug(
        f"Zone {zone} ({zone_name}): sync_word=0x{sync_word:08x} at addr=0x{sync_addr:06x}"
    )

    # Zone was never used (BRISC clears sync to 0 before each run)
    if sync_word == 0:
        logger.debug(f"Zone {zone}: sync_word is 0, skipping (zone not used)")
        return []

    # Lightweight stop writes only SYNC_ZONE_COMPLETE (0xFF); the high bytes are
    # unused under the current protocol. Any other low-byte value signals
    # corrupted or partially-written sync state.
    if (sync_word & 0xFF) != _SYNC_ZONE_COMPLETE:
        logger.warning(
            f"Zone {zone}: unexpected sync word 0x{sync_word:08x} "
            f"(expected SYNC_ZONE_COMPLETE=0xFF in low byte)"
        )
        return []

    # Shared config (same for all zones): read the metadata layout once.
    config_addr = _zone_config_addr(zone)
    metadata = read_words_from_device(
        location=location, addr=config_addr, word_count=COUNTER_SLOT_COUNT
    )
    if not metadata:
        return []

    valid_count = sum(1 for m in metadata if (m & PERF_CFG_VALID_BIT) != 0)
    if valid_count == 0:
        return []

    # Per-zone data: 5 bank-cycle words (one OUT_L per bank) + N counter counts.
    data_addr = _zone_data_addr(zone)
    bank_cycles_words = TestConfig._PERF_COUNTERS_BANK_CYCLES_WORDS
    data = read_words_from_device(
        location=location, addr=data_addr, word_count=bank_cycles_words + valid_count
    )
    if len(data) < bank_cycles_words + valid_count:
        return []

    # Bank cycles are the first 5 words: indexed by bank_id (0..4).
    # Order matches llk::perf::Bank: INSTRN, FPU, TDMA_UNPACK, L1, TDMA_PACK
    bank_cycles = data[:bank_cycles_words]
    counter_counts = data[bank_cycles_words:]

    results = []
    count_idx = 0
    for i in range(COUNTER_SLOT_COUNT):
        config_word = metadata[i]
        if (config_word & PERF_CFG_VALID_BIT) == 0:
            continue

        bank_id, bank_name, counter_id, counter_name, l1_mux = decode_config_word(
            config_word
        )

        cycles = bank_cycles[bank_id] if bank_id < bank_cycles_words else 0
        count = counter_counts[count_idx]
        count_idx += 1

        if count == _COUNTER_SELECT_MISSED:
            logger.warning(
                f"Zone {zone_name}: {bank_name}.{counter_name} (sel {counter_id}) missed its select "
                "readback; reporting it as missing"
            )
            count = float("nan")

        results.append(
            {
                "zone": zone_name,
                "bank": bank_name,
                "counter_name": counter_name,
                "counter_id": counter_id,
                "cycles": cycles,
                "count": count,
                "l1_mux": l1_mux,
            }
        )

    return results


def decode_config_word(config_word: int) -> tuple:
    """(bank_id, bank_name, counter_id, counter_name, l1_mux) for one valid config word."""
    bank_id = config_word & PERF_CFG_BANK_MASK
    counter_id = (config_word >> PERF_CFG_COUNTER_SHIFT) & PERF_CFG_COUNTER_MASK
    l1_mux = (config_word >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK

    bank_name = COUNTER_BANK_NAMES.get(bank_id, f"UNKNOWN_{bank_id}")

    if bank_name == "L1" and l1_mux != TestConfig.PERF_L1_MUX_GROUP:
        # The group is baked into brisc.elf and nothing keys a rebuild on it, so a stale ELF
        # would otherwise return a self-consistently mislabelled dataset.
        raise RuntimeError(
            f"L1 counters were captured with mux group {l1_mux}, but "
            f"LLK_PERF_L1_MUX_GROUP={TestConfig.PERF_L1_MUX_GROUP} was requested. The ELF predates "
            "the change; recompile the producer (the group is a compile-time constant)."
        )

    if bank_name == "L1":
        counter_name = COUNTER_NAMES["L1"].get(
            (counter_id, l1_mux), f"L1_UNKNOWN_{counter_id}_{l1_mux}"
        )
    elif bank_name == "L1_CLIENT":
        counter_name = quasar_l1_client_label(counter_id)
    else:
        counter_name = COUNTER_NAMES.get(bank_name, {}).get(
            counter_id, f"{bank_name}_UNKNOWN_{counter_id}"
        )

    return (
        bank_id,
        bank_name,
        counter_id,
        counter_name,
        l1_mux if bank_name == "L1" else None,
    )


def read_counters(location: str = "0,0") -> pd.DataFrame:
    """
    Read performance counter results from all zones.

    Iterates zones 0..MAX_ZONES-1; zones with sync_word=0 (never used) are
    skipped silently. MAX_ZONES is set by counters.h (currently 8).

    Args:
        location: Tensix core coordinates (e.g., "0,0").

    Returns:
        DataFrame with columns:
        zone, bank, counter_name, counter_id, cycles, count, l1_mux
    """
    all_results = []

    for zone_idx in range(TestConfig.PERF_COUNTERS_MAX_ZONES):
        zone_name = f"ZONE_{zone_idx}"
        zone_results = _read_zone_counters(location, zone_idx, zone_name)
        all_results.extend(zone_results)

    return pd.DataFrame(all_results)


# Quasar has no BRISC: the host does counters.h configure_and_arm's work for a counting launch, so no TRISC runs counter
# code before the windows. Bank (ref_period, mode) offsets in NEO0's register window: perf_counters/registers.h.
_QUASAR_NEO0_WINDOW = 0x01800000
_QUASAR_BANK_REGS = {0: (0x0, 0x4), 1: (0x18, 0x1C), 2: (0xC, 0x10), 4: (0x8C, 0x90)}
_REF_PERIOD_MAX = 0xFFFFFFFF


def builtin_counter_config() -> list:
    """counters.h BUILTIN_COUNTER_CONFIG: the INSTRN, FPU, TDMA_UNPACK and TDMA_PACK tables, then the l1_client selection."""
    tables = bank_tables(get_chip_architecture())
    words = [
        PERF_CFG_VALID_BIT
        | ((entry.select & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT)
        | bank
        for bank, key in (
            (0, "INSTRN"),
            (1, "FPU"),
            (2, "TDMA_UNPACK"),
            (4, "TDMA_PACK"),
        )
        for entry in tables[key]
    ]
    if TestConfig.PERF_L1_CLIENT_SEL >= 0:
        sel = TestConfig.PERF_L1_CLIENT_SEL & PERF_CFG_COUNTER_MASK
        words.append(PERF_CFG_VALID_BIT | (sel << PERF_CFG_COUNTER_SHIFT) | 3)
    return words


def configure_quasar_counters(location: str) -> None:
    """Before a counting Quasar launch: the readout's config, cleared zones and the bank modes. Nothing starts here
    and the l1_client stays off: the zone arms route, enable and start them, the same instructions as counters off.
    """
    config = builtin_counter_config()
    zones = TestConfig.PERF_COUNTERS_MAX_ZONES
    bank_mask = 0
    for word in config:
        bank_mask |= 1 << (word & PERF_CFG_BANK_MASK)
    write_words_to_device(
        location,
        TestConfig.PERF_COUNTERS_CONFIG_ADDR,
        config + [0] * (COUNTER_SLOT_COUNT - len(config)),
    )
    # every zone's data and sync words, then ENABLED_FLAG, BANK_MASK and the valid counts after them
    write_words_to_device(
        location,
        TestConfig.PERF_COUNTERS_ZONES_BASE,
        [0] * (TestConfig.PERF_COUNTERS_ZONE_SIZE // 4 * zones)
        + [1, bank_mask]
        + [len(config)] * zones,
    )
    for bank, (ref_period, mode) in _QUASAR_BANK_REGS.items():
        if bank_mask >> bank & 1:
            for offset, value in ((ref_period, _REF_PERIOD_MAX), (mode, 0)):
                write_words_to_device(
                    location, _QUASAR_NEO0_WINDOW + offset, [value], safe_mode=False
                )
