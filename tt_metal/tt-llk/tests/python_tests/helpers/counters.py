# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pandas as pd
from loguru import logger

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .device_io import read_words_from_device
from .test_config import TestConfig

COUNTER_SLOT_COUNT = TestConfig._PERF_COUNTERS_CONFIG_WORDS

COUNTER_BANK_NAMES = {
    0: "INSTRN_THREAD",
    1: "FPU",
    2: "TDMA_UNPACK",
    3: "L1",
    4: "TDMA_PACK",
}


# --- Counter id -> name tables, parsed live from the canonical metal hw_counters.h ---

_ARRAY_TO_BANK = {
    "instrn_counters": "INSTRN_THREAD",
    "fpu_counters": "FPU",
    "unpack_counters": "TDMA_UNPACK",
    "pack_counters": "TDMA_PACK",
}
_ARCH_DIR = {
    ChipArchitecture.WORMHOLE: "wormhole",
    ChipArchitecture.BLACKHOLE: "blackhole",
}


def _metal_root() -> Path:
    """Walk up from this file until the metal hw_counters.h tree is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "tt_metal/hw/inc/internal/tt-1xx").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate tt_metal/hw/inc/internal/tt-1xx above this file"
    )


def _parse_perf_cfg(text: str) -> dict:
    """PERF_CFG_* constants from counters.h: a literal or literal << literal, u/U suffixes allowed."""
    cfg = {}
    for name, expr in re.findall(r"PERF_CFG_([A-Z0-9_]+)\s*=\s*([^;]+);", text):
        expr = re.sub(r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]+", r"\1", expr).strip()
        m = re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)(?:\s*<<\s*(\d+))?", expr)
        if m is None:
            raise RuntimeError(
                f"PERF_CFG_{name} in counters.h is not a plain literal or shift: {expr!r}"
            )
        cfg[name] = int(m.group(1), 0) << int(m.group(2) or "0")
    return cfg


# Config word layout, parsed from the device-side header so the two cannot drift apart.
_PERF_CFG = _parse_perf_cfg(
    (_metal_root() / "tt_metal/tt-llk/tests/helpers/include/counters.h").read_text()
)
PERF_CFG_VALID_BIT = _PERF_CFG["VALID_BIT"]
PERF_CFG_L1_MUX_SHIFT = _PERF_CFG["L1_MUX_SHIFT"]
PERF_CFG_L1_MUX_MASK = _PERF_CFG["L1_MUX_MASK"]
PERF_CFG_COUNTER_SHIFT = _PERF_CFG["COUNTER_SHIFT"]
PERF_CFG_COUNTER_MASK = _PERF_CFG["COUNTER_MASK"]
PERF_CFG_BANK_MASK = _PERF_CFG["BANK_MASK"]


def _parse_hw_counters(text: str) -> dict:
    """Parse one hw_counters.h into {bank: {id: name}}; L1 is keyed by (id, mux)."""
    banks = {bank: {} for bank in COUNTER_BANK_NAMES.values()}

    decls = list(re.finditer(r"(\w+_counters)\s*=", text))
    for i, decl in enumerate(decls):
        name = decl.group(1)
        start = decl.end()
        end = decls[i + 1].start() if i + 1 < len(decls) else len(text)
        chunk = text[start:end]
        term = chunk.find("};")
        if term != -1:
            chunk = chunk[:term]

        pairs = [
            (cname, int(cid))
            for cname, cid in re.findall(r"PerfCounterType::(\w+)\s*,\s*(\d+)", chunk)
        ]
        if not pairs:
            continue

        if name.startswith("l1_"):
            mux = int(name.split("_")[1])
            for cname, cid in pairs:
                banks["L1"][(cid, mux)] = cname
        elif name in _ARRAY_TO_BANK:
            bank = _ARRAY_TO_BANK[name]
            for cname, cid in pairs:
                banks[bank][cid] = cname

    return banks


def _load_counter_names(arch: ChipArchitecture) -> dict:
    arch_dir = _ARCH_DIR.get(arch)
    if arch_dir is None:  # Quasar / unsupported: no hw_counters.h yet.
        return {bank: {} for bank in COUNTER_BANK_NAMES.values()}
    header = _metal_root() / f"tt_metal/hw/inc/internal/tt-1xx/{arch_dir}/hw_counters.h"
    banks = _parse_hw_counters(header.read_text())
    # Silently, every column becomes UNKNOWN and metrics.py turns the misses into a page of zeros.
    missing = [
        b
        for b in ("INSTRN_THREAD", "FPU", "TDMA_UNPACK", "TDMA_PACK", "L1")
        if not banks.get(b)
    ]
    if missing:
        raise RuntimeError(
            f"Could not parse counter names from {header}: empty banks {missing}. "
            "The pair syntax in hw_counters.h probably changed; update _parse_hw_counters."
        )
    return banks


# These L1 group names are unverified and several are known wrong: they label mux indices, not
# confirmed client functions.
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
_SYNC_ZONE_COMPLETE = 0xFF


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

    # Shared config (same for all zones) — read once metadata layout.
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
    # Order matches counter_bank enum (see counters.h): INSTRN, FPU, TDMA_UNPACK, L1, TDMA_PACK
    bank_cycles = data[:bank_cycles_words]
    counter_counts = data[bank_cycles_words:]

    results = []
    count_idx = 0
    for i in range(COUNTER_SLOT_COUNT):
        config_word = metadata[i]
        if (config_word & PERF_CFG_VALID_BIT) == 0:
            continue

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
        else:
            counter_name = COUNTER_NAMES.get(bank_name, {}).get(
                counter_id, f"{bank_name}_UNKNOWN_{counter_id}"
            )

        cycles = bank_cycles[bank_id] if bank_id < bank_cycles_words else 0
        count = counter_counts[count_idx]
        count_idx += 1

        results.append(
            {
                "zone": zone_name,
                "bank": bank_name,
                "counter_name": counter_name,
                "counter_id": counter_id,
                "cycles": cycles,
                "count": count,
                "l1_mux": l1_mux if bank_name == "L1" else None,
            }
        )

    return results


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


def print_counters(results: pd.DataFrame) -> None:
    """
    Log all counter results in a readable format.

    Args:
        results: DataFrame with counter results (from read_counters).
    """
    if results.empty:
        logger.info("No counter results to display.")
        return

    if "zone" in results.columns:
        zones = results["zone"].unique()
        for zone in zones:
            zone_results = results[results["zone"] == zone]
            logger.info("\n{}\nZONE: {}\n{}", "═" * 100, zone, "═" * 100)
            _print_zone_counters(zone_results)
    else:
        logger.info("\n{}\nPERFORMANCE COUNTER RESULTS\n{}", "=" * 100, "=" * 100)
        _print_zone_counters(results)


def _print_zone_counters(results: pd.DataFrame) -> None:
    """Helper to log counters for a single zone."""
    lines = []

    for bank in ["INSTRN_THREAD", "FPU", "TDMA_UNPACK", "L1", "TDMA_PACK"]:
        bank_df = results[results["bank"] == bank]
        if bank_df.empty:
            continue

        cycles = bank_df["cycles"].iloc[0] if len(bank_df) > 0 else 0

        lines.append(f"\n  ┌─ {bank} (cycles: {cycles:,})")
        lines.append(f"  │ {'Counter Name':<40} {'Count':>15} {'Rate':>12}")
        lines.append(f"  │ {'─' * 40} {'─' * 15} {'─' * 12}")

        for _, row in bank_df.iterrows():
            name = row["counter_name"]
            if pd.notna(row["l1_mux"]):
                name = f"{name} (mux{int(row['l1_mux'])})"
            count = row["count"]
            rate = (count / cycles) if cycles else 0.0
            lines.append(f"  │ {name:<40} {count:>15,} {rate:>12.4f}")

        lines.append(f"  └{'─' * 70}")

    if lines:
        logger.info("\n".join(lines))
