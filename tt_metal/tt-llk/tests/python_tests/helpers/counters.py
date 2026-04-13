# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pandas as pd
from loguru import logger
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from .test_config import TestConfig

# ============================================================================
# Constants and Configuration
# ============================================================================

# Derive all constants from TestConfig (single source of truth)
COUNTER_SLOT_COUNT = TestConfig._PERF_COUNTERS_CONFIG_WORDS  # 137 config slots
COUNTER_DATA_WORD_COUNT = (
    TestConfig._PERF_COUNTERS_DATA_WORDS
)  # 274 data words (137 * 2)
PERF_COUNTERS_STARTER_MASK = 0x3  # 2 bits for thread ID 0-3
PERF_COUNTERS_STOPPER_MASK = 0x3

# Single shared buffer addresses (all threads use the same location)
# These are already computed in TestConfig - use them directly
PERF_COUNTERS_CONFIG_ADDR = TestConfig.PERF_COUNTERS_CONFIG_ADDR
PERF_COUNTERS_DATA_ADDR = TestConfig.PERF_COUNTERS_DATA_ADDR
PERF_COUNTERS_SYNC_CTRL_ADDR = TestConfig.PERF_COUNTERS_SYNC_CTRL_ADDR
PERF_COUNTERS_START_COUNTER_ADDR = TestConfig.PERF_COUNTERS_SYNC_CTRL_ADDR + 4

PERF_COUNTERS_THREAD_COUNT = len(TestConfig.KERNEL_COMPONENTS)
PERF_COUNTERS_STOP_COUNTER_ADDR = PERF_COUNTERS_START_COUNTER_ADDR + (
    PERF_COUNTERS_THREAD_COUNT * 4
)
PERF_COUNTERS_STOP_ELECT_ADDR = PERF_COUNTERS_STOP_COUNTER_ADDR + (
    PERF_COUNTERS_THREAD_COUNT * 4
)

# TRISC id -> name. BH uses ids 0–2; Quasar uses 0–3. Same mapping for missing-thread errors and starter/stopper.
PERF_COUNTER_TRISC_NAMES = {0: "UNPACK", 1: "MATH", 2: "PACK", 3: "SFPU"}

COUNTER_BANK_NAMES = {
    0: "INSTRN_THREAD",
    1: "FPU",
    2: "TDMA_UNPACK",
    3: "L1",
    4: "TDMA_PACK",
}

# Reverse lookup: bank name -> bank id (computed once at module load)
_BANK_NAME_TO_ID = {v: k for k, v in COUNTER_BANK_NAMES.items()}

COUNTER_NAMES = {
    "INSTRN_THREAD": {
        # Instruction availability counters (per-thread)
        0: "CFG_INSTRN_AVAILABLE_0",
        1: "CFG_INSTRN_AVAILABLE_1",
        2: "CFG_INSTRN_AVAILABLE_2",
        3: "SYNC_INSTRN_AVAILABLE_0",
        4: "SYNC_INSTRN_AVAILABLE_1",
        5: "SYNC_INSTRN_AVAILABLE_2",
        6: "THCON_INSTRN_AVAILABLE_0",
        7: "THCON_INSTRN_AVAILABLE_1",
        8: "THCON_INSTRN_AVAILABLE_2",
        9: "XSEARCH_INSTRN_AVAILABLE_0",
        10: "XSEARCH_INSTRN_AVAILABLE_1",
        11: "XSEARCH_INSTRN_AVAILABLE_2",
        12: "MOVE_INSTRN_AVAILABLE_0",
        13: "MOVE_INSTRN_AVAILABLE_1",
        14: "MOVE_INSTRN_AVAILABLE_2",
        15: "FPU_INSTRN_AVAILABLE_0",
        16: "FPU_INSTRN_AVAILABLE_1",
        17: "FPU_INSTRN_AVAILABLE_2",
        18: "UNPACK_INSTRN_AVAILABLE_0",
        19: "UNPACK_INSTRN_AVAILABLE_1",
        20: "UNPACK_INSTRN_AVAILABLE_2",
        21: "PACK_INSTRN_AVAILABLE_0",
        22: "PACK_INSTRN_AVAILABLE_1",
        23: "PACK_INSTRN_AVAILABLE_2",
        # Thread stalls
        24: "THREAD_STALLS_0",
        25: "THREAD_STALLS_1",
        26: "THREAD_STALLS_2",
        # Wait counters (shared across threads)
        27: "WAITING_FOR_SRCA_CLEAR",
        28: "WAITING_FOR_SRCB_CLEAR",
        29: "WAITING_FOR_SRCA_VALID",
        30: "WAITING_FOR_SRCB_VALID",
        # Per-thread wait counters
        31: "WAITING_FOR_THCON_IDLE_0",
        32: "WAITING_FOR_THCON_IDLE_1",
        33: "WAITING_FOR_THCON_IDLE_2",
        34: "WAITING_FOR_UNPACK_IDLE_0",
        35: "WAITING_FOR_UNPACK_IDLE_1",
        36: "WAITING_FOR_UNPACK_IDLE_2",
        37: "WAITING_FOR_PACK_IDLE_0",
        38: "WAITING_FOR_PACK_IDLE_1",
        39: "WAITING_FOR_PACK_IDLE_2",
        40: "WAITING_FOR_MATH_IDLE_0",
        41: "WAITING_FOR_MATH_IDLE_1",
        42: "WAITING_FOR_MATH_IDLE_2",
        43: "WAITING_FOR_NONZERO_SEM_0",
        44: "WAITING_FOR_NONZERO_SEM_1",
        45: "WAITING_FOR_NONZERO_SEM_2",
        46: "WAITING_FOR_NONFULL_SEM_0",
        47: "WAITING_FOR_NONFULL_SEM_1",
        48: "WAITING_FOR_NONFULL_SEM_2",
        49: "WAITING_FOR_MOVE_IDLE_0",
        50: "WAITING_FOR_MOVE_IDLE_1",
        51: "WAITING_FOR_MOVE_IDLE_2",
        52: "WAITING_FOR_MMIO_IDLE_0",
        53: "WAITING_FOR_MMIO_IDLE_1",
        54: "WAITING_FOR_MMIO_IDLE_2",
        55: "WAITING_FOR_SFPU_IDLE_0",
        56: "WAITING_FOR_SFPU_IDLE_1",
        57: "WAITING_FOR_SFPU_IDLE_2",
        # Thread instruction counts (grant counters, bit 8 set = ID 256+n)
        # CFG instructions issued per thread
        256: "CFG_INSTRUCTIONS_0",
        257: "CFG_INSTRUCTIONS_1",
        258: "CFG_INSTRUCTIONS_2",
        # SYNC instructions issued per thread
        259: "SYNC_INSTRUCTIONS_0",
        260: "SYNC_INSTRUCTIONS_1",
        261: "SYNC_INSTRUCTIONS_2",
        # THCON instructions issued per thread
        262: "THCON_INSTRUCTIONS_0",
        263: "THCON_INSTRUCTIONS_1",
        264: "THCON_INSTRUCTIONS_2",
        # XSEARCH instructions issued per thread
        265: "XSEARCH_INSTRUCTIONS_0",
        266: "XSEARCH_INSTRUCTIONS_1",
        267: "XSEARCH_INSTRUCTIONS_2",
        # MOVE instructions issued per thread
        268: "MOVE_INSTRUCTIONS_0",
        269: "MOVE_INSTRUCTIONS_1",
        270: "MOVE_INSTRUCTIONS_2",
        # MATH instructions issued per thread
        271: "MATH_INSTRUCTIONS_0",
        272: "MATH_INSTRUCTIONS_1",
        273: "MATH_INSTRUCTIONS_2",
        # UNPACK instructions issued per thread
        274: "UNPACK_INSTRUCTIONS_0",
        275: "UNPACK_INSTRUCTIONS_1",
        276: "UNPACK_INSTRUCTIONS_2",
        # PACK instructions issued per thread
        277: "PACK_INSTRUCTIONS_0",
        278: "PACK_INSTRUCTIONS_1",
        279: "PACK_INSTRUCTIONS_2",
    },
    "FPU": {
        0: "FPU_INSTRUCTION",
        1: "SFPU_INSTRUCTION",
        257: "FPU_OR_SFPU_INSTRN",  # Combined FPU/SFPU
    },
    "TDMA_UNPACK": {
        0: "MATH_NOT_BLOCKED_BY_SRC",
        1: "DATA_HAZARD_STALLS_MOVD2A",
        2: "FIDELITY_PHASE_STALLS",
        3: "MATH_INSTRN_STARTED",
        4: "MATH_INSTRN_AVAILABLE",
        5: "SRCB_WRITE_AVAILABLE",
        6: "SRCA_WRITE_AVAILABLE",
        7: "UNPACK0_BUSY_THREAD0",
        8: "UNPACK1_BUSY_THREAD0",
        9: "UNPACK0_BUSY_THREAD1",
        10: "UNPACK1_BUSY_THREAD1",
        256: "MATH_NOT_BLOCKED_BY_SRC_GRANT",  # Grant version of counter 0
        257: "INSTRN_2HF_CYCLES",
        258: "INSTRN_1HF_CYCLE",
        259: "SRCB_WRITE",
        260: "SRCA_WRITE_NOT_BLOCKED_OVERWRITE",
        261: "SRCA_WRITE",
        262: "SRCB_WRITE_NOT_BLOCKED_PORT",
        263: "SRCA_WRITE_THREAD0",
        264: "SRCB_WRITE_THREAD0",
        265: "SRCA_WRITE_THREAD1",
        266: "SRCB_WRITE_THREAD1",
    },
    "L1": {
        # Format: (counter_id, l1_mux) -> name
        (0, 0): "NOC_RING0_INCOMING_1",
        (1, 0): "NOC_RING0_INCOMING_0",
        (2, 0): "NOC_RING0_OUTGOING_1",
        (3, 0): "NOC_RING0_OUTGOING_0",
        (4, 0): "L1_ARB_TDMA_BUNDLE_1",
        (5, 0): "L1_ARB_TDMA_BUNDLE_0",
        (6, 0): "L1_ARB_UNPACKER",
        (7, 0): "L1_NO_ARB_UNPACKER",
        (0, 1): "NOC_RING1_INCOMING_1",
        (1, 1): "NOC_RING1_INCOMING_0",
        (2, 1): "NOC_RING1_OUTGOING_1",
        (3, 1): "NOC_RING1_OUTGOING_0",
        (4, 1): "TDMA_BUNDLE_1_ARB",
        (5, 1): "TDMA_BUNDLE_0_ARB",
        (6, 1): "TDMA_EXT_UNPACK_9_10",
        (7, 1): "TDMA_PACKER_2_WR",
    },
    "TDMA_PACK": {
        11: "PACKER_DEST_READ_AVAILABLE_0",
        12: "PACKER_DEST_READ_AVAILABLE_1",
        13: "PACKER_DEST_READ_AVAILABLE_2",
        14: "PACKER_DEST_READ_AVAILABLE_3",
        15: "PACKER_BUSY_0",
        16: "PACKER_BUSY_1",
        17: "PACKER_BUSY_2",
        18: "PACKER_BUSY",  # Any packer engine busy
        267: "DEST_READ_GRANTED_0",
        268: "DEST_READ_GRANTED_1",
        269: "DEST_READ_GRANTED_2",
        270: "DEST_READ_GRANTED_3",
        271: "MATH_NOT_STALLED_BY_DEST_PORT",
        272: "AVAILABLE_MATH",
    },
}

# Reverse lookups for O(1) counter name -> id resolution (computed once at module load)
_L1_NAME_TO_ID = {(name, mux): cid for (cid, mux), name in COUNTER_NAMES["L1"].items()}

_COUNTER_NAME_TO_ID = {
    bank: {name: cid for cid, name in counters.items()}
    for bank, counters in COUNTER_NAMES.items()
    if bank != "L1"
}


def _build_all_counters() -> List[Dict]:
    """Build the complete list of performance counters across all banks."""
    counters = []

    # INSTRN_THREAD bank (61 counters)
    for counter_id in COUNTER_NAMES["INSTRN_THREAD"].keys():
        counters.append({"bank": "INSTRN_THREAD", "counter_id": counter_id})

    # FPU bank (3 counters)
    for counter_id in COUNTER_NAMES["FPU"].keys():
        counters.append({"bank": "FPU", "counter_id": counter_id})

    # TDMA_UNPACK bank (11 counters)
    for counter_id in COUNTER_NAMES["TDMA_UNPACK"].keys():
        counters.append({"bank": "TDMA_UNPACK", "counter_id": counter_id})

    # TDMA_PACK bank (3 counters)
    for counter_id in COUNTER_NAMES["TDMA_PACK"].keys():
        counters.append({"bank": "TDMA_PACK", "counter_id": counter_id})

    # L1 bank with l1_mux=0 (8 counters)
    for (counter_id, l1_mux), name in COUNTER_NAMES["L1"].items():
        if l1_mux == 0:
            counters.append({"bank": "L1", "counter_id": counter_id, "l1_mux": 0})

    # L1 bank with l1_mux=1 (8 counters)
    for (counter_id, l1_mux), name in COUNTER_NAMES["L1"].items():
        if l1_mux == 1:
            counters.append({"bank": "L1", "counter_id": counter_id, "l1_mux": 1})

    return counters


# Pre-built list of all counters (computed once at module load)
# Total: 137 counters (82 INSTRN_THREAD + 3 FPU + 22 TDMA_UNPACK + 14 TDMA_PACK + 16 L1)
# All Wormhole hardware performance counters are included.
ALL_COUNTERS = _build_all_counters()


def configure_counters(location: str = "0,0") -> None:
    """
    Configure performance counters in the shared buffer for all threads (UNPACK, MATH, PACK, and in Quasar, isolated SFPU).

    Writes counter configuration to L1 memory that all threads access. Configures all 137
    Wormhole hardware counter definitions (82 INSTRN_THREAD + 3 FPU + 22 TDMA_UNPACK + 14 TDMA_PACK + 16 L1).

    The counters are started/stopped by all threads via MEASURE_PERF_COUNTERS("name"),
    but only the last thread to finish (last stopper) reads the hardware and writes results.

    Args:
        location: Tensix core coordinates (e.g., "0,0").
    """
    # Encode counter configurations
    config_words = []
    for counter in ALL_COUNTERS:
        valid_bit = 1 << 31
        l1_mux = counter.get("l1_mux", 0) & 0x1
        l1_mux_shifted = l1_mux << 17
        counter_id_shifted = (counter["counter_id"] & 0x1FF) << 8
        bank_id = _BANK_NAME_TO_ID[counter["bank"]] & 0xFF
        config_words.append(valid_bit | l1_mux_shifted | counter_id_shifted | bank_id)

    # Pad config words to full slot count
    config_words.extend([0] * (COUNTER_SLOT_COUNT - len(config_words)))

    # Write config to the single shared buffer (all threads share one location)
    write_words_to_device(
        location=location, addr=PERF_COUNTERS_CONFIG_ADDR, data=config_words
    )

    # Clear data buffer completely (remove any stale values from previous runs)
    write_words_to_device(
        location=location,
        addr=PERF_COUNTERS_DATA_ADDR,
        data=[0] * COUNTER_DATA_WORD_COUNT,
    )

    # Clear sync state and ATINCGET counters before kernel runs (layout matches counters.h).
    # 1 word sync ctrl + PERF_COUNTERS_THREAD_COUNT start + PERF_COUNTERS_THREAD_COUNT stop + 1 word stop_elect
    write_words_to_device(
        location=location,
        addr=PERF_COUNTERS_SYNC_CTRL_ADDR,
        data=[0] * (1 + 2 * PERF_COUNTERS_THREAD_COUNT + 1),
    )


def _zone_config_addr(zone: int) -> int:
    """Compute L1 config address for a given zone (matches counters.h)."""
    return (
        TestConfig.PERF_COUNTERS_BASE_ADDR + zone * TestConfig.PERF_COUNTERS_ZONE_SIZE
    )


def _zone_data_addr(zone: int) -> int:
    """Compute L1 data address for a given zone (matches counters.h)."""
    return _zone_config_addr(zone) + TestConfig._PERF_COUNTERS_CONFIG_WORDS * 4


def _zone_sync_ctrl_addr(zone: int) -> int:
    """Compute L1 sync control address for a given zone (matches counters.h)."""
    return _zone_config_addr(zone) + TestConfig._PERF_COUNTERS_BUFFER_SIZE


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

    # Zone was never used (BRISC clears sync to 0 before each run)
    if sync_word == 0:
        return []

    # Lightweight stop writes SYNC_ZONE_COMPLETE (0xFF) in low byte + stopper thread ID.
    if (sync_word & 0xFF) != _SYNC_ZONE_COMPLETE:
        logger.warning(
            f"Zone {zone}: unexpected sync word 0x{sync_word:08x} "
            f"(expected SYNC_ZONE_COMPLETE=0xFF in low byte)"
        )
        return []

    # Extract stopper thread ID
    thread_count = len(TestConfig.KERNEL_COMPONENTS)
    stopper_shift = 2 * thread_count + 2 + 2  # SYNC_STOPPER_SHIFT from counters.h
    stopper_id = (sync_word >> stopper_shift) & PERF_COUNTERS_STOPPER_MASK
    stopper_thread = PERF_COUNTER_TRISC_NAMES.get(stopper_id, f"UNKNOWN_{stopper_id}")

    # Read config from this zone's L1 buffer
    config_addr = _zone_config_addr(zone)
    metadata = read_words_from_device(
        location=location, addr=config_addr, word_count=COUNTER_SLOT_COUNT
    )
    if not metadata:
        return []

    valid_count = sum(1 for m in metadata if (m & 0x80000000) != 0)
    if valid_count == 0:
        return []

    # Read data from this zone's L1 buffer
    data_addr = _zone_data_addr(zone)
    data = read_words_from_device(
        location=location, addr=data_addr, word_count=valid_count * 2
    )
    if not data or len(data) < valid_count * 2:
        return []

    results = []
    data_idx = 0
    for i in range(COUNTER_SLOT_COUNT):
        config_word = metadata[i]
        if (config_word & 0x80000000) == 0:
            continue

        bank_id = config_word & 0xFF
        counter_id = (config_word >> 8) & 0x1FF
        l1_mux = (config_word >> 17) & 0x1

        bank_name = COUNTER_BANK_NAMES.get(bank_id, f"UNKNOWN_{bank_id}")

        if bank_name == "L1":
            counter_name = COUNTER_NAMES["L1"].get(
                (counter_id, l1_mux), f"L1_UNKNOWN_{counter_id}_{l1_mux}"
            )
        else:
            counter_name = COUNTER_NAMES.get(bank_name, {}).get(
                counter_id, f"{bank_name}_UNKNOWN_{counter_id}"
            )

        cycles = data[data_idx * 2]
        count = data[data_idx * 2 + 1]
        data_idx += 1

        results.append(
            {
                "zone": zone_name,
                "starter_thread": "N/A",
                "stopper_thread": stopper_thread,
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

    Iterates over zone 0 (INIT) and zone 1 (TILE_LOOP), reading from
    per-zone L1 buffers. Zones that were never used are skipped silently.

    Args:
        location: Tensix core coordinates (e.g., "0,0").

    Returns:
        DataFrame containing counter results with columns:
        zone, starter_thread, stopper_thread, bank, counter_name, counter_id, cycles, count, l1_mux
    """
    all_results = []

    for zone_idx in range(2):  # INIT (0) and TILE_LOOP (1)
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


def export_counters(
    results: pd.DataFrame,
    filename: str,
    test_params: dict = None,
    worker_id: str = "gw0",
) -> None:
    """
    Export counter DataFrame to CSV file in perf_data directory.

    Args:
        results: DataFrame with counter results (from read_counters).
        filename: Base filename (without extension), e.g., "test_matmul_counters".
        test_params: Optional dictionary of test parameters to add as columns.
        worker_id: Worker ID for parallel test runs (e.g., "gw0", "master").
    """
    if results.empty:
        return

    perf_dir = TestConfig.LLK_ROOT / "perf_data"
    perf_dir.mkdir(parents=True, exist_ok=True)

    df = results.copy()

    # Add test params as columns
    if test_params:
        for key, value in test_params.items():
            df[key] = value

    output_path = perf_dir / f"{filename}.{worker_id}.csv"

    if output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(output_path, index=False)
