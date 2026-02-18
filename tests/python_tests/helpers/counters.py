# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pandas as pd
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from .test_config import TestConfig

COUNTER_SLOT_COUNT = 86  # Max counters per thread
COUNTER_DATA_WORD_COUNT = COUNTER_SLOT_COUNT * 2  # 2 words per counter (cycles + count)

_THREAD_ADDRESSES = {
    "UNPACK": (
        TestConfig.PERF_COUNTER_UNPACK_CONFIG_ADDR,
        TestConfig.PERF_COUNTER_UNPACK_DATA_ADDR,
    ),
    "MATH": (
        TestConfig.PERF_COUNTER_MATH_CONFIG_ADDR,
        TestConfig.PERF_COUNTER_MATH_DATA_ADDR,
    ),
    "PACK": (
        TestConfig.PERF_COUNTER_PACK_CONFIG_ADDR,
        TestConfig.PERF_COUNTER_PACK_DATA_ADDR,
    ),
}

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
        # Thread instruction counts (bit 8 set = ID 256+n)
        256: "THREAD_INSTRUCTIONS_0",
        257: "THREAD_INSTRUCTIONS_1",
        258: "THREAD_INSTRUCTIONS_2",
    },
    "FPU": {
        0: "FPU_INSTRUCTION",
        1: "SFPU_INSTRUCTION",
        257: "FPU_OR_SFPU_INSTRN",  # Combined FPU/SFPU
    },
    "TDMA_UNPACK": {
        1: "DATA_HAZARD_STALLS_MOVD2A",
        3: "MATH_INSTRN_STARTED",
        4: "MATH_INSTRN_AVAILABLE",
        5: "SRCB_WRITE_AVAILABLE",
        6: "SRCA_WRITE_AVAILABLE",
        7: "UNPACK0_BUSY_THREAD0",
        8: "UNPACK1_BUSY_THREAD0",
        9: "UNPACK0_BUSY_THREAD1",
        10: "UNPACK1_BUSY_THREAD1",
        259: "SRCB_WRITE",  # Bit 8 set
        261: "SRCA_WRITE",  # Bit 8 set
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
        11: "PACKER_DEST_READ_AVAILABLE",
        18: "PACKER_BUSY",
        272: "AVAILABLE_MATH",  # Bit 8 set
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


# Pre-built list of all 66 counters (computed once at module load)
ALL_COUNTERS = _build_all_counters()

# All threads that support performance counters
ALL_THREADS = ["UNPACK", "MATH", "PACK"]


def configure_counters(location: str = "0,0") -> None:
    """
    Configure all performance counters on all threads (UNPACK, MATH, PACK).

    Args:
        location: Tensix core coordinates (e.g., "0,0").
    """
    # Encode counter configurations
    config_words = []
    for counter in ALL_COUNTERS:
        bank_id = _BANK_NAME_TO_ID[counter["bank"]]
        l1_mux = counter.get("l1_mux", 0)
        counter_id = counter["counter_id"]
        # Config word format: [valid(31), l1_mux(17), counter_sel(8-16), bank_id(0-7)]
        config_word = (
            (1 << 31) | (l1_mux << 17) | (counter_id << 8) | bank_id  # Valid bit
        )
        config_words.append(config_word)

    # Pad and combine with zero data
    config_words.extend([0] * (COUNTER_SLOT_COUNT - len(config_words)))
    combined_data = config_words + [0] * COUNTER_DATA_WORD_COUNT

    # Write to all threads
    for thread in ALL_THREADS:
        config_addr, _ = _THREAD_ADDRESSES[thread]
        write_words_to_device(location=location, addr=config_addr, data=combined_data)


def read_counters(location: str = "0,0") -> pd.DataFrame:
    """
    Read performance counter results from all threads.

    Args:
        location: Tensix core coordinates (e.g., "0,0").

    Returns:
        DataFrame with columns: thread, bank, counter_name, counter_id, cycles, count, l1_mux
    """
    all_results = []

    for thread in ALL_THREADS:
        config_addr, data_addr = _THREAD_ADDRESSES[thread]

        # Read metadata
        metadata = read_words_from_device(
            location=location, addr=config_addr, word_count=COUNTER_SLOT_COUNT
        )

        if not metadata:
            continue

        # Count valid configs (check bit 31)
        valid_count = sum(1 for m in metadata if (m & 0x80000000) != 0)

        if valid_count == 0:
            continue

        # Read ONLY data for valid counters (no UNKNOWN counters)
        data = read_words_from_device(
            location=location, addr=data_addr, word_count=valid_count * 2
        )

        if not data or len(data) < valid_count * 2:
            continue

        data_idx = 0
        for i in range(COUNTER_SLOT_COUNT):
            config_word = metadata[i]
            if (config_word & 0x80000000) == 0:  # Check valid bit
                continue  # Unused slot

            # Decode metadata: [valid(31), l1_mux(17), counter_id(8-16), bank(0-7)]
            bank_id = config_word & 0xFF
            counter_id = (config_word >> 8) & 0x1FF  # 9 bits for counter_id
            l1_mux = (config_word >> 17) & 0x1

            bank_name = COUNTER_BANK_NAMES.get(bank_id, f"UNKNOWN_{bank_id}")

            # Get counter name
            if bank_name == "L1":
                counter_name = COUNTER_NAMES["L1"].get(
                    (counter_id, l1_mux), f"L1_UNKNOWN_{counter_id}_{l1_mux}"
                )
            else:
                counter_name = COUNTER_NAMES.get(bank_name, {}).get(
                    counter_id, f"{bank_name}_UNKNOWN_{counter_id}"
                )

            # Extract results using data_idx
            cycles = data[data_idx * 2]
            count = data[data_idx * 2 + 1]
            data_idx += 1

            all_results.append(
                {
                    "thread": thread,
                    "bank": bank_name,
                    "counter_name": counter_name,
                    "counter_id": counter_id,
                    "cycles": cycles,
                    "count": count,
                    "l1_mux": l1_mux if bank_name == "L1" else None,
                }
            )

    return pd.DataFrame(all_results)


def print_counters(results: pd.DataFrame) -> None:
    """
    Print all counter results to console in a readable format.

    Args:
        results: DataFrame with counter results (from read_counters).
    """
    if results.empty:
        print("No counter results to display.")
        return

    print("\n" + "=" * 100)
    print("PERFORMANCE COUNTER RESULTS")
    print("=" * 100)
    print("NOTE: Each bank has one output register. Counters are read sequentially")
    print("      after stopping. L1 mux0/mux1 cannot be measured simultaneously.")
    print("=" * 100)

    for thread in ALL_THREADS:
        thread_df = results[results["thread"] == thread]
        if thread_df.empty:
            continue

        print(f"\n{'─' * 100}")
        print(f"  THREAD: {thread}")
        print(f"{'─' * 100}")

        for bank in ["INSTRN_THREAD", "FPU", "TDMA_UNPACK", "L1", "TDMA_PACK"]:
            bank_df = thread_df[thread_df["bank"] == bank]
            if bank_df.empty:
                continue

            cycles = bank_df["cycles"].iloc[0] if len(bank_df) > 0 else 0

            print(f"\n  ┌─ {bank} (cycles: {cycles:,})")
            print(f"  │ {'Counter Name':<40} {'Count':>15} {'Rate':>12}")
            print(f"  │ {'─' * 40} {'─' * 15} {'─' * 12}")

            for _, row in bank_df.iterrows():
                name = row["counter_name"]
                # Add mux info for L1 counters
                if pd.notna(row["l1_mux"]):
                    name = f"{name} (mux{int(row['l1_mux'])})"
                count = row["count"]
                rate = (count / cycles) if cycles else 0.0
                print(f"  │ {name:<40} {count:>15,} {rate:>12.4f}")

            print(f"  └{'─' * 70}")

    print("\n" + "=" * 100 + "\n")


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
