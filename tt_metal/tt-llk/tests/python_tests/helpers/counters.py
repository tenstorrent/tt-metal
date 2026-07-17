# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .test_config import TestConfig

# Length of the shared config array (matches counters.h COUNTER_SLOT_COUNT).
COUNTER_SLOT_COUNT = TestConfig._PERF_COUNTERS_CONFIG_WORDS

COUNTER_BANK_NAMES = {
    0: "INSTRN_THREAD",
    1: "FPU",
    2: "TDMA_UNPACK",
    3: "L1",
    4: "TDMA_PACK",
}

# Config word layout — parsed from the canonical C++ header (counters.h) so the Python decode and the
# on-device layout can't drift. (Was hand-copied with a "must match counters.h" comment.)


def _counters_h_path() -> Path:
    """Locate the LLK test-harness counters.h (authoritative PERF_CFG_* bit layout)."""
    rel = Path("tt_metal/tt-llk/tests/helpers/include/counters.h")
    for parent in Path(__file__).resolve().parents:
        if (parent / rel).is_file():
            return parent / rel
    raise RuntimeError(f"Could not locate {rel} above {__file__}")


def _eval_int_expr(expr: str) -> int:
    """Safely evaluate a C++ integer constant expression (literals + << >> | & ~ + - *).

    No eval() — parse to an AST and walk only integer nodes/operators, so a malformed header can't
    execute code (ast.literal_eval can't help here because it rejects `1 << 31`).
    """
    ops = {
        ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b,
        ast.BitOr: lambda a, b: a | b,
        ast.BitAnd: lambda a, b: a & b,
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
    }

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return ~ev(node.operand)
        raise ValueError(f"Unsupported integer expression: {expr!r}")

    return ev(ast.parse(expr.strip(), mode="eval").body)


def _parse_perf_cfg(text: str) -> dict:
    """Parse PERF_CFG_* config-word constants (hex/dec literals, u/U/l/L suffixes, `<<`)."""
    cfg = {}
    for name, expr in re.findall(r"PERF_CFG_([A-Z0-9_]+)\s*=\s*([^;]+);", text):
        expr = re.sub(
            r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]+", r"\1", expr
        )  # strip C++ integer suffixes
        cfg[name] = _eval_int_expr(expr)
    return cfg


_PERF_CFG = _parse_perf_cfg(_counters_h_path().read_text())
PERF_CFG_VALID_BIT = _PERF_CFG["VALID_BIT"]
PERF_CFG_L1_MUX_SHIFT = _PERF_CFG["L1_MUX_SHIFT"]
PERF_CFG_L1_MUX_MASK = _PERF_CFG["L1_MUX_MASK"]
PERF_CFG_COUNTER_SHIFT = _PERF_CFG["COUNTER_SHIFT"]
PERF_CFG_COUNTER_MASK = _PERF_CFG["COUNTER_MASK"]
PERF_CFG_BANK_MASK = _PERF_CFG["BANK_MASK"]


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


def _parse_hw_counters(text: str) -> dict:
    """Parse one hw_counters.h into {bank: {id: name}}; L1 is keyed by (id, mux)."""
    banks = {bank: {} for bank in COUNTER_BANK_NAMES.values()}

    # Locate each `<name>_counters = ...` array, then read up to its terminating `};`.
    # Slicing by declaration position is robust to the nested `{{...},{...}}` braces.
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
    return _parse_hw_counters(header.read_text())


# Active-arch table: {bank: {id: name}} with L1 keyed by (id, mux).
COUNTER_NAMES = _load_counter_names(get_chip_architecture())


def l1_mux_groups() -> list[int]:
    """
    Distinct L1 mux groups present in the active-arch inventory.

    The L1 perf-cnt mux is a count-time input selector, so only one 8-client group can be measured
    per count window. Callers re-run the kernel once per group (see write_l1_mux_sel) so every L1
    client reads real data. Returns [0] when no L1 counters are configured.
    """
    muxes = sorted({mux for (_cid, mux) in COUNTER_NAMES.get("L1", {})})
    return muxes or [0]


def write_l1_mux_sel(location: str, mux: int) -> None:
    """Select which L1 mux group the kernel measures during the next run's count window."""
    write_words_to_device(
        location, TestConfig.PERF_COUNTERS_L1_MUX_SEL_ADDR, [mux & PERF_CFG_L1_MUX_MASK]
    )


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


# ── Counter CSV Export ────────────────────────────────────────────────


def export_counters(
    all_counters: pd.DataFrame,
    run_type_name: str,
    zone_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Export raw hardware counter values as a DataFrame for a separate counters CSV.

    Produces one row per zone with columns: marker, then
    ``{run_type_name}_mean({bank}.{counter_name})`` and
    ``{run_type_name}_std({bank}.{counter_name})`` for every counter observed.

    Args:
        all_counters: Concatenated raw counter DataFrame from read_counters()
                      (with ``zone`` and ``run_index`` columns).
        run_type_name: Run type prefix for column names (e.g., "L1_TO_L1").
        zone_names: Optional list mapping zone index to display name.

    Returns:
        DataFrame with one row per zone.
    """
    if all_counters.empty:
        return pd.DataFrame()

    zone_to_marker = {}
    if zone_names:
        for i, name in enumerate(zone_names):
            zone_to_marker[f"ZONE_{i}"] = name

    zones = sorted(all_counters["zone"].unique())
    has_runs = "run_index" in all_counters.columns
    rows = []

    for zone in zones:
        zone_df = all_counters[all_counters["zone"] == zone]
        marker_name = zone_to_marker.get(zone, zone)
        row = {"marker": marker_name}

        # Get unique counters in this zone (preserving discovery order)
        counter_keys = (
            zone_df[["bank", "counter_name"]].drop_duplicates().values.tolist()
        )

        for bank, counter_name in counter_keys:
            mask = (zone_df["bank"] == bank) & (zone_df["counter_name"] == counter_name)
            col_name = f"{bank}.{counter_name}"

            if has_runs:
                per_run = zone_df.loc[mask].groupby("run_index")["count"].mean()
                if len(per_run) >= 2:
                    row[f"{run_type_name}_mean({col_name})"] = float(per_run.mean())
                    row[f"{run_type_name}_std({col_name})"] = float(per_run.std())
                elif len(per_run) == 1:
                    row[f"{run_type_name}_{col_name}"] = float(per_run.iloc[0])
            else:
                values = zone_df.loc[mask, "count"]
                row[f"{run_type_name}_{col_name}"] = float(values.mean())

            # Also export cycles for this counter
            col_cycles = f"{col_name}.cycles"
            if has_runs:
                per_run_cyc = zone_df.loc[mask].groupby("run_index")["cycles"].mean()
                if len(per_run_cyc) >= 2:
                    row[f"{run_type_name}_mean({col_cycles})"] = float(
                        per_run_cyc.mean()
                    )
                    row[f"{run_type_name}_std({col_cycles})"] = float(per_run_cyc.std())
                elif len(per_run_cyc) == 1:
                    row[f"{run_type_name}_{col_cycles}"] = float(per_run_cyc.iloc[0])
            else:
                cyc_values = zone_df.loc[mask, "cycles"]
                row[f"{run_type_name}_{col_cycles}"] = float(cyc_values.mean())

        rows.append(row)

    return pd.DataFrame(rows)
