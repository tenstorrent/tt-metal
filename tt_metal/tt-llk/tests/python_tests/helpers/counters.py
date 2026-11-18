# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from tt_llk_perf import headers as _llk_headers
from tt_llk_perf import noc_counters as _noc_counters
from tt_llk_perf import riscv_pmc as _riscv_pmc
from tt_llk_perf import router_counters as _router_counters
from tt_llk_perf.headers import bank_tables
from tt_llk_perf.metrics import quasar_l1_client_label
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from .chip_architecture import ChipArchitecture, get_chip_architecture
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


class CounterReadoutMissing(RuntimeError):
    """A zone the launch counted has no complete readout from that launch."""


def _read_zone_counters(
    location: str, zone: int, zone_name: str, token: int | None = None
) -> list[dict]:
    """
    Read performance counter results for a single zone from L1.

    Returns list of result dicts, or empty list if zone was not used. With a token the launch counted the zone, so a
    missing or incomplete readout raises CounterReadoutMissing: the window would otherwise reach the report blank.
    """

    def missing(reason: str) -> list[dict]:
        if token is not None:
            raise CounterReadoutMissing(
                f"Zone {zone_name}: {reason}, but the launch counted it (token {token})"
            )
        return []

    sync_addr = _zone_sync_ctrl_addr(zone)
    sync_ctrl = read_words_from_device(location=location, addr=sync_addr, word_count=1)
    if not sync_ctrl:
        return missing("the sync word read failed")

    sync_word = sync_ctrl[0]
    logger.debug(
        f"Zone {zone} ({zone_name}): sync_word=0x{sync_word:08x} at addr=0x{sync_addr:06x}"
    )

    if token is not None and sync_word >> TestConfig._LAUNCH_TOKEN_SHIFT != token:
        return missing(f"sync word 0x{sync_word:08x} is not this launch's readout")

    # Zone was never used
    if sync_word == 0:
        logger.debug(f"Zone {zone}: sync_word is 0, skipping (zone not used)")
        return []

    # The readout writes SYNC_ZONE_COMPLETE (0xFF) and its launch's token above it.
    # Any other low-byte value signals corrupted or partially-written sync state.
    if (sync_word & 0xFF) != _SYNC_ZONE_COMPLETE:
        logger.warning(
            f"Zone {zone}: unexpected sync word 0x{sync_word:08x} "
            f"(expected SYNC_ZONE_COMPLETE=0xFF in low byte)"
        )
        return missing(f"sync word 0x{sync_word:08x} is incomplete")

    # Shared config (same for all zones): read the metadata layout once.
    config_addr = _zone_config_addr(zone)
    metadata = read_words_from_device(
        location=location, addr=config_addr, word_count=COUNTER_SLOT_COUNT
    )
    if not metadata:
        return missing("the config read failed")

    valid_count = sum(1 for m in metadata if (m & PERF_CFG_VALID_BIT) != 0)
    if valid_count == 0:
        return missing("the readout stored no counter config")

    # Per-zone data: 5 bank-cycle words (one OUT_L per bank) + N counter counts.
    data_addr = _zone_data_addr(zone)
    bank_cycles_words = TestConfig._PERF_COUNTERS_BANK_CYCLES_WORDS
    data = read_words_from_device(
        location=location, addr=data_addr, word_count=bank_cycles_words + valid_count
    )
    if len(data) < bank_cycles_words + valid_count:
        return missing("the data read was short")

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


def read_counters(
    location: str = "0,0", zones: int | None = None, token: int | None = None
) -> pd.DataFrame:
    """
    Read performance counter results from all zones.

    Iterates zones 0..MAX_ZONES-1; without a token, zones with sync_word=0 (never used) are
    skipped silently. MAX_ZONES is set by counters.h (currently 8).

    Args:
        location: Tensix core coordinates (e.g., "0,0").
        zones: the zones the launch counted (a bit per zone id), None for all.
        token: the launch's readout token (counters.h launch_word), None to skip the check. With it, a zone in
            `zones` without this launch's complete readout raises CounterReadoutMissing.

    Returns:
        DataFrame with columns:
        zone, bank, counter_name, counter_id, cycles, count, l1_mux
    """
    all_results = []

    for zone_idx in range(TestConfig.PERF_COUNTERS_MAX_ZONES):
        if zones is not None and not zones >> zone_idx & 1:
            continue
        zone_name = f"ZONE_{zone_idx}"
        zone_results = _read_zone_counters(location, zone_idx, zone_name, token)
        all_results.extend(zone_results)

    return pd.DataFrame(all_results)


# Quasar has no BRISC: the host programs the bank registers through NEO0's register window
# before a counting launch, so no TRISC runs counter code before the windows. Offsets: perf_counters/registers.h.
_QUASAR_NEO0_WINDOW = 0x01800000
_QUASAR_BANK_REGS = {0: (0x0, 0x4), 1: (0x18, 0x1C), 2: (0xC, 0x10), 4: (0x8C, 0x90)}
_REF_PERIOD_MAX = 0xFFFFFFFF


def configure_quasar_counters(location: str) -> None:
    """Before a counting Quasar launch: each bank's reference period and continuous mode. Nothing starts here and the
    l1_client stays off: the zone arms route, enable and start them.
    """
    for ref_period, mode in _QUASAR_BANK_REGS.values():
        for offset, value in ((ref_period, _REF_PERIOD_MAX), (mode, 0)):
            write_words_to_device(
                location, _QUASAR_NEO0_WINDOW + offset, [value], safe_mode=False
            )


# ---------------------------------------------------------------------------------------------------------------------
# Optional families next to the Tensix banks: the RISC-V monitor of each TRISC (per zone) and the NoC interface and
# router counters of the tile (per launch, read by BRISC). Selected with --perf-riscv-pmc, --perf-noc-counters and
# --perf-router-counters; the device side is counters.h.
# ---------------------------------------------------------------------------------------------------------------------

# Thread order of the per-zone RISC-V blocks, as llk_barrier::THREAD_ID.
RISCV_THREAD_NAMES = ("UNPACK", "MATH", "PACK", "SFPU")
RISCV_BANK_PREFIX = "RISCV_"
NOC_BANK_PREFIX = "NOC"
ROUTER_BANK_PREFIX = "ROUTER"
# A launch's tile counters span the whole launch, so they are reported under the zone that launch counted.
LAUNCH_ZONE_PREFIX = "LAUNCH_"

# counters.h RISCV_WORD_* and TILE_WORD_*
_RISCV_WORD = {
    name: index
    for index, name in enumerate(
        ("FLAG", "EVENT3", "EVENT4", "CYCLES", "INSTRET", "PMC3", "PMC4")
    )
}
_TILE_WORD_NOC_FLAG, _TILE_WORD_NOC_SELECT = 0, 1
_TILE_WORD_ROUTER_FLAG, _TILE_WORD_ROUTER_SELECT = 2, 3
_TILE_WORD_NOC_VALUES = 4
_TILE_WORD_ROUTER_VALUES = (
    _TILE_WORD_NOC_VALUES + TestConfig._PERF_COUNTERS_TILE_VALUE_WORDS
)


def _arch_name(arch=None) -> str:
    return _llk_headers.normalize_arch(
        arch if arch is not None else get_chip_architecture()
    )


def riscv_core_kind(arch=None) -> str:
    """Select encoding of a TRISC: 'baby' on Blackhole, 'trisc' on Quasar. Wormhole cores have no counters."""
    name = _arch_name(arch)
    if name.startswith("wormhole"):
        raise ValueError(
            "--perf-riscv-pmc: Wormhole RISC-V cores implement no performance counter CSRs, so there is nothing to capture"
        )
    return "trisc" if name == "quasar" else "baby"


def _split_option(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _riscv_or_select(arch: str, kind: str, names: list[str]) -> int:
    """The select that arms all the named events at once. An mhpmevent has one 4-bit field per pipeline domain, so two
    events of one domain cannot be armed together: ORing their selects would arm a third event.
    """
    by_name = dict(_llk_headers.riscv_event_tables(arch).get(kind, []))
    domains = {}
    for name in names:
        if name not in by_name:
            raise ValueError(f"{name} is not an event of {kind} cores on {arch}")
        domain = (by_name[name].bit_length() - 1) // 4
        if domain in domains:
            raise ValueError(
                f"--perf-riscv-pmc: {domains[domain]} and {name} are events of the same pipeline domain and cannot share "
                "a counter"
            )
        domains[domain] = name
    return _llk_headers.riscv_event_select(arch, kind, names)


def parse_riscv_events(text: str, arch=None) -> tuple[int, int]:
    """(mhpmevent3, mhpmevent4) of --perf-riscv-pmc=EV3[,EV4]. Each is a number (hex ok) or event names joined with '+'
    (the counter counts their OR); an omitted EV4 is 0, which counts nothing."""
    kind = riscv_core_kind(arch)
    parts = _split_option(text)
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"--perf-riscv-pmc takes one or two events, got {text!r}")
    selects = []
    for part in parts:
        try:
            selects.append(int(part, 0))
            continue
        except ValueError:
            pass
        names = [name.strip().upper() for name in part.split("+")]
        selects.append(_riscv_or_select(_arch_name(arch), kind, names))
    while len(selects) < 2:
        selects.append(0)
    if any(not 0 <= select <= 0xFFFFFFFF for select in selects):
        raise ValueError(f"--perf-riscv-pmc selects must fit 32 bits, got {text!r}")
    return selects[0], selects[1]


def parse_noc_groups(text: str, arch=None) -> int:
    """Group mask of --perf-noc-counters: a number or group names (MASTER, SLAVE, ATOMIC, ALL) joined with '+'."""
    name = _arch_name(arch)
    if name == "quasar":
        raise ValueError(
            "--perf-noc-counters: the Quasar harness has no BRISC and no TRISC is known to read the tile's NoC registers"
        )
    groups = _llk_headers.NOC_GROUPS
    try:
        mask = int(text, 0)
    except ValueError:
        mask = 0
        for part in text.split("+"):
            part = part.strip().upper()
            if part == "ALL":
                mask |= (1 << len(groups)) - 1
            elif part in groups:
                mask |= 1 << groups.index(part)
            else:
                raise ValueError(
                    f"--perf-noc-counters: {part!r} is not one of {', '.join(groups)}, ALL or a mask"
                )
    if mask <= 0 or mask >> len(groups):
        raise ValueError(
            f"--perf-noc-counters: mask {mask:#x} selects none of the {len(groups)} groups or sets bits beyond them"
        )
    return mask


def parse_router_option(text: str, arch=None) -> int:
    """Option mask of --perf-router-counters: a number (virtual channel bits 0 to 15, bank bits from 16, bit 24 keeps the
    channels apart) or ALL (every channel, every bank), optionally +PER_VC."""
    name = _arch_name(arch)
    if name == "quasar":
        raise ValueError(
            "--perf-router-counters: the Quasar harness has no BRISC and no TRISC is known to read the tile's NoC registers"
        )
    consts = _llk_headers.router_constants()
    table = _llk_headers.router_counter_table(name)
    if not table.banks:
        raise ValueError(f"--perf-router-counters: {name} has no router counters")
    option = 0
    try:
        option = int(text, 0)
    except ValueError:
        for part in text.split("+"):
            part = part.strip().upper()
            if part == "ALL":
                option |= consts["VC_MASK_ALL"] | consts["BANKS_ALL"]
            elif part == "PER_VC":
                option |= consts["PER_VC"]
            else:
                raise ValueError(
                    f"--perf-router-counters: {part!r} is not ALL, PER_VC or a mask"
                )
    valid = consts["VC_MASK_ALL"] | consts["BANKS_ALL"] | consts["PER_VC"]
    if option & ~valid:
        raise ValueError(
            f"--perf-router-counters: option {option:#x} has bits outside the router option mask {valid:#x}"
        )
    selected = _llk_headers.router_option(name, option)
    if not selected.banks or not selected.vcs:
        raise ValueError(
            f"--perf-router-counters: option {option:#x} selects no bank or no virtual channel on {name}"
        )
    values = (
        table.num_nocs
        * len(selected.banks)
        * (len(selected.vcs) if selected.per_vc else 1)
    )
    if values > consts["MAX_VALUES"]:
        raise ValueError(
            f"--perf-router-counters: option {option:#x} captures {values} values, at most {consts['MAX_VALUES']}"
        )
    return option


def family_row(zone, bank, name, select, cycles, count) -> dict:
    """One row in the layout of read_counters: the family values travel through the same report frame."""
    return {
        "zone": zone,
        "bank": bank,
        "counter_name": name,
        "counter_id": select,
        "cycles": cycles,
        "count": count,
        "l1_mux": None,
    }


def _slot_complete(words, token, what) -> bool:
    """True when a block's flag word carries this launch's token; raises when a counted launch left none."""
    flag = words[0]
    if (flag & 0xFF) == _SYNC_ZONE_COMPLETE and (
        flag >> TestConfig._LAUNCH_TOKEN_SHIFT
    ) == token:
        return True
    raise CounterReadoutMissing(
        f"{what}: flag 0x{flag:08x} is not this launch's readout (token {token})"
    )


def decode_riscv_block(words, arch, thread: str, zone_name: str, token) -> list[dict]:
    """Rows of one thread's RISC-V monitor block of one zone: cycles, instructions and the armed events."""
    _slot_complete(words, token, f"Zone {zone_name} {thread}: RISC-V monitor block")
    event3, event4 = words[_RISCV_WORD["EVENT3"]], words[_RISCV_WORD["EVENT4"]]
    launch = _riscv_pmc.Launch(
        words[_RISCV_WORD["PMC3"]],
        words[_RISCV_WORD["PMC4"]],
        words[_RISCV_WORD["CYCLES"]],
        words[_RISCV_WORD["INSTRET"]],
        event3,
        event4,
    )
    named = _riscv_pmc.decode_launch(_arch_name(arch), riscv_core_kind(arch), launch)
    bank = RISCV_BANK_PREFIX + thread
    rows = [
        family_row(zone_name, bank, "cycles", -1, launch.mcycle, launch.mcycle),
        family_row(zone_name, bank, "instructions", -1, launch.mcycle, launch.minstret),
    ]
    for label, select, count in (
        (named["event3"], event3, launch.pmc3),
        (named["event4"], event4, launch.pmc4),
    ):
        if label != "DISABLED":
            rows.append(
                family_row(zone_name, bank, label, select, launch.mcycle, count)
            )
    return rows


def read_riscv_counters(
    location: str, zones: int, token: int, arch=None, threads: int | None = None
) -> list[dict]:
    """Rows of the RISC-V monitor of every counted zone, each TRISC of the arch. A counted zone whose thread left no
    readout of this launch raises CounterReadoutMissing."""
    arch = arch if arch is not None else get_chip_architecture()
    threads = threads or (4 if _arch_name(arch) == "quasar" else 3)
    rows = []
    for zone in range(TestConfig.PERF_COUNTERS_MAX_ZONES):
        if not zones >> zone & 1:
            continue
        zone_name = f"ZONE_{zone}"
        for thread in range(threads):
            addr = (
                _zone_data_addr(zone)
                + TestConfig._PERF_COUNTERS_RISCV_ZONE_OFFSET
                + thread * TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS * 4
            )
            words = read_words_from_device(
                location=location,
                addr=addr,
                word_count=TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS,
            )
            if len(words) < TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS:
                raise CounterReadoutMissing(
                    f"Zone {zone_name} {RISCV_THREAD_NAMES[thread]}: the RISC-V monitor read was short"
                )
            rows += decode_riscv_block(
                words, arch, RISCV_THREAD_NAMES[thread], zone_name, token
            )
    return rows


def decode_tile_block(
    words, arch, zone_name: str, noc_groups, router_option, token
) -> list[dict]:
    """Rows of one launch's NoC interface and router captures from the launch block words, named by the counter tables.
    The echoed masks must equal the requested ones: a stale BRISC would otherwise label another selection's values.
    """
    name = _arch_name(arch)
    zone = LAUNCH_ZONE_PREFIX + zone_name
    rows = []
    if noc_groups is not None:
        _slot_complete(words[_TILE_WORD_NOC_FLAG:], token, f"{zone}: NoC counter block")
        echoed = words[_TILE_WORD_NOC_SELECT]
        if echoed != noc_groups:
            raise RuntimeError(
                f"NoC counters were captured with group mask {echoed:#x}, but {noc_groups:#x} was requested. The BRISC ELF "
                "predates the change; recompile the producer."
            )
        table = _llk_headers.noc_counter_table(name)
        selected = _llk_headers.noc_selected_counters(name, noc_groups)
        count = len(selected) * table.num_nocs
        launch = _noc_counters.Launch(
            noc_groups,
            table.num_nocs,
            list(words[_TILE_WORD_NOC_VALUES : _TILE_WORD_NOC_VALUES + count]),
        )
        decoded = _noc_counters.decode_noc_launch(name, launch)
        for noc in range(table.num_nocs):
            bank = f"{NOC_BANK_PREFIX}{noc}"
            for counter, value in decoded["counts"][noc].items():
                rows.append(family_row(zone, bank, counter, -1, float("nan"), value))
            for metric, value in decoded["metrics"][noc].items():
                if (
                    value is not None
                    and "bytes" in metric
                    and not metric.endswith("_ratio")
                ):
                    rows.append(family_row(zone, bank, metric, -1, float("nan"), value))
    if router_option is not None:
        _slot_complete(
            words[_TILE_WORD_ROUTER_FLAG:], token, f"{zone}: router counter block"
        )
        echoed = words[_TILE_WORD_ROUTER_SELECT]
        if echoed != router_option:
            raise RuntimeError(
                f"Router counters were captured with option {echoed:#x}, but {router_option:#x} was requested. The BRISC ELF "
                "predates the change; recompile the producer."
            )
        labels = _router_counters.value_labels(name, router_option)
        table = _llk_headers.router_counter_table(name)
        total = len(labels) * table.num_nocs
        values = list(
            words[_TILE_WORD_ROUTER_VALUES : _TILE_WORD_ROUTER_VALUES + total]
        )
        launch = _router_counters.Launch(
            router_option, table.num_nocs, values + [0] * (total % 2)
        )
        decoded = _router_counters.decode_router_launch(name, launch)
        for noc in range(table.num_nocs):
            bank = f"{ROUTER_BANK_PREFIX}{noc}"
            for label, value in decoded["counts"][noc].items():
                rows.append(family_row(zone, bank, label, -1, float("nan"), value))
    return rows


def read_tile_counters(
    location: str, zones: int, token: int, noc_groups, router_option, arch=None
) -> list[dict]:
    """Rows of the tile counters of one launch, reported under the one zone the launch counted."""
    if noc_groups is None and router_option is None:
        return []
    arch = arch if arch is not None else get_chip_architecture()
    counted = [
        zone for zone in range(TestConfig.PERF_COUNTERS_MAX_ZONES) if zones >> zone & 1
    ]
    if not counted:
        return []
    words = read_words_from_device(
        location=location,
        addr=TestConfig.PERF_COUNTERS_LAUNCH_BLOCK_ADDR,
        word_count=_TILE_WORD_ROUTER_VALUES
        + TestConfig._PERF_COUNTERS_TILE_VALUE_WORDS,
    )
    rows = []
    for zone in counted:
        rows += decode_tile_block(
            words, arch, f"ZONE_{zone}", noc_groups, router_option, token
        )
    return rows


def read_family_counters(
    location: str = "0,0", zones: int = 0, token: int | None = None
) -> pd.DataFrame:
    """Rows of the optional families the build compiled in (TestConfig.PERF_*), in the layout of read_counters. Empty
    when none is on."""
    rows = []
    if TestConfig.PERF_RISCV_EVENTS is not None:
        rows += read_riscv_counters(location, zones, token)
    rows += read_tile_counters(
        location,
        zones,
        token,
        TestConfig.PERF_NOC_GROUPS,
        TestConfig.PERF_ROUTER_OPTION,
    )
    return pd.DataFrame(rows)
