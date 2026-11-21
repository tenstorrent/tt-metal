# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parsers for the perf counter headers in tools/include/perf_counters, the single source of counter names.
types.h gives the ordinal -> name table the firmware tags records with; <arch>.h gives (bank, select) -> name.
"""

import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent

# Array name in <arch>.h -> bank key. l1_<mux>_counters arrays are handled separately.
_ARRAY_TO_BANK = {
    "instrn_counters": "INSTRN",
    "fpu_counters": "FPU",
    "unpack_counters": "TDMA_UNPACK",
    "pack_counters": "TDMA_PACK",
}
BANK_KEYS = ("INSTRN", "FPU", "TDMA_UNPACK", "TDMA_PACK", "L1")

_ARCH_HEADER = {
    "blackhole": "blackhole.h",
    "wormhole": "wormhole.h",
    "wormhole_b0": "wormhole.h",
    "quasar": "quasar.h",
}
# Quasar has no L1 counter bank (its l1_client CSR is not a select table).
_ARCHES_WITHOUT_L1 = ("quasar",)


class CounterEntry(NamedTuple):
    name: str
    select: int
    l1_mux: Optional[int]


def _candidate_include_dirs(explicit) -> List[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    llk_home = os.environ.get("LLK_HOME")
    if llk_home:
        candidates.append(Path(llk_home) / "tools" / "include" / "perf_counters")
    candidates.append(_PACKAGE_DIR.parents[1] / "include" / "perf_counters")
    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        candidates.append(
            Path(metal_home)
            / "tt_metal"
            / "tt-llk"
            / "tools"
            / "include"
            / "perf_counters"
        )
    # An installed ttnn wheel carries the headers as package data next to ttnn/__init__.py.
    ttnn_dir = _ttnn_package_dir()
    if ttnn_dir is not None:
        candidates.append(
            ttnn_dir / "tt_metal" / "tt-llk" / "tools" / "include" / "perf_counters"
        )
    return candidates


def _ttnn_package_dir() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("ttnn")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin != "namespace":
        return Path(spec.origin).resolve().parent
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return None


def find_include_dir(explicit=None) -> Path:
    """The perf_counters header directory; the first existing candidate wins."""
    candidates = _candidate_include_dirs(explicit)
    for candidate in candidates:
        if (candidate / "types.h").is_file():
            return candidate
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find the perf_counters headers (types.h). Tried:\n  " + tried
    )


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _enum_body(text: str, enum_name: str) -> str:
    match = re.search(
        rf"enum\s+(?:class\s+|struct\s+)?{enum_name}\b[^{{]*\{{(.*?)\}}\s*;", text, re.S
    )
    if match is None:
        raise ValueError(f"enum {enum_name} not found")
    return match.group(1)


def parse_enum(text: str, enum_name: str = "PerfCounterType") -> Dict[int, str]:
    """Ordinal -> name for a C++ enum body, honouring explicit `= N` assignments."""
    body = _strip_comments(_enum_body(text, enum_name))
    names: Dict[int, str] = {}
    value = -1
    for token in body.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*(?:=\s*(0[xX][0-9a-fA-F]+|\d+))?", token
        )
        if match is None:
            raise ValueError(f"unexpected enumerator {token!r} in enum {enum_name}")
        value = int(match.group(2), 0) if match.group(2) else value + 1
        if value in names:
            raise ValueError(f"duplicate ordinal {value} in enum {enum_name}")
        names[value] = match.group(1)
    return names


def num_neos(include_dir=None) -> int:
    """NUM_NEOS from registers.h: the NEO execution units a Quasar core carries."""
    header = find_include_dir(include_dir) / "registers.h"
    match = re.search(r"NUM_NEOS\s*=\s*(\d+)", _strip_comments(header.read_text()))
    if match is None:
        raise ValueError(f"NUM_NEOS not found in {header}")
    return int(match.group(1))


def counter_type_names(include_dir=None) -> Dict[int, str]:
    """Ordinal -> name table of PerfCounterType, parsed from types.h."""
    header = find_include_dir(include_dir) / "types.h"
    names = parse_enum(header.read_text())
    if not names:
        raise ValueError(f"enum PerfCounterType in {header} parsed empty")
    return names


# LLK_PERF_TABLE_SECTION (quasar.h) may sit between the array name and =
_TABLE_DECL = re.compile(
    r"std::array<\s*Entry\s*,\s*(\d+)\s*>\s+(\w+_counters)\b[^=;]*="
)
_TABLE_ENTRY = re.compile(r"\{\s*PerfCounterType::(\w+)\s*,\s*([^{}]*?)\s*\}")
_SELECT_LITERAL = re.compile(r"0[xX][0-9a-fA-F]+|0|[1-9]\d*")


def _table_entries(name: str, size: int, chunk: str) -> List[tuple]:
    """(name, select) pairs of one table; raises on a non-literal select or a count other than the array size."""
    pairs = []
    for counter, select in _TABLE_ENTRY.findall(chunk):
        if _SELECT_LITERAL.fullmatch(select) is None:
            raise ValueError(
                f"{name}: select {select!r} of {counter} is not an integer literal"
            )
        pairs.append((counter, int(select, 0)))
    if len(pairs) != size:
        raise ValueError(
            f"{name}: parsed {len(pairs)} entries but the array declares {size}"
        )
    return pairs


def parse_tables(text: str) -> Dict[str, List[CounterEntry]]:
    """Bank -> entries for one <arch>.h; empty arrays (Wormhole L1 banks 2-5) are skipped."""
    banks: Dict[str, List[CounterEntry]] = {bank: [] for bank in BANK_KEYS}
    text = _strip_comments(text)
    decls = list(_TABLE_DECL.finditer(text))
    for i, decl in enumerate(decls):
        size, name = int(decl.group(1)), decl.group(2)
        chunk = text[decl.end() : decls[i + 1].start() if i + 1 < len(decls) else None]
        chunk = chunk.split("};", 1)[0]
        pairs = _table_entries(name, size, chunk)
        if not pairs:
            continue
        l1 = re.fullmatch(r"l1_(\d+)_counters", name)
        if l1 is not None:
            mux = int(l1.group(1))
            banks["L1"].extend(CounterEntry(n, s, mux) for n, s in pairs)
        elif name in _ARRAY_TO_BANK:
            banks[_ARRAY_TO_BANK[name]].extend(
                CounterEntry(n, s, None) for n, s in pairs
            )
    return banks


def normalize_arch(arch) -> str:
    """Lower-case arch name; accepts an enum through its .value (the harness ChipArchitecture)."""
    return str(getattr(arch, "value", arch)).lower()


def bank_tables(arch, include_dir=None) -> Dict[str, List[CounterEntry]]:
    """Bank -> [CounterEntry] for one arch; every bank key is present, Quasar's L1 is empty."""
    arch = normalize_arch(arch)
    if arch not in _ARCH_HEADER:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of {sorted(_ARCH_HEADER)}"
        )
    header = find_include_dir(include_dir) / _ARCH_HEADER[arch]
    banks = parse_tables(header.read_text())
    optional = ("L1",) if arch in _ARCHES_WITHOUT_L1 else ()
    empty = [bank for bank in BANK_KEYS if not banks[bank] and bank not in optional]
    if empty:
        raise ValueError(
            f"{header}: banks {empty} parsed empty. The table syntax probably changed; "
            "the parser expects {PerfCounterType::NAME, <select>} entries in *_counters arrays."
        )
    return banks


# RISC-V performance monitor: riscv_pmc_types.h has the PmcEvent names, riscv_pmc_<arch>.h the {name, select} tables
# of each core kind. A profiler record carries the raw mhpmevent select, so decoding goes select -> names.
RISCV_CORE_KINDS = ("baby", "trisc", "dm")
_RISCV_ARCH_HEADER = {
    "blackhole": "riscv_pmc_blackhole.h",
    "wormhole": "riscv_pmc_wormhole.h",
    "wormhole_b0": "riscv_pmc_wormhole.h",
    "quasar": "riscv_pmc_quasar.h",
}
_RISCV_TABLE_DECL = re.compile(
    r"std::array<\s*Entry\s*,\s*(\d+)\s*>\s+(baby|trisc|dm)_events\b[^=;]*="
)
_RISCV_ENTRY = re.compile(r"\{\s*PmcEvent::(\w+)\s*,\s*(0[xX][0-9a-fA-F]+|\d+)\s*\}")


def riscv_event_names(include_dir=None) -> List[str]:
    """PmcEvent enumerator names in declaration order (UNDEF first), parsed from riscv_pmc_types.h."""
    header = find_include_dir(include_dir) / "riscv_pmc_types.h"
    names = parse_enum(header.read_text(), "PmcEvent")
    if not names:
        raise ValueError(f"enum PmcEvent in {header} parsed empty")
    return [names[i] for i in sorted(names)]


def parse_riscv_tables(text: str) -> Dict[str, List[tuple]]:
    """Core kind -> [(event name, select)] for one riscv_pmc_<arch>.h; raises on a count other than the array size."""
    text = _strip_comments(text)
    decls = list(_RISCV_TABLE_DECL.finditer(text))
    tables: Dict[str, List[tuple]] = {}
    for i, decl in enumerate(decls):
        size, kind = int(decl.group(1)), decl.group(2)
        chunk = text[decl.end() : decls[i + 1].start() if i + 1 < len(decls) else None]
        chunk = chunk.split("}};", 1)[0] if size else ""
        pairs = [(n, int(s, 0)) for n, s in _RISCV_ENTRY.findall(chunk)]
        if len(pairs) != size:
            raise ValueError(
                f"{kind}_events: parsed {len(pairs)} entries but the array declares {size}"
            )
        tables[kind] = pairs
    return tables


def riscv_event_tables(arch, include_dir=None) -> Dict[str, List[tuple]]:
    """Core kind -> [(event name, select)] for one arch. Wormhole returns {'baby': []}: it has no counters."""
    arch = normalize_arch(arch)
    if arch not in _RISCV_ARCH_HEADER:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of {sorted(_RISCV_ARCH_HEADER)}"
        )
    header = find_include_dir(include_dir) / _RISCV_ARCH_HEADER[arch]
    tables = parse_riscv_tables(header.read_text())
    if not tables:
        raise ValueError(f"{header}: no *_events tables parsed")
    return tables


def riscv_select_events(arch, kind, select, include_dir=None) -> List[str]:
    """Names of the events an mhpmevent select arms; the counter counts their OR. [] when the select is 0."""
    entries = riscv_event_tables(arch, include_dir).get(kind)
    if entries is None:
        raise ValueError(f"arch {arch!r} has no {kind!r} cores")
    select = int(select)
    if kind == "dm":
        event_set, mask = select & 0xFF, select >> 8
        return [n for n, s in entries if (s & 0xFF) == event_set and (s >> 8) & mask]
    names = []
    for n, s in entries:
        position = (s.bit_length() - 1) // 4 * 4
        if (select >> position) & 0xF == s >> position:
            names.append(n)
    return names


def riscv_event_select(arch, kind, names, include_dir=None) -> int:
    """The mhpmevent select that arms every named event of one core kind (inverse of riscv_select_events)."""
    by_name = dict(riscv_event_tables(arch, include_dir).get(kind, []))
    select = 0
    chosen = {}
    for name in names:
        if name not in by_name:
            raise ValueError(f"{name} is not an event of {kind} cores on {arch}")
        event = by_name[name]
        # A DM select arms one event set, a baby or TRISC nibble one value; OR-ing two would arm a third event.
        if kind == "dm":
            slot, value = "set", event & 0xFF
        else:
            slot = (event.bit_length() - 1) // 4 * 4
            value = event >> slot
        if chosen.setdefault(slot, (value, name))[0] != value:
            raise ValueError(
                f"{name} and {chosen[slot][1]} cannot share one {kind} select"
            )
        select |= event
    return select


# NoC interface (NIU) counters: noc_counters_types.h has the names and groups, noc_counters_<arch>.h the table of
# {name, status register index, group} in capture order plus the data word size and NoC count.
_NOC_ARCH_HEADER = {
    "blackhole": "noc_counters_blackhole.h",
    "wormhole": "noc_counters_wormhole.h",
    "wormhole_b0": "noc_counters_wormhole.h",
    "quasar": "noc_counters_quasar.h",
}
NOC_GROUPS = ("MASTER", "SLAVE", "ATOMIC")
_NOC_ENTRY = re.compile(
    r"\{\s*NocCounter::(\w+)\s*,\s*(0[xX][0-9a-fA-F]+|\d+)\s*,\s*NocGroup::(\w+)\s*\}"
)
_NOC_TABLE_DECL = re.compile(
    r"std::array<\s*Entry\s*,\s*(\d+)\s*>\s+noc_counters\b[^=;]*="
)


class NocEntry(NamedTuple):
    name: str
    reg: int
    group: str


class NocTable(NamedTuple):
    entries: List[NocEntry]
    num_nocs: int
    word_bytes: int


def noc_counter_names(include_dir=None) -> List[str]:
    """NocCounter enumerator names in declaration order (UNDEF first), parsed from noc_counters_types.h."""
    header = find_include_dir(include_dir) / "noc_counters_types.h"
    names = parse_enum(header.read_text(), "NocCounter")
    if not names:
        raise ValueError(f"enum NocCounter in {header} parsed empty")
    return [names[i] for i in sorted(names)]


def _noc_constant(text: str, name: str, header: Path) -> int:
    match = re.search(rf"\b{name}\s*=\s*(0[xX][0-9a-fA-F]+|\d+)", text)
    if match is None:
        raise ValueError(f"{header}: constant {name} not found")
    return int(match.group(1), 0)


def noc_counter_table(arch, include_dir=None) -> NocTable:
    """Capture-order counter table, NoC count and data word bytes of one arch."""
    arch = normalize_arch(arch)
    if arch not in _NOC_ARCH_HEADER:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of {sorted(_NOC_ARCH_HEADER)}"
        )
    header = find_include_dir(include_dir) / _NOC_ARCH_HEADER[arch]
    text = _strip_comments(header.read_text())
    decl = _NOC_TABLE_DECL.search(text)
    if decl is None:
        raise ValueError(f"{header}: no noc_counters table found")
    chunk = text[decl.end() :].split("}};", 1)[0]
    entries = [NocEntry(n, int(r, 0), g) for n, r, g in _NOC_ENTRY.findall(chunk)]
    if len(entries) != int(decl.group(1)):
        raise ValueError(
            f"noc_counters: parsed {len(entries)} entries but the array declares {decl.group(1)}"
        )
    return NocTable(
        entries,
        _noc_constant(text, "NIU_NOCS", header),
        _noc_constant(text, "NIU_DATA_WORD_BYTES", header),
    )


def noc_selected_counters(arch, group_mask, include_dir=None) -> List[NocEntry]:
    """Entries of the groups a mask selects (bit n is NOC_GROUPS[n]), in capture order."""
    table = noc_counter_table(arch, include_dir)
    return [e for e in table.entries if (group_mask >> NOC_GROUPS.index(e.group)) & 1]


# DRAM channel tile counters: dram_counters_types.h has the DramCounter names, dram_counters_<arch>.h the
# {name, register offset} table of the NIU and the Layout the offsets are relative to.
_DRAM_ARCH_HEADER = {
    "blackhole": "dram_counters_blackhole.h",
    "wormhole": "dram_counters_wormhole.h",
    "wormhole_b0": "dram_counters_wormhole.h",
    "quasar": "dram_counters_quasar.h",
}
_DRAM_TABLE_DECL = re.compile(
    r"std::array<\s*Entry\s*,\s*(\d+)\s*>\s+niu_counters\b[^=;]*="
)
_DRAM_ENTRY = re.compile(r"\{\s*DramCounter::(\w+)\s*,\s*(0[xX][0-9a-fA-F]+|\d+)\s*\}")
_DRAM_LAYOUT = re.compile(
    r"Layout\s+layout\s*=\s*\{" + r",".join([r"\s*(\w+)\s*"] * 7) + r"\}"
)


def dram_counter_names(include_dir=None) -> List[str]:
    """DramCounter enumerator names in declaration order (UNDEF first), parsed from dram_counters_types.h."""
    header = find_include_dir(include_dir) / "dram_counters_types.h"
    names = parse_enum(header.read_text(), "DramCounter")
    if not names:
        raise ValueError(f"enum DramCounter in {header} parsed empty")
    return [names[i] for i in sorted(names)]


def parse_dram_table(text: str) -> List[tuple]:
    """[(counter name, offset)] of niu_counters in one dram_counters_<arch>.h."""
    text = _strip_comments(text)
    decl = _DRAM_TABLE_DECL.search(text)
    if decl is None:
        raise ValueError("niu_counters not found")
    size = int(decl.group(1))
    chunk = text[decl.end() :]
    chunk = chunk.split("}};", 1)[0] if size else ""
    pairs = [(n, int(s, 0)) for n, s in _DRAM_ENTRY.findall(chunk)]
    if len(pairs) != size:
        raise ValueError(
            f"niu_counters: parsed {len(pairs)} entries but the array declares {size}"
        )
    return pairs


def _dram_header_text(arch, include_dir) -> str:
    arch = normalize_arch(arch)
    if arch not in _DRAM_ARCH_HEADER:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of {sorted(_DRAM_ARCH_HEADER)}"
        )
    header = find_include_dir(include_dir) / _DRAM_ARCH_HEADER[arch]
    return header.read_text()


def dram_counter_table(arch, include_dir=None) -> List[tuple]:
    """[(counter name, offset)] of the NIU table for one arch, empty when the map is not known."""
    return parse_dram_table(_dram_header_text(arch, include_dir))


class DramLayout(NamedTuple):
    niu_base: int
    noc_instance_stride: int
    tile_instance_stride: int
    tile_instance_count: int
    tile_instance_bias: int
    word_bytes: int
    num_nocs: int


def dram_layout(arch, include_dir=None) -> DramLayout:
    """What the NIU offsets are relative to. Zeros mean the architecture's map is not known."""
    text = _strip_comments(_dram_header_text(arch, include_dir))
    match = _DRAM_LAYOUT.search(text)
    if match is None:
        raise ValueError(f"no Layout found for arch {arch!r}")
    return DramLayout(*(int(g.rstrip("uUlL"), 0) for g in match.groups()))


def dram_niu_address(arch, noc, tile_noc0_y, offset, include_dir=None) -> int:
    """Address of an NIU register of a DRAM tile with the given noc0 y, in the space the host reads it in."""
    layout = dram_layout(arch, include_dir)
    instance = (tile_noc0_y + layout.tile_instance_bias) % layout.tile_instance_count
    return (
        layout.niu_base
        + noc * layout.noc_instance_stride
        + instance * layout.tile_instance_stride
        + offset
    )
