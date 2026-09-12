# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parsers for the perf counter headers in tools/include/perf_counters.

The C++ headers are the single source: the PerfCounterType enum gives the
ordinal -> name table the firmware tags records with, and <arch>.h gives the
(bank, select) -> name tables the LLK harness decodes its config words with.
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
}
_ARCHES_WITHOUT_TABLES = ("quasar",)


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


def counter_type_names(include_dir=None) -> Dict[int, str]:
    """Ordinal -> name table of PerfCounterType, parsed from types.h."""
    header = find_include_dir(include_dir) / "types.h"
    names = parse_enum(header.read_text())
    if not names:
        raise ValueError(f"enum PerfCounterType in {header} parsed empty")
    return names


def parse_tables(text: str) -> Dict[str, List[CounterEntry]]:
    """Bank -> entries for one <arch>.h. Empty arrays (Wormhole's L1 banks 2-5) are skipped."""
    banks: Dict[str, List[CounterEntry]] = {bank: [] for bank in BANK_KEYS}
    text = _strip_comments(text)
    decls = list(re.finditer(r"\b(\w+_counters)\s*=", text))
    for i, decl in enumerate(decls):
        name = decl.group(1)
        chunk = text[decl.end() : decls[i + 1].start() if i + 1 < len(decls) else None]
        chunk = chunk.split("};", 1)[0]
        pairs = re.findall(r"PerfCounterType::(\w+)\s*,\s*(\d+)", chunk)
        if not pairs:
            continue
        l1 = re.fullmatch(r"l1_(\d+)_counters", name)
        if l1 is not None:
            mux = int(l1.group(1))
            banks["L1"].extend(CounterEntry(n, int(s), mux) for n, s in pairs)
        elif name in _ARRAY_TO_BANK:
            banks[_ARRAY_TO_BANK[name]].extend(
                CounterEntry(n, int(s), None) for n, s in pairs
            )
    return banks


def normalize_arch(arch) -> str:
    """Lower-case arch name; accepts enums whose str() is the name (the harness ChipArchitecture)."""
    return str(getattr(arch, "value", arch)).lower()


def bank_tables(arch, include_dir=None) -> Dict[str, List[CounterEntry]]:
    """Bank -> [CounterEntry] for one arch; quasar has no tables yet and returns {}."""
    arch = normalize_arch(arch)
    if arch in _ARCHES_WITHOUT_TABLES:
        return {}
    if arch not in _ARCH_HEADER:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of "
            f"{sorted(_ARCH_HEADER) + list(_ARCHES_WITHOUT_TABLES)}"
        )
    header = find_include_dir(include_dir) / _ARCH_HEADER[arch]
    banks = parse_tables(header.read_text())
    empty = [bank for bank in BANK_KEYS if not banks[bank]]
    if empty:
        raise ValueError(
            f"{header}: banks {empty} parsed empty. The table syntax probably changed; "
            "the parser expects {PerfCounterType::NAME, <select>} entries in *_counters arrays."
        )
    return banks
