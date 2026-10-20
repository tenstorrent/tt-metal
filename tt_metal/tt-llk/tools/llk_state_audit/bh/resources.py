# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Blackhole persistent-state resource taxonomy and name normalizers.

Every Effect names exactly one canonical resource. Names are ``CLASS.rest``;
the class decides scope:

* ``shared``     one copy for the whole Tensix core (all three TRISCs see it);
* ``per_thread`` every TRISC has its own copy, so two effects only touch the same
  storage when their ``Effect.thread`` matches (or one of them is ``any``);
* ``thread_named`` the thread is part of the name (``GPR.<thread>.<NAME>``).

See SCHEMA.md for the full naming table. Normalizers turn the source text of an
instruction/macro operand (e.g. ``ALU_ACC_CTRL_Fp32_enabled_RMW``,
``LO_16(p_gpr_pack::TMP0)``, ``0xB``) into canonical names, using tables parsed
from the analysed tree itself (``cfg_defines.h``, ``ckernel_gpr_map.h``, scanned
``constexpr`` constants), so the same code works on both trees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import scan
from .model import Resource

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
#: class -> (scope, description, example)
RESOURCE_CLASSES: dict[str, tuple[str, str, str]] = {
    "CFG": (
        "shared",
        "Tensix config register field (ALU/PACK/UNPACK/THCON/GLOBAL sections of cfg_defines.h), "
        "written by cfg_reg_rmw_tensix, cfg_rmw, WRCFG, RMWCIB*, CFGSHIFTMASK, cfg[]/TT_CFG_STORE MMIO",
        "CFG.ALU_ACC_CTRL_Fp32_enabled",
    ),
    "THDCFG": (
        "per_thread",
        "Per-thread config register (THREAD section of cfg_defines.h), written by SETC16",
        "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
    ),
    "ADDR_MOD": (
        "per_thread",
        "Math/unpack address-modifier slot n (addr_mod_t::set)",
        "ADDR_MOD.3",
    ),
    "ADDR_MOD_PACK": (
        "per_thread",
        "Packer address-modifier slot n (addr_mod_pack_t::set)",
        "ADDR_MOD_PACK.0",
    ),
    "GPR": (
        "thread_named",
        "Per-thread GPR (SETDMAREG, regfile[], RDCFG; read by WRCFG)",
        "GPR.pack.TMP0",
    ),
    "MOP": (
        "per_thread",
        "MOP expander configuration (ckernel_template / ckernel_unpack_template program)",
        "MOP",
    ),
    "REPLAY": (
        "per_thread",
        "Replay buffer contents; Effect.detail holds start=..;len=..",
        "REPLAY",
    ),
    "ADC": (
        "per_thread",
        "Unpacker/packer address counters (SETADC*, INCADC*, ADDRCR*)",
        "ADC.UNP_A.XY",
    ),
    "RWC": (
        "per_thread",
        "Math src/dest read-write counters (SETRWC/INCRWC)",
        "RWC.DEST",
    ),
    "SFPU": (
        "shared",
        "SFPU state: LREGs, programmable constants LREG11-14, LOADMACRO instruction templates, "
        "macro sequence registers, LoadMacro control, SFPU control register",
        "SFPU.LREG12",
    ),
    "SEM": (
        "shared",
        "Tensix semaphore (SEMINIT/SEMPOST/SEMGET/SEMWAIT, t6_semaphore_*)",
        "SEM.MATH_PACK",
    ),
    "MAILBOX": ("shared", "Inter-thread mailbox", "MAILBOX.math"),
    "DEST": (
        "shared",
        "Dest register offset/section/bank state (family-defined names)",
        "DEST.SECTION",
    ),
    "PACK": (
        "shared",
        "Packer state without a single CFG field (family-defined names)",
        "PACK.RELU",
    ),
    "UNPACK": (
        "shared",
        "Unpacker state without a single CFG field (family-defined names)",
        "UNPACK.TILIZE",
    ),
    "SW": (
        "per_thread",
        "Software shadow / LLK global variable in TRISC-local memory",
        "SW.unp_cfg_context",
    ),
    "param": (
        "shared",
        "A compute-API parameter treated as state (param:<api>.<param>)",
        "param:pack_tile.relu",
    ),
}

SFPCONFIG_DESTS: dict[int, str] = {
    0: "SFPU.LOADMACRO.INSTR0",
    1: "SFPU.LOADMACRO.INSTR1",
    2: "SFPU.LOADMACRO.INSTR2",
    3: "SFPU.LOADMACRO.INSTR3",
    4: "SFPU.LOADMACRO.SEQ0",
    5: "SFPU.LOADMACRO.SEQ1",
    6: "SFPU.LOADMACRO.SEQ2",
    7: "SFPU.LOADMACRO.SEQ3",
    8: "SFPU.LOADMACRO.CTRL",
    11: "SFPU.LREG11",
    12: "SFPU.LREG12",
    13: "SFPU.LREG13",
    14: "SFPU.LREG14",
    15: "SFPU.CTRL",
}

#: sfpi programmable constants -> LREG (sfpi_constants.h: CREG_IDX_PRGM0..3 = 11..14,
#: vConst*Prgm0 = PRGM1 = 12, Prgm1 = 13, Prgm2 = 14, Prgm3 = PRGM0 = 11).
SFPI_CONSTANTS: dict[str, str] = {
    "vConstFloatPrgm0": "SFPU.LREG12",
    "vConstIntPrgm0": "SFPU.LREG12",
    "vConstFloatPrgm1": "SFPU.LREG13",
    "vConstIntPrgm1": "SFPU.LREG13",
    "vConstFloatPrgm2": "SFPU.LREG14",
    "vConstIntPrgm2": "SFPU.LREG14",
    "vConstFloatPrgm3": "SFPU.LREG11",
    "vConstIntPrgm3": "SFPU.LREG11",
}

SEMAPHORE_NAMES = (
    "FPU_SFPU",
    "MATH_PACK",
    "UNPACK_TO_DEST",
    "UNPACK_OPERAND_SYNC",
    "PACK_DONE",
    "UNPACK_SYNC",
    "UNPACK_MATH_DONE",
    "MATH_DONE",
)

#: (group, regex on canonical name). Families may append via ``GROUPS``.
BUILTIN_GROUPS: list[tuple[str, str]] = [
    ("pack.relu", r"^CFG\.STACC_RELU_"),
    (
        "pack.edge_mask",
        r"^CFG\.(?:PCK_EDGE_|TILE_ROW_SET_MAPPING_|TILE_FACE_SET_MAPPING_)",
    ),
    ("pack.l1_acc", r"^CFG\..*(?i:l1_acc)"),
    (
        "pack.strides",
        r"^CFG\.PCK0_ADDR_CTRL_|^CFG\.PCK\d?_ADDR_|^CFG\.PCK.*(?i:stride)",
    ),
    (
        "pack.format",
        r"^CFG\.THCON_SEC[01]_REG1_|^CFG\.PCK_DEST_RD_CTRL_|^CFG\.THCON_SEC[01]_REG8_",
    ),
    ("pack.counters", r"^CFG\.PACK_COUNTERS_"),
    ("unpack.format", r"^CFG\.THCON_SEC[01]_REG[02]_|^CFG\.UNP[01]_"),
    ("unpack.address", r"^CFG\.THCON_SEC[01]_REG3_"),
    ("unpack.tilize", r"^CFG\..*(?i:tilize|haloize|Tile_x_dim|Dest_cntx)"),
    (
        "alu.format",
        r"^CFG\.ALU_FORMAT_SPEC_|^CFG\.ALU_ACC_CTRL_|^CFG\.ALU_ROUNDING_MODE_",
    ),
    ("thread.dest_offset", r"^THDCFG\.DEST_TARGET_REG_CFG_"),
    (
        "thread.srca",
        r"^THDCFG\.(?:SRCA_|SRCB_|CLR_DVALID_|FP16A_FORCE_|DISABLE_IMPLIED_)",
    ),
    ("thread.cfg_state", r"^THDCFG\.CFG_STATE_ID_"),
    ("addr_mod", r"^ADDR_MOD(?:_PACK)?\."),
    ("sfpu.constants", r"^SFPU\.LREG1[1-4]$"),
    ("sfpu.loadmacro", r"^SFPU\.LOADMACRO\."),
    ("sfpu.lreg", r"^SFPU\.LREG[0-9]$|^SFPU\.LREG10$"),
    ("sfpu.control", r"^SFPU\.CTRL$"),
    ("gpr.unpack", r"^GPR\.unpack\."),
    ("gpr.math", r"^GPR\.math\."),
    ("gpr.pack", r"^GPR\.pack\."),
    ("mop", r"^MOP$"),
    ("replay", r"^REPLAY"),
    ("counters", r"^(?:ADC|RWC)\."),
    ("semaphore", r"^SEM\."),
    ("software", r"^SW\."),
]


def resource_class(name: str) -> str:
    if name.startswith("param:"):
        return "param"
    return name.split(".", 1)[0]


def scope_of(name: str) -> str:
    return RESOURCE_CLASSES.get(resource_class(name), ("shared", "", ""))[0]


def is_per_thread(name: str) -> bool:
    return scope_of(name) == "per_thread"


def taxonomy() -> list[Resource]:
    """Resource classes as :class:`Resource` records (sorted)."""
    return [
        Resource(
            name=f"{cls}.*" if cls != "param" else "param:*",
            cls=cls,
            scope=scope,
            description=description,
            example=example,
        )
        for cls, (scope, description, example) in sorted(RESOURCE_CLASSES.items())
    ]


# --------------------------------------------------------------------------- #
# Tables parsed from the analysed tree
# --------------------------------------------------------------------------- #
CFG_DEFINES = "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h"
GPR_MAP = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_gpr_map.h"

_SECTION = re.compile(r"//\s*Registers for (\w+)")
_FIELD_DEFINE = re.compile(
    r"^#define\s+(\w+?)_(ADDR32|SHAMT|MASK)\s+(0x[0-9a-fA-F]+|\d+)\s*$"
)


@dataclass
class CfgField:
    name: str
    addr32: int
    shamt: int
    mask: int
    section: str

    @property
    def space(self) -> str:
        return "thread" if self.section == "THREAD" else "cfg"


@dataclass
class CfgTable:
    fields: dict[str, CfgField] = field(default_factory=dict)
    words: dict[tuple[str, int], list[str]] = field(
        default_factory=dict
    )  # (space, addr32) -> field names

    @classmethod
    def load(cls, repo_root: Path | str) -> "CfgTable":
        table = cls()
        path = Path(repo_root) / CFG_DEFINES
        if not path.is_file():
            return table
        section = "THREAD"
        partial: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            header = _SECTION.search(line)
            if header:
                section = header.group(1)
                continue
            match = _FIELD_DEFINE.match(line.strip())
            if not match:
                continue
            name, kind, value = match.groups()
            if name.endswith("_CFGREG_BASE"):
                continue
            entry = partial.setdefault(name, {"section": section})
            entry[kind] = int(value, 0)
        for name, entry in partial.items():
            if "ADDR32" not in entry:
                continue
            item = CfgField(
                name=name,
                addr32=entry["ADDR32"],
                shamt=entry.get("SHAMT", 0),
                mask=entry.get("MASK", 0xFFFFFFFF),
                section=entry["section"],
            )
            table.fields[name] = item
            table.words.setdefault((item.space, item.addr32), []).append(name)
        for names in table.words.values():
            names.sort(key=lambda n: (table.fields[n].shamt, n))
        return table

    def field_for_word(self, space: str, addr32: int) -> str | None:
        names = self.words.get((space, addr32))
        return names[0] if names else None


@dataclass
class GprTable:
    by_name: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # thread -> NAME -> index
    by_index: dict[str, dict[int, str]] = field(
        default_factory=dict
    )  # thread -> index -> NAME

    @classmethod
    def load(cls, repo_root: Path | str) -> "GprTable":
        table = cls()
        path = Path(repo_root) / GPR_MAP
        if not path.is_file():
            return table
        text = scan.mask_comments_and_strings(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for struct in re.finditer(
            r"struct\s+(p_gpr(?:_unpack|_math|_pack)?)\s*\{", text
        ):
            close = scan.match_close(text, struct.end() - 1, "{}")
            if close is None:
                continue
            thread = {
                "p_gpr": "any",
                "p_gpr_unpack": "unpack",
                "p_gpr_math": "math",
                "p_gpr_pack": "pack",
            }[struct.group(1)]
            for member in re.finditer(
                r"\b([A-Za-z_]\w*)\s*=\s*(\d+)\s*;", text[struct.end() : close]
            ):
                table.by_name.setdefault(thread, {})[member.group(1)] = int(
                    member.group(2)
                )
                table.by_index.setdefault(thread, {}).setdefault(
                    int(member.group(2)), member.group(1)
                )
        return table


@dataclass(frozen=True)
class Normalized:
    """Result of normalizing one operand expression."""

    name: str
    granularity: str = "field"  # field | word | byte | slot | range | whole
    detail: str = ""
    resolved: bool = True


class ResourceTables:
    """Per-tree tables plus every normalizer (``tables.normalize(kind, expr, thread)``)."""

    def __init__(self, repo_root: Path | str, constants: dict[str, str] | None = None):
        self.repo_root = Path(repo_root)
        self.cfg = CfgTable.load(repo_root)
        self.gpr = GprTable.load(repo_root)
        self._constants_text = dict(constants or {})
        self._int_cache: dict[str, int | None] = {}
        self.groups: list[tuple[str, re.Pattern[str]]] = [
            (name, re.compile(rx)) for name, rx in BUILTIN_GROUPS
        ]
        self.normalizers: dict[str, Callable[..., list[Normalized]]] = {
            "literal": self._literal,
            "cfg_rmw": self.cfg_rmw,
            "cfg_addr32": self.cfg_addr32,
            "cfg_byte": self.cfg_byte,
            "thread_cfg": self.thread_cfg,
            "gpr": self.gpr_name,
            "addr_mod": self.addr_mod,
            "addr_mod_pack": self.addr_mod_pack,
            "sfpconfig_dest": self.sfpconfig_dest,
            "lreg": self.lreg,
            "sfpi_const": self.sfpi_const,
            "adc": self.adc,
            "rwc": self.rwc,
            "semaphore": self.semaphore,
            "mop": lambda expr, thread=None: [Normalized("MOP", "whole")],
            "replay": lambda expr, thread=None: [
                Normalized("REPLAY", "range", detail=scan.compact(expr or ""))
            ],
            "sw": self.sw,
            "mailbox": self.mailbox,
        }

    # ---- registry -------------------------------------------------------- #
    def register(
        self,
        name: str,
        function: Callable[..., list[Normalized] | Normalized | str | None],
    ) -> None:
        """Families may add normalizers (name must be family-prefixed, e.g. ``pack.relu``)."""
        self.normalizers[name] = function

    def add_group(self, group: str, regex: str) -> None:
        self.groups.append((group, re.compile(regex)))

    def normalize(
        self, kind: str, expr: str | None, thread: str | None = None
    ) -> list[Normalized]:
        function = self.normalizers.get(kind)
        if function is None:
            raise KeyError(f"unknown resource normalizer {kind!r}")
        result = function(expr, thread=thread)
        if result is None:
            return []
        if isinstance(result, str):
            return [Normalized(result)]
        if isinstance(result, Normalized):
            return [result]
        return list(result)

    def groups_of(self, name: str) -> tuple[str, ...]:
        return tuple(
            sorted({group for group, regex in self.groups if regex.search(name)})
        )

    # ---- constants -------------------------------------------------------- #
    def int_value(self, expr: str | None) -> int | None:
        """Evaluate an integer constant expression (scanned constexpr names allowed)."""
        if expr is None:
            return None
        text = scan.compact(expr)
        if text in self._int_cache:
            return self._int_cache[text]
        self._int_cache[text] = None  # recursion guard
        identifiers: dict[str, object] = {}
        for name in set(re.findall(r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*", text)):
            if name in ("true", "false"):
                continue
            candidates = [name, name.split("::", 1)[-1] if "::" in name else name]
            if "::" in name:
                candidates.append("::".join(name.split("::")[-2:]))
            for candidate in candidates:
                if candidate in self._constants_text:
                    value = self.int_value(self._constants_text[candidate])
                    if value is not None:
                        identifiers[name] = value
                    break
        cleaned = re.sub(
            r"static_cast<[^<>]*>|\b(?:std::)?u?int(?:8|16|32|64)_t\s*\(", "(", text
        )
        cleaned = re.sub(r"\bto_underlying\s*\(", "(", cleaned)
        value = scan.evaluate(cleaned, identifiers=identifiers, symbolic_qualified=True)
        result = (
            value
            if isinstance(value, int) and not isinstance(value, bool)
            else (int(value) if isinstance(value, bool) else None)
        )
        self._int_cache[text] = result
        return result

    # ---- normalizers ------------------------------------------------------ #
    @staticmethod
    def _literal(expr, thread=None):
        return [Normalized(scan.compact(expr or ""), "whole")] if expr else []

    def _cfg_field_name(self, token: str) -> tuple[str | None, str]:
        """Strip _ADDR32/_RMW/_SHAMT/_MASK and return (field, suffix)."""
        match = re.fullmatch(
            r"(?:ckernel::)?([A-Za-z_]\w*?)_(ADDR32|RMW|SHAMT|MASK)", token
        )
        if match:
            return match.group(1), match.group(2)
        return None, ""

    def cfg_rmw(self, expr, thread=None):
        """``X_RMW`` or ``X_ADDR32, X_SHAMT, X_MASK`` -> field-granular CFG write."""
        parts = scan.split_args(expr or "")
        if not parts:
            return []
        name, suffix = self._cfg_field_name(parts[0])
        if name is not None:
            prefix = "THDCFG" if self._section_of(name) == "THREAD" else "CFG"
            if suffix == "ADDR32" and len(parts) >= 3:
                shamt_name, _ = self._cfg_field_name(parts[1])
                if shamt_name and shamt_name != name:
                    name = (
                        shamt_name  # field named by SHAMT when ADDR32 names a sibling
                    )
            return [Normalized(f"{prefix}.{name}", "field")]
        if len(parts) >= 3:
            word = self._cfg_word(parts[0], "cfg")
            shamt, mask = self.int_value(parts[1]), self.int_value(parts[2])
            item = self.storage(word.name) if word.resolved else None
            if item is not None and shamt is not None and mask is not None:
                space, addr32, _ = item
                field_mask = (
                    (mask << shamt) & 0xFFFFFFFF
                    if mask >> shamt == 0 and shamt
                    else mask
                )
                candidates = [
                    n
                    for n in self.cfg.words.get((space, addr32), [])
                    if self.cfg.fields[n].mask & field_mask
                ]
                if len(candidates) == 1:
                    return [Normalized(f"CFG.{candidates[0]}", "field")]
                if candidates:
                    return [
                        Normalized(
                            f"CFG.{candidates[0]}",
                            "byte",
                            detail=f"mask=0x{field_mask:08x}",
                        )
                    ]
            if word.resolved:
                return [
                    Normalized(
                        word.name, "byte", detail=f"shamt={parts[1]};mask={parts[2]}"
                    )
                ]
        return [Normalized(f"CFG.?{scan.compact(parts[0])}", "field", resolved=False)]

    def _section_of(self, name: str) -> str | None:
        item = self.cfg.fields.get(name)
        return item.section if item else None

    def _cfg_word(self, expr: str, space: str) -> Normalized:
        text = scan.compact(expr)
        match = re.fullmatch(
            r"\(?\s*(?:ckernel::)?([A-Za-z_]\w*?_ADDR32)\s*(?:([+\-])\s*(.+?))?\s*\)?",
            text,
        )
        prefix = "THDCFG" if space == "thread" else "CFG"
        if match:
            base = match.group(1)[: -len("_ADDR32")]
            offset_text = match.group(3)
            if offset_text is None:
                return Normalized(f"{prefix}.{base}", "word")
            offset = self.int_value(offset_text)
            item = self.cfg.fields.get(base)
            if offset is not None and item is not None:
                word = item.addr32 + (offset if match.group(2) == "+" else -offset)
                name = (
                    self.cfg.field_for_word(item.space, word)
                    or f"{base}{match.group(2)}{offset}"
                )
                return Normalized(f"{prefix}.{name}", "word")
            return Normalized(
                f"{prefix}.{base}",
                "word_range",
                detail=f"{match.group(2)}{scan.compact(offset_text)}",
            )
        number = self.int_value(text)
        if number is not None:
            name = self.cfg.field_for_word(space, number)
            return Normalized(
                f"{prefix}.{name}" if name else f"{prefix}.WORD{number}", "word"
            )
        return Normalized(f"{prefix}.?{text}", "word", resolved=False)

    def cfg_addr32(self, expr, thread=None):
        """A 32-bit config word written whole (cfg[], TT_CFG_STORE, WRCFG, RDCFG)."""
        if not expr:
            return []
        return [self._cfg_word(expr, "cfg")]

    def cfg_byte(self, expr, thread=None):
        """A byte-masked config word write (RMWCIB0..3, CFGSHIFTMASK)."""
        if not expr:
            return []
        word = self._cfg_word(expr, "cfg")
        return [
            Normalized(word.name, "byte", detail=word.detail, resolved=word.resolved)
        ]

    def thread_cfg(self, expr, thread=None):
        """SETC16 target: ``X_ADDR32`` of a THREAD register or an addr_mod slot array."""
        if not expr:
            return []
        text = scan.compact(expr)
        slot = re.fullmatch(
            r"addr_mod_(src|dest|bias|pack)_reg_addr\s*\[\s*(.+?)\s*\]", text
        )
        if slot:
            index = self._slot_index(slot.group(2), "ADDR_MOD_")
            cls = "ADDR_MOD_PACK" if slot.group(1) == "pack" else "ADDR_MOD"
            if index is None:
                return [
                    Normalized(
                        f"{cls}.?{slot.group(2)}",
                        "slot",
                        detail=slot.group(1),
                        resolved=False,
                    )
                ]
            return [Normalized(f"{cls}.{index}", "slot", detail=slot.group(1))]
        match = re.fullmatch(r"(?:ckernel::)?([A-Za-z_]\w*?)_ADDR32", text)
        if match:
            section = self._section_of(match.group(1))
            prefix = "THDCFG" if section in (None, "THREAD") else "CFG"
            return [Normalized(f"{prefix}.{match.group(1)}", "word")]
        word = self._cfg_word(text, "thread")
        return [word]

    def _slot_index(self, expr: str, prefix: str) -> int | None:
        text = scan.compact(expr)
        match = re.fullmatch(rf"(?:ckernel::)?{prefix}(\d+)", text)
        if match:
            return int(match.group(1))
        return self.int_value(text)

    def addr_mod(self, expr, thread=None):
        if not expr:
            return []
        index = self._slot_index(expr, "ADDR_MOD_")
        if index is None:
            return [
                Normalized(f"ADDR_MOD.?{scan.compact(expr)}", "slot", resolved=False)
            ]
        return [Normalized(f"ADDR_MOD.{index}", "slot")]

    def addr_mod_pack(self, expr, thread=None):
        if not expr:
            return []
        index = self._slot_index(expr, "ADDR_MOD_")
        if index is None:
            return [
                Normalized(
                    f"ADDR_MOD_PACK.?{scan.compact(expr)}", "slot", resolved=False
                )
            ]
        return [Normalized(f"ADDR_MOD_PACK.{index}", "slot")]

    def gpr_name(self, expr, thread=None):
        """``[LO_16|HI_16](p_gpr_<t>::NAME [+ k])`` or an index -> ``GPR.<t>.<NAME>``."""
        if not expr:
            return []
        text = scan.compact(expr)
        detail = ""
        half = re.fullmatch(r"(LO_16|HI_16)\s*\((.*)\)", text)
        if half:
            detail, text = half.group(1), scan.compact(half.group(2))
        match = re.fullmatch(
            r"(?:ckernel::)?(p_gpr(?:_unpack|_math|_pack)?)::([A-Za-z_]\w*)\s*(?:\+\s*(.+))?",
            text,
        )
        if match:
            owner = {
                "p_gpr": "any",
                "p_gpr_unpack": "unpack",
                "p_gpr_math": "math",
                "p_gpr_pack": "pack",
            }[match.group(1)]
            name = match.group(2)
            if match.group(3):
                offset = self.int_value(match.group(3))
                base = self.gpr.by_name.get(owner, {}).get(name)
                if offset is not None and base is not None:
                    name = self._gpr_name_at(owner, base + offset)
                else:
                    return [
                        Normalized(
                            f"GPR.{owner}.{name}", "range", detail=f"+{match.group(3)}"
                        )
                    ]
            return [Normalized(f"GPR.{owner}.{name}", "slot", detail=detail)]
        index = self.int_value(text)
        owner = thread if thread in scan.THREADS else "any"
        if index is not None:
            return [
                Normalized(
                    f"GPR.{owner}.{self._gpr_name_at(owner, index)}",
                    "slot",
                    detail=detail,
                )
            ]
        return [
            Normalized(f"GPR.{owner}.?{text}", "slot", detail=detail, resolved=False)
        ]

    def _gpr_name_at(self, owner: str, index: int) -> str:
        for table_owner in (owner, "any"):
            name = self.gpr.by_index.get(table_owner, {}).get(index)
            if name is not None:
                return name
        if owner == "any":
            for table_owner in ("unpack", "math", "pack"):
                name = self.gpr.by_index.get(table_owner, {}).get(index)
                if name is not None:
                    return f"R{index}"
        return f"R{index}"

    def sfpconfig_dest(self, expr, thread=None):
        if not expr:
            return []
        index = self.int_value(expr)
        if index is None:
            return [Normalized(f"SFPU.?{scan.compact(expr)}", "slot", resolved=False)]
        return [Normalized(SFPCONFIG_DESTS.get(index, f"SFPU.CONFIG{index}"), "slot")]

    def lreg(self, expr, thread=None):
        if not expr:
            return []
        text = scan.compact(expr)
        match = re.fullmatch(
            r"(?:(?:ckernel::)?p_sfpu::|(?:sfpi::)?LRegs::)?L[Rr][Ee][Gg](\d+)", text
        )
        index = int(match.group(1)) if match else self.int_value(text)
        if index is None:
            return [Normalized(f"SFPU.?{text}", "slot", resolved=False)]
        return [Normalized(f"SFPU.LREG{index}", "slot")]

    def sfpi_const(self, expr, thread=None):
        if not expr:
            return []
        name = scan.compact(expr).split("::")[-1]
        if name in SFPI_CONSTANTS:
            return [Normalized(SFPI_CONSTANTS[name], "slot", detail=name)]
        return [Normalized(f"SFPU.?{name}", "slot", resolved=False)]

    def adc(self, expr, thread=None):
        """``"<cnt_set_mask>|<dims>"`` -> ADC.<UNP_A|UNP_B|PAC>.<dims>."""
        text = scan.compact(expr or "")
        mask_text, _, dims = text.rpartition("|") if "|" in text else (text, "", "")
        dims = re.sub(r"^.*SET_", "", scan.compact(dims)) or "ALL"
        if not re.fullmatch(r"[A-Z_]+", dims):
            dims = {"0": "X", "1": "Y", "2": "Z", "3": "W"}.get(dims, "ALL")
        mask_text = scan.compact(mask_text)
        names = {
            "p_setadc::UNP0": 1,
            "p_setadc::UNP1": 2,
            "p_setadc::UNP_A": 1,
            "p_setadc::UNP_B": 2,
            "p_setadc::UNP_AB": 3,
            "p_setadc::PAC": 4,
        }
        mask = names.get(mask_text.replace("ckernel::", ""))
        if mask is None:
            mask = self.int_value(mask_text)
        if mask is None:
            return [Normalized(f"ADC.?{mask_text}.{dims}", "slot", resolved=False)]
        result = []
        for bit, unit in ((1, "UNP_A"), (2, "UNP_B"), (4, "PAC")):
            if mask & bit:
                result.append(Normalized(f"ADC.{unit}.{dims}", "slot"))
        return result

    def rwc(self, expr, thread=None):
        """``SETRWC`` bitmask (p_setrwc::SET_*) -> RWC.<SRCA|SRCB|DEST|FIDELITY|CR>."""
        text = scan.compact(expr or "")
        value = self.int_value(text)
        if value is None:
            symbolic = re.findall(r"SET_([ABDF]+)", text)
            if symbolic:
                letters = "".join(symbolic)
                value = (
                    (1 if "A" in letters else 0)
                    | (2 if "B" in letters else 0)
                    | (4 if "D" in letters else 0)
                    | (8 if "F" in letters else 0)
                )
        if value is None:
            return [Normalized(f"RWC.?{text}", "slot", resolved=False)]
        result = [
            Normalized(f"RWC.{unit}", "slot")
            for bit, unit in ((1, "SRCA"), (2, "SRCB"), (4, "DEST"), (8, "FIDELITY"))
            if value & bit
        ]
        return result or [Normalized("RWC.CR", "slot")]

    def semaphore(self, expr, thread=None):
        text = scan.compact(expr or "")
        wrapped = re.fullmatch(r"(?:(?:ckernel::)?semaphore::)?t6_sem\s*\((.*)\)", text)
        if wrapped:
            text = scan.compact(wrapped.group(1))
        match = re.search(r"semaphore::([A-Z_]+)", text)
        if match and match.group(1) in SEMAPHORE_NAMES:
            return [Normalized(f"SEM.{match.group(1)}", "slot")]
        match = re.search(r"p_stall::SEMAPHORE_(\d)", text)
        if match:
            return [Normalized(f"SEM.{SEMAPHORE_NAMES[int(match.group(1))]}", "slot")]
        value = self.int_value(text)
        if value is not None and 0 <= value < len(SEMAPHORE_NAMES):
            return [Normalized(f"SEM.{SEMAPHORE_NAMES[value]}", "slot")]
        return [Normalized(f"SEM.?{text}", "slot", resolved=False)]

    @staticmethod
    def sw(expr, thread=None):
        text = scan.compact(expr or "")
        base = re.match(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)", text)
        return [
            Normalized(
                f"SW.{base.group(1)}" if base else f"SW.?{text}",
                "whole",
                resolved=base is not None,
            )
        ]

    @staticmethod
    def mailbox(expr, thread=None):
        text = scan.compact(expr or "")
        match = re.search(r"(?:ThreadId::)?\b(Unpack|Math|Pack)ThreadId\b", text)
        if match:
            return [Normalized(f"MAILBOX.{match.group(1).lower()}", "slot")]
        return [Normalized(f"MAILBOX.?{text}", "slot", resolved=False)]

    # ---- relations -------------------------------------------------------- #
    def storage(self, name: str) -> tuple[str, int, int] | None:
        """(space, addr32, mask) for CFG/THDCFG names that map to cfg_defines fields."""
        cls = resource_class(name)
        if cls not in ("CFG", "THDCFG"):
            return None
        field_name = name.split(".", 1)[1]
        item = self.cfg.fields.get(field_name)
        if item is None:
            return None
        return (item.space, item.addr32, item.mask)

    def overlaps(
        self, a: str, b: str, granularity_a: str = "field", granularity_b: str = "field"
    ) -> bool:
        """True when two canonical resources may share storage.

        Word/byte granular CFG writes overlap every field of the word. Unresolved
        names (containing ``?``) overlap anything of the same class.
        """
        if a == b:
            return True
        cls_a, cls_b = resource_class(a), resource_class(b)
        if cls_a != cls_b:
            return False
        if "?" in a or "?" in b:
            return True
        storage_a, storage_b = self.storage(a), self.storage(b)
        if storage_a is None or storage_b is None:
            return False
        if storage_a[0] != storage_b[0] or storage_a[1] != storage_b[1]:
            return False
        mask_a = (
            0xFFFFFFFF
            if granularity_a in ("word", "byte", "word_range")
            else storage_a[2]
        )
        mask_b = (
            0xFFFFFFFF
            if granularity_b in ("word", "byte", "word_range")
            else storage_b[2]
        )
        return bool(mask_a & mask_b)


def same_thread(thread_a: str, thread_b: str) -> bool:
    """Whether two effect threads can refer to the same per-thread copy."""
    return thread_a == thread_b or "any" in (thread_a, thread_b)


def expand_rmw_args(args: Iterable[str]) -> tuple[str, ...]:
    """Expand ``X_RMW`` template arguments into ``X_ADDR32, X_SHAMT, X_MASK``."""
    result: list[str] = []
    for arg in args:
        match = re.fullmatch(r"(?:ckernel::)?([A-Za-z_]\w*?)_RMW", scan.compact(arg))
        if match:
            base = match.group(1)
            result.extend((f"{base}_ADDR32", f"{base}_SHAMT", f"{base}_MASK"))
        else:
            result.append(arg)
    return tuple(result)
