# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole unpack family: THCON/UNPACK state of the unpack TRISC (``--arch blackhole``).

Scope (:func:`in_scope`): ``tt_llk_blackhole/llk_lib/llk_unpack_*.h``,
``llk_lib/experimental/llk_unpack_*.h``, ``common/inc/cunpack_common.h``,
``common/inc/ckernel_gpr_map.h`` (GPR names, no functions) and the BH llk_api wrappers
``hw/ckernels/blackhole/metal/llk_api/llk_unpack*.h`` (their parameter flow is the
call graph's substitution of ``unpack_src_format[operand_id]`` & co.).

The core patterns record the primitives (cfg_reg_rmw_tensix, TT_CFG_STORE, cfg[]=,
WRCFG/RDCFG, SETDMAREG, SETADC*, SETC16, MOP program/run, load_replay_buf, semaphores,
``unp_cfg_context`` assignments). This module adds what they cannot see. Everything is
emitted from :func:`extract`, because each fact needs the whole function: replay
recording regions, GPR data flow, struct staging, the operands of a MOP program.

Any function (UNPACR is unpack-owned):

* ``TT[I|_OP]_UNPACR[_COMMON[_EXPLICIT_CONTEXT[_AND_COUNTER]]]`` **reads** the selected
  unpacker's tile descriptor (``UNPACK.THCON_SEC<u>_TILE_DESC.<field>`` plus the two
  descriptor CFG words), its two REG2 config words, ``Unp_LF8_4b_exp``, the four
  ch0/ch1 stride words, (unpacker 0) the REG5 dest-address and x-dim words,
  ``UNP0_ADD_DEST_ADDR_CNTR`` and ``THDCFG.SRCA_SET_Base``, the thread's
  ``THDCFG.UNPACK_MISC_CFG_CfgContextOffset_0`` and its ADC counters; it **writes**
  (auto-increments) ``ADC.UNP_<A|B>.ZW`` / ``.XY`` when AddrMode increments them.
  Base addresses (REG3/REG4) are omitted: the issuing execute writes them itself.
* Instructions built as ``TT_OP_*`` words, and instructions issued inside a
  ``load_replay_buf(..)`` lambda or a lambda handed to ``code_seq::load``, are
  recorded rather than issued: their effects carry ``detail="deferred"``.

Unpack files only (other families own theirs):

* ``unpack.wrcfg_value``: the word a ``WRCFG(gpr, .., ADDR)`` writes, from the GPR's
  ``SETDMAREG`` LO_16/HI_16 halves, ``regfile[]`` store, ``RDCFG`` or ``*DMAREG``
  result earlier in the same function and recording region (if/else halves fold to
  ``(c) ? a : b``; a write that does not reach every path leaves it unresolved).
* ``unpack.cfg_struct_field``: words stored from a staged ``unpack_config_u`` /
  ``unpack_tile_descriptor_u`` / ``alu_config_u`` (``TT_CFG_STORE`` / ``cfg[]=`` loops,
  GPR + ``WRCFG``, ``cfg_reg_rmw_tensix(var.val)``) decomposed into field writes. Field
  values follow ``var.f.x = ..`` in body order (conditional ones as
  ``(cond) ? v : prev``); fields of a zero-initialised struct not assigned are ``0``.
  Layouts are parsed from the tree's ``cunpack_common.h``.
* ``unpack.cfg_field``: field writes where the core reports a word or the wrong field:
  tile-descriptor words (one 128-bit cfg field, split with the struct layout into
  ``UNPACK.THCON_SEC<u>_TILE_DESC.<field>``), multi-field words written whole, RMW masks
  spanning several fields, and RMWs whose ``SHAMT`` belongs to a field of another word
  (``<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, ..>``
  is reported by the core as ``XY_REG_0_Ystride``).
* ``unpack.cfg_ternary``: ``cfg[upk0_reg] = v`` with ``upk0_reg = c ? A_ADDR32 : B_ADDR32``
  as two conditional writes (the core leaves ``CFG.?upk0_reg``).
* ``unpack.cfg_mmio_read``: ``.. = cfg[X]``.
* MOP programs: the deferred UNPACR reads of the instruction words a
  ``tmp.program()`` uses (constructor operands and ``set_*_op`` setters, through locals
  and ternaries, with the setter's extra condition) are attributed to that ``program()``
  site, not to the ``static constexpr`` definitions, which may be unused in the active
  branch. The MOP write itself is ``math.mop_program`` (a site signature; it replaces
  ``core.mop_program`` everywhere); :data:`DESCRIBE_MOP` re-enables a content
  description ``unpack.mop_program`` for runs without the math family.
  ``unpack.mop_cfg``: ``TTI_MOP_CFG`` (MOP zmask hi16).
* ``unpack.replay_program``: REPLAY write for ``load_replay_buf`` / ``code_seq::load``
  with ``detail="start=..;len=.."`` and value ``replay#<digest>(<params it depends on>)``,
  so two different recorded programs never compare equal (the core and
  ``math.load_replay_buf`` carry no content).
* Reads of ``unp_cfg_context`` (``SW.unp_cfg_context``), ``semaphore_read`` (SEM),
  ``CFGSHIFTMASK`` (SCRATCH source and target word), the OpB register of ``*DMAREG``;
  ``SETADC*_THREAD_OVERRIDE`` writes on the overridden thread's ADC set; and
  ``TT_INSN(table[i])`` (REPLAY read, plus MOP when the table holds ``TT_OP_MOP``).

Companions sit at the evidence of the core effect they refine; :data:`SUPERSEDES`
maps them (the foundation may drop the coarser core effect, see
out/requests/unpack.md).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import effects_db, resources, scan
from ..model import compact, identifiers
from ..resources import Normalized

FAMILY = "unpack"
ENABLED = True

#: No declarative patterns: every recognizer needs function context (module docstring).
PATTERNS: list = []

#: Family pattern -> core patterns it refines at the same (evidence, resource, access).
SUPERSEDES: dict[str, tuple[str, ...]] = {
    "unpack.cfg_ternary": ("core.cfg_assign",),
    "unpack.mop_program": ("core.mop_program", "math.mop_program"),
    "unpack.replay_program": ("core.load_replay_buf", "math.load_replay_buf"),
    "unpack.wrcfg_value": ("core.wrcfg.cfg",),
}

DISABLE_CORE: tuple[str, ...] = ()

GROUPS = [
    ("unpack.adc", r"^ADC\.UNP_[AB]\."),
    ("unpack.context", r"^THDCFG\.UNPACK_MISC_CFG_|^SW\.unp_cfg_context$"),
    (
        "unpack.dest_addr",
        r"^CFG\.THCON_SEC[01]_REG5_Dest_cntx|^CFG\.UNP0_ADD_DEST_ADDR_CNTR_",
    ),
    ("unpack.format", r"^UNPACK\.THCON_SEC[01]_TILE_DESC\.in_data_format$"),
    ("unpack.scratch", r"^CFG\.SCRATCH_SEC[0-3]_val$"),
    ("unpack.strides", r"^CFG\.UNP[01]_ADDR_CTRL_"),
    (
        "unpack.tile_descriptor",
        r"^UNPACK\.THCON_SEC[01]_TILE_DESC\.|^CFG\.THCON_SEC[01]_REG0_TileDescriptor",
    ),
    ("unpack.transpose", r"^CFG\.THCON_SEC[01]_REG2_Haloize_mode$"),
    (
        "unpack.x_dim",
        r"^CFG\.THCON_SEC[01]_REG5_Tile_x_dim_|^UNPACK\.THCON_SEC[01]_TILE_DESC\.x_dim$",
    ),
]

# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
_SCOPE = re.compile(
    r"^(?:tt_metal/tt-llk/tt_llk_blackhole/llk_lib/(?:experimental/)?llk_unpack_[^/]*"
    r"|tt_metal/tt-llk/tt_llk_blackhole/common/inc/(?:cunpack_common|ckernel_gpr_map)\.h"
    r"|tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack[^/]*)$"
)
CUNPACK_COMMON = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h"


def in_scope(path: str) -> bool:
    """True for files owned by the unpack family (repo-relative path)."""
    return bool(_SCOPE.match(path or ""))


# --------------------------------------------------------------------------- #
# UNPACR read set
# --------------------------------------------------------------------------- #
#: Tile-descriptor fields of the two words the LLK programs (TILE_DESC_SIZE = 2).
TD_FIELDS = (
    "in_data_format",
    "uncompressed",
    "blobs_per_xy_plane",
    "x_dim",
    "y_dim",
    "z_dim",
)

_UNPACKER_NAMES = {
    "SrcA": 0,
    "SrcB": 1,
    "Srcs::SrcA": 0,
    "Srcs::SrcB": 1,
    "p_unpacr::UNP0": 0,
    "p_unpacr::UNP1": 1,
    "p_unpacr_nop::UNP0": 0,
    "p_unpacr_nop::UNP1": 1,
    "0": 0,
    "1": 1,
}


def unpacker_index(expr: str | None) -> int | None:
    """``SrcA`` / ``Srcs::SrcB`` / ``0`` / ``p_unpacr::UNP1`` -> 0 or 1 (None when unknown)."""
    text = compact(expr or "").replace("ckernel::", "")
    while (
        text.startswith("(")
        and text.endswith(")")
        and scan.match_close(text, 0, "()") == len(text) - 1
    ):
        text = compact(text[1:-1])
    return _UNPACKER_NAMES.get(text)


def unpacr_read_names(unpacker: int) -> list[tuple[str, str]]:
    """(canonical name, granularity) read by one UNPACR on ``unpacker`` (0 = SrcA, 1 = SrcB).

    CFG words are named after their first field (``CfgTable.field_for_word``), the
    convention of the core's word-granular effects; ``tables.overlaps`` matches a
    word read against every field write of that word.
    """
    sec = f"THCON_SEC{unpacker}"
    unp = f"UNP{unpacker}"
    names = [(f"UNPACK.{sec}_TILE_DESC.{name}", "field") for name in TD_FIELDS]
    names += [
        (f"CFG.{sec}_REG0_TileDescriptor", "word"),
        (f"CFG.{sec}_REG0_TileDescriptor+1", "word"),
        (f"CFG.{sec}_REG2_Out_data_format", "word"),
        (f"CFG.{sec}_REG2_Disable_zero_compress_cntx0", "word"),
        (f"CFG.{sec}_REG1_Unp_LF8_4b_exp", "field"),
        (f"CFG.{unp}_ADDR_CTRL_XY_REG_0_Xstride", "word"),
        (f"CFG.{unp}_ADDR_CTRL_ZW_REG_0_Zstride", "word"),
        (f"CFG.{unp}_ADDR_CTRL_XY_REG_1_Xstride", "word"),
        (f"CFG.{unp}_ADDR_CTRL_ZW_REG_1_Zstride", "word"),
        ("THDCFG.UNPACK_MISC_CFG_CfgContextOffset_0", "field"),
    ]
    if unpacker == 0:
        names += [
            ("CFG.THCON_SEC0_REG5_Dest_cntx0_address", "word"),
            ("CFG.THCON_SEC0_REG5_Tile_x_dim_cntx0", "word"),
            ("CFG.UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr", "field"),
            ("THDCFG.SRCA_SET_Base", "field"),
        ]
    adc = "UNP_A" if unpacker == 0 else "UNP_B"
    names += [(f"ADC.{adc}.{dims}", "slot") for dims in ("XY", "ZW", "XX")]
    return names


def _norm_unpacr_reads(expr, thread=None):
    """Normalizer ``unpack.unpacr_reads``: unpacker-select operand -> the read set."""
    index = unpacker_index(expr)
    if index is None:
        return [Normalized(f"UNPACK.?{compact(expr or '')}", "whole", resolved=False)]
    return [
        Normalized(name, granularity) for name, granularity in unpacr_read_names(index)
    ]


def _norm_unpacr_adc(expr, thread=None):
    """Normalizer ``unpack.unpacr_adc``: ``"<select>|<dims>"`` -> ``ADC.UNP_<A|B>.<dims>``."""
    select, _, dims = compact(expr or "").rpartition("|")
    index = unpacker_index(select)
    if index is None:
        return [Normalized(f"ADC.?{select}.{dims or 'ALL'}", "slot", resolved=False)]
    return [
        Normalized(f"ADC.{'UNP_A' if index == 0 else 'UNP_B'}.{dims or 'ALL'}", "slot")
    ]


NORMALIZERS = {
    "unpack.unpacr_adc": _norm_unpacr_adc,
    "unpack.unpacr_reads": _norm_unpacr_reads,
}

# --------------------------------------------------------------------------- #
# Struct layouts (cunpack_common.h typedefs)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StructField:
    name: str
    word: int
    bit: int
    width: int


@dataclass(frozen=True)
class StructLayout:
    name: str
    fields: tuple[StructField, ...]
    words: int

    def in_word(self, word: int) -> list[StructField]:
        return [f for f in self.fields if f.word == word]


#: Used only when the tree has no cunpack_common.h (synthetic test trees).
_FALLBACK_STRUCTS = {
    "unpack_tile_descriptor_t": [
        ("in_data_format", 4),
        ("uncompressed", 1),
        ("reserved_0", 3),
        ("blobs_per_xy_plane", 4),
        ("reserved_1", 4),
        ("x_dim", 16),
        ("y_dim", 16),
        ("z_dim", 16),
        ("w_dim", 16),
        ("blobs_y_start_lo", 16),
        ("blobs_y_start_hi", 16),
        ("digest_type", 8),
        ("digest_size", 8),
    ],
    "unpack_config_t": [
        ("out_data_format", 4),
        ("throttle_mode", 2),
        ("context_count", 2),
        ("haloize_mode", 1),
        ("tileize_mode", 1),
        ("unpack_src_reg_set_update", 1),
        ("unpack_if_sel", 1),
        ("upsample_rate", 2),
        ("reserved_1", 1),
        ("upsamle_and_interlave", 1),
        ("shift_amount", 16),
        ("uncompress_cntx0_3", 4),
        ("unpack_if_sel_cntx0_3", 4),
        ("force_shared_exp", 1),
        ("reserved_2", 7),
        ("uncompress_cntx4_7", 4),
        ("unpack_if_sel_cntx4_7", 4),
        ("reserved_3", 8),
        ("limit_addr", 17),
        ("reserved_4", 15),
        ("fifo_size", 17),
        ("reserved_5", 15),
    ],
    "alu_config_t": [
        ("ALU_ROUNDING_MODE_Fpu_srnd_en", 1),
        ("ALU_ROUNDING_MODE_Gasket_srnd_en", 1),
        ("ALU_ROUNDING_MODE_Packer_srnd_en", 1),
        ("ALU_ROUNDING_MODE_Padding", 10),
        ("ALU_ROUNDING_MODE_GS_LF", 1),
        ("ALU_ROUNDING_MODE_Bfp8_HF", 1),
        ("ALU_FORMAT_SPEC_REG0_SrcAUnsigned", 1),
        ("ALU_FORMAT_SPEC_REG0_SrcBUnsigned", 1),
        ("ALU_FORMAT_SPEC_REG0_SrcA", 4),
        ("ALU_FORMAT_SPEC_REG1_SrcB", 4),
        ("ALU_FORMAT_SPEC_REG2_Dstacc", 4),
        ("ALU_ACC_CTRL_Fp32_enabled", 1),
        ("ALU_ACC_CTRL_SFPU_Fp32_enabled", 1),
        ("ALU_ACC_CTRL_INT8_math_enabled", 1),
    ],
}
_FALLBACK_UNIONS = {
    "alu_config_u": "alu_config_t",
    "unpack_config_u": "unpack_config_t",
    "unpack_tile_descriptor_u": "unpack_tile_descriptor_t",
}
#: Union types whose staging is modelled.
UNION_TYPES = tuple(sorted(_FALLBACK_UNIONS))


def layout_from_widths(name: str, widths: list[tuple[str, int]]) -> StructLayout:
    """GCC packing of ``std::uint32_t`` bit-fields: a field never straddles a 32-bit unit."""
    position = 0
    fields = []
    for field_name, width in widths:
        if position % 32 + width > 32:
            position = (position // 32 + 1) * 32
        fields.append(StructField(field_name, position // 32, position % 32, width))
        position += width
    return StructLayout(name, tuple(fields), (position + 31) // 32)


_LAYOUT_CACHE: dict[str, tuple[dict[str, StructLayout], dict[str, str]]] = {}


def struct_layouts(tables) -> tuple[dict[str, StructLayout], dict[str, str]]:
    """(struct -> layout, union -> struct) for the analysed tree (built-ins as fallback)."""
    root = str(getattr(tables, "repo_root", ""))
    cached = _LAYOUT_CACHE.get(root)
    if cached is not None:
        return cached
    structs: dict[str, StructLayout] = {}
    unions: dict[str, str] = {}
    path = Path(root) / CUNPACK_COMMON if root else None
    if path is not None and path.is_file():
        text = scan.mask_comments_and_strings(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for match in re.finditer(
            r"typedef\s+struct\s*\{(?P<body>[^{}]*)\}\s*(?P<name>\w+)\s*;", text
        ):
            widths = []
            for member in re.finditer(
                r"std::uint32_t\s+(\w+)\s*:\s*([\w:]+)\s*;", match.group("body")
            ):
                width = tables.int_value(member.group(2))
                if width is None:
                    widths = []
                    break
                widths.append((member.group(1), width))
            if widths:
                structs[match.group("name")] = layout_from_widths(
                    match.group("name"), widths
                )
        for match in re.finditer(
            r"typedef\s+union\s*\{(?P<body>[^{}]*)\}\s*(?P<name>\w+)\s*;", text
        ):
            member = re.search(r"\b(\w+)\s+f\s*;", match.group("body"))
            if member and member.group(1) in structs:
                unions[match.group("name")] = member.group(1)
    for name, widths in _FALLBACK_STRUCTS.items():
        structs.setdefault(name, layout_from_widths(name, widths))
    for name, struct in _FALLBACK_UNIONS.items():
        unions.setdefault(name, struct)
    _LAYOUT_CACHE[root] = (structs, unions)
    return structs, unions


# --------------------------------------------------------------------------- #
# Expression helpers
# --------------------------------------------------------------------------- #
_CFG_TOKEN = re.compile(r"(?:\bckernel::)?\b([A-Za-z_]\w*?)_(ADDR32|SHAMT|MASK)\b")


def cfg_int(tables, text: str | None) -> int | None:
    """Integer value of ``text`` with cfg_defines ``X_ADDR32/_SHAMT/_MASK`` tokens resolved."""
    if text is None:
        return None

    def replace(match: re.Match[str]) -> str:
        item = tables.cfg.fields.get(match.group(1))
        if item is None:
            return match.group(0)
        return str(
            {"ADDR32": item.addr32, "SHAMT": item.shamt, "MASK": item.mask}[
                match.group(2)
            ]
        )

    cleaned = _CFG_TOKEN.sub(replace, compact(text))
    cleaned = re.sub(r"(?<=[0-9a-fA-F])'(?=[0-9a-fA-F])", "", cleaned)
    cleaned = re.sub(r"\b(0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]+\b", r"\1", cleaned)
    return tables.int_value(cleaned)


def strip_parens(text: str) -> str:
    text = compact(text)
    while text.startswith("(") and scan.match_close(text, 0, "()") == len(text) - 1:
        text = compact(text[1:-1])
    return text


def split_top(text: str, operator: str) -> list[str]:
    """Split on a top-level ``|`` (not ``||``) or ``<<`` (not ``<<=``)."""
    parts, depth, start, index = [], 0, 0, 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(operator, index):
            if operator == "|" and "|" in (
                text[index + 1 : index + 2],
                text[index - 1 : index],
            ):
                index += 1
                continue
            if operator == "<<" and text[index + 2 : index + 3] == "=":
                index += 3
                continue
            parts.append(text[start:index])
            index += len(operator)
            start = index
            continue
        index += 1
    parts.append(text[start:])
    return [compact(part) for part in parts]


def fmt_int(value: int) -> str:
    return str(value) if value < 10 else hex(value)


def extract_bits(tables, value: str | None, shift: int, bit: int, width: int) -> str:
    """Text of bits ``[bit, bit+width)`` of ``value << shift`` ("" when ``value`` is unknown).

    Constants fold; ``a | (b << k)`` compositions split by term (a term entirely above
    the field contributes nothing); ``LOWER_HALFWORD(x)`` is ``x`` for low-half fields.
    Anything else renders as ``((v) >> bit) & mask``.
    """
    if value is None:
        return ""
    text = strip_parens(value)
    if not text:
        return ""
    mask = (1 << width) - 1
    number = cfg_int(tables, text)
    if number is not None:
        return fmt_int(((number << shift) >> bit) & mask)
    half = re.fullmatch(r"LOWER_HALFWORD\s*\((.*)\)", text)
    if half and bit + width <= shift + 16 and bit >= shift:
        return extract_bits(tables, half.group(1), shift, bit, width)
    terms = []  # (base text, bit position, constant value or None)
    for term in split_top(text, "|"):
        term = strip_parens(term)
        pieces = split_top(term, "<<")
        base = strip_parens(pieces[0] if len(pieces) == 1 else " << ".join(pieces[:-1]))
        amount = 0 if len(pieces) == 1 else cfg_int(tables, pieces[-1])
        if amount is None:
            terms = None
            break
        terms.append((base, amount + shift, cfg_int(tables, base)))
    contributions: list[str] | None = None
    if terms is not None and len(terms) == 1:
        base, total, _ = terms[0]
        if total == bit:
            contributions = [
                base
            ]  # the value starts at the field (the core's convention)
        elif total >= bit + width:
            contributions = []
    elif terms is not None:
        # a | (b << k) packs fields: each term is taken to occupy the bits up to the next term
        contributions = []
        ordered = sorted(terms, key=lambda item: item[1])
        for position, (base, total, const) in enumerate(ordered):
            if const is not None:
                piece = ((const << total) >> bit) & mask
                if piece:
                    contributions.append(fmt_int(piece))
                continue
            end = ordered[position + 1][1] if position + 1 < len(ordered) else 1 << 30
            if end <= bit or total >= bit + width:
                continue
            if total == bit:
                contributions.append(base)
            elif total > bit:
                contributions.append(f"({base}) << {total - bit}")
            else:
                contributions.append(f"({base}) >> {bit - total}")
    if contributions is not None:
        if not contributions:
            return "0"
        return (
            contributions[0]
            if len(contributions) == 1
            else " | ".join(f"({c})" for c in contributions)
        )
    shifted = text if shift == 0 else f"({text}) << {shift}"
    if bit == 0 and bit + width >= 32:
        return shifted
    if bit == 0:
        return f"({shifted}) & {hex(mask)}"
    return f"(({shifted}) >> {bit}) & {hex(mask)}"


def digest(text: str, size: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:size]


def relative_condition(condition: str | None, ambient: str | None) -> str:
    """The part of an event's condition not already implied by the sink's condition."""
    condition = compact(condition or "")
    ambient = compact(ambient or "")
    if not condition or condition == ambient:
        return ""
    if ambient and condition.startswith(ambient + " && "):
        return condition[len(ambient) + 4 :]
    return condition


_UNKNOWN = "?"


def conditional_value(
    condition: str, value: str | None, previous: str | None
) -> str | None:
    """``(c) ? v : prev``. An else-branch write after an if-branch one folds:
    ``(c) ? a : x`` followed by a write ``b`` under ``!(c)`` is ``(c) ? a : b``."""
    if not condition:
        return value
    if value is None:
        return None
    if previous is not None:
        parts = split_ternary(previous)
        if parts and condition == f"!({parts[0]})":
            return f"({parts[0]}) ? ({parts[1]}) : ({value})"
    return f"({condition}) ? ({value}) : ({previous if previous is not None else _UNKNOWN})"


def is_known(value: str | None) -> bool:
    return value is not None and "(?)" not in value


# --------------------------------------------------------------------------- #
# Per-function context
# --------------------------------------------------------------------------- #
_LOCAL_DEF = re.compile(
    r"(?<![\w.>:])(?:(?:static|constexpr|const|inline)\s+)*"
    r"(?:(?:std::)?u?int(?:8|16|32|64)_t|unsigned|int|bool|auto)\s+(?P<name>[A-Za-z_]\w*)\s*=(?!=)\s*"
)


@dataclass
class LocalDef:
    name: str
    start: int  # offset of the initializer
    end: int
    text: str  # initializer (masked body)
    raw: str  # initializer (raw body, comments stripped): keeps string literals for digests


@dataclass
class FnContext:
    fn: object
    ctx: object
    defs: dict[str, list[LocalDef]] = field(default_factory=dict)
    replay: list[tuple[int, int]] = field(default_factory=list)
    blocks: list | None = None

    @property
    def tables(self):
        return self.ctx.tables

    def region(self, offset: int) -> int:
        """Index of the replay-recording region holding ``offset`` (-1 outside)."""
        for index, (start, end) in enumerate(self.replay):
            if start <= offset <= end:
                return index
        return -1

    def local(self, name: str, offset: int | None = None) -> LocalDef | None:
        """The definition of local ``name`` in effect at ``offset`` (the nearest preceding one)."""
        found = None
        for item in self.defs.get(name, ()):
            if offset is None or item.start <= offset:
                found = item
        return found

    def cond(self, offset: int) -> str:
        """Enclosing ``if``/``else`` and undecidable ``#if`` conditions at ``offset``.

        Same as ``FunctionDef.condition_at`` except that every part holding a top-level
        ``||`` or ``?:`` is parenthesized, so ``else if (a || b)`` keeps its precedence
        when joined with ``&&`` (``scan.condition_at`` joins the raw texts).
        """
        if self.blocks is None:
            self.blocks = scan.control_blocks(self.fn.masked_body, self.fn.body)
        parts = []
        _, pp = self.fn.pp_info_at_line(self.fn.line_at(offset))
        if pp and pp != self.fn.pp_condition:
            parts.append(pp)
        parts += [
            block["condition"]
            for block in self.blocks
            if block["start"] <= offset < block["end"]
        ]
        return _join(*parts)

    def local_at(self, offset: int) -> str | None:
        """Name of the local whose initializer holds ``offset``."""
        for items in self.defs.values():
            for item in items:
                if item.start <= offset <= item.end:
                    return item.name
        return None

    def resolve_local(
        self, text: str, offset: int | None = None, depth: int = 0, raw: bool = False
    ) -> str:
        """Substitute locals (the definition in effect at ``offset``) into ``text``."""
        text = compact(text)
        if depth > 4 or not self.defs:
            return text

        def replace(match: re.Match[str]) -> str:
            item = self.local(match.group(1), offset)
            if item is None:
                return match.group(0)
            return (
                "("
                + self.resolve_local(
                    item.raw if raw else item.text, item.start, depth + 1, raw
                )
                + ")"
            )

        return compact(
            re.sub(r"(?<![\w.>:$#])\b([A-Za-z_]\w*)\b(?!\s*::)(?!\s*\()", replace, text)
        )


def _locals(fn) -> dict[str, list[LocalDef]]:
    """Typed local definitions never re-assigned outside their declarations."""
    body = fn.masked_body
    raw_body = fn.body if len(fn.body) == len(body) else body
    found: dict[str, list[LocalDef]] = {}
    for match in _LOCAL_DEF.finditer(body):
        end = scan.statement_end(body, match.end())
        raw = re.sub(
            r"//[^\n]*|/\*.*?\*/", " ", raw_body[match.end() : end], flags=re.S
        )
        found.setdefault(match.group("name"), []).append(
            LocalDef(
                match.group("name"),
                match.end(),
                end,
                compact(body[match.end() : end]),
                compact(raw),
            )
        )
    keep = {}
    for name, items in found.items():
        assignments = len(
            re.findall(
                rf"(?<![\w.>:]){re.escape(name)}\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)", body
            )
        )
        mutated = re.search(
            rf"(?:\+\+|--)\s*{re.escape(name)}\b|\b{re.escape(name)}\s*(?:\+\+|--)",
            body,
        )
        if assignments == len(items) and not mutated:
            keep[name] = items
    return keep


def _paren_span(body: str, offset: int) -> tuple[int, int] | None:
    """(open, close) of the argument list of the call whose name starts at ``offset``."""
    opening = body.find("(", offset)
    if opening < 0:
        return None
    close = scan.match_close(body, opening, "()")
    return (opening, close) if close is not None else None


_LAMBDA = re.compile(
    r"\bauto\s+([A-Za-z_]\w*)\s*=\s*\[[^\]]*\]\s*(?:\([^)]*\))?\s*(?:->\s*[\w:<>]+\s*)?\{"
)


def _lambda_bodies(body: str) -> dict[str, tuple[int, int]]:
    found = {}
    for match in _LAMBDA.finditer(body):
        close = scan.match_close(body, match.end() - 1, "{}")
        if close is not None:
            found.setdefault(match.group(1), (match.end() - 1, close))
    return found


def _replay_regions(fn) -> list[tuple[int, int]]:
    """Body spans whose instructions are recorded into the replay buffer, not issued."""
    body = fn.masked_body
    spans: set[tuple[int, int]] = set()
    lambdas = _lambda_bodies(body) if "load" in body else {}
    for call in fn.calls:
        if call.via_macro:
            continue
        if call.name == "load_replay_buf" and not call.is_member:
            span = _paren_span(body, call.offset)
            if span:
                spans.add(span)
        elif call.name == "load" and call.is_member and call.args:
            argument = compact(call.args[0])
            if argument in lambdas:
                spans.add(lambdas[argument])
            elif argument.startswith("["):
                span = _paren_span(body, call.offset)
                if span:
                    spans.add(span)
    return sorted(spans)


def _enclosing_for(body: str, offset: int) -> tuple[str, str] | None:
    """(variable, bound) of the innermost ``for (v = 0; v < bound; ..)`` around ``offset``."""
    best = None
    for match in re.finditer(r"\bfor\s*\(", body[:offset]):
        opening = match.end() - 1
        close = scan.match_close(body, opening, "()")
        if close is None or close > offset:
            continue
        loop = re.match(
            r"\s*(?:[\w:]+\s+)?([A-Za-z_]\w*)\s*=\s*0\s*;\s*\1\s*<\s*([^;]+);",
            body[opening + 1 : close],
        )
        if not loop:
            continue
        after = close + 1
        while after < len(body) and body[after].isspace():
            after += 1
        end = (
            scan.match_close(body, after, "{}")
            if after < len(body) and body[after] == "{"
            else scan.statement_end(body, after)
        )
        if end is not None and after <= offset <= end:
            best = (loop.group(1), compact(loop.group(2)))
    return best


def _emit(
    fc: FnContext,
    out: list,
    *,
    pattern: str,
    resource: str,
    normalizer: str = "literal",
    access: str = "write",
    value: str = "",
    line: int,
    condition: str | None,
    thread: str | None,
    detail: str = "",
    granularity: str | None = None,
    confidence: str = "high",
) -> None:
    out.extend(
        fc.ctx.effects(
            fc.fn,
            pattern_id=pattern,
            resource_expr=resource,
            normalizer=normalizer,
            access=access,
            value=value,
            line=line,
            condition=condition or "",
            thread=thread,
            detail=detail,
            granularity=granularity,
            confidence=confidence,
        )
    )


def _thread(fc: FnContext, call) -> str | None:
    return call.thread or fc.fn.thread_hint


def _deferred(fc: FnContext, call) -> str:
    return (
        "deferred"
        if call.name.startswith("TT_OP_") or fc.region(call.offset) >= 0
        else ""
    )


def _low_precedence(text: str) -> bool:
    """Whether ``text`` has a top-level ``||`` or ``?`` (needs parens inside an ``&&`` chain)."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and (char == "?" or text.startswith("||", index)):
            return True
    return False


def _join(*conditions: str | None) -> str:
    """``&&``-join conditions (deduplicated), parenthesizing ``||`` / ``?:`` parts."""
    parts = []
    for condition in conditions:
        text = compact(condition or "")
        if (
            text
            and _low_precedence(text)
            and not (
                text.startswith("(")
                and scan.match_close(text, 0, "()") == len(text) - 1
            )
        ):
            text = f"({text})"
        if text and text not in parts:
            parts.append(text)
    return " && ".join(parts)


# --------------------------------------------------------------------------- #
# UNPACR
# --------------------------------------------------------------------------- #
_UNPACR = re.compile(
    r"TT(?:I|_OP)?_UNPACR(?:_COMMON(?:_EXPLICIT_CONTEXT(?:_AND_COUNTER)?)?)?"
)
_ZW_INC = 0b00110011  # ch0_z [1:0], ch1_z [5:4]
_XY_INC = 0b11001100  # ch0_y [3:2], ch1_y [7:6]


def _unpacr_effects(
    fc: FnContext,
    out: list,
    args,
    *,
    line: int,
    condition: str | None,
    thread: str | None,
    detail: str,
    offset: int | None = None,
) -> None:
    if len(args) < 2:
        return
    select, addr_mode = compact(args[0]), compact(args[1])
    _emit(
        fc,
        out,
        pattern="unpack.unpacr.read",
        resource=select,
        normalizer="unpack.unpacr_reads",
        access="read",
        line=line,
        condition=condition,
        thread=thread,
        detail=detail,
    )
    mode = cfg_int(fc.tables, fc.resolve_local(addr_mode, offset))
    if mode == 0:
        return
    for bits, dims in ((_ZW_INC, "ZW"), (_XY_INC, "XY")):
        if mode is not None and not mode & bits:
            continue
        _emit(
            fc,
            out,
            pattern="unpack.unpacr.adc_inc",
            resource=f"{select}|{dims}",
            normalizer="unpack.unpacr_adc",
            value=f"inc:{addr_mode}",
            line=line,
            condition=condition,
            thread=thread,
            detail=detail,
            confidence="high" if mode is not None else "medium",
        )


def _unpacr(fc: FnContext, call, out: list, scoped: bool) -> None:
    if scoped and call.name.startswith("TT_OP_") and fc.local_at(call.offset):
        return  # an instruction-word constant: attributed to the MOP programs that use it
    _unpacr_effects(
        fc,
        out,
        call.args,
        line=call.line,
        condition=fc.cond(call.offset),
        thread=_thread(fc, call),
        detail=_deferred(fc, call),
        offset=call.offset,
    )


_INSN = re.compile(
    r"(?:::)?(?:ckernel::)?(TT_OP_UNPACR(?:_COMMON(?:_EXPLICIT_CONTEXT(?:_AND_COUNTER)?)?)?)\s*\("
)


def split_ternary(text: str) -> tuple[str, str, str] | None:
    """``c ? a : b`` at top level -> (c, a, b). ``::``, brackets and nested ternaries are respected."""
    text = strip_parens(text)
    depth, question, nested = 0, -1, 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == ":" and text[index : index + 2] == "::":
            index += 2
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            if question < 0:
                question = index
            else:
                nested += 1
        elif depth == 0 and char == ":" and question >= 0:
            if nested:
                nested -= 1
            else:
                return (
                    strip_parens(text[:question]),
                    strip_parens(text[question + 1 : index]),
                    strip_parens(text[index + 1 :]),
                )
        index += 1
    return None


def instruction_words(
    fc: FnContext, text: str, offset: int, condition: str = "", depth: int = 0
):
    """Yield (``TT_OP_UNPACR*`` call text, condition) reachable from an operand expression at ``offset``."""
    text = strip_parens(text)
    if depth > 5 or not text:
        return
    parts = split_ternary(text)
    if parts:
        cond, first, second = parts
        yield from instruction_words(
            fc, first, offset, _join(condition, cond), depth + 1
        )
        yield from instruction_words(
            fc, second, offset, _join(condition, f"!({cond})"), depth + 1
        )
        return
    item = fc.local(text, offset) if re.fullmatch(r"[A-Za-z_]\w*", text) else None
    if item is not None:
        yield from instruction_words(fc, item.text, item.start, condition, depth + 1)
        return
    op = _INSN.match(text)
    if op:
        close = scan.match_close(text, op.end() - 1, "()")
        if close == len(text) - 1:
            yield text, condition


# --------------------------------------------------------------------------- #
# Instruction refinements (unpack files)
# --------------------------------------------------------------------------- #
_THREAD_OVRD = {
    "p_setadc::THREAD_OVRD_UNPACK": "unpack",
    "p_setadc::THREAD_OVRD_MATH": "math",
    "p_setadc::THREAD_OVRD_PACK": "pack",
    "1": "unpack",
    "2": "math",
    "3": "pack",
}


def _setadc_override(fc: FnContext, call, out: list) -> None:
    base = re.sub(r"^TT(?:I|_OP)?_", "", call.name)
    args = call.args
    if base == "SETADC_THREAD_OVERRIDE" and len(args) == 5:
        mask, dims, owner, value = args[0], args[2], args[3], args[4]
    elif (
        base in ("SETADCXY_THREAD_OVERRIDE", "SETADCZW_THREAD_OVERRIDE")
        and len(args) == 7
    ):
        mask, dims, owner, value = args[0], base[6:8], args[1], ", ".join(args[2:])
    else:
        return
    target = _THREAD_OVRD.get(compact(owner).replace("ckernel::", "")) or _thread(
        fc, call
    )
    _emit(
        fc,
        out,
        pattern="unpack.setadc_override",
        resource=f"{compact(mask)}|{compact(dims)}",
        normalizer="adc",
        value=compact(value),
        line=call.line,
        condition=fc.cond(call.offset),
        thread=target,
        detail=_deferred(fc, call),
    )


def _cfgshiftmask(fc: FnContext, call, out: list) -> None:
    if len(call.args) != 6:
        return
    detail = _deferred(fc, call)
    thread = _thread(fc, call)
    select = cfg_int(fc.tables, call.args[4])
    if select is not None and 0 <= select <= 3:
        # scratch_sel 0..2 select SCRATCH_SEC<n>_val. 0b11 is used with the SCRATCH_SEC0_val
        # preload of _llk_unpack_tilize_init_ (llk_unpack_tilize.h), so it reads SEC0.
        _emit(
            fc,
            out,
            pattern="unpack.cfgshiftmask.scratch",
            resource=f"SCRATCH_SEC{select if select < 3 else 0}_val_ADDR32",
            normalizer="cfg_addr32",
            access="read",
            line=call.line,
            condition=fc.cond(call.offset),
            thread=thread,
            detail=detail,
            confidence="high" if select < 3 else "medium",
        )
    _emit(
        fc,
        out,
        pattern="unpack.cfgshiftmask.target",
        resource=compact(call.args[5]),
        normalizer="cfg_addr32",
        access="read",
        line=call.line,
        condition=fc.cond(call.offset),
        thread=thread,
        detail=detail,
    )


def _dmareg_opb(fc: FnContext, call, out: list) -> None:
    if len(call.args) != 4 or cfg_int(fc.tables, call.args[0]) != 0:
        return
    _emit(
        fc,
        out,
        pattern="unpack.dmareg.opb_read",
        resource=compact(call.args[2]),
        normalizer="gpr",
        access="read",
        line=call.line,
        condition=fc.cond(call.offset),
        thread=_thread(fc, call),
        detail=_deferred(fc, call),
    )


def _mop_cfg(fc: FnContext, call, out: list) -> None:
    _emit(
        fc,
        out,
        pattern="unpack.mop_cfg",
        resource="MOP",
        normalizer="mop",
        value=f"zmask_hi16={', '.join(compact(a) for a in call.args)}",
        line=call.line,
        condition=fc.cond(call.offset),
        thread=_thread(fc, call),
        detail=_deferred(fc, call),
    )


def _semaphore_read(fc: FnContext, call, out: list) -> None:
    if call.args and not _CORE_SEMAPHORE_READ:
        _emit(
            fc,
            out,
            pattern="unpack.semaphore_read",
            resource=compact(call.args[0]),
            normalizer="semaphore",
            access="read",
            line=call.line,
            condition=fc.cond(call.offset),
            thread=_thread(fc, call),
        )


_CORE_SEMAPHORE_READ = any(
    p.kind == "call" and re.fullmatch(p.match, "semaphore_read")
    for p in effects_db.CORE_PATTERNS
)

_TABLE_CACHE: dict[tuple[str, str, str], bool] = {}


def _table_has_mop(fc: FnContext, table: str) -> bool:
    """Whether instruction table ``table`` (defined in the same file) holds ``TT_OP_MOP`` words."""
    root = str(getattr(fc.tables, "repo_root", ""))
    key = (root, fc.fn.path, table)
    if key not in _TABLE_CACHE:
        found = False
        path = Path(root) / fc.fn.path
        if root and path.is_file():
            text = scan.mask_comments_and_strings(
                path.read_text(encoding="utf-8", errors="replace")
            )
            match = re.search(rf"\b{re.escape(table)}\s*(?:\[[^\]]*\])?\s*=", text)
            if match:
                found = (
                    "TT_OP_MOP"
                    in text[match.end() : scan.statement_end(text, match.end())]
                )
        _TABLE_CACHE[key] = found
    return _TABLE_CACHE[key]


def _tt_insn_table(fc: FnContext, call, out: list) -> None:
    match = (
        re.fullmatch(r"([A-Za-z_]\w*)\s*\[.*\]", compact(call.args[0]))
        if len(call.args) == 1
        else None
    )
    if not match:
        return
    table = match.group(1)
    thread = _thread(fc, call)
    _emit(
        fc,
        out,
        pattern="unpack.insn_table",
        resource="REPLAY",
        access="read",
        value=f"table={table}",
        line=call.line,
        condition=fc.cond(call.offset),
        thread=thread,
        detail="dynamic",
        granularity="range",
        confidence="medium",
    )
    if _table_has_mop(fc, table):
        _emit(
            fc,
            out,
            pattern="unpack.insn_table",
            resource="MOP",
            normalizer="mop",
            access="read",
            value=f"table={table}",
            line=call.line,
            condition=fc.cond(call.offset),
            thread=thread,
            detail="dynamic",
            confidence="medium",
        )


# --------------------------------------------------------------------------- #
# Struct staging
# --------------------------------------------------------------------------- #
@dataclass
class StructVar:
    name: str
    layout: StructLayout
    zero_words: frozenset
    events: list = field(
        default_factory=list
    )  # (offset, kind, target, expr, condition)

    def state(self, tables, offset: int, ambient: str | None) -> dict[str, str | None]:
        """Field values just before ``offset``, relative to the sink condition ``ambient``."""
        values: dict[str, str | None] = {
            f.name: ("0" if f.word in self.zero_words else None)
            for f in self.layout.fields
        }
        for event_offset, kind, target, expr, condition in self.events:
            if event_offset >= offset:
                break
            extra = relative_condition(condition, ambient)
            if kind == "field" and target in values:
                values[target] = conditional_value(extra, expr, values[target])
            elif kind == "zero":
                for f in self.layout.fields:
                    if f.word in target:
                        values[f.name] = conditional_value(extra, "0", values[f.name])
            elif kind == "word":
                for f in self.layout.in_word(target):
                    values[f.name] = conditional_value(
                        extra,
                        extract_bits(tables, expr, 0, f.bit, f.width) or None,
                        values[f.name],
                    )
        return values


def _struct_vars(fc: FnContext) -> dict[str, StructVar]:
    body = fc.fn.masked_body
    if not any(union in body for union in UNION_TYPES):
        return {}
    structs, unions = struct_layouts(fc.tables)
    found: dict[str, StructVar] = {}
    declaration = re.compile(
        r"\b("
        + "|".join(UNION_TYPES)
        + r")\s+([A-Za-z_]\w*)\s*(=\s*\{([^;{}]*)\})?\s*;"
    )
    for match in declaration.finditer(body):
        layout = structs.get(unions.get(match.group(1), ""))
        if layout is None:
            continue
        name, init = match.group(2), match.group(4)
        zero = (
            init is not None
            and re.fullmatch(r"\s*(?:\.val\s*=\s*)?0\s*", init) is not None
        )
        var = StructVar(
            name, layout, frozenset(range(layout.words)) if zero else frozenset()
        )
        for assign in re.finditer(
            rf"(?<![\w.>]){re.escape(name)}\s*\.\s*f\s*\.\s*([A-Za-z_]\w*)\s*=(?!=)\s*",
            body,
        ):
            if assign.start() > match.end():
                end = scan.statement_end(body, assign.end())
                var.events.append(
                    (
                        assign.start(),
                        "field",
                        assign.group(1),
                        compact(body[assign.end() : end]),
                        fc.cond(assign.start()),
                    )
                )
        for assign in re.finditer(
            rf"(?<![\w.>]){re.escape(name)}\s*\.\s*val\s*(?:\[\s*([^\]]+?)\s*\])?\s*=(?!=)\s*",
            body,
        ):
            if assign.start() <= match.end():
                continue
            end = scan.statement_end(body, assign.end())
            expr = compact(body[assign.end() : end])
            index = assign.group(1)
            condition = fc.cond(assign.start())
            loop = _enclosing_for(body, assign.start())
            if index and loop and index == loop[0]:
                bound = cfg_int(fc.tables, loop[1])
                words = frozenset(range(bound if bound is not None else layout.words))
                if expr == "0":
                    var.events.append((assign.start(), "zero", words, expr, condition))
                else:
                    var.events.extend(
                        (assign.start(), "word", w, None, condition)
                        for w in sorted(words)
                    )
                continue
            word = cfg_int(fc.tables, index) if index else 0
            if word is None:
                continue
            if expr == "0":
                var.events.append(
                    (assign.start(), "zero", frozenset({word}), expr, condition)
                )
            else:
                var.events.append((assign.start(), "word", word, expr, condition))
        var.events.sort(key=lambda event: event[0])
        found[name] = var
    return found


_STRUCT_REF = re.compile(r"^([A-Za-z_]\w*)\s*\.\s*val\s*(?:\[\s*([^\]]+?)\s*\])?$")


def _struct_ref(text: str | None, structs: dict[str, StructVar]):
    """``var.val[k]`` / ``var.val`` -> (StructVar, index text) for a staged ``var``."""
    match = _STRUCT_REF.match(strip_parens(text or ""))
    if not match or match.group(1) not in structs:
        return None
    return structs[match.group(1)], (match.group(2) or "0")


# --------------------------------------------------------------------------- #
# Field-level emission
# --------------------------------------------------------------------------- #
def container_of(tables, address: int) -> tuple[str, int] | None:
    """(field, base address) when ``address`` lies in a cfg field wider than 32 bits."""
    for base in range(address, address - 4, -1):
        for name in tables.cfg.words.get(("cfg", base), []):
            item = tables.cfg.fields[name]
            if (
                item.mask > 0xFFFFFFFF
                and address - base < (item.mask.bit_length() + 31) // 32
            ):
                return name, base
    return None


def container_prefix(container: str) -> str:
    """``THCON_SEC0_REG0_TileDescriptor`` -> ``UNPACK.THCON_SEC0_TILE_DESC``."""
    return "UNPACK." + re.sub(r"_REG\d+_TileDescriptor$", "_TILE_DESC", container)


def _word_fields(tables, address: int) -> list:
    names = tables.cfg.words.get(("cfg", address), [])
    return [
        tables.cfg.fields[n] for n in names if tables.cfg.fields[n].mask <= 0xFFFFFFFF
    ]


def compose_struct(
    tables, layout: StructLayout, word: int, values: dict, shamt: int, width: int
) -> str | None:
    """Value of cfg bits ``[shamt, shamt+width)`` from the struct fields of ``word``."""
    pieces = []
    for f in layout.in_word(word):
        low, high = max(f.bit, shamt), min(f.bit + f.width, shamt + width)
        if low >= high:
            continue
        value = values.get(f.name)
        if not is_known(value):
            return None
        if f.bit == shamt and f.width == width:
            return value
        part = extract_bits(tables, value, 0, low - f.bit, high - low)
        if not part:
            return None
        if part != "0":
            pieces.append(part if low == shamt else f"({part}) << {low - shamt}")
    if not pieces:
        return "0"
    folded = cfg_int(tables, " | ".join(f"({p})" for p in pieces))
    if folded is not None:
        return fmt_int(folded)
    return pieces[0] if len(pieces) == 1 else " | ".join(f"({p})" for p in pieces)


def emit_word_fields(
    fc: FnContext,
    out: list,
    *,
    address: int,
    line: int,
    condition: str | None,
    thread: str | None,
    detail: str,
    mask: int | None = None,
    shift: int = 0,
    value: str | None = None,
    struct: tuple | None = None,
    core_names: tuple[str, ...] = (),
) -> None:
    """Field writes for a write of cfg word ``address``: a struct word or an opaque value.

    For an opaque value nothing is emitted when the core name is already exact: the
    word holds one full-width field, or the RMW mask hits exactly the reported field.
    """
    tables = fc.tables
    pattern = "unpack.cfg_struct_field" if struct is not None else "unpack.cfg_field"
    container = container_of(tables, address)
    if container is not None:
        name, base = container
        structs, _ = struct_layouts(tables)
        layout = (
            struct[0].layout
            if struct is not None
            else structs.get("unpack_tile_descriptor_t")
        )
        if layout is None:
            return
        for f in layout.in_word(address - base):
            if f.name.startswith("reserved"):
                continue
            if mask is not None and not ((((1 << f.width) - 1) << f.bit) & mask):
                continue
            if struct is not None:
                text = struct[2].get(f.name)
                text = text if is_known(text) else ""
            else:
                text = extract_bits(tables, value, shift, f.bit, f.width)
            _emit(
                fc,
                out,
                pattern=pattern,
                resource=f"{container_prefix(name)}.{f.name}",
                value=text or "",
                line=line,
                condition=condition,
                thread=thread,
                detail=detail,
                granularity="field",
            )
        return
    fields = _word_fields(tables, address)
    if mask is not None:
        fields = [f for f in fields if f.mask & mask]
    if not fields:
        return
    if struct is None:
        reported = {n.split(".", 1)[-1] for n in core_names}
        if len(fields) == 1 and (
            fields[0].mask == 0xFFFFFFFF or fields[0].name in reported
        ):
            return
    for item in fields:
        width = (item.mask >> item.shamt).bit_length()
        if struct is not None:
            var, word, values = struct
            text = compose_struct(tables, var.layout, word, values, item.shamt, width)
        else:
            text = extract_bits(tables, value, shift, item.shamt, width)
        _emit(
            fc,
            out,
            pattern=pattern,
            resource=f"CFG.{item.name}",
            value=text or "",
            line=line,
            condition=condition,
            thread=thread,
            detail=detail,
            granularity="field",
        )


# --------------------------------------------------------------------------- #
# GPR data flow
# --------------------------------------------------------------------------- #
def _gpr_events(fc: FnContext) -> list:
    """(offset, kind, call or None, gpr name, payload, condition) in body order.

    kind: LO_16 / HI_16 / FULL (writes) or SINK (WRCFG; payload is the cfg address).
    """
    tables = fc.tables
    events = []

    def gpr(text, thread):
        found = tables.normalize("gpr", text, thread)
        return found[0] if found and found[0].resolved else None

    for call in fc.fn.calls:
        if not call.name.startswith(("TT_", "TTI_")) or call.name.startswith("TT_OP_"):
            continue
        base = re.sub(r"^TTI?_", "", call.name)
        thread = _thread(fc, call)
        condition = fc.cond(call.offset)
        if base == "SETDMAREG" and len(call.args) == 4:
            target = gpr(call.args[3], thread)
            if target is not None:
                events.append(
                    (
                        call.offset,
                        target.detail or "FULL",
                        call,
                        target.name,
                        compact(call.args[1]),
                        condition,
                    )
                )
        elif base == "RDCFG" and len(call.args) == 2:
            target = gpr(call.args[0], thread)
            if target is not None:
                events.append(
                    (
                        call.offset,
                        "FULL",
                        call,
                        target.name,
                        f"CFG({compact(call.args[1])})",
                        condition,
                    )
                )
        elif (
            re.fullmatch(r"(?:ADD|SUB|MUL|SHIFT|CMP)DMAREG", base)
            and len(call.args) == 4
        ):
            target = gpr(call.args[1], thread)
            if target is not None:
                events.append(
                    (
                        call.offset,
                        "FULL",
                        call,
                        target.name,
                        f"{base}({compact(call.args[3])}, {compact(call.args[2])})",
                        condition,
                    )
                )
        elif base == "WRCFG" and len(call.args) == 3:
            source = gpr(call.args[0], thread)
            if source is not None:
                events.append(
                    (
                        call.offset,
                        "SINK",
                        call,
                        source.name,
                        compact(call.args[2]),
                        condition,
                    )
                )
    body = fc.fn.masked_body
    for match in re.finditer(r"(?<![\w.>])regfile\s*\[", body):
        close = scan.match_close(body, match.end() - 1, "[]")
        operator = (
            re.match(r"\s*=(?!=)\s*", body[close + 1 :]) if close is not None else None
        )
        if not operator:
            continue
        start = close + 1 + operator.end()
        target = gpr(body[match.end() : close], fc.fn.thread_hint)
        if target is not None:
            events.append(
                (
                    match.start(),
                    "FULL",
                    None,
                    target.name,
                    compact(body[start : scan.statement_end(body, start)]),
                    fc.cond(match.start()),
                )
            )
    events.sort(key=lambda event: (event[0], event[1]))
    return events


@dataclass
class GprValue:
    lo: str | None = None
    hi: str | None = None
    full: str | None = None
    point: int = -1  # offset of the latest contributing write (struct state point)

    def text(self, tables) -> str | None:
        if self.full is not None:
            return self.full if is_known(self.full) else None
        if not is_known(self.lo) or not is_known(self.hi):
            return None
        lo, hi = strip_parens(self.lo), strip_parens(self.hi)
        low = re.fullmatch(r"LOWER_HALFWORD\s*\((.*)\)", lo)
        high = re.fullmatch(r"UPPER_HALFWORD\s*\((.*)\)", hi)
        if low and high and strip_parens(low.group(1)) == strip_parens(high.group(1)):
            return strip_parens(low.group(1))
        lo_value, hi_value = cfg_int(tables, lo), cfg_int(tables, hi)
        if lo_value is not None and hi_value is not None:
            return fmt_int((lo_value & 0xFFFF) | ((hi_value & 0xFFFF) << 16))
        return f"({lo}) | (({hi}) << 16)"

    def halves(self, name: str) -> str:
        """``lo | (hi << 16)`` with an unknown half kept symbolic (for field extraction)."""
        lo = self.lo if is_known(self.lo) else f"GPR_LO({name})"
        hi = self.hi if is_known(self.hi) else f"GPR_HI({name})"
        return f"({lo}) | (({hi}) << 16)"


def _gpr_value(
    events: list,
    gpr: str,
    region_of,
    region: int,
    sink_offset: int,
    sink_condition: str,
) -> GprValue | None:
    """What ``gpr`` holds at a WRCFG, folding the writes of the same region in order."""
    value = None
    for offset, kind, _, name, payload, condition in events:
        if offset >= sink_offset:
            break
        if kind == "SINK" or name != gpr or region_of(offset) != region:
            continue
        value = value or GprValue()
        extra = relative_condition(condition, sink_condition)
        if kind == "FULL":
            value.full = conditional_value(
                extra, payload, value.full if value.full is not None else None
            )
            if not extra:
                value.lo = value.hi = None
        elif kind == "LO_16":
            value.lo = conditional_value(extra, payload, value.lo)
            value.full = None if not extra else value.full
        elif kind == "HI_16":
            value.hi = conditional_value(extra, payload, value.hi)
            value.full = None if not extra else value.full
        value.point = offset
    return value


#: ``p_gpr::ZERO`` ("Always stores 0", ckernel_gpr_map.h).
ZERO_GPR = "GPR.any.ZERO"


def _gpr_flow(fc: FnContext, structs: dict[str, StructVar], out: list) -> None:
    tables = fc.tables
    events = _gpr_events(fc)
    for offset, kind, call, gpr, address_text, condition in events:
        if kind != "SINK":
            continue
        address = cfg_int(tables, address_text)
        value = _gpr_value(events, gpr, fc.region, fc.region(offset), offset, condition)
        thread = _thread(fc, call)
        detail = _deferred(fc, call)
        text = value.text(tables) if value is not None else None
        if text is None and gpr == ZERO_GPR and value is None:
            text = "0"  # p_gpr::ZERO always holds 0 (ckernel_gpr_map.h)
        if text is not None:
            _emit(
                fc,
                out,
                pattern="unpack.wrcfg_value",
                resource=address_text,
                normalizer="cfg_addr32",
                value=text,
                line=call.line,
                condition=fc.cond(call.offset),
                thread=thread,
                detail=detail,
            )
        if address is None:
            continue
        ref = _struct_ref(text, structs) if text is not None else None
        if ref is not None:
            var, index = ref
            word = cfg_int(tables, index)
            if word is not None:
                values = var.state(tables, value.point, condition)
                emit_word_fields(
                    fc,
                    out,
                    address=address,
                    line=call.line,
                    condition=fc.cond(call.offset),
                    thread=thread,
                    detail=detail,
                    struct=(var, word, values),
                )
            continue
        name = gpr.split(".", 2)[-1]
        if text is None:
            text = (
                value.halves(name)
                if value is not None
                else f"GPR({compact(call.args[0])})"
            )
        emit_word_fields(
            fc,
            out,
            address=address,
            line=call.line,
            condition=fc.cond(call.offset),
            thread=thread,
            detail=detail,
            value=text,
        )


# --------------------------------------------------------------------------- #
# Config writes / reads
# --------------------------------------------------------------------------- #
def index_alternatives(
    fc: FnContext, index: str, offset: int
) -> list[tuple[str, str]] | None:
    """A cfg index naming a local: [(address text, extra condition)] (None otherwise)."""
    item = fc.local(compact(index), offset)
    if item is None:
        return None
    initializer = strip_parens(item.text)
    parts = split_ternary(initializer)
    if parts:
        cond, first, second = parts
        return [(first, cond), (second, f"!({cond})")]
    if "_ADDR32" in initializer:
        return [(initializer, "")]
    return None


def _cfg_writes(fc: FnContext, structs: dict[str, StructVar], out: list) -> None:
    """TT_CFG_STORE / ``cfg[i] = v``: struct words, ternary indices, field companions."""
    tables = fc.tables
    body = fc.fn.masked_body
    sites = []  # (offset, line, condition, thread, index, value, is_assign)
    for call in fc.fn.calls:
        if call.name == "TT_CFG_STORE" and len(call.args) == 3:
            sites.append(
                (
                    call.offset,
                    call.line,
                    fc.cond(call.offset),
                    _thread(fc, call),
                    compact(call.args[1]),
                    compact(call.args[2]),
                    False,
                )
            )
    for match in re.finditer(r"(?<![\w.>])cfg\s*\[", body):
        close = scan.match_close(body, match.end() - 1, "[]")
        operator = (
            re.match(r"\s*=(?!=)\s*", body[close + 1 :]) if close is not None else None
        )
        if not operator:
            continue
        start = close + 1 + operator.end()
        sites.append(
            (
                match.start(),
                fc.fn.line_at(match.start()),
                fc.cond(match.start()),
                fc.fn.thread_hint,
                compact(body[match.end() : close]),
                compact(body[start : scan.statement_end(body, start)]),
                True,
            )
        )
    for offset, line, condition, thread, index, value, is_assign in sites:
        detail = "deferred" if fc.region(offset) >= 0 else ""
        ref = _struct_ref(value, structs)
        if ref is not None:
            var, word_index = ref
            values = var.state(tables, offset, condition)
            loop = _enclosing_for(body, offset)
            indexed = re.fullmatch(r"(.+?)\s*\+\s*([A-Za-z_]\w*)", index)
            if indexed and loop and indexed.group(2) == word_index == loop[0]:
                base, bound = cfg_int(tables, indexed.group(1)), cfg_int(
                    tables, loop[1]
                )
                pairs = (
                    []
                    if base is None or bound is None
                    else [(base + w, w) for w in range(bound)]
                )
            else:
                address, word = cfg_int(tables, index), cfg_int(tables, word_index)
                pairs = [] if address is None or word is None else [(address, word)]
            for address, word in pairs:
                emit_word_fields(
                    fc,
                    out,
                    address=address,
                    line=line,
                    condition=condition,
                    thread=thread,
                    detail=detail,
                    struct=(var, word, values),
                )
            continue
        alternatives = index_alternatives(fc, index, offset) if is_assign else None
        if alternatives and any(extra for _, extra in alternatives):
            for address_text, extra in alternatives:
                full = _join(condition, extra)
                _emit(
                    fc,
                    out,
                    pattern="unpack.cfg_ternary",
                    resource=address_text,
                    normalizer="cfg_addr32",
                    value=value,
                    line=line,
                    condition=full,
                    thread=thread,
                    detail=detail,
                )
                address = cfg_int(tables, address_text)
                if address is not None:
                    names = tuple(
                        n.name for n in tables.normalize("cfg_addr32", address_text)
                    )
                    emit_word_fields(
                        fc,
                        out,
                        address=address,
                        line=line,
                        condition=full,
                        thread=thread,
                        detail=detail,
                        value=value,
                        core_names=names,
                    )
            continue
        address = cfg_int(tables, index)
        if address is not None:
            names = tuple(n.name for n in tables.normalize("cfg_addr32", index))
            emit_word_fields(
                fc,
                out,
                address=address,
                line=line,
                condition=condition,
                thread=thread,
                detail=detail,
                value=value,
                core_names=names,
            )


def _cfg_reads(fc: FnContext, out: list) -> None:
    """``.. = cfg[X]`` MMIO reads (and the read half of ``cfg[X] op= v``)."""
    body = fc.fn.masked_body
    for match in re.finditer(r"(?<![\w.>])cfg\s*\[", body):
        close = scan.match_close(body, match.end() - 1, "[]")
        if close is None or re.match(r"\s*=(?!=)", body[close + 1 :]):
            continue
        index = compact(body[match.end() : close])
        line = fc.fn.line_at(match.start())
        condition = fc.cond(match.start())
        for address_text, extra in index_alternatives(fc, index, match.start()) or [
            (index, "")
        ]:
            _emit(
                fc,
                out,
                pattern="unpack.cfg_mmio_read",
                resource=address_text,
                normalizer="cfg_addr32",
                access="read",
                line=line,
                condition=_join(condition, extra),
                thread=fc.fn.thread_hint,
            )


def _rmw_fields(fc: FnContext, structs: dict[str, StructVar], out: list) -> None:
    """``cfg_reg_rmw_tensix<A, S, M>(v)``: fields where the core name is coarse or wrong."""
    tables = fc.tables
    for call in fc.fn.calls:
        if call.name != "cfg_reg_rmw_tensix" or len(call.args) != 1:
            continue
        targs = resources.expand_rmw_args(call.template_args)
        if len(targs) != 3:
            continue
        address, shift, mask = (
            cfg_int(tables, fc.resolve_local(t, call.offset)) for t in targs
        )
        if address is None or shift is None or mask is None:
            continue
        thread = _thread(fc, call)
        detail = _deferred(fc, call)
        ref = _struct_ref(call.args[0], structs)
        if ref is not None:
            var, index = ref
            word = cfg_int(tables, index)
            if word is not None:
                emit_word_fields(
                    fc,
                    out,
                    address=address,
                    line=call.line,
                    condition=fc.cond(call.offset),
                    thread=thread,
                    detail=detail,
                    mask=mask,
                    shift=shift,
                    struct=(
                        var,
                        word,
                        var.state(tables, call.offset, fc.cond(call.offset)),
                    ),
                )
            continue
        core = tuple(n.name for n in tables.normalize("cfg_rmw", ", ".join(targs)))
        emit_word_fields(
            fc,
            out,
            address=address,
            line=call.line,
            condition=fc.cond(call.offset),
            thread=thread,
            detail=detail,
            mask=mask,
            shift=shift,
            value=compact(call.args[0]),
            core_names=core,
        )


# --------------------------------------------------------------------------- #
# MOP and replay programs
# --------------------------------------------------------------------------- #
_TEMPLATES = ("ckernel_template", "ckernel_unpack_template")
#: Emit ``unpack.mop_program`` (a content description of each MOP program). Off: the math
#: family's ``math.mop_program`` replaces ``core.mop_program`` for every file and already
#: gives each program site a distinct value; a second MOP write per site would duplicate it.
DESCRIBE_MOP = False
_TEMPLATE_LABELS = {
    "ckernel_template": ("outer", "inner", "loop0", "loop1"),
    "ckernel_unpack_template": (
        "unpackB",
        "halo",
        "A0",
        "A1",
        "A2",
        "A3",
        "skipA",
        "B",
        "skipB",
    ),
}
_COUNT_LABELS = frozenset({"outer", "inner", "unpackB", "halo"})
_MODIFIERS = {
    "set_start_op": ("start",),
    "set_end_op": ("end",),
    "set_end_ops": ("end", "end1"),
    "set_loop_op0": ("loop0",),
    "set_loop_op1": ("loop1",),
    "set_last_inner_loop_instr": ("last_inner",),
    "set_last_outer_loop_instr": ("last_outer",),
}


def render_insn(fc: FnContext, text: str, offset: int, depth: int = 0) -> str:
    """Compact rendering of an instruction-word operand that substitution cannot mangle."""
    text = strip_parens(text)
    if depth > 4:
        return "$" + digest(text, 6)
    parts = split_ternary(text)
    if parts:
        cond, first, second = parts
        return f"({cond}?{render_insn(fc, first, offset, depth + 1)}:{render_insn(fc, second, offset, depth + 1)})"
    if text == "TT_OP_NOP":
        return "NOP"
    if cfg_int(fc.tables, text) is not None:
        return text
    replay = re.fullmatch(r"(?:::)?(?:ckernel::)?lltt::replay_insn\s*\((.*)\)", text)
    if replay:
        return (
            "replay("
            + ",".join(compact(a) for a in scan.split_args(replay.group(1)))
            + ")"
        )
    if re.fullmatch(r"[A-Za-z_]\w*", text):
        item = fc.local(text, offset)
        if item is not None:
            inner = strip_parens(item.text)
            if split_ternary(inner):
                return render_insn(fc, inner, item.start, depth + 1)
            return f"${text}#{digest(fc.resolve_local(text, offset, raw=True), 4)}"
        if text in fc.fn.param_names:
            return text
    op = re.match(r"(?:::)?(?:ckernel::)?TT_OP_([A-Z0-9_]+)\s*\(", text)
    if op:
        return f"${op.group(1)}#{digest(fc.resolve_local(text, offset, raw=True), 4)}"
    if re.fullmatch(r"(?:::)?[A-Za-z_][\w:]*", text):
        return "$" + text.split("::")[-1]
    return "$" + digest(fc.resolve_local(text, offset, raw=True), 6)


def _template_decl(fc: FnContext, receiver: str, before: int):
    """(type, args, offset) of the latest declaration of template variable ``receiver``."""
    best = None
    for call in fc.fn.calls:
        if call.offset >= before:
            break
        if (
            call.is_ctor
            and call.receiver_type in _TEMPLATES
            and compact(call.receiver or "").split("::")[-1] == receiver
        ):
            best = (call.receiver_type, call.args, call.offset)
    body = fc.fn.masked_body
    for match in re.finditer(
        rf"\b({'|'.join(_TEMPLATES)})\s+{re.escape(receiver)}\s*=\s*\1\s*\(",
        body[:before],
    ):
        close = scan.match_close(body, match.end() - 1, "()")
        if close is not None and (best is None or match.start() > best[2]):
            best = (
                match.group(1),
                scan.split_args(body[match.end() : close]),
                match.start(),
            )
    return best


def _mop_programs(fc: FnContext, out: list) -> None:
    for call in fc.fn.calls:
        if (
            not call.is_member
            or call.name not in ("program", "program_and_run")
            or call.receiver_type not in _TEMPLATES
        ):
            continue
        receiver = compact(call.receiver or "")
        temporary = re.fullmatch(rf"({'|'.join(_TEMPLATES)})\s*\((.*)\)", receiver)
        decl = (
            (temporary.group(1), scan.split_args(temporary.group(2)), call.offset)
            if temporary
            else _template_decl(fc, receiver, call.offset)
        )
        if decl is None:
            continue
        kind, args, decl_offset = decl
        thread = _thread(fc, call)
        operands = (
            []
        )  # (label, operand text, extra condition, is a constructor argument)
        for label, arg in zip(_TEMPLATE_LABELS[kind], args):
            operands.append((label, arg, "", True))
        for other in fc.fn.calls:
            if other.offset <= decl_offset or other.offset >= call.offset:
                continue
            if (
                other.is_member
                and compact(other.receiver or "") == receiver
                and other.name in _MODIFIERS
            ):
                labels, values = _MODIFIERS[other.name], other.args
            elif (
                not other.is_member
                and not other.is_ctor
                and other.args
                and compact(other.args[0]) == receiver
            ):
                labels, values = (
                    tuple(f"{other.name}.{i}" for i in range(1, len(other.args))),
                    other.args[1:],
                )
            else:
                continue
            extra = relative_condition(fc.cond(other.offset), fc.cond(call.offset))
            for label, arg in zip(labels, values):
                operands.append((label, arg, extra, False))
        parts, modifiers = [], []
        for label, arg, extra, from_ctor in operands:
            counts = from_ctor and label in _COUNT_LABELS
            if DESCRIBE_MOP:
                rendered = compact(arg) if counts else render_insn(fc, arg, call.offset)
                if from_ctor:
                    parts.append(f"{label}={rendered}")
                else:
                    modifiers.append(
                        f"{label}=({extra}?{rendered}:-)"
                        if extra
                        else f"{label}={rendered}"
                    )
            if counts:
                continue
            for word, word_condition in instruction_words(fc, arg, call.offset):
                inner = scan.split_args(word[word.index("(") + 1 : -1])
                _unpacr_effects(
                    fc,
                    out,
                    inner,
                    line=call.line,
                    condition=_join(fc.cond(call.offset), extra, word_condition),
                    thread=thread,
                    detail="deferred",
                    offset=call.offset,
                )
        if DESCRIBE_MOP:
            text = f"({','.join(parts)})" + "".join(f"+{m}" for m in modifiers)
            value = (
                f"{kind}#{digest(fc.resolve_local(text, call.offset, raw=True))}{text}"
            )
            _emit(
                fc,
                out,
                pattern="unpack.mop_program",
                resource="MOP",
                normalizer="mop",
                value=value,
                line=call.line,
                condition=fc.cond(call.offset),
                thread=thread,
                detail=_deferred(fc, call),
            )


def _replay_programs(fc: FnContext, out: list) -> None:
    body = fc.fn.masked_body
    params = set(fc.fn.param_names)
    lambdas = _lambda_bodies(body) if "load" in body else {}
    for call in fc.fn.calls:
        if call.via_macro:
            continue
        if (
            call.name == "load_replay_buf"
            and not call.is_member
            and len(call.args) >= 3
        ):
            span = _paren_span(body, call.offset)
            detail = f"start={compact(call.args[0])};len={compact(call.args[1])}"
        elif (
            call.name == "load"
            and call.is_member
            and call.args
            and compact(call.args[0]) in lambdas
        ):
            span = lambdas[compact(call.args[0])]
            detail = f"seq={compact(call.receiver or '')}"
        else:
            continue
        if span is None:
            continue
        text = compact(body[span[0] : span[1] + 1])
        raw_body = fc.fn.body if len(fc.fn.body) == len(body) else body
        raw = compact(
            re.sub(
                r"//[^\n]*|/\*.*?\*/", " ", raw_body[span[0] : span[1] + 1], flags=re.S
            )
        )
        depends = sorted(identifiers(text) & params)
        _emit(
            fc,
            out,
            pattern="unpack.replay_program",
            resource="REPLAY",
            value=f"replay#{digest(fc.resolve_local(raw, span[0], raw=True))}({','.join(depends)})",
            line=call.line,
            condition=fc.cond(call.offset),
            thread=_thread(fc, call),
            detail=detail,
            granularity="range",
        )


# --------------------------------------------------------------------------- #
# Software state reads
# --------------------------------------------------------------------------- #
_SW_READ = re.compile(
    r"(?<![\w.>])(?:ckernel::)?unp_cfg_context\b(?!\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=))"
)


def _sw_reads(fc: FnContext, out: list) -> None:
    body = fc.fn.masked_body
    lines: set[int] = set()
    for match in _SW_READ.finditer(body):
        line = fc.fn.line_at(match.start())
        if line in lines:
            continue
        lines.add(line)
        _emit(
            fc,
            out,
            pattern="unpack.sw_read",
            resource="unp_cfg_context",
            normalizer="sw",
            access="read",
            line=line,
            condition=fc.cond(match.start()),
            thread=fc.fn.thread_hint,
        )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
_SCOPED_CALLS = {
    "TTI_CFGSHIFTMASK": _cfgshiftmask,
    "TTI_MOP_CFG": _mop_cfg,
    "TT_CFGSHIFTMASK": _cfgshiftmask,
    "TT_INSN": _tt_insn_table,
    "TT_MOP_CFG": _mop_cfg,
    "TT_OP_CFGSHIFTMASK": _cfgshiftmask,
    "semaphore_read": _semaphore_read,
}
_DMAREG = re.compile(r"TT(?:I|_OP)?_(?:ADD|SUB|MUL|SHIFT|CMP)DMAREG")
_OVERRIDE = re.compile(r"TT(?:I|_OP)?_SETADC(?:XY|ZW)?_THREAD_OVERRIDE")


def extract(fn, ctx):
    """Unpack effects of one scanned function (see the module docstring)."""
    body = fn.masked_body
    if not body or not body.strip():
        return []
    scoped = in_scope(fn.path)
    if (
        not scoped
        and "UNPACR" not in body
        and not any("UNPACR" in e.text for e in fn.expansions)
    ):
        return []
    fc = FnContext(fn, ctx)
    fc.replay = _replay_regions(fn)
    fc.defs = _locals(fn)
    out: list = []
    for call in fn.calls:
        if call.is_member or call.is_ctor:
            continue
        if _UNPACR.fullmatch(call.name):
            _unpacr(fc, call, out, scoped)
        elif scoped:
            handler = _SCOPED_CALLS.get(call.name)
            if handler is not None:
                handler(fc, call, out)
            elif _DMAREG.fullmatch(call.name):
                _dmareg_opb(fc, call, out)
            elif _OVERRIDE.fullmatch(call.name):
                _setadc_override(fc, call, out)
    if scoped:
        structs = _struct_vars(fc)
        _gpr_flow(fc, structs, out)
        _cfg_writes(fc, structs, out)
        _cfg_reads(fc, out)
        _rmw_fields(fc, structs, out)
        _mop_programs(fc, out)
        _replay_programs(fc, out)
        _sw_reads(fc, out)
    return out


#: Per-tile helpers that executes call; their name-based role (config/init) misleads.
HELPER_NAMES = frozenset(
    {
        "_llk_unpack_configure_addresses_",
        "_llk_unpack_configure_single_address_",
        "read_alu_config",
        "read_unpack_config",
        "read_unpack_config_helper",
        "read_unpack_tile_descriptor",
        "read_unpack_tile_descriptor_helper",
        "reset_config_context",
        "set_dst_write_addr",
        "switch_config_context",
        "wait_for_dest_available",
        "wait_for_idle",
        "wait_for_next_context",
    }
)


def role(fn, default: str) -> str | None:
    """Role overrides for unpack files (helpers; ``*_reinit_*`` is an init)."""
    if not in_scope(fn.path):
        return None
    if fn.name in HELPER_NAMES:
        return "helper"
    if re.search(r"(?:^|_)reinit(?:_|$)", fn.name.strip("_")):
        return "init"
    return None


def thread(fn, default: str) -> str | None:
    """Everything in the unpack files runs on the unpack TRISC."""
    if in_scope(fn.path) and default == "any":
        return "unpack"
    return None
