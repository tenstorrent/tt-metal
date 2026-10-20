# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole PACK family: packer configuration, pack GPRs and pack-thread execution.

Scope (:data:`SCOPE_RE`, repo-relative):

* ``tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack*.h`` and ``llk_lib/experimental/llk_pack*.h``
* ``tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h``
* ``tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack*.h``, ``llk_api/experimental/llk_pack*.h``,
  ``llk_api/experimental/2_0/llk_pack*.h`` and ``llk_io/llk_io_pack.h``

The core patterns already turn every primitive in these files into effects (cfg_reg_rmw_tensix,
WRCFG, SETDMAREG, regfile[], cfg[] / TT_CFG_STORE, addr_mod_pack_t::set, PACR ADDR_MOD operands,
MOP program/run, replay, SETADC*, semaphores, SW globals). This family keeps all of them and adds,
for scope functions only, what the generic patterns cannot know:

``pack.cfg_field``
    Per-field decomposition of a CFG write at the same sink. The core names one field per write
    (``CFG.<first field>``, word granularity for WRCFG/cfg[]) and records the raw source text as
    the value (``GPR(p_gpr_pack::TMP0)``, ``pack_edge_offset.val``). This pattern emits one
    field-granular effect per ``cfg_defines.h`` field the write covers, with that field's value
    resolved through the function's own dataflow: the SETDMAREG halves that fed the GPR, local
    variables and the bitfield unions of ``cpack_common.h`` (``pack_config_u``, ``relu_config_u``,
    ``pck_edge_offset_u``, ``pack_counters_u``, ``dest_rd_ctrl_u``), per ``if``/``if constexpr``
    path. A value that cannot be derived is ``""`` (unknown) -- never a guess. ``detail`` holds
    ``src=wrcfg|mmio|rmw`` (``mmio`` = RISC store, not ordered with the Tensix stream) plus
    ``word=0x..`` when the whole written word is a constant (``data=0x..`` for an RMW, whose
    constant covers only the masked bits). Array-like words (``TILE_ROW_SET_MAPPING_k``) get one
    word-granular effect carrying the whole-word value instead of 16 two-bit fields. Multi-field RMWs
    (``cfg_reg_rmw_tensix<ADDR32, SHAMT, A_MASK | B_MASK>``) therefore also report the fields the
    core misses (e.g. ``STACC_RELU_ReluThreshold``, ``THCON_SEC0_REG1_Exp_threshold_en``).
``pack.exec.read`` / ``pack.exec.busy``
    A pack-thread MOP run (``ckernel_template::run()``, ``TTI_MOP``), an immediate PACR or a
    replay execution reads the packer state it consumes (:data:`PACK_EXEC_READS`: formats, L1
    destination, strides, edge masks / row-set mapping, pack counters, relu, L1 accumulate,
    exponent threshold, Dstacc format, dest offset, packer ADCs, REPLAY) and makes the packer busy
    (``PACK.BUSY`` write). PACRs recorded into the replay buffer (``load_replay_buf`` lambdas,
    ``lltt::record`` / ``TTI_REPLAY`` load windows) and ``TT_OP_PACR`` words are not executions.
``pack.stall``
    ``TTI_STALLWAIT(.., <mask with PACK>)`` and ``tensix_sync()`` wait for the packer
    (``PACK.BUSY`` read).
``pack.cfg_read``
    ``cfg[..]`` MMIO reads (``read_pack_config_helper`` & co).
``pack.sw_read``
    Reads of ``dest_offset_id`` / ``pack_sync_tile_dst_ptr`` (the core models only the writes).
``pack.gpr_read.dest_offset``
    ``TT_WRCFG(get_packer_dest_offset_index(), ..)`` reads ``GPR.pack.DEST_OFFSET_HI`` when
    ``dest_offset_id`` else ``GPR.pack.DEST_OFFSET_LO``.
``pack.storereg``
    ``TT_STOREREG(gpr, addr)`` reads the GPR.
``pack.cb_state``
    Circular-buffer software pointers of the pack side (``fifo_wr_ptr``, ``fifo_wr_tile_ptr``,
    ``tiles_received``, ...) as ``SW.cb_interface.<field>`` with ``detail=cb=<operand>``.

Roles of a few scope functions whose names mislead the default rules are overridden
(:data:`ROLE_OVERRIDES`). :func:`ownership` summarizes, per pack resource, which roles write it
-- the "set only by hw-configure / a separate config call" versus "(re)written by pack init /
reconfig / execute" classification the checker needs.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import scan
from ..model import combine_conditions, compact, identifiers
from ..resources import Normalized

FAMILY = "pack"
ENABLED = True

#: Repo-relative paths owned by this family.
SCOPE_RE = re.compile(
    r"^tt_metal/(?:"
    r"tt-llk/tt_llk_blackhole/(?:llk_lib/(?:experimental/)?llk_pack[^/]*|common/inc/cpack_common)\.h"
    r"|hw/ckernels/blackhole/metal/(?:llk_api/(?:experimental/(?:2_0/)?)?llk_pack[^/]*"
    r"|llk_io/llk_io_pack)\.h"
    r")$"
)

CPACK_COMMON = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h"


def in_scope(path: str) -> bool:
    """Whether a repo-relative path belongs to the pack family."""
    return bool(SCOPE_RE.match(path or ""))


# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #
#: Packer state consumed by every pack execution: (cfg_defines field, granularity). ``word``
#: entries cover a config word that only the packer uses (a word read overlaps every field of
#: the word); ``field`` entries sit in words shared with other threads (relu shares ALU_ACC_CTRL
#: zero flags, Dstacc shares the math-owned SrcA/SrcB format overrides, REG1 word 3 holds the
#: unpacker's Unp_LF8_4b_exp), so only the packer's own fields are read.
PACK_EXEC_READS: tuple[tuple[str, str], ...] = (
    ("THCON_SEC0_REG1_Row_start_section_size", "word"),
    ("THCON_SEC0_REG1_L1_Dest_addr", "word"),
    ("THCON_SEC0_REG1_Disable_zero_compress", "word"),
    ("THCON_SEC0_REG1_Downsample_mask", "field"),
    ("THCON_SEC0_REG1_Downsample_rate", "field"),
    ("THCON_SEC0_REG1_Pack_L1_Acc", "field"),
    ("THCON_SEC0_REG1_Exp_threshold_en", "field"),
    ("THCON_SEC0_REG1_Pac_LF8_4b_exp", "field"),
    ("THCON_SEC0_REG1_Exp_threshold", "field"),
    ("PCK0_ADDR_CTRL_XY_REG_0_Xstride", "word"),
    ("PCK0_ADDR_CTRL_ZW_REG_0_Zstride", "word"),
    ("PCK0_ADDR_CTRL_XY_REG_1_Xstride", "word"),
    ("PCK0_ADDR_CTRL_ZW_REG_1_Zstride", "word"),
    ("PCK0_ADDR_BASE_REG_0_Base", "word"),
    ("PCK_DEST_RD_CTRL_Read_32b_data", "word"),
    ("TILE_ROW_SET_MAPPING_0_row_set_mapping_0", "word"),
    ("TILE_ROW_SET_MAPPING_1_row_set_mapping_0", "word"),
    ("TILE_ROW_SET_MAPPING_2_row_set_mapping_0", "word"),
    ("TILE_ROW_SET_MAPPING_3_row_set_mapping_0", "word"),
    ("PCK_EDGE_OFFSET_SEC0_mask", "word"),
    ("PCK_EDGE_OFFSET_SEC1_mask", "word"),
    ("PCK_EDGE_OFFSET_SEC2_mask", "word"),
    ("PCK_EDGE_OFFSET_SEC3_mask", "word"),
    ("PACK_COUNTERS_SEC0_pack_per_xy_plane", "word"),
    ("STACC_RELU_ApplyRelu", "field"),
    ("STACC_RELU_ReluThreshold", "field"),
    ("ALU_FORMAT_SPEC_REG2_Dstacc", "field"),
    ("ALU_FORMAT_SPEC_REG_Dstacc_val", "field"),
    ("ALU_FORMAT_SPEC_REG_Dstacc_override", "field"),
    ("DEST_TARGET_REG_CFG_PACK_SEC0_Offset", "word"),
)

#: Packer address counters read (and advanced) by every PACR, named as the core names writes.
PACK_EXEC_ADC = (
    "ADC.PAC.X",
    "ADC.PAC.Y",
    "ADC.PAC.Z",
    "ADC.PAC.W",
    "ADC.PAC.XY",
    "ADC.PAC.ZW",
    "ADC.PAC.XX",
)

#: GPR halves with a documented fixed value (ckernel_gpr_map.h): (gpr name, half) -> value.
GPR_FIXED_HALVES = {
    ("GPR.any.ZERO", "lo"): 0,
    ("GPR.any.ZERO", "hi"): 0,
    ("GPR.pack.TMP_LO", "hi"): 0,  # "Temp data, upper 16-bit always 0"
    ("GPR.pack.TMP_HI", "lo"): 0,  # "Temp data, lower 16-bit always 0"
}

#: Role corrections for scope functions whose names mislead ``effects_db.default_role``.
ROLE_OVERRIDES = {
    # per-tile helpers of execute functions, not configuration entry points
    "program_packer_destination": "helper",
    "program_packer_dest_offset_registers": "helper",
    "program_packer_untilized_destination": "helper",
    "set_dst_write_addr": "helper",
    "get_output_tile_address": "helper",
    # debug readers ("config" in the name)
    "read_pack_config": "helper",
    "read_pack_config_helper": "helper",
    "read_relu_config": "helper",
    # init-time programming named like execute
    "_llk_pack_fast_tilize_load_replay_": "init",
    "_llk_pack_fast_tilize_reinit_unit_dim_": "init",
    "llk_pack_fast_tilize_reinit_unit_dim": "init",
    # undo of _llk_pack_reduce_mask_config_ (called by reduce_uninit)
    "_llk_pack_reduce_mask_clear_": "uninit",
    "llk_pack_reduce_mask_clear": "uninit",
    # pack half of the mid-kernel fp32 dest-acc reconfiguration
    "llk_pack_wait_fp32_dest_acc": "reconfig",
    # canonical-state cleanup: a full re-run of the packer hw configure
    "_llk_pack_hw_cleanup_canonical_": "hw_startup",
    "_llk_pack_hw_cleanup_configure_current_bank_": "hw_startup",
    "_llk_pack_hw_cleanup_poison_pack_ambient_": "config",
}

#: Semantic groups over canonical names (``tables.groups_of``). The ``*_owned`` groups record
#: which kind of call (re)writes each packer field on Blackhole, derived with :func:`ownership`
#: over the llk_api pack wrappers of both trees:
#: ``pack.hw_config_owned`` -- written only by hw-configure or a separate config call (relu by
#: llk_pack_relu_config, L1 accumulate by llk_pack_reconfig_l1_acc, edge masks / row-set mapping /
#: pack counters by the reduce mask config/clear, Dstacc override, PCK_DEST_RD_CTRL_Read_int8);
#: no pack init and no data-format reconfig touches them.
#: ``pack.reconfig_owned`` -- rewritten by every pack data-format reconfig (THCON_SEC0_REG1 words
#: 0 and 2 -- the WRCFG of word 2 clobbers every field of it --, LF8, exponent threshold, Dstacc,
#: PCK_DEST_RD_CTRL except Read_int8).
#: ``pack.init_owned`` -- rewritten by pack init (ADDR_MODs unless skipped, strides unless
#: skipped -- reconfig rewrites them too --, the PAC X counter).
#: ``pack.execute_owned`` -- rewritten by every pack execute (L1 destination, W/Z counters).
#: Downsample mask/rate are read by every pack but written by no pack code (hardware default).
GROUPS = [
    ("pack.dest_rd_ctrl", r"^CFG\.PCK_DEST_RD_CTRL_"),
    ("pack.exp_threshold", r"^CFG\.THCON_SEC0_REG1_Exp_threshold"),
    (
        "pack.l1_dest_addr",
        r"^CFG\.THCON_SEC0_REG1_L1_Dest_addr$|^GPR\.pack\.OUTPUT_ADDR",
    ),
    (
        "pack.dest_offset",
        r"^CFG\.DEST_TARGET_REG_CFG_PACK_|^GPR\.pack\.(?:DEST_OFFSET_|TEMP_TILE_OFFSET)"
        r"|^SW\.dest_offset_id$|^CFG\.PCK0_ADDR_BASE_REG_",
    ),
    (
        "pack.data_format",
        r"^CFG\.THCON_SEC0_REG1_(?:Out_data_format|In_data_format|Exp_section_size"
        r"|Row_start_section_size|Pac_LF8_4b_exp)$|^CFG\.ALU_FORMAT_SPEC_REG2_Dstacc$"
        r"|^CFG\.ALU_FORMAT_SPEC_REG_Dstacc_(?:val|override)$",
    ),
    ("pack.adc", r"^ADC\.PAC\."),
    ("pack.addr_mod", r"^ADDR_MOD_PACK\."),
    ("pack.tile_header", r"^GPR\.pack\.(?:TILE_HEADER|EXP\d_SEC_SIZE_BFP)"),
    ("pack.busy", r"^PACK\.BUSY$"),
    ("pack.cb", r"^SW\.cb_interface\."),
    (
        "pack.hw_config_owned",
        r"^CFG\.(?:STACC_RELU_\w+|THCON_SEC0_REG1_Pack_L1_Acc|PCK_EDGE_OFFSET_SEC\d_mask"
        r"|PCK_EDGE_MODE_mode|PCK_EDGE_TILE_ROW_SET_SELECT_select"
        r"|TILE_ROW_SET_MAPPING_\d_row_set_mapping_\d+|PACK_COUNTERS_SEC0_\w+"
        r"|ALU_FORMAT_SPEC_REG_Dstacc_(?:val|override)|PCK_DEST_RD_CTRL_Read_int8)$",
    ),
    (
        "pack.reconfig_owned",
        r"^CFG\.(?:THCON_SEC0_REG1_(?:Row_start_section_size|Exp_section_size|Disable_zero_compress"
        r"|Add_l1_dest_addr_offset|Disable_pack_zero_flags|ovrd_default_throttle_mode"
        r"|Out_data_format|In_data_format|Dis_shared_exp_assembler|Auto_set_last_pacr_intf_sel"
        r"|Enable_out_fifo|Sub_l1_tile_header_size|Source_interface_selection|pack_start_intf_pos"
        r"|All_pack_disable_zero_compress_ovrd|Add_tile_header_size|pack_dis_y_pos_start_offset"
        r"|L1_source_addr|Pac_LF8_4b_exp|Exp_threshold|Exp_threshold_en)"
        r"|ALU_FORMAT_SPEC_REG2_Dstacc|PCK_DEST_RD_CTRL_(?:Read_32b_data|Read_unsigned"
        r"|Round_10b_mant))$",
    ),
    (
        "pack.init_owned",
        r"^ADDR_MOD_PACK\.[0-3]$|^CFG\.PCK0_ADDR_CTRL_(?:XY|ZW)_REG_[01]_\w+$|^ADC\.PAC\.XX$",
    ),
    (
        "pack.execute_owned",
        r"^CFG\.THCON_SEC0_REG1_L1_Dest_addr$|^GPR\.pack\.OUTPUT_ADDR$|^ADC\.PAC\.(?:W|ZW|XY)$",
    ),
]

PATTERNS: list = []  # every pack effect needs function context; see extract()
DISABLE_CORE: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
def _norm_cfg_field(expr, thread=None):
    """``FIELD`` (a cfg_defines field name) -> ``CFG.FIELD`` (field granularity)."""
    name = compact(expr or "")
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None
    return Normalized(f"CFG.{name}", "field")


def _norm_cfg_word(expr, thread=None):
    """``FIELD`` -> ``CFG.FIELD`` at word granularity (the whole config word)."""
    name = compact(expr or "")
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None
    return Normalized(f"CFG.{name}", "word")


def _norm_state(expr, thread=None):
    """Family-defined packer state names (``PACK.BUSY``)."""
    name = compact(expr or "")
    return Normalized(name, "whole") if re.fullmatch(r"PACK\.[A-Z_]+", name) else None


def _norm_cb(expr, thread=None):
    """CB interface member -> ``SW.cb_interface.<member>``."""
    name = compact(expr or "")
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None
    return Normalized(f"SW.cb_interface.{name}", "slot")


NORMALIZERS = {
    "pack.cfg_field": _norm_cfg_field,
    "pack.cfg_word": _norm_cfg_word,
    "pack.state": _norm_state,
    "pack.cb": _norm_cb,
}


# --------------------------------------------------------------------------- #
# Hooks: roles and threads
# --------------------------------------------------------------------------- #
def role(fn, default: str) -> str | None:
    if not in_scope(fn.path):
        return None
    return ROLE_OVERRIDES.get(fn.name)


def thread(fn, default: str) -> str | None:
    if in_scope(fn.path) and default == "any":
        return "pack"
    return None


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
_INSN = re.compile(r"TT(?:I)?_(?!OP_)[A-Z0-9_]+")
_CFG_SYMBOL = re.compile(r"\b(?:ckernel::)?([A-Za-z_]\w*?)_(ADDR32|SHAMT|MASK)\b")
_QUALIFIED = re.compile(r"\b(?:[A-Za-z_]\w*::)+([A-Za-z_]\w*)\b")


def _fmt(value) -> str:
    """Canonical text of a derived value: ints as decimal below 10, else hex."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value) if 0 <= value < 10 else hex(value)
    if isinstance(value, (_Word, _Garbage)):
        return ""  # a whole union word is rendered through its fields, never as text
    return compact(str(value))


def _split_literals(condition: str | None) -> list[tuple[str, bool]]:
    """``a && !(b) && c`` -> [(a, True), (b, False), (c, True)] (top-level ``&&`` only)."""
    text = compact(condition or "")
    if not text:
        return []
    parts: list[str] = []
    depth = 0
    start = 0
    position = 0
    while position < len(text):
        char = text[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith("&&", position):
            parts.append(text[start:position])
            start = position + 2
            position += 2
            continue
        position += 1
    parts.append(text[start:])
    result = []
    for part in parts:
        part = compact(part)
        if not part:
            continue
        negated = re.fullmatch(r"!\s*\((.*)\)", part)
        if negated and scan.match_close(part, part.index("("), "()") == len(part) - 1:
            result.append((compact(negated.group(1)), False))
        else:
            result.append((part, True))
    return result


def _literal_text(atom: str, polarity: bool) -> str:
    return atom if polarity else f"!({atom})"


_EQUALITY = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*==\s*((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*|-?\d+)\s*$"
)


def _equality(atom: str) -> tuple[str, str] | None:
    match = _EQUALITY.match(atom)
    return (match.group(1), match.group(2)) if match else None


def _feasible(assignment: dict[str, bool]) -> bool:
    """Reject assignments where ``X == A`` and ``X == B`` (A != B) are both true."""
    seen: dict[str, str] = {}
    for atom, value in assignment.items():
        if not value:
            continue
        equality = _equality(atom)
        if equality is None:
            continue
        lhs, rhs = equality
        if lhs in seen and seen[lhs] != rhs:
            return False
        seen[lhs] = rhs
    return True


def _simplify_cube(cube: dict[str, bool]) -> dict[str, bool]:
    """Drop ``!(X == B)`` literals implied by a positive ``X == A`` in the same cube."""
    positive = {}
    for atom, value in cube.items():
        equality = _equality(atom)
        if value and equality is not None:
            positive[equality[0]] = equality[1]
    result = {}
    for atom, value in cube.items():
        equality = _equality(atom)
        if (
            not value
            and equality is not None
            and equality[0] in positive
            and positive[equality[0]] != equality[1]
        ):
            continue
        result[atom] = value
    return result


def _forced(atom: str, partial: dict[str, bool]) -> bool | None:
    """``X == B`` is forced false when the partial assignment already has ``X == A`` true."""
    equality = _equality(atom)
    if equality is None:
        return None
    for other, value in partial.items():
        if not value or other == atom:
            continue
        other_equality = _equality(other)
        if (
            other_equality is not None
            and other_equality[0] == equality[0]
            and other_equality[1] != equality[1]
        ):
            return False
    return None


def _decide(
    atoms: list[str], table: dict[tuple, str], fixed: dict[str, bool]
) -> dict[str, list[dict[str, bool]]]:
    """Ordered decision-tree cover: ``{value: [cube, ...]}`` with pairwise disjoint cubes.

    ``table`` maps a full assignment (sorted items) to the value under it; atoms are split in
    the given (source nesting) order and an atom whose two branches give the same result is
    dropped, so each cube keeps only the literals that decide the value. Disjoint cubes fold
    cleanly when a caller binds the template parameters (exactly one cube survives).
    """

    def recurse(index: int, partial: dict[str, bool]):
        if index == len(atoms):
            value = table.get(tuple(sorted(partial.items())))
            return {} if value is None else {value: [{}]}
        atom = atoms[index]
        if atom in partial:
            return recurse(index + 1, partial)
        forced = _forced(atom, partial)
        if forced is not None:
            return recurse(index + 1, {**partial, atom: forced})
        when_true = recurse(index + 1, {**partial, atom: True})
        when_false = recurse(index + 1, {**partial, atom: False})
        if not when_true or when_true == when_false:
            return when_false
        if not when_false:
            return when_true
        merged: dict[str, list[dict[str, bool]]] = {}
        for polarity, branch in ((True, when_true), (False, when_false)):
            for value, cubes in branch.items():
                merged.setdefault(value, []).extend(
                    {atom: polarity, **cube} for cube in cubes
                )
        return merged

    return recurse(0, dict(fixed))


# --------------------------------------------------------------------------- #
# Config-register arithmetic bound to one tree's tables
# --------------------------------------------------------------------------- #
class _Cfg:
    def __init__(self, tables):
        self.tables = tables
        self.fields = tables.cfg.fields
        self.words = tables.cfg.words

    def _symbol(self, match: re.Match) -> str:
        item = self.fields.get(match.group(1))
        if item is None:
            return match.group(0)
        kind = match.group(2)
        value = (
            item.addr32
            if kind == "ADDR32"
            else item.shamt if kind == "SHAMT" else item.mask
        )
        return str(value)

    def int(self, expr) -> int | None:
        if expr is None:
            return None
        if isinstance(expr, bool):
            return int(expr)
        if isinstance(expr, int):
            return expr
        text = compact(str(expr))
        if not text:
            return None
        text = _CFG_SYMBOL.sub(self._symbol, text)
        text = re.sub(r"\bLOWER_HALFWORD\s*\(", "0xFFFF & (", text)
        value = self.tables.int_value(text)
        if value is None and "::" in text:
            value = self.tables.int_value(_QUALIFIED.sub(lambda m: m.group(1), text))
        return value

    def word(self, expr: str) -> int | None:
        """Config-space word address of an ``X_ADDR32 [+ k]`` / integer expression."""
        return self.int(expr)

    def word_fields(self, addr32: int):
        return [self.fields[name] for name in self.words.get(("cfg", addr32), [])]

    def name_of(self, addr32: int) -> str | None:
        names = self.words.get(("cfg", addr32))
        return names[0] if names else None


# --------------------------------------------------------------------------- #
# Bitfield unions of cpack_common.h
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _UField:
    name: str
    word: int
    shamt: int
    width: int


@dataclass(frozen=True)
class _Union:
    name: str
    words: int
    array: bool  # ``val[N]`` versus scalar ``val``
    struct_member: str  # ``f`` / ``r``
    fields: tuple[_UField, ...]

    def field(self, name: str) -> _UField | None:
        for item in self.fields:
            if item.name == name:
                return item
        return None


_LAYOUT_CACHE: dict[str, dict[str, _Union]] = {}


def _layouts(tables, cfg: _Cfg) -> dict[str, _Union]:
    """Parse the ``typedef struct/union`` bitfield layouts of cpack_common.h (cached per tree)."""
    root = str(Path(tables.repo_root).resolve())
    cached = _LAYOUT_CACHE.get(root)
    if cached is not None:
        return cached
    path = Path(root) / CPACK_COMMON
    unions: dict[str, _Union] = {}
    if path.is_file():
        text = scan.mask_comments_and_strings(
            path.read_text(encoding="utf-8", errors="replace")
        )
        structs: dict[str, tuple[_UField, ...]] = {}
        for match in re.finditer(r"typedef\s+struct\s*\{", text):
            close = scan.match_close(text, match.end() - 1, "{}")
            if close is None:
                continue
            name = re.match(r"\s*([A-Za-z_]\w*)\s*;", text[close + 1 :])
            if name is None:
                continue
            fields: list[_UField] = []
            word = bit = 0
            ok = True
            for member in re.finditer(
                r"(?:std::)?u?int32_t\s+([A-Za-z_]\w*)\s*(?::\s*([^;]+))?;",
                text[match.end() : close],
            ):
                width = cfg.int(member.group(2)) if member.group(2) else 32
                if width is None or not 0 < width <= 32:
                    ok = False
                    break
                if bit + width > 32:
                    word, bit = word + 1, 0
                fields.append(_UField(member.group(1), word, bit, width))
                bit += width
                if bit == 32:
                    word, bit = word + 1, 0
            if ok:
                structs[name.group(1)] = tuple(fields)
        for match in re.finditer(r"typedef\s+union\s*\{", text):
            close = scan.match_close(text, match.end() - 1, "{}")
            if close is None:
                continue
            name = re.match(r"\s*([A-Za-z_]\w*)\s*;", text[close + 1 :])
            body = text[match.end() : close]
            val = re.search(
                r"(?:std::)?u?int32_t\s+val\s*(?:\[\s*(\d+)\s*\])?\s*;", body
            )
            member = re.search(
                r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;",
                re.sub(r"(?:std::)?u?int32_t\s+val[^;]*;", "", body),
            )
            if (
                name is None
                or val is None
                or member is None
                or member.group(1) not in structs
            ):
                continue
            layout = structs[member.group(1)]
            words = int(val.group(1)) if val.group(1) else 1
            unions[name.group(1)] = _Union(
                name.group(1), words, val.group(1) is not None, member.group(2), layout
            )
    _LAYOUT_CACHE[root] = unions
    return unions


class _Garbage:
    """An uninitialized value (declared without initializer)."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<garbage>"


GARBAGE = _Garbage()


@dataclass
class _Word:
    """One 32-bit word of a union variable: base value plus per-bitfield overrides."""

    layout: _Union
    index: int
    base: object  # int | str | GARBAGE | None
    overrides: dict = field(default_factory=dict)  # field name -> int | str | None

    def bits(self, shamt: int, width: int, cfg: _Cfg):
        """Value of bits [shamt, shamt+width) of this word (int | str | None)."""
        pieces = []
        covered = 0
        for item in self.layout.fields:
            if item.word != self.index:
                continue
            low, high = max(item.shamt, shamt), min(
                item.shamt + item.width, shamt + width
            )
            if low >= high:
                continue
            if item.name in self.overrides:
                value = self.overrides[item.name]
            else:
                value = _extract(self.base, item.shamt, item.width, cfg)
            if item.shamt == shamt and item.width == width:
                return value
            pieces.append((item, low, high, value))
            covered |= ((1 << (high - low)) - 1) << low
        total = 0
        for item, low, high, value in pieces:
            number = cfg.int(value) if not isinstance(value, _Garbage) else None
            if number is None:
                return None
            total |= ((number >> (low - item.shamt)) & ((1 << (high - low)) - 1)) << (
                low - shamt
            )
        gaps = (((1 << width) - 1) << shamt) & ~covered
        if gaps:
            rest = _extract(self.base, 0, 32, cfg)
            number = cfg.int(rest) if not isinstance(rest, _Garbage) else None
            if number is None:
                return None
            total |= (number & gaps) >> shamt
        return total


def _extract(value, shamt: int, width: int, cfg: _Cfg, sole: bool = False):
    """Bits [shamt, shamt+width) of a value (int | str | _Word | GARBAGE | None)."""
    if value is None or isinstance(value, _Garbage):
        return value
    if isinstance(value, _Word):
        return value.bits(shamt, width, cfg)
    number = cfg.int(value)
    if number is not None:
        return (number >> shamt) & ((1 << width) - 1)
    text = compact(str(value))
    terms = _shift_terms(text, cfg)
    if terms is not None:
        # field-packing idiom: every symbolic term is assumed to fit the field it is shifted to
        wmask = (1 << width) - 1
        constant = 0
        hit = None
        for term, position, extent in terms:
            number = cfg.int(term)
            if number is not None:
                constant |= ((number << position) >> shamt) & wmask
                continue
            if position == shamt and hit is None:
                hit = term
            elif position >= shamt + width:
                continue  # a higher field
            elif extent is not None and position + extent <= shamt:
                continue  # a lower named field that ends below this one
            else:
                return None  # symbolic bits that cannot be separated from this field
        if hit is None:
            return constant
        hit_term = next(item for item in terms if item[0] == hit and item[1] == shamt)
        if hit_term[2] is None and shamt + width < 32:
            hit = f"({hit}) & {_fmt(wmask)}"  # unnamed shift: the term may be wider than the field
        return hit if not constant else f"{hit} | {_fmt(constant)}"
    if shamt == 0 and sole:
        return text
    if shamt == 0:
        return f"({text}) & {_fmt((1 << width) - 1)}"
    return f"(({text}) >> {shamt}) & {_fmt((1 << width) - 1)}"


def _top_level_split(text: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    position = 0
    length = len(text)
    while position < length:
        char = text[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(operator, position):
            if operator == "|" and text.startswith("||", position):
                position += 2
                continue
            if operator == "<<" and text.startswith("<<=", position):
                position += 3
                continue
            parts.append(text[start:position])
            start = position + len(operator)
            position += len(operator)
            continue
        position += 1
    parts.append(text[start:])
    return [compact(p) for p in parts]


def _strip_parens(text: str) -> str:
    text = compact(text)
    while text.startswith("(") and scan.match_close(text, 0, "()") == len(text) - 1:
        text = compact(text[1:-1])
    return text


def _shift_terms(text: str, cfg: _Cfg) -> list[tuple[str, int, int | None]] | None:
    """``(a << X_SHAMT) | (b << 4) | 1`` -> [(a, shamt_X, width_X), (b, 4, None), (1, 0, None)].

    ``None`` when the text is not that form. A term shifted by ``X_SHAMT`` is taken to occupy
    exactly field X (its width is the third item: the field-packing idiom); a term shifted by a
    plain constant has unknown width.
    """
    body = _strip_parens(text)
    terms = _top_level_split(body, "|")
    result = []
    for term in terms:
        term = _strip_parens(term)
        pieces = _top_level_split(term, "<<")
        if len(pieces) == 1:
            if len(terms) == 1:
                return None
            if cfg.int(pieces[0]) is None and not re.fullmatch(
                r"[\w:.()\[\] ]+", pieces[0]
            ):
                return None
            result.append((pieces[0], 0, None))
            continue
        if len(pieces) != 2:
            return None
        position = cfg.int(pieces[1])
        if position is None:
            return None
        extent = None
        name = re.fullmatch(
            r"\(?\s*(?:ckernel::)?([A-Za-z_]\w*?)_SHAMT\s*\)?", pieces[1]
        )
        if name:
            item = cfg.fields.get(name.group(1))
            if item is not None and item.mask:
                extent = bin(item.mask >> item.shamt).count("1")
        result.append((_strip_parens(pieces[0]), position, extent))
    return result


# --------------------------------------------------------------------------- #
# Statement scan: local variable events, loops, replay-record windows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Event:
    offset: int
    var: str
    path: tuple[str, ...]
    op: str  # "=", "|=", ..., "decl" (= expr), "decl_init" ({..} or (..)), "decl0" (none), "taint"
    rhs: str
    literals: tuple[tuple[str, bool], ...]
    loop: bool = False  # inside a loop that could not be unrolled


@dataclass
class _Loop:
    start: int
    end: int
    var: str | None
    values: (
        tuple[int, ...] | None
    )  # unrolled iteration values, None when not unrollable


_DECL = re.compile(
    r"^(?P<quals>(?:(?:const|constexpr|static|volatile|inline)\s+)*)"
    r"(?P<type>(?:::)?[A-Za-z_][\w:]*(?:\s*<[^;=]*?>)?)\s*(?:[*&]+\s*|\s+)"
    r"(?P<name>[A-Za-z_]\w*)\s*(?P<rest>=.*|\{.*\}|\(.*\))?$",
    re.S,
)
_ASSIGN = re.compile(
    r"^(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]))*)\s*"
    r"(?P<op><<=|>>=|[+\-*/%&|^]?=)(?!=)\s*(?P<rhs>.*)$",
    re.S,
)
#: Callees known to take their arguments by value (a local passed to them stays unchanged).
_BY_VALUE_CALLEES = frozenset(
    {"cfg_reg_rmw_tensix", "cfg_rmw", "cfg_rmw_gpr", "cfg_write", "LLK_ASSERT"}
)

_NOT_TYPES = frozenset(
    {
        "return",
        "else",
        "case",
        "goto",
        "throw",
        "delete",
        "new",
        "typedef",
        "using",
        "do",
        "if",
        "for",
        "while",
        "switch",
    }
)


def _statements(masked: str):
    """(start, end) spans of statements at paren depth 0, splitting at ``;`` and block braces."""
    spans = []
    depth = 0
    init_depth = 0
    start = 0
    position = 0
    length = len(masked)
    while position < length:
        char = masked[position]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and char == "{":
            before = masked[start:position].rstrip()
            last = before[-1:] if before else ""
            word = re.search(r"([A-Za-z_]\w*)$", before)
            is_block = (
                not before
                or last in ");{}"
                or (word is not None and word.group(1) in ("else", "do", "try"))
            )
            if is_block and init_depth == 0:
                if masked[start:position].strip():
                    spans.append((start, position))
                start = position + 1
            else:
                init_depth += 1
        elif depth == 0 and char == "}":
            if init_depth:
                init_depth -= 1
            else:
                if masked[start:position].strip():
                    spans.append((start, position))
                start = position + 1
        elif depth == 0 and init_depth == 0 and char == ";":
            spans.append((start, position))
            start = position + 1
        position += 1
    if masked[start:].strip():
        spans.append((start, length))
    return spans


def _strip_control(text: str) -> tuple[str, int]:
    """Remove leading ``else`` / ``if (..)`` / ``while (..)`` headers; return (rest, skipped)."""
    skipped = 0
    while True:
        stripped = text.lstrip()
        skipped += len(text) - len(stripped)
        text = stripped
        match = re.match(r"else\b", text)
        if match:
            text = text[match.end() :]
            skipped += match.end()
            continue
        match = re.match(r"(?:if(?:\s+constexpr)?|while|switch)\s*\(", text)
        if match:
            close = scan.match_close(text, match.end() - 1, "()")
            if close is None:
                return "", skipped
            text = text[close + 1 :]
            skipped += close + 1
            continue
        return text, skipped


class _Frame:
    """Events of the locals of one function body and helpers to evaluate expressions."""

    MAX_ATOMS = 7
    MAX_DEPTH = 12

    def __init__(self, fn, cfg: _Cfg, layouts: dict[str, _Union]):
        self.fn = fn
        self.cfg = cfg
        self.layouts = layouts
        self.masked = fn.masked_body
        self.params = set(fn.param_names)
        self.loops = self._loops()
        self.types: dict[str, str] = {}
        self.consts: set[str] = set()
        self.events: dict[str, list[_Event]] = {}
        self._literal_cache: dict[int, tuple] = {}
        self._parse()
        self._taint_by_reference()

    # ---- conditions -------------------------------------------------------- #
    def literals(self, offset: int) -> tuple[tuple[str, bool], ...]:
        cached = self._literal_cache.get(offset)
        if cached is None:
            cached = tuple(_split_literals(self.fn.condition_at(offset)))
            self._literal_cache[offset] = cached
        return cached

    # ---- loops ------------------------------------------------------------- #
    def _loops(self) -> list[_Loop]:
        loops = []
        masked = self.masked
        for match in re.finditer(r"(?<![\w.])(for|while)\s*\(", masked):
            opening = match.end() - 1
            close = scan.match_close(masked, opening, "()")
            if close is None:
                continue
            after = close + 1
            while after < len(masked) and masked[after].isspace():
                after += 1
            if after < len(masked) and masked[after] == "{":
                end = scan.match_close(masked, after, "{}")
                end = len(masked) if end is None else end
            else:
                end = scan.statement_end(masked, after)
            var = None
            values = None
            if match.group(1) == "for":
                header = compact(masked[opening + 1 : close])
                unroll = re.fullmatch(
                    r"(?:const\s+)?(?:[\w:]+\s+)?([A-Za-z_]\w*)\s*=\s*([^;]+);\s*\1\s*(<|<=)\s*([^;]+);\s*(?:\1\s*\+\+|\+\+\s*\1|\1\s*\+=\s*1)",
                    header,
                )
                if unroll:
                    var = unroll.group(1)
                    low, high = self.cfg.int(unroll.group(2)), self.cfg.int(
                        unroll.group(4)
                    )
                    if low is not None and high is not None:
                        high = high + 1 if unroll.group(3) == "<=" else high
                        if 0 <= high - low <= 8:
                            values = tuple(range(low, high))
            loops.append(_Loop(after, end, var, values))
        # ``switch`` bodies (case labels are not branch conditions) and ``do { } while`` bodies
        # are opaque: assignments inside them make the variable unknown afterwards
        for match in re.finditer(r"(?<![\w.])switch\s*\(", masked):
            close = scan.match_close(masked, match.end() - 1, "()")
            brace = masked.find("{", close) if close is not None else -1
            if brace >= 0 and not masked[close + 1 : brace].strip():
                end = scan.match_close(masked, brace, "{}")
                loops.append(
                    _Loop(brace, len(masked) if end is None else end, None, None)
                )
        for match in re.finditer(r"(?<![\w.])do\s*\{", masked):
            end = scan.match_close(masked, match.end() - 1, "{}")
            loops.append(
                _Loop(match.end() - 1, len(masked) if end is None else end, None, None)
            )
        loops.sort(key=lambda loop: (loop.start, loop.end))
        return loops

    def loop_at(self, offset: int) -> list[_Loop]:
        return [loop for loop in self.loops if loop.start <= offset < loop.end]

    # ---- events ------------------------------------------------------------ #
    def _parse(self) -> None:
        masked = self.masked
        for start, end in _statements(masked):
            raw = masked[start:end]
            text, skipped = _strip_control(raw)
            text = text.strip()
            if not text or re.match(r"(?:for|do|switch|case|default|return)\b", text):
                continue
            self._statement(compact(text), start + skipped)

    def _add(self, event: _Event) -> None:
        loops = self.loop_at(event.offset)
        expanded = [event]
        for loop in loops:
            if loop.values is None:
                expanded = [
                    _Event(e.offset, e.var, e.path, e.op, e.rhs, e.literals, True)
                    for e in expanded
                ]
                continue
            uses = loop.var and any(
                loop.var in identifiers(part) or re.search(rf"\b{loop.var}\b", part)
                for e in expanded
                for part in (e.rhs, *e.path)
            )
            if not uses:
                continue
            unrolled = []
            for e in expanded:
                for value in loop.values:
                    path = tuple(
                        re.sub(rf"\b{loop.var}\b", str(value), p) for p in e.path
                    )
                    rhs = re.sub(rf"\b{loop.var}\b", str(value), e.rhs)
                    unrolled.append(
                        _Event(e.offset, e.var, path, e.op, rhs, e.literals, e.loop)
                    )
            expanded = unrolled
        for item in expanded:
            bucket = self.events.setdefault(item.var, [])
            bucket.append(item)
            bucket.sort(key=lambda e: e.offset)

    def _statement(self, text: str, offset: int) -> None:
        literals = self.literals(offset)
        decl = _DECL.match(text)
        if (
            decl
            and decl.group("type").split("::")[-1] not in _NOT_TYPES
            and decl.group("name") not in _NOT_TYPES
        ):
            type_name = re.sub(
                r"^(?:ckernel::packer::|ckernel::|::)", "", compact(decl.group("type"))
            )
            name = decl.group("name")
            rest = compact(decl.group("rest") or "")
            if type_name in ("auto", "auto&") or name in self.params:
                return
            self.types[name] = type_name
            if re.search(r"\b(?:const|constexpr)\b", decl.group("quals") or ""):
                self.consts.add(name)
            if not rest:
                self._add(_Event(offset, name, (), "decl0", "", literals))
            elif rest.startswith("="):
                self._add(_Event(offset, name, (), "decl", compact(rest[1:]), literals))
            elif rest.startswith("{") or rest.startswith("("):
                self._add(_Event(offset, name, (), "decl_init", rest, literals))
            return
        assign = _ASSIGN.match(text)
        if assign:
            lhs = compact(assign.group("lhs"))
            base = re.match(r"[A-Za-z_]\w*", lhs).group(0)
            if base not in self.types:
                return
            path = tuple(
                part.strip("[] ").strip()
                for part in re.findall(
                    r"\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]", lhs[len(base) :]
                )
            )
            path = tuple(p.lstrip(".").strip() for p in path)
            self._add(
                _Event(
                    offset,
                    base,
                    path,
                    assign.group("op"),
                    compact(assign.group("rhs")),
                    literals,
                )
            )
            return
        for match in re.finditer(
            r"(?:\+\+|--)\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?:\+\+|--)", text
        ):
            name = match.group(1) or match.group(2)
            if name in self.types:
                self._add(_Event(offset, name, (), "taint", "", literals))

    def _taint_by_reference(self) -> None:
        """A non-const local passed whole (or by address) to a function may be modified there."""
        for call in self.fn.calls:
            if (
                call.is_macro
                or call.is_member
                or _INSN.fullmatch(call.name)
                or call.name.startswith("TT_OP_")
                or call.name in _BY_VALUE_CALLEES
            ):
                continue
            for arg in call.args:
                text = compact(arg)
                name = text[1:].strip() if text.startswith("&") else text
                if (
                    name in self.types
                    and name not in self.consts
                    and re.fullmatch(r"[A-Za-z_]\w*", name)
                ):
                    self._add(
                        _Event(
                            call.offset,
                            name,
                            (),
                            "taint",
                            "",
                            self.literals(call.offset),
                        )
                    )

    # ---- evaluation --------------------------------------------------------- #
    def atoms_for(self, texts, offset: int) -> list[str]:
        """Branch atoms that can change the value of ``texts`` evaluated at ``offset``."""
        seen_vars: set[str] = set()
        first: dict[str, tuple[int, int]] = {}
        pending = list(texts)
        while pending:
            text = pending.pop()
            for name in sorted(identifiers(text)):
                if name in seen_vars or name not in self.events:
                    continue
                seen_vars.add(name)
                for event in self.events[name]:
                    if event.offset >= offset:
                        continue
                    for position, (atom, _) in enumerate(event.literals):
                        rank = (event.offset, position)
                        first[atom] = min(first.get(atom, rank), rank)
                    pending.append(event.rhs)
                    pending.extend(event.path)
        return sorted(first, key=lambda atom: (first[atom], atom))

    def assignments(self, atoms: list[str], base: tuple[tuple[str, bool], ...]):
        """Feasible truth assignments of ``atoms`` consistent with the use's literals."""
        fixed = {atom: value for atom, value in base}
        free = [atom for atom in atoms if atom not in fixed]
        if len(free) > self.MAX_ATOMS:
            return None
        result = []
        for values in itertools.product((True, False), repeat=len(free)):
            assignment = dict(fixed)
            assignment.update(zip(free, values))
            if _feasible(assignment):
                result.append(assignment)
        return result

    @staticmethod
    def _holds(literals, assignment: dict[str, bool]) -> bool:
        return all(assignment.get(atom, None) == value for atom, value in literals)

    def evaluate(
        self, text: str, offset: int, assignment: dict[str, bool], depth: int = 0
    ):
        """Value of an expression at ``offset`` under ``assignment``: int | str | _Word | None."""
        if depth > self.MAX_DEPTH:
            return None
        text = compact(text)
        if not text:
            return None
        whole = re.fullmatch(
            r"([A-Za-z_]\w*)((?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]))*)", text
        )
        if whole and whole.group(1) in self.events:
            return self.var_value(
                whole.group(1), whole.group(2), offset, assignment, depth + 1
            )
        failed = False

        def replace(match: re.Match) -> str:
            nonlocal failed
            name = match.group(1)
            if name not in self.events:
                return match.group(0)
            value = self.var_value(
                name, match.group(2) or "", offset, assignment, depth + 1
            )
            if isinstance(value, _Word):
                value = value.bits(0, 32, self.cfg)
            if value is None or isinstance(value, _Garbage):
                failed = True
                return match.group(0)
            return _fmt(value) if isinstance(value, int) else f"({value})"

        substituted = re.sub(
            r"(?<![\w.:>])([A-Za-z_]\w*)((?:\s*(?:\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]))*)(?!\s*\()",
            replace,
            text,
        )
        if failed:
            return None
        number = self.cfg.int(substituted)
        return number if number is not None else compact(substituted)

    def _events_before(self, name: str, offset: int, assignment):
        return [
            event
            for event in self.events.get(name, ())
            if event.offset < offset and self._holds(event.literals, assignment)
        ]

    def var_value(self, name: str, chain: str, offset: int, assignment, depth: int):
        events = self._events_before(name, offset, assignment)
        if any(event.loop or event.op == "taint" for event in events):
            return None
        chain_parts = [
            p.lstrip(".").strip("[] ").strip()
            for p in re.findall(r"\.\s*[A-Za-z_]\w*|\[[^\[\]]*\]", chain or "")
        ]
        layout = self.layouts.get(self.types.get(name, "").split("::")[-1])
        if layout is None:
            if chain_parts:
                return None
            return self._scalar(events, assignment, depth)
        words = self._union(layout, events, assignment, depth)
        if words is None:
            return None
        if not chain_parts:
            return None
        if chain_parts[0] == "val":
            index = 0
            if layout.array:
                if len(chain_parts) < 2:
                    return None
                index = self.cfg.int(chain_parts[1])
                if index is None or not 0 <= index < layout.words:
                    return None
            return words[index]
        if chain_parts[0] == layout.struct_member and len(chain_parts) == 2:
            item = layout.field(chain_parts[1])
            if item is None:
                return None
            word = words[item.word]
            if item.name in word.overrides:
                return word.overrides[item.name]
            return _extract(word.base, item.shamt, item.width, self.cfg)
        return None

    def _scalar(self, events, assignment, depth):
        value = None
        defined = False
        for event in events:
            if event.path:
                return None
            if event.op == "decl0":
                value, defined = GARBAGE, True
            elif event.op in ("decl", "decl_init", "="):
                rhs = event.rhs
                braced = (
                    rhs.startswith("{")
                    and scan.match_close(rhs, 0, "{}") == len(rhs) - 1
                )
                ctor = event.op == "decl_init" and rhs.startswith("(")
                if braced or ctor:
                    inner = rhs[1:-1].strip()
                    if not inner or not re.fullmatch(r"[^{}=,]+", inner):
                        return None
                    rhs = inner
                value = self.evaluate(rhs, event.offset, assignment, depth)
                defined = True
                if value is None:
                    return None
            else:
                if not defined or value is None or isinstance(value, _Garbage):
                    return None
                operand = self.evaluate(event.rhs, event.offset, assignment, depth)
                if operand is None:
                    return None
                operator = event.op[:-1]
                combined = f"({_fmt(value)}) {operator} ({_fmt(operand)})"
                number = self.cfg.int(combined)
                value = number if number is not None else combined
        if not defined:
            return None
        return value

    def _union(self, layout: _Union, events, assignment, depth):
        words = [_Word(layout, index, GARBAGE) for index in range(layout.words)]
        for event in events:
            if event.op == "decl0":
                words = [_Word(layout, index, GARBAGE) for index in range(layout.words)]
            elif event.op in ("decl", "decl_init"):
                rhs = event.rhs
                match = re.fullmatch(r"\{\s*(?:\.val\s*=\s*)?([^{}]*)\}", rhs)
                if not match:
                    return None
                first = match.group(1).strip()
                value = (
                    self.evaluate(first, event.offset, assignment, depth)
                    if first
                    else 0
                )
                if value is None:
                    return None
                words = [
                    _Word(layout, index, value if index == 0 else 0)
                    for index in range(layout.words)
                ]
            elif event.op == "=" and event.path:
                if event.path[0] == "val":
                    index = 0
                    if layout.array:
                        if len(event.path) < 2:
                            return None
                        index = self.cfg.int(event.path[1])
                        if index is None or not 0 <= index < layout.words:
                            return None
                    value = self.evaluate(event.rhs, event.offset, assignment, depth)
                    if isinstance(value, _Word):
                        value = value.bits(0, 32, self.cfg)
                    words[index] = _Word(layout, index, value)
                elif event.path[0] == layout.struct_member and len(event.path) == 2:
                    item = layout.field(event.path[1])
                    if item is None:
                        return None
                    value = self.evaluate(event.rhs, event.offset, assignment, depth)
                    if isinstance(value, _Word):
                        value = value.bits(0, 32, self.cfg)
                    word = words[item.word]
                    overrides = dict(word.overrides)
                    overrides[item.name] = (
                        value & ((1 << item.width) - 1)
                        if isinstance(value, int) and not isinstance(value, bool)
                        else value
                    )
                    words[item.word] = _Word(layout, item.word, word.base, overrides)
                else:
                    return None
            else:
                return None
        return words


# --------------------------------------------------------------------------- #
# Deferred (replay-recorded) spans and pack-execution sites
# --------------------------------------------------------------------------- #
def _deferred_offsets(fn, cfg: _Cfg) -> tuple[list[tuple[int, int]], set[int]]:
    """Spans and call indices whose instructions are recorded into the replay buffer."""
    masked = fn.masked_body
    spans: list[tuple[int, int]] = []
    calls = list(enumerate(fn.calls))
    for _, call in calls:
        if call.name == "load_replay_buf" and not any(
            re.search(r"(?<!No)Exec\b", t) for t in call.template_args
        ):
            opening = masked.find("(", call.offset)
            if opening >= 0:
                close = scan.match_close(masked, opening, "()")
                if close is not None:
                    spans.append((opening, close))
    indices: set[int] = set()
    for position, call in calls:
        window = None
        if (
            call.name == "record"
            and call.qualifier.endswith("lltt")
            and not any(re.search(r"(?<!No)Exec\b", t) for t in call.template_args)
        ):
            window = cfg.int(call.args[1]) if len(call.args) > 1 else None
            window = -1 if window is None else window
        elif re.fullmatch(r"TTI?_REPLAY", call.name) and len(call.args) >= 4:
            load, execute = cfg.int(call.args[3]), cfg.int(call.args[2])
            if load and not execute:
                window = cfg.int(call.args[1])
                window = -1 if window is None else window
        if window is None:
            continue
        block_end = _block_end(masked, call.offset)
        remaining = window
        for later, other in calls[position + 1 :]:
            if other.offset > block_end:
                break
            if remaining == 0:
                break
            indices.add(later)
            if remaining > 0 and (
                _INSN.fullmatch(other.name) or other.name == "TT_INSN"
            ):
                remaining -= 1
    return spans, indices


def _block_end(masked: str, offset: int) -> int:
    """End offset of the innermost ``{}`` block containing ``offset`` (or the body end)."""
    depth = 0
    for position in range(offset, len(masked)):
        char = masked[position]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return position
            depth -= 1
    return len(masked)


def _in_spans(offset: int, spans) -> bool:
    return any(start < offset < end for start, end in spans)


def _is_mop_run(call) -> bool:
    if call.name == "run":
        if call.is_member:
            return bool(
                re.fullmatch(
                    r"ckernel_template|ckernel_unpack_template",
                    call.receiver_type or "",
                )
            )
        return call.qualifier.split("::")[-1:] == ["ckernel_template"]
    return bool(re.fullmatch(r"TTI?_MOP", call.name))


def _is_replay_exec(call, cfg: _Cfg) -> bool:
    if call.name == "replay" and call.qualifier.endswith("lltt"):
        return True
    if re.fullmatch(r"TTI?_REPLAY", call.name) and len(call.args) >= 4:
        load = cfg.int(call.args[3])
        return load == 0
    return False


# --------------------------------------------------------------------------- #
# extract(): every pack-family effect
# --------------------------------------------------------------------------- #
def extract(fn, ctx):
    if not in_scope(fn.path):
        return []
    cfg = _Cfg(ctx.tables)
    layouts = _layouts(ctx.tables, cfg)
    spans, deferred_calls = _deferred_offsets(fn, cfg)
    effects: list = []
    frame = None

    def get_frame() -> _Frame:
        nonlocal frame
        if frame is None:
            frame = _Frame(fn, cfg, layouts)
        return frame

    pack_thread = (fn.thread_hint or "pack") == "pack"
    for index, call in enumerate(fn.calls):
        deferred = (
            index in deferred_calls
            or _in_spans(call.offset, spans)
            or call.name.startswith("TT_OP_")
        )
        call_thread = call.thread or fn.thread_hint
        # ---- pack execution ------------------------------------------------- #
        if not deferred and (call_thread or "pack") == "pack" and pack_thread:
            kind = None
            if _is_mop_run(call):
                kind = "mop_run"
            elif re.fullmatch(r"TTI?_PACR", call.name):
                kind = "pacr"
            elif _is_replay_exec(call, cfg):
                kind = "replay_run"
            if kind is not None:
                effects.extend(_exec_effects(fn, ctx, cfg, call, kind))
        # ---- stalls ----------------------------------------------------------- #
        if (
            re.fullmatch(r"TTI?_STALLWAIT", call.name)
            and not deferred
            and len(call.args) >= 2
        ):
            if re.search(r"\bPACK\b", call.args[1]):
                effects.extend(
                    ctx.effects(
                        fn,
                        pattern_id="pack.stall",
                        resource_expr="PACK.BUSY",
                        normalizer="pack.state",
                        access="read",
                        value=f"stall={call.args[0]};wait={call.args[1]}",
                        line=call.line,
                        condition=call.condition,
                        detail="stallwait",
                    )
                )
        elif call.name == "tensix_sync" and not call.is_member:
            effects.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.stall",
                    resource_expr="PACK.BUSY",
                    normalizer="pack.state",
                    access="read",
                    value="",
                    line=call.line,
                    condition=call.condition,
                    detail="tensix_sync",
                )
            )
        # ---- GPR reads the core cannot name ----------------------------------- #
        if re.fullmatch(r"TTI?_STOREREG", call.name) and call.args:
            effects.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.storereg",
                    resource_expr=call.args[0],
                    normalizer="gpr",
                    access="read",
                    value="",
                    line=call.line,
                    condition=call.condition,
                    detail="deferred" if deferred else "",
                )
            )
        if (
            re.fullmatch(r"TTI?_WRCFG", call.name)
            and call.args
            and re.fullmatch(
                r"(?:ckernel::packer::)?get_packer_dest_offset_index\s*\(\s*\)",
                call.args[0],
            )
        ):
            effects.extend(_dest_offset_index_reads(fn, ctx, call))
        # ---- per-field CFG writes ------------------------------------------- #
        if deferred:
            continue
        if re.fullmatch(r"TTI?_WRCFG", call.name) and len(call.args) >= 3:
            effects.extend(_wrcfg_fields(fn, ctx, cfg, get_frame(), call, index))
        elif call.name == "TT_CFG_STORE" and len(call.args) >= 3:
            effects.extend(_cfg_store_call(fn, ctx, cfg, get_frame(), call))
        elif (
            call.name == "cfg_reg_rmw_tensix"
            and len(call.template_args) == 3
            and call.args
        ):
            effects.extend(_rmw_fields(fn, ctx, cfg, get_frame(), call))
    effects.extend(_cfg_assign_fields(fn, ctx, cfg, get_frame, spans))
    effects.extend(_cfg_reads(fn, ctx))
    effects.extend(_sw_reads(fn, ctx))
    effects.extend(_cb_state(fn, ctx))
    return effects


def _exec_effects(fn, ctx, cfg: _Cfg, call, kind: str):
    found = []
    common = dict(line=call.line, condition=call.condition, detail=kind)
    for name, granularity in PACK_EXEC_READS:
        if name not in cfg.fields:
            continue
        found.extend(
            ctx.effects(
                fn,
                pattern_id="pack.exec.read",
                resource_expr=name,
                normalizer=(
                    "pack.cfg_word" if granularity == "word" else "pack.cfg_field"
                ),
                access="read",
                value="",
                **common,
            )
        )
    for name in PACK_EXEC_ADC:
        found.extend(
            ctx.effects(
                fn,
                pattern_id="pack.exec.read",
                resource_expr=name,
                normalizer="literal",
                access="read",
                value="",
                granularity="slot",
                **common,
            )
        )
    if kind != "pacr":
        # a MOP may hold lltt::replay_insn words; a replay run reads the buffer by definition
        found.extend(
            ctx.effects(
                fn,
                pattern_id="pack.exec.read",
                resource_expr="REPLAY",
                normalizer="literal",
                access="read",
                value="",
                granularity="range",
                **common,
            )
        )
    found.extend(
        ctx.effects(
            fn,
            pattern_id="pack.exec.busy",
            resource_expr="PACK.BUSY",
            normalizer="pack.state",
            access="write",
            value=kind,
            **common,
        )
    )
    return found


_DEST_OFFSET_INDEX = re.compile(
    r"get_packer_dest_offset_index\s*\(\s*\)\s*\{\s*return\s*\(?\s*([A-Za-z_]\w*)\s*\?\s*([\w:]+)\s*:\s*([\w:]+)\s*\)?\s*;"
)
_DEST_INDEX_CACHE: dict[str, tuple[str, str, str] | None] = {}


def _dest_offset_index_reads(fn, ctx, call):
    root = str(Path(ctx.tables.repo_root).resolve())
    if root not in _DEST_INDEX_CACHE:
        path = Path(root) / CPACK_COMMON
        found = None
        if path.is_file():
            text = scan.mask_comments_and_strings(
                path.read_text(encoding="utf-8", errors="replace")
            )
            match = _DEST_OFFSET_INDEX.search(text)
            if match:
                found = (match.group(1), match.group(2), match.group(3))
        _DEST_INDEX_CACHE[root] = found
    info = _DEST_INDEX_CACHE[root]
    if info is None:
        return []
    variable, when_true, when_false = info
    found = []
    for gpr, polarity in ((when_true, True), (when_false, False)):
        found.extend(
            ctx.effects(
                fn,
                pattern_id="pack.gpr_read.dest_offset",
                resource_expr=gpr,
                normalizer="gpr",
                access="read",
                value="",
                line=call.line,
                condition=combine_conditions(
                    call.condition, _literal_text(variable, polarity)
                ),
            )
        )
    found.extend(
        ctx.effects(
            fn,
            pattern_id="pack.gpr_read.dest_offset",
            resource_expr=variable,
            normalizer="sw",
            access="read",
            value="",
            line=call.line,
            condition=call.condition,
        )
    )
    return found


# ---- per-field writes ------------------------------------------------------- #
def _emit_fields(
    fn,
    ctx,
    cfg: _Cfg,
    frame: _Frame,
    *,
    offset,
    line,
    fields,
    evaluate,
    src,
    base_condition,
    sources,
    extra_atoms=(),
):
    """Emit ``pack.cfg_field`` effects for ``fields`` (CfgField list).

    ``evaluate(assignment) -> {field name: value}`` computes the field values for one path;
    ``sources`` are the expression texts whose locals decide which paths exist, and
    ``extra_atoms`` further branch atoms (e.g. of the SETDMAREGs that fed a GPR).
    """
    base = tuple(_split_literals(base_condition))
    atoms = frame.atoms_for(sources, offset)
    for atom in extra_atoms:
        if atom not in atoms:
            atoms.append(atom)
    assignments = frame.assignments(atoms, base)
    results: list[tuple[dict, dict]] = []
    if assignments is not None:
        for assignment in assignments:
            results.append((assignment, evaluate(assignment)))
    if not results or all(
        all(v is None for v in values.values()) for _, values in results
    ):
        return [], False
    fixed = {atom for atom, _ in base}
    found = []

    def whole_word(values):
        if not fields or not all(isinstance(values.get(f.name), int) for f in fields):
            return None
        total = 0
        for item in fields:
            total |= (values[item.name] << item.shamt) & item.mask
        return total

    word_constant = None
    if (
        len(
            {
                tuple(sorted((k, _fmt(v)) for k, v in vals.items()))
                for _, vals in results
            }
        )
        == 1
    ):
        word_constant = whole_word(results[0][1])
    # array-like words (TILE_ROW_SET_MAPPING_k_row_set_mapping_0..15): one word-granular effect
    # carrying the whole-word value instead of 16 two-bit fields
    array = len(fields) > 4 and len({re.sub(r"\d+$", "", f.name) for f in fields}) == 1
    if array:
        targets = [(fields[0].name, "pack.cfg_word")]
        results = [
            (assignment, {fields[0].name: whole_word(values)})
            for assignment, values in results
        ]
        word_constant = None
    else:
        targets = [(item.name, "pack.cfg_field") for item in fields]
    fixed_values = {atom: value for atom, value in base}
    for name, normalizer in targets:
        table = {
            tuple(sorted(assignment.items())): _fmt(values.get(name))
            for assignment, values in results
        }
        rows = []
        for value_text, cubes in sorted(_decide(atoms, table, fixed_values).items()):
            for cube in cubes:
                rows.append(
                    (
                        value_text,
                        _simplify_cube(
                            {a: v for a, v in cube.items() if a not in fixed}
                        ),
                    )
                )
        if len(rows) > 8:
            rows = [("", {})]
        for value_text, cube in rows:
            ordered = [atom for atom in atoms if atom in cube]
            condition = combine_conditions(
                base_condition, *(_literal_text(atom, cube[atom]) for atom in ordered)
            )
            detail = f"src={src}"
            if word_constant is not None:
                # rmw: only the masked bits are written, so the constant is the masked data
                detail += f";{'data' if src == 'rmw' else 'word'}={_fmt(word_constant)}"
            found.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.cfg_field",
                    resource_expr=name,
                    normalizer=normalizer,
                    access="write",
                    value=value_text,
                    line=line,
                    condition=condition,
                    detail=detail,
                )
            )
    return found, True


def _gpr_events(fn, ctx, cfg: _Cfg, frame: _Frame):
    """SETDMAREG / regfile[] / DMA-ALU writes per GPR half, in body order."""
    events = []
    for index, call in enumerate(fn.calls):
        if call.name.startswith("TT_OP_"):
            continue
        if re.fullmatch(r"TTI?_SETDMAREG", call.name) and len(call.args) >= 4:
            for item in ctx.tables.normalize("gpr", call.args[3], "pack"):
                half = "hi" if item.detail == "HI_16" else "lo"
                events.append(
                    (
                        call.offset,
                        index,
                        item.name,
                        half,
                        call.args[1],
                        frame.literals(call.offset),
                    )
                )
        elif (
            re.fullmatch(r"TTI?_(?:ADD|SUB|MUL|BITWOP|SHIFT|CMP)DMAREG", call.name)
            and len(call.args) >= 2
        ):
            for item in ctx.tables.normalize("gpr", call.args[1], "pack"):
                events.append(
                    (
                        call.offset,
                        index,
                        item.name,
                        "lo",
                        None,
                        frame.literals(call.offset),
                    )
                )
                events.append(
                    (
                        call.offset,
                        index,
                        item.name,
                        "hi",
                        None,
                        frame.literals(call.offset),
                    )
                )
        elif re.fullmatch(r"TTI?_RDCFG", call.name) and call.args:
            for item in ctx.tables.normalize("gpr", call.args[0], "pack"):
                events.append(
                    (
                        call.offset,
                        index,
                        item.name,
                        "lo",
                        None,
                        frame.literals(call.offset),
                    )
                )
                events.append(
                    (
                        call.offset,
                        index,
                        item.name,
                        "hi",
                        None,
                        frame.literals(call.offset),
                    )
                )
    for match in re.finditer(r"(?<![\w.>])regfile\s*\[", fn.masked_body):
        close = scan.match_close(fn.masked_body, match.end() - 1, "[]")
        if close is None:
            continue
        operator = re.match(r"\s*=(?!=)", fn.masked_body[close + 1 :])
        if operator is None:
            continue
        end = scan.statement_end(fn.masked_body, close + 1 + operator.end())
        rhs = compact(fn.masked_body[close + 1 + operator.end() : end])
        for item in ctx.tables.normalize(
            "gpr", fn.masked_body[match.end() : close], "pack"
        ):
            events.append(
                (
                    match.start(),
                    -1,
                    item.name,
                    "lo",
                    ("whole", rhs),
                    frame.literals(match.start()),
                )
            )
            events.append(
                (
                    match.start(),
                    -1,
                    item.name,
                    "hi",
                    ("whole", rhs),
                    frame.literals(match.start()),
                )
            )
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _half_source(value_text, half):
    """(expr, kind): kind lo = bits 0-15 of expr, hi = bits 16-31 of expr, imm = 16-bit immediate."""
    if isinstance(value_text, tuple):
        return value_text[1], half  # regfile whole-word write
    text = compact(value_text)
    match = re.fullmatch(r"LOWER_HALFWORD\s*\((.*)\)", text)
    if match and scan.match_close(text, text.index("("), "()") == len(text) - 1:
        return compact(match.group(1)), "lo"
    match = re.fullmatch(r"UPPER_HALFWORD\s*\((.*)\)", text)
    if match and scan.match_close(text, text.index("("), "()") == len(text) - 1:
        return compact(match.group(1)), "hi"
    return text, "imm"


def _wrcfg_fields(fn, ctx, cfg: _Cfg, frame: _Frame, call, index):
    gpr_arg, mode_arg, addr_arg = call.args[0], call.args[1], call.args[2]
    mode = cfg.int(mode_arg)
    if (
        mode is None
    ):  # p_cfg::WRCFG_32b / WRCFG_128b when ckernel_instr_params.h is not scanned
        mode = {"WRCFG_32b": 0, "WRCFG_128b": 1}.get(
            compact(mode_arg).rsplit("::", 1)[-1]
        )
    addr = cfg.word(addr_arg)
    if mode != 0 or addr is None:
        return []
    fields = cfg.word_fields(addr)
    if not fields:
        return []
    names = ctx.tables.normalize("gpr", gpr_arg, "pack")
    if len(names) != 1 or not names[0].resolved:
        return []
    gpr = names[0].name
    events = [
        e
        for e in _gpr_events(fn, ctx, cfg, frame)
        if e[2] == gpr and (e[0], e[1]) < (call.offset, index)
    ]
    sources = [e[4][1] if isinstance(e[4], tuple) else (e[4] or "") for e in events]

    def halves(assignment):
        result = {}
        for half in ("lo", "hi"):
            chosen = None
            for event in events:
                if event[3] == half and frame._holds(event[5], assignment):
                    chosen = event
            if chosen is None:
                fixed = GPR_FIXED_HALVES.get((gpr, half))
                result[half] = (fixed, "imm") if fixed is not None else None
            elif chosen[4] is None:
                result[half] = None  # computed by a DMA ALU op / RDCFG
            elif any(
                loop.values is None and not (loop.start <= call.offset < loop.end)
                for loop in frame.loop_at(chosen[0])
            ):
                result[half] = None  # last iteration of a loop the WRCFG is outside of
            else:
                result[half] = _half_source(chosen[4], half)
        return result

    def evaluate(assignment):
        parts = halves(assignment)
        values = {}
        resolved = {}
        for half in ("lo", "hi"):
            if parts[half] is None:
                resolved[half] = None
                continue
            expr, kind = parts[half]
            value = (
                expr
                if isinstance(expr, int)
                else frame.evaluate(expr, call.offset, assignment)
            )
            resolved[half] = (value, kind)
        whole = _combine_halves(resolved.get("lo"), resolved.get("hi"), cfg)
        for item in fields:
            width = bin(item.mask >> item.shamt).count("1") if item.mask else 32
            low, high = item.shamt, item.shamt + width
            value = None
            if high <= 16 and resolved["lo"] is not None:
                source, kind = resolved["lo"]
                value = _extract(source, low + 16 if kind == "hi" else low, width, cfg)
            elif low >= 16 and resolved["hi"] is not None:
                source, kind = resolved["hi"]
                value = _extract(source, low if kind == "hi" else low - 16, width, cfg)
            elif whole is not None:
                value = _extract(whole, low, width, cfg, len(fields) == 1)
            values[item.name] = None if isinstance(value, _Garbage) else value
        return values

    found, _ = _emit_fields(
        fn,
        ctx,
        cfg,
        frame,
        offset=call.offset,
        line=call.line,
        fields=fields,
        evaluate=evaluate,
        src=f"wrcfg;gpr={gpr.rsplit('.', 1)[-1]}",
        base_condition=call.condition,
        sources=sources,
        extra_atoms=[atom for event in events for atom, _ in event[5]],
    )
    return found


def _half_bits(number: int, kind: str) -> int:
    """The 16 bits a SETDMAREG half receives: ``hi`` = bits 16-31 of the source, else bits 0-15."""
    return (number >> 16) & 0xFFFF if kind == "hi" else number & 0xFFFF


def _combine_halves(lo, hi, cfg: _Cfg):
    """Whole 32-bit value from two resolved halves, when both come from one expression."""
    if lo is None or hi is None:
        return None
    (lo_value, lo_kind), (hi_value, hi_kind) = lo, hi
    if isinstance(lo_value, _Word):
        lo_value = lo_value.bits(0, 32, cfg)
    if isinstance(hi_value, _Word):
        hi_value = hi_value.bits(0, 32, cfg)
    if lo_value is None or hi_value is None:
        return None
    lo_int, hi_int = cfg.int(lo_value), cfg.int(hi_value)
    if lo_int is not None and hi_int is not None:
        return (_half_bits(hi_int, hi_kind) << 16) | _half_bits(lo_int, lo_kind)
    if lo_kind == "lo" and hi_kind == "hi":
        if compact(str(lo_value)) == compact(str(hi_value)):
            return lo_value
        terms = _top_level_split(_strip_parens(str(hi_value)), "|")
        others = [t for t in terms if _strip_parens(t) != _strip_parens(str(lo_value))]
        if len(others) == len(terms) - 1 and all(
            cfg.int(t) is not None and cfg.int(t) & 0xFFFF == 0 for t in others
        ):
            return hi_value
    return None


def _store_fields(
    fn,
    ctx,
    cfg: _Cfg,
    frame: _Frame,
    addr_text,
    value_text,
    offset,
    line,
    src,
    condition=None,
):
    addr = cfg.word(addr_text)
    if addr is None:
        return []
    fields = cfg.word_fields(addr)
    if not fields:
        return []

    def evaluate(assignment):
        value = frame.evaluate(value_text, offset, assignment)
        result = {}
        for item in fields:
            width = bin(item.mask >> item.shamt).count("1") if item.mask else 32
            extracted = _extract(value, item.shamt, width, cfg, len(fields) == 1)
            result[item.name] = None if isinstance(extracted, _Garbage) else extracted
        return result

    found, _ = _emit_fields(
        fn,
        ctx,
        cfg,
        frame,
        offset=offset,
        line=line,
        fields=fields,
        evaluate=evaluate,
        src=src,
        base_condition=condition if condition is not None else fn.condition_at(offset),
        sources=[value_text],
    )
    return found


def _unrolled(
    frame: _Frame, offset: int, index_text: str, value_text: str
) -> list[tuple[str, str]]:
    """Index/value variants of a store inside unrollable loops whose variable the index uses."""
    variants = [(index_text, value_text)]
    for loop in frame.loop_at(offset):
        if loop.values is None or not loop.var:
            continue
        if re.search(rf"\b{loop.var}\b", index_text):
            variants = [
                (
                    re.sub(rf"\b{loop.var}\b", str(v), i),
                    re.sub(rf"\b{loop.var}\b", str(v), t),
                )
                for i, t in variants
                for v in loop.values
            ]
    return variants


def _cfg_store_call(fn, ctx, cfg: _Cfg, frame: _Frame, call):
    """``TT_CFG_STORE(cfg, ADDR32 (+ k), value)`` (folded tree's RISC MMIO store)."""
    found = []
    for index_variant, value_variant in _unrolled(
        frame, call.offset, call.args[1], call.args[2]
    ):
        found.extend(
            _store_fields(
                fn,
                ctx,
                cfg,
                frame,
                index_variant,
                value_variant,
                call.offset,
                call.line,
                "mmio",
                condition=call.condition or "",
            )
        )
    return found


def _cfg_assign_fields(fn, ctx, cfg: _Cfg, get_frame, spans):
    """``cfg[ADDR32 (+ k)] = value`` stores (RISC MMIO), loop-unrolled when the index uses the loop var."""
    masked = fn.masked_body
    found = []
    for match in re.finditer(r"(?<![\w.>])cfg\s*\[", masked):
        if _in_spans(match.start(), spans):
            continue
        close = scan.match_close(masked, match.end() - 1, "[]")
        if close is None:
            continue
        operator = re.match(r"\s*=(?!=)", masked[close + 1 :])
        if operator is None:
            continue
        end = scan.statement_end(masked, close + 1 + operator.end())
        index_text = compact(masked[match.end() : close])
        value_text = compact(masked[close + 1 + operator.end() : end])
        frame = get_frame()
        for index_variant, value_variant in _unrolled(
            frame, match.start(), index_text, value_text
        ):
            found.extend(
                _store_fields(
                    fn,
                    ctx,
                    cfg,
                    frame,
                    index_variant,
                    value_variant,
                    match.start(),
                    fn.line_at(match.start()),
                    "mmio",
                )
            )
    return found


def _rmw_fields(fn, ctx, cfg: _Cfg, frame: _Frame, call):
    addr_text, shamt_text, mask_text = call.template_args
    addr = cfg.word(addr_text)
    shamt = cfg.int(shamt_text)
    mask = cfg.int(mask_text)
    if mask is None:
        mask = frame.evaluate(mask_text, call.offset, dict(frame.literals(call.offset)))
        mask = mask if isinstance(mask, int) else None
    if addr is None or shamt is None or mask is None:
        return []
    fields = [item for item in cfg.word_fields(addr) if item.mask & mask]
    if not fields:
        return []
    data_text = call.args[0]

    def evaluate(assignment):
        data = frame.evaluate(data_text, call.offset, assignment)
        result = {}
        for item in fields:
            width = bin(item.mask >> item.shamt).count("1") if item.mask else 32
            position = item.shamt - shamt
            if position < 0:
                result[item.name] = None
                continue
            extracted = _extract(data, position, width, cfg, len(fields) == 1)
            result[item.name] = None if isinstance(extracted, _Garbage) else extracted
        return result

    found, emitted = _emit_fields(
        fn,
        ctx,
        cfg,
        frame,
        offset=call.offset,
        line=call.line,
        fields=fields,
        evaluate=evaluate,
        src="rmw",
        base_condition=call.condition,
        sources=[data_text],
    )
    if not emitted and len(fields) > 1:
        # value unknown, but report every field the mask covers (the core names only one)
        for item in fields:
            found.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.cfg_field",
                    resource_expr=item.name,
                    normalizer="pack.cfg_field",
                    access="write",
                    value="",
                    line=call.line,
                    condition=call.condition,
                    detail="src=rmw",
                )
            )
    return found


# ---- reads --------------------------------------------------------------------- #
def _cfg_reads(fn, ctx):
    masked = fn.masked_body
    found = []
    for match in re.finditer(r"(?<![\w.>])cfg\s*\[", masked):
        close = scan.match_close(masked, match.end() - 1, "[]")
        if close is None:
            continue
        if re.match(r"\s*=(?!=)", masked[close + 1 :]):
            continue
        index_text = compact(masked[match.end() : close])
        found.extend(
            ctx.effects(
                fn,
                pattern_id="pack.cfg_read",
                resource_expr=index_text,
                normalizer="cfg_addr32",
                access="read",
                value="",
                offset=match.start(),
                detail="mmio",
            )
        )
    return found


_SW_READ_NAMES = ("dest_offset_id", "pack_sync_tile_dst_ptr")


def _sw_reads(fn, ctx):
    masked = fn.masked_body
    found = []
    for name in _SW_READ_NAMES:
        for match in re.finditer(rf"(?<![\w.>:]){name}\b", masked):
            after = masked[match.end() :]
            if re.match(r"\s*=(?!=)", after):
                continue
            found.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.sw_read",
                    resource_expr=name,
                    normalizer="sw",
                    access="read",
                    value="",
                    offset=match.start(),
                )
            )
    return found


#: Mutable CB pointers of the pack side (immutable CB metadata such as fifo_page_size is not state).
_CB_FIELDS = r"fifo_wr_ptr|fifo_wr_tile_ptr|tiles_received|tiles_acked"


def _cb_state(fn, ctx):
    masked = fn.masked_body
    found = []
    aliases = {
        m.group(1): compact(m.group(2))
        for m in re.finditer(
            r"auto\s*&\s*([A-Za-z_]\w*)\s*=\s*get_local_cb_interface\s*\(([^()]*)\)",
            masked,
        )
    }
    pattern = re.compile(
        rf"(?:get_local_cb_interface\s*\((?P<cb>[^()]*)\)|(?<![\w.>])(?P<alias>{'|'.join(map(re.escape, sorted(aliases))) or 'a^'})\b)\s*\.\s*(?P<field>{_CB_FIELDS})\b"
    )
    for match in pattern.finditer(masked):
        cb = (
            compact(match.group("cb"))
            if match.group("cb") is not None
            else aliases.get(match.group("alias"), "")
        )
        after = masked[match.end() :]
        operator = re.match(r"\s*(<<=|>>=|[+\-*/%&|^]?=)(?!=)", after)
        accesses = []
        if operator is None:
            accesses = ["read"]
        elif operator.group(1) == "=":
            accesses = ["write"]
        else:
            accesses = ["read", "write"]
        for access in accesses:
            value = ""
            if access == "write":
                end = scan.statement_end(masked, match.end() + operator.end())
                rhs = compact(masked[match.end() + operator.end() : end])
                value = (
                    rhs
                    if operator.group(1) == "="
                    else f"{operator.group(1)[:-1]}= {rhs}"
                )
            found.extend(
                ctx.effects(
                    fn,
                    pattern_id="pack.cb_state",
                    resource_expr=match.group("field"),
                    normalizer="pack.cb",
                    access=access,
                    value=value,
                    offset=match.start(),
                    detail=f"cb={cb}",
                )
            )
    return found


# --------------------------------------------------------------------------- #
# Ownership summary (for the checker and the report)
# --------------------------------------------------------------------------- #
_PACK_RESOURCE = re.compile(
    r"^(?:CFG\.(?:THCON_SEC0_REG1_|PCK|TILE_ROW_SET_MAPPING_|PACK_COUNTERS_|STACC_RELU_|ALU_FORMAT_SPEC_REG2?_Dstacc|DEST_TARGET_REG_CFG_PACK_|SCRATCH_SEC2_)"
    r"|GPR\.pack\.|ADDR_MOD_PACK\.|ADC\.PAC\.|PACK\.|SW\.(?:dest_offset_id|pack_sync_tile_dst_ptr|cb_interface\.))"
)


def ownership(db, *, layers=("llk_api",)) -> dict[str, dict[str, list[str]]]:
    """Which roles write each pack resource: ``{resource: {role: [function names]}}``.

    Built from the transitive effects of the pack-scope functions of ``layers`` (default: the
    llk_api pack wrappers, i.e. what the compute API calls). Thread-local classes are kept to the
    pack thread. Sorted and deterministic.
    """
    table: dict[str, dict[str, set[str]]] = {}
    for key in db.keys():
        model = db.function(key)
        if model is None or model.layer not in layers or not in_scope(model.path):
            continue
        for effect in db.effects(key):
            if effect.access != "write" or not _PACK_RESOURCE.match(effect.resource):
                continue
            table.setdefault(effect.resource, {}).setdefault(model.role, set()).add(
                model.name
            )
    return {
        resource: {role: sorted(names) for role, names in sorted(roles.items())}
        for resource, roles in sorted(table.items())
    }
