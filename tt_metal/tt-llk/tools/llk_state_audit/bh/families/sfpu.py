# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole SFPU family: SFPCONFIG / LOADMACRO / programmable constants / LREG flow.

Everything is recognized by the imperative :func:`extract` hook because the SFPU
state model needs instruction *order* inside a body (SFPLOADI staging feeds the
following SFPCONFIG, LREG live-in / live-out, replay recording windows).

Effects (``Effect.pattern``):

* ``sfpu.sfpconfig`` - ``SFPCONFIG(imm16, dest, mod1)`` write of ``SFPU.<dest>``
  (dest 0-3 ``LOADMACRO.INSTRn``, 4-7 ``LOADMACRO.SEQn``, 8 ``LOADMACRO.CTRL``,
  11-14 ``LREGn``, 15 ``SFPU.CTRL``). ``mod1 & 1``: the value is ``imm16``;
  otherwise the value is the 32-bit word staged in LREG0 by the preceding
  ``SFPLOADI`` (``mod0`` 8 = upper half, 10 = lower half, 0/1/2/4 = full) in the same
  function, as a C expression (folded to hex when constant, ``bits(X)`` when the
  halves are ``hi16(X)``/``lo16(X)``), or ``?LREG0`` when unknown. ``mod1 & 8``:
  ``imm16`` is a lane mask (``detail`` ``lane_mask=..``); ``mod1 & 6``: AND/OR/XOR
  read-modify-write (adds a ``sfpu.sfpconfig.rmw`` read). A constant sequence word
  (dest 4-7) is decoded into ``detail`` ``uses=INSTRn,..``.
* ``sfpu.loadmacro.seq`` / ``.ctrl`` / ``.instr`` - ``SFPLOADMACRO(lreg_ind, ..)``
  reads ``LOADMACRO.SEQ[(lreg_ind >> 2) & 3]``, ``LOADMACRO.CTRL`` and the
  instruction templates ``INSTR0-3``; its ``VD = (lreg_ind & 3) | (addr & 1) << 2``
  is an LREG write in the LREG flow.
* ``sfpu.backdoor`` - an arithmetic SFPU instruction whose VD is 12..15 installs
  itself into ``LOADMACRO.INSTR(VD-12)`` (value: the instruction text) instead of
  executing; ``sfpu.backdoor.ctrl`` reads ``SFPU.CTRL`` (DISABLE_BACKDOOR_LOAD).
* ``sfpu.swap.ctrl`` - ``SFPSWAP`` reads ``SFPU.CTRL`` (swap direction / index
  tracking, e.g. topk ``SFPCONFIG(0x104, 0xF, 1)``).
* ``sfpu.const_read`` - an instruction operand reading programmable constant
  LREG11-14 (``p_sfpu::LREG12``, ``p_sfpu::LCONST_neg1`` ...).
* ``sfpu.lreg.live_in`` / ``sfpu.lreg.live_out`` - LREG0-7: a read not preceded by a
  dominating full write in the function (state the function expects from earlier
  code, e.g. a LUT loaded by ``gelu_init``), and a write not overwritten later by a
  dominating write (state left behind). TTI operands, ``sfpi::l_reg[LRegs::LRegN]``
  and LOADMACRO VD. Data-dependent values are ``?<INSN>``.
* ``sfpu.sfpi_clobber`` - a function with sfpi vector code may overwrite LREG0-7
  through register allocation (``confidence=low``, value ``?sfpi``), except LREGs
  it explicitly rewrites afterwards.
* ``sfpu.dst_reg.addr_mod`` - ``sfpi::dst_reg[..]`` load/store uses
  ``SFPLOAD/SFPSTORE_ADDR_MODE_NOINC`` = ADDR_MOD_7 on Blackhole (read).
* ``sfpu.loadmacro.template_const`` - a LOADMACRO runs templates that the file's inits
  installed through the backdoor; the programmable constants those templates read
  (exp's ``SFPMAD(12, 0, 13, 13, 0)`` reads LREG12/13) are reported at the LOADMACRO
  (``confidence=low``, ``detail`` ``via_template=<init>:INSTRn``).
* ``sfpu.replay.*`` - ``lltt::replay`` / ``TTI_REPLAY`` of a range recorded in the same
  file: the recorded LOADMACRO/constant/CTRL reads are reported at the replay site
  (``detail`` ``via_replay=<recorder>``).
* ``sfpu.sfpi_constant`` - ``sfpi::vConst{Float,Int}PrgmN = v`` (replaces
  ``core.sfpi_constant``): same resource and ``detail``, value with block-scoped local
  constants inlined (hex when integral).

``TT_INSN(word)`` whose word is built by ``TT_OP_SFP*(..)`` (directly or through a constexpr
local) is treated as that instruction; operands that a patched word (``^ imm``) overrides
become ``?``.

Instructions inside a replay recording window (``lltt::record``, ``load_replay_buf``,
``TTI_REPLAY(.., load=1)``) that are not executed while recording get ``detail``
``deferred:replay``; ``TT_OP_*`` words get ``deferred``. Deferred instructions are
left out of the LREG flow.

Conditions: every effect carries the enclosing ``if``/``if constexpr``/``else`` and
undecidable ``#if`` text of its sink (``APPROXIMATION_MODE``,
``!defined(DISABLE_SFPLOADMACRO)``, ``is_fp32_dest_acc_en`` ...); branches are never
merged.

Live-in dominance uses the enclosing conditions: a write dominates a read when its
conjuncts are implied by the read's (subset, equality/flag facts such as ``N == 1`` =>
``N >= 1``, single-parameter enumeration over 0..32 with the body's ``static_assert``s),
or when both arms of an enclosing ``if``/``else`` write the LREG. A helper's live-in reads
are ``low`` (its caller normally loads them inside the same op), others ``medium``.

:func:`init_execute_pairs` (used by the checker, not by ``build``) pairs compute-API SFPU
executes with their inits.

Call-graph repair: the SFPU params wrappers forward their callable parameter
(``_llk_math_eltwise_sfpu_apply_vector_mode_(std::forward<Callable>(sfpu_func), ..)``,
``std::forward<Callable>(sfpu_func)(..)``). The call graph binds one level, so the
``calculate_*`` callee is lost. :func:`extract` appends a direct ``sfpu_func(args..)``
call site to such functions so the param-callable edge resolves from the caller's
binding (see ``out/requests/sfpu.md``).
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import effects_db, scan
from ..model import Effect, classify_value, compact
from ..resources import Normalized

FAMILY = "sfpu"
ENABLED = True

#: The family replaces the core SFPCONFIG / LOADMACRO recognizers (value-less) with
#: staged-value aware ones, and the sfpi constant assignment with one whose value has the
#: block-scoped local constants inlined. ``core.sfpi_constant_read`` and the ADDR_MOD
#: operand reads (SFPLOAD/SFPSTORE/SFPLOADMACRO) stay with their owners.
DISABLE_CORE: tuple[str, ...] = (
    "core.sfpconfig",
    "core.sfploadmacro",
    "core.sfpi_constant",
)

#: Work around the one-level callable binding of the call graph (out/requests/sfpu.md R1).
INJECT_FORWARDED_CALLABLES = True

#: No declarative patterns: see the module docstring for the extract pattern ids.
PATTERNS: list = []

#: Pattern ids produced by :func:`extract` (documentation / tests).
EXTRACT_PATTERNS = (
    "sfpu.backdoor",
    "sfpu.backdoor.ctrl",
    "sfpu.const_read",
    "sfpu.dst_reg.addr_mod",
    "sfpu.loadmacro.ctrl",
    "sfpu.loadmacro.instr",
    "sfpu.loadmacro.seq",
    "sfpu.loadmacro.template_const",
    "sfpu.lreg.live_in",
    "sfpu.lreg.live_out",
    "sfpu.replay.const_read",
    "sfpu.replay.loadmacro.ctrl",
    "sfpu.replay.loadmacro.instr",
    "sfpu.replay.loadmacro.seq",
    "sfpu.replay.swap.ctrl",
    "sfpu.sfpconfig",
    "sfpu.sfpconfig.rmw",
    "sfpu.sfpi_clobber",
    "sfpu.sfpi_constant",
    "sfpu.swap.ctrl",
)

GROUPS = [
    ("sfpu.loadmacro.seq", r"^SFPU\.LOADMACRO\.SEQ\d$"),
    ("sfpu.loadmacro.instr", r"^SFPU\.LOADMACRO\.INSTR\d$"),
    ("sfpu.loadmacro.ctrl", r"^SFPU\.LOADMACRO\.CTRL$"),
    ("sfpu.state", r"^SFPU\."),
]


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
_TABLES = None  # set by extract (normalizers do not receive the tables)


def _norm_int(text: str) -> int | None:
    if _TABLES is None:
        try:
            return int(scan.evaluate(text))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return _int_value(_TABLES, text)


def _lm_seq(expr, thread=None):
    """``SFPLOADMACRO`` ``lreg_ind`` operand -> ``SFPU.LOADMACRO.SEQ<(v >> 2) & 3>``."""
    text = compact(expr or "")
    if not text:
        return []
    value = _norm_int(text)
    if value is None:
        return [Normalized(f"SFPU.LOADMACRO.?{text}", "slot", resolved=False)]
    return [Normalized(f"SFPU.LOADMACRO.SEQ{(value >> 2) & 3}", "slot")]


def _slot(expr, thread=None):
    """A canonical SFPU resource name used as is (``slot`` granularity)."""
    text = compact(expr or "")
    return [Normalized(text, "slot", resolved="?" not in text)] if text else []


NORMALIZERS = {"sfpu.lm_seq": _lm_seq, "sfpu.slot": _slot}


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
_LLK_LIB_SFPU = re.compile(
    r"^tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_(?:eltwise_(?:unary|binary|ternary)_sfpu|"
    r"eltwise_sfpu_common|welfords_sfpu)\w*\.h$"
)
_SFPU_CALL_HINT = re.compile(
    r"\bSFPU_[A-Z_]+\b|_sfpu_params_\b|llk_math_eltwise_(?:unary|binary|ternary)_sfpu\w*|"
    r"\bsfpu::\w+|ckernel::sfpu::\w+|llk_math_welfords_sfpu\w*"
)


def scope_kind(path: str) -> str | None:
    """Which part of the SFPU scope a repo-relative path belongs to (``None`` outside)."""
    if "/tt_llk_blackhole/" in path and "/common/inc/sfpu/" in path:
        return "common_sfpu"
    if path.startswith(
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/"
    ) or path.startswith(
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/"
    ):
        return "llk_api_sfpu"
    if path.startswith("tt_metal/hw/ckernels/blackhole/") and path.rsplit("/", 1)[
        -1
    ].startswith("ckernel_sfpu"):
        return "llk_api_sfpu"
    if _LLK_LIB_SFPU.match(path):
        return "llk_lib_sfpu"
    if path.startswith("tt_metal/hw/inc/api/compute/"):
        return "compute_api_sfpu"
    return None


def in_scope(fn) -> bool:
    """SFPU-scope function: an SFPU file, or a compute-API function calling SFPU entry points."""
    kind = scope_kind(fn.path)
    if kind is None:
        return False
    if kind != "compute_api_sfpu":
        return True
    if _SFPU_CALL_HINT.search(fn.masked_body or ""):
        return True
    return any(
        _SFPU_CALL_HINT.search(c.callee or c.name or "")
        or (c.via_macro or "").startswith("SFPU_")
        for c in fn.calls
    )


def thread(fn, default: str) -> str | None:
    """SFPU LLK wrappers run on whichever thread calls them (MATH or PACK)."""
    if default == "math" and scope_kind(fn.path) in (
        "llk_lib_sfpu",
        "llk_api_sfpu",
        "common_sfpu",
    ):
        return "any"
    return None


# --------------------------------------------------------------------------- #
# Constants and operand evaluation
# --------------------------------------------------------------------------- #
#: sfpi_constants.h values used as TTI operands (environment-independent copy).
_SFPI_CONSTANTS = {
    "SFPLOADI_MOD0_FLOATB": 0,
    "SFPLOADI_MOD0_FLOATA": 1,
    "SFPLOADI_MOD0_USHORT": 2,
    "SFPLOADI_MOD0_SHORT": 4,
    "SFPLOADI_MOD0_UPPER": 8,
    "SFPLOADI_MOD0_LOWER": 10,
    "SFPCONFIG_MOD1_ARG_IMM": 1,
    "SFPCONFIG_MOD1_AND": 2,
    "SFPCONFIG_MOD1_OR": 4,
    "SFPCONFIG_MOD1_XOR": 6,
    "SFPCONFIG_DEST_MACRO_INST0": 0,
    "SFPCONFIG_DEST_MACRO_INST1": 1,
    "SFPCONFIG_DEST_MACRO_INST2": 2,
    "SFPCONFIG_DEST_MACRO_INST3": 3,
    "SFPCONFIG_DEST_MACRO_SEQ0": 4,
    "SFPCONFIG_DEST_MACRO_SEQ1": 5,
    "SFPCONFIG_DEST_MACRO_SEQ2": 6,
    "SFPCONFIG_DEST_MACRO_SEQ3": 7,
    "SFPCONFIG_DEST_MACRO_CTRL": 8,
    "SFPCONFIG_DEST_LREG11": 11,
    "SFPCONFIG_DEST_LREG12": 12,
    "SFPCONFIG_DEST_LREG13": 13,
    "SFPCONFIG_DEST_LREG14": 14,
    "SFPCONFIG_DEST_SFPU_CTRL": 15,
    "SFPCONFIG_MOD1_SRC_R0_LREG0": 0,
    "SFPSHFT2_MOD1_COPY4": 0,
    "SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4": 1,
    "SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4": 2,
    "SFPSHFT2_MOD1_SUBVEC_SHFLROR1": 3,
    "SFPSHFT2_MOD1_SUBVEC_SHFLSHR1": 4,
    "SFPSHFT2_MOD1_SHFT_LREG": 5,
    "SFPSHFT2_MOD1_SHFT_IMM": 6,
    "SFPLOAD_ADDR_MODE_NOINC": 7,
    "SFPSTORE_ADDR_MODE_NOINC": 7,
}
#: llk_defs.h ``enum class InstrModLoadStore``.
_INSTR_MOD_LOAD_STORE = {
    "DEFAULT": 0,
    "FP16A": 1,
    "FP16B": 2,
    "FP32": 3,
    "INT32": 4,
    "INT8": 5,
    "LO16": 6,
    "HI16": 7,
    "INT32_2S_COMP": 12,
    "INT8_2S_COMP": 13,
    "LO16_ONLY": 14,
    "HI16_ONLY": 15,
}
_ENUM_REF = re.compile(
    r"(?<![\w:])(?:(?:::)?(?:ckernel::)?(?:sfpi::)?)(?:InstrModLoadStore::(?P<ims>\w+)|(?P<sfpi>SFP[A-Z0-9]+_[A-Z0-9_]+))\b"
)
_C_CAST = re.compile(
    r"\(\s*(?:const\s+)?(?:std::)?(?:u?int(?:8|16|32|64)_t|unsigned(?:\s+int)?|int|uint)\s*\)"
)


def _int_value(tables, text: str | None) -> int | None:
    if text is None:
        return None
    expr = compact(text)
    if not expr:
        return None

    def enum(match: re.Match) -> str:
        if match.group("ims") is not None:
            value = _INSTR_MOD_LOAD_STORE.get(match.group("ims"))
        else:
            value = _SFPI_CONSTANTS.get(match.group("sfpi"))
        return match.group(0) if value is None else str(value)

    expr = _ENUM_REF.sub(enum, expr)
    expr = _C_CAST.sub("", expr)
    expr = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", expr)
    return tables.int_value(expr)


def _hex(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:x}"


# --------------------------------------------------------------------------- #
# Instruction table: LREG operand roles (BH ckernel_ops.h operand names)
# --------------------------------------------------------------------------- #
# role per operand position: "r" read, "w" write, "rw" both, None not an LREG.
_A_B_C_D = ("r", "r", "r", "w", None)  # SFPMAD/SFPADD/SFPMUL/SFPMUL24(a, b, c, d, mod1)
_IMM_C_D = (None, "r", "w", None)  # (imm12, lreg_c, lreg_dest, mod1): VD = f(VC)
_IMM_C_DRW = (None, "r", "rw", None)  # VD = f(VD, VC)
_OPERANDS: dict[str, tuple] = {
    "SFPMAD": _A_B_C_D,
    "SFPADD": _A_B_C_D,
    "SFPMUL": _A_B_C_D,
    "SFPMUL24": _A_B_C_D,
    "SFPABS": _IMM_C_D,
    "SFPARECIP": _IMM_C_D,
    "SFPEXEXP": _IMM_C_D,
    "SFPEXMAN": _IMM_C_D,
    "SFPLZ": _IMM_C_D,
    "SFPMOV": _IMM_C_D,
    "SFPNOT": _IMM_C_D,
    "SFPDIVP2": _IMM_C_D,
    "SFPAND": _IMM_C_DRW,
    "SFPOR": _IMM_C_DRW,
    "SFPXOR": _IMM_C_DRW,
    "SFPIADD": _IMM_C_DRW,  # refined by mod1 below
    "SFPSHFT": _IMM_C_DRW,  # refined by mod1 below
    "SFPSETSGN": _IMM_C_DRW,  # refined by mod1 below
    "SFPSETEXP": _IMM_C_DRW,
    "SFPSETMAN": _IMM_C_DRW,
    "SFPSWAP": (None, "rw", "rw", None),
    "SFPGT": (None, "r", "r", None),  # refined by mod1 below (SET_ALL_ONES writes VD)
    "SFPLE": (None, "r", "r", None),
    "SFPSETCC": (None, "r", None, None),
    "SFPCAST": ("r", "w", None),  # (lreg_src_c, lreg_dest, mod1)
    "SFP_STOCH_RND": (None, None, None, "r", "w", None),  # (mode, imm8, b, c, d, mod1)
    "SFPADDI": (None, "rw", None),  # (imm16, lreg_dest, mod1)
    "SFPMULI": (None, "rw", None),
    "SFPLOAD": ("w", None, None, None),
    "SFPSTORE": ("r", None, None, None),
    "SFPLOADI": ("w", None, None),
    "SFPSHFT2": (None, "r", "w", None),  # refined by mod1 below
    "SFPLUTFP32": ("w", None),
    "SFPLUT": ("w", None, None),
}
#: Instructions whose VD operand position may install a LOADMACRO template (VD 12..15).
_BACKDOOR_VD = {
    name: roles.index("w") if "w" in roles else roles.index("rw")
    for name, roles in _OPERANDS.items()
    if name
    not in ("SFPLOAD", "SFPSTORE", "SFPLOADI", "SFPLUT", "SFPLUTFP32", "SFPSETCC")
    and ("w" in roles or "rw" in roles)
}
_BACKDOOR_VD.update({"SFPGT": 2, "SFPLE": 2, "SFPSWAP": 2})
_NO_LREG = frozenset(
    {
        "SFPNOP",
        "SFPENCC",
        "SFPPUSHC",
        "SFPPOPC",
        "SFPCOMPC",
        "SFPCONFIG",
        "SFPLOADMACRO",
        "SFPTRANSP",
    }
)


def _roles(op: str, args: tuple[str, ...], ival) -> list[tuple[int, str]]:
    """(position, role) of LREG operands for one instruction, refined by mod1 when known."""
    base = _OPERANDS.get(op)
    if base is None:
        return []
    roles = list(base[: len(args)])
    mod = ival(args[-1]) if args else None
    if op == "SFPIADD" and mod is not None and mod & 1 and len(roles) >= 3:
        roles[2] = "w"  # VD = VC + imm
    if op == "SFPSHFT" and mod is not None and len(roles) >= 3:
        if mod & 4:
            roles[1], roles[2] = "r", "w"  # VD = VC << imm
        elif mod & 1:
            roles[1], roles[2] = (
                None,
                "rw",
            )  # VD = VD << imm (raw encoding, see exp init)
        else:
            roles[1], roles[2] = "r", "rw"  # VD = VD << VC
    if (
        op in ("SFPSETSGN", "SFPSETEXP", "SFPSETMAN")
        and mod is not None
        and mod & 1
        and len(roles) >= 3
    ):
        roles[2] = "w"  # sign/exp/man from imm
    if op in ("SFPGT", "SFPLE") and mod is not None and mod & 8 and len(roles) >= 3:
        roles[2] = "rw"  # SET_ALL_ONES: VD written
    if op == "SFPSHFT2" and len(args) >= 4:
        if mod == 5:
            roles = ["r", "r", "w", None]  # VD = VB(imm12 slot) << VC
        elif mod is not None and mod <= 4:
            roles = [None, "r", None, None]  # LREG0-3 group shuffle, handled separately
        elif mod is None:
            roles = [None, "r", "w", None]
    return [(position, role) for position, role in enumerate(roles) if role]


# --------------------------------------------------------------------------- #
# Condition helpers
# --------------------------------------------------------------------------- #
def _split_and(condition: str) -> list[str]:
    """Top-level conjuncts of a condition (``a && !(b && c)`` -> ``a``, ``!(b && c)``)."""
    text = compact(condition)
    if not text:
        return []
    parts, depth, start, index = [], 0, 0, 0
    has_or = False
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith("||", index):
            has_or = True
        elif depth == 0 and text.startswith("&&", index):
            parts.append(text[start:index])
            start = index + 2
            index += 1
        index += 1
    parts.append(text[start:])
    if has_or:
        return [_canon_part(text)]
    return [_canon_part(p) for p in parts if compact(p)]


def _split_raw(condition: str) -> list[str]:
    """Top-level ``&&`` pieces of a condition, original text (no canonicalization)."""
    text = compact(condition)
    if not text:
        return []
    parts, depth, start, index = [], 0, 0, 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith("||", index):
            return [text]
        elif depth == 0 and text.startswith("&&", index):
            parts.append(compact(text[start:index]))
            start = index + 2
            index += 1
        index += 1
    parts.append(compact(text[start:]))
    return [p for p in parts if p]


def _strip_parens(text: str) -> str:
    text = compact(text)
    while text.startswith("(") and scan.match_close(text, 0, "()") == len(text) - 1:
        text = compact(text[1:-1])
    return text


def _canon_part(text: str) -> str:
    text = _strip_parens(text)
    match = re.fullmatch(r"!\s*\((.*)\)", text)
    if match and scan.match_close(text, text.index("("), "()") == len(text) - 1:
        inner = _strip_parens(match.group(1))
        if re.fullmatch(
            r"!?\s*defined\s*\(\s*\w+\s*\)|!?\s*defined\s+\w+|!?\s*[\w:.]+", inner
        ):
            return _negate(inner)
        return f"!({inner})"
    return re.sub(r"\s+", " ", text)


def _negate(text: str) -> str:
    text = _strip_parens(text)
    if text.startswith("!"):
        rest = _strip_parens(text[1:])
        return rest
    if re.fullmatch(r"defined\s*\(\s*\w+\s*\)|defined\s+\w+|[\w:.]+", text):
        return f"!{text}"
    return f"!({text})"


def _parts(condition: str | None) -> frozenset[str]:
    return frozenset(_split_and(condition or ""))


def _exclusive(a: frozenset[str], b: frozenset[str]) -> bool:
    if any(_negate(x) in b for x in a):
        return True
    facts = _facts(a)
    return bool(facts) and any(scan.truth(_bind(x, facts)) is False for x in b)


_FACT_EQ = re.compile(
    r"([A-Za-z_]\w*)\s*==\s*(-?\d+|true|false|(?:::)?[A-Za-z_][\w]*(?:::[A-Za-z_]\w*)+)"
)


def _facts(parts: frozenset[str]) -> dict[str, str]:
    """Bindings implied by conjuncts: ``N == 3`` -> N=3, ``FLAG`` -> true, ``!FLAG`` -> false."""
    facts: dict[str, str] = {}
    for part in parts:
        match = _FACT_EQ.fullmatch(part)
        if match:
            facts[match.group(1)] = match.group(2)
        elif re.fullmatch(r"[A-Za-z_]\w*", part):
            facts[part] = "true"
        elif re.fullmatch(r"![A-Za-z_]\w*", part):
            facts[part[1:]] = "false"
    return facts


def _bind(text: str, facts: dict[str, str]) -> str:
    return re.sub(
        r"(?<![\w.:])([A-Za-z_]\w*)\b(?!\s*(?:::|\())",
        lambda m: facts.get(m.group(1), m.group(1)),
        text,
    )


def _implies(have: frozenset[str], need: frozenset[str]) -> bool:
    """Whether the conjunction ``have`` implies every conjunct of ``need`` (subset, or by
    evaluating the missing conjuncts under the equalities/flags that ``have`` fixes)."""
    missing = need - have
    if not missing:
        return True
    facts = _facts(have)
    if facts and all(scan.truth(_bind(part, facts)) is True for part in missing):
        return True
    return _implies_by_enumeration(have, missing)


_ENUM_DOMAIN = range(0, 33)


def _implies_by_enumeration(have: frozenset[str], missing: frozenset[str]) -> bool:
    """One small-integer template parameter (``POLY_DEGREE``): ``have`` implies ``missing`` when
    every value in 0..32 that satisfies the conjuncts of ``have`` over it satisfies ``missing``.
    """
    names = set()
    for part in missing:
        names |= {n for n in _WORD.findall(part) if n not in ("true", "false")}
    if len(names) != 1:
        return False
    (name,) = names
    relevant = [
        p
        for p in have
        if {n for n in _WORD.findall(p) if n not in ("true", "false")} == {name}
    ]
    if not relevant:
        return False
    satisfied = False
    for value in _ENUM_DOMAIN:
        facts = {name: str(value)}
        verdicts = [scan.truth(_bind(p, facts)) for p in relevant]
        if any(v is None for v in verdicts):
            return False
        if not all(verdicts):
            continue
        satisfied = True
        for part in missing:
            if scan.truth(_bind(part, facts)) is not True:
                return False
    return satisfied


# --------------------------------------------------------------------------- #
# Body helpers
# --------------------------------------------------------------------------- #
_DECL = re.compile(
    r"(?<![\w.>:])(?:(?:static|inline)\s+)*(?:constexpr|const)\s+(?:(?:static|const|constexpr|volatile)\s+)*"
    r"(?:(?:::)?[A-Za-z_][\w:]*(?:\s*<[^;{}()]*?>)?)(?:\s*[*&])?\s+([A-Za-z_]\w*)\s*=(?!=)\s*"
)
_VAR_DECL = re.compile(
    r"(?<![\w.>:,(])(?:int|unsigned(?:\s+int)?|bool|auto|(?:std::)?u?int(?:8|16|32|64)_t)\s+([A-Za-z_]\w*)\s*=(?!=)\s*"
)
_WORD = re.compile(r"(?<![\w.$])(?<!->)(?<!::)([A-Za-z_]\w*)\b(?!\s*(?:::|\())")


@dataclass
class _Decl:
    name: str
    expr: str
    offset: int
    block: tuple[int, int]


def _atomic(text: str) -> bool:
    """Identifier, literal or ``name(...)`` call: needs no parentheses when substituted."""
    if re.fullmatch(r"-?[\w:.]+", text):
        return True
    match = re.match(r"[A-Za-z_][\w:]*\s*(?:<[^<>]*>)?\s*\(", text)
    return (
        bool(match) and scan.match_close(text, match.end() - 1, "()") == len(text) - 1
    )


def _blocks(body: str) -> list[tuple[int, int]]:
    stack, pairs = [], []
    for index, char in enumerate(body):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            pairs.append((stack.pop(), index))
    pairs.sort()
    return pairs


def _innermost(
    blocks: list[tuple[int, int]], offset: int, length: int
) -> tuple[int, int]:
    best = (-1, length)
    for start, end in blocks:
        if start < offset < end and start >= best[0]:
            best = (start, end)
    return best


@dataclass
class _Write:
    reg: object  # int (LREG index) or str (unresolved operand text)
    expr: str
    offset: int
    line: int
    condition: str
    parts: frozenset
    thread: str | None
    value: str
    half: str | None  # None full, "lo" / "hi" partial
    lo: str | None = None
    hi: str | None = None
    detail: str = ""
    killed: bool = False
    lo_lines: tuple = ()
    hi_lines: tuple = ()

    @property
    def lines(self) -> tuple:
        return tuple(sorted(set(self.lo_lines) | set(self.hi_lines)))

    pattern_value: str = ""


@dataclass
class _Event:
    index: int
    offset: int
    kind: str  # insn | tinsn | record | replay | lrb | lreg_w | lreg_r
    op: str = ""
    form: str = ""  # TTI | TT | TT_OP
    call: object = None
    args: tuple = ()
    line: int = 0
    condition: str = ""
    thread: str | None = None
    deferred: str = ""  # "", "deferred", "deferred:replay"
    rec: tuple | None = None  # (start, len) of the recording window
    extra: dict = field(default_factory=dict)


_INSN = re.compile(r"TT(I|_OP|)_(SFP[A-Z0-9_]+)$")
_TENSIX = re.compile(r"TTI?_[A-Z0-9_]+$")
_LREG_REF = re.compile(r"(?<![\w.])(?:sfpi::)?l_reg\s*\[")
_SFPI_VECTOR = re.compile(
    r"(?<![\w])(?:sfpi::)?(?:v(?:Float|Int|UInt|SMag|Bool|Float16a|Float16b|UInt16|UInt8|Lut\w*)\b|"
    r"dst_reg\s*\[|v_if\b|v_elseif\b|v_else\b|v_and\b|vConst(?:Float|Int)Prgm[0-3]\s*=(?!=))|__builtin_rvtt_sfp\w+|"
    r"(?<![\w:])sfpi::[a-z_]\w*\s*(?:<[^<>;]*>)?\s*\("
)
_FLOAT_LITERAL = re.compile(
    r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+|0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*[pP][-+]?\d+)[fFlL]?(?![\w.])|\d[fF]\b"
)
_SFPI_CONST_ASSIGN = re.compile(
    r"(?<![\w.])(?:sfpi::)?(vConst(?:Float|Int)Prgm[0-3])\s*=(?!=)\s*"
)
_DST_REG = re.compile(r"(?<![\w.])(?:sfpi::)?dst_reg\s*\[")
_QUICK = re.compile(
    r"SFP|l_reg|dst_reg|v_if|vFloat|vInt|vUInt|vConst|sfpi::|sfpu_func|Callable|record|replay"
)

#: (wrapper, forwarded argument positions before the variadic tail, tail start)
_FORWARDERS = {
    "_llk_math_eltwise_sfpu_apply_vector_mode_": ((), 2),
    "_llk_math_eltwise_unary_sfpu_params_": ((), 3),
    "_llk_math_eltwise_binary_sfpu_params_": ((1, 2, 3), 5),
    "_llk_math_eltwise_ternary_sfpu_params_": ((1, 2, 3, 4), 6),
    "_llk_math_welfords_sfpu_params_": ((), 2),
}
_FORWARD = re.compile(
    r"^(?:::)?std::(?:forward\s*<[^()]*>|move)\s*\((.*)\)(\s*\.\.\.)?$"
)


def _unforward(arg: str) -> str:
    text = compact(arg)
    match = _FORWARD.match(text)
    if match:
        return compact(match.group(1)) + ("..." if match.group(2) else "")
    return text


# --------------------------------------------------------------------------- #
# Per-function analysis
# --------------------------------------------------------------------------- #
class _Analysis:
    def __init__(self, fn, ctx):
        self.fn = fn
        self.ctx = ctx
        self.tables = ctx.tables
        self.body = fn.masked_body or ""
        self._blocks: list[tuple[int, int]] | None = None
        self._decls: dict[str, list[_Decl]] | None = None
        self.params = set(fn.param_names)
        self.constants = set(getattr(ctx.tables, "_constants_text", {}) or {})
        self.out: list[Effect] = []
        self._seen: set[tuple] = set()
        self.writes: dict[object, list[_Write]] = {}
        self.defined: dict[object, list[frozenset]] = {}
        self.live_in: dict[tuple, tuple] = {}
        self._asserts: list | None = None

    # ---- local constants ------------------------------------------------- #
    def decls(self) -> dict[str, list[_Decl]]:
        if self._decls is None:
            self._blocks = _blocks(self.body)
            found: dict[str, list[_Decl]] = {}
            for match in _DECL.finditer(self.body):
                name = match.group(1)
                if name in self.params:
                    continue
                end = scan.statement_end(self.body, match.end())
                expr = compact(self.body[match.end() : end])
                if not expr or len(expr) > 240 or "{" in expr or "[" in expr:
                    continue
                block = _innermost(self._blocks, match.start(), len(self.body))
                found.setdefault(name, []).append(
                    _Decl(name, expr, match.start(), block)
                )
            for match in _VAR_DECL.finditer(self.body):
                name = match.group(1)
                if name in self.params or name in found:
                    continue
                assignments = re.findall(
                    rf"(?<![\w.>:]){re.escape(name)}\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)|\+\+\s*{re.escape(name)}\b|"
                    rf"\b{re.escape(name)}\s*(?:\+\+|--)",
                    self.body,
                )
                if len(assignments) != 1:
                    continue
                end = scan.statement_end(self.body, match.end())
                expr = compact(self.body[match.end() : end])
                if not expr or len(expr) > 240 or "{" in expr or "[" in expr:
                    continue
                block = _innermost(self._blocks, match.start(), len(self.body))
                found.setdefault(name, []).append(
                    _Decl(name, expr, match.start(), block)
                )
            self._decls = found
        return self._decls

    def visible(self, name: str, offset: int) -> _Decl | None:
        best = None
        for decl in self.decls().get(name, ()):
            start, end = decl.block
            if decl.offset < offset and start < offset < end:
                if best is None or decl.offset > best.offset:
                    best = decl
        return best

    def inline(self, text: str, offset: int, depth: int = 0) -> str:
        text = compact(text)
        if depth > 6 or not text or not self.decls():
            return text

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            decl = self.visible(name, offset)
            if decl is None:
                return name
            inner = self.inline(decl.expr, decl.offset, depth + 1)
            value = _int_value(self.tables, inner)
            if value is not None:
                return str(value)
            return inner if _atomic(inner) else f"({inner})"

        return compact(_WORD.sub(substitute, text))

    def ival(self, text: str, offset: int | None = None) -> int | None:
        return _int_value(
            self.tables, self.inline(text, offset) if offset is not None else text
        )

    def possible(self, text: str, offset: int, transform=None) -> set[int] | None:
        """Every value ``text`` can take when its unknown non-parameter locals range over 0..7.

        ``None`` when it depends on a parameter of the function (left symbolic, so that call
        sites can bind it) or is not an integer expression.
        """
        inlined = self.inline(text, offset)
        value = _int_value(self.tables, inlined)
        if value is not None:
            return {transform(value) if transform else value}
        unknown = []
        for name in _WORD.findall(inlined):
            if name in ("true", "false") or name in unknown:
                continue
            if name in self.params:
                return None
            if _int_value(self.tables, name) is None:
                unknown.append(name)
        if not unknown or len(unknown) > 3:
            return None
        found: set[int] = set()
        for combo in itertools.product(range(8), repeat=len(unknown)):
            bound = inlined
            for name, number in zip(unknown, combo):
                bound = re.sub(rf"(?<![\w.:]){re.escape(name)}\b", str(number), bound)
            result = _int_value(self.tables, bound)
            if result is None:
                return None
            found.add(transform(result) if transform else result)
        return found

    def canon(self, text: str, offset: int) -> str:
        inlined = self.inline(text, offset)
        value = _int_value(self.tables, inlined)
        return _hex(value) if value is not None else inlined

    # ---- effect emission --------------------------------------------------- #
    def emit(
        self,
        pattern: str,
        resource_expr: str,
        normalizer: str,
        access: str,
        *,
        line: int,
        condition: str,
        thread: str | None,
        value: str = "",
        detail: str = "",
        confidence: str = "high",
        collapse: bool = False,
        granularity: str | None = "slot",
    ) -> None:
        effects = self.ctx.effects(
            self.fn,
            pattern_id=pattern,
            resource_expr=resource_expr,
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
        for effect in effects:
            if collapse:
                key = (
                    pattern,
                    effect.resource,
                    access,
                    effect.value,
                    effect.condition,
                    effect.detail,
                    effect.thread,
                )
                if key in self._seen:
                    continue
                self._seen.add(key)
            self.out.append(
                replace(
                    effect,
                    value_kind=classify_value(
                        effect.value, self.params, self.constants
                    ),
                )
            )

    # ---- events ------------------------------------------------------------ #
    def events(self) -> list[_Event]:
        fn = self.fn
        events: list[_Event] = []
        for index, call in enumerate(fn.calls):
            if call.is_member or call.is_ctor:
                continue
            name = call.name
            match = _INSN.match(name)
            base = dict(
                index=index,
                offset=call.offset,
                call=call,
                args=tuple(call.args),
                line=call.line,
                condition=call.condition or "",
                thread=call.thread,
            )
            if match:
                form = {"I": "TTI", "": "TT", "_OP": "TT_OP"}[match.group(1)]
                events.append(_Event(kind="insn", op=match.group(2), form=form, **base))
            elif name in ("TT_INSN", "TTI_INSN") and call.args:
                word = self._insn_word(call)
                if word is not None:
                    op, args = word
                    base["args"] = args
                    events.append(_Event(kind="insn", op=op, form="TT_INSN", **base))
                else:
                    events.append(_Event(kind="tinsn", op=name, form="TTI", **base))
            elif _TENSIX.match(name):
                if re.fullmatch(r"TTI?_REPLAY", name):
                    events.append(
                        _Event(kind="replay_insn", op="REPLAY", form="TTI", **base)
                    )
                else:
                    events.append(_Event(kind="tinsn", op=name, form="TTI", **base))
            elif name == "record" and call.qualifier.endswith("lltt"):
                events.append(_Event(kind="record", **base))
            elif name in ("replay", "replay_insn") and call.qualifier.endswith("lltt"):
                events.append(_Event(kind="replay", **base))
            elif name == "load_replay_buf":
                events.append(_Event(kind="lrb", **base))

        for match in _LREG_REF.finditer(self.body):
            close = scan.match_close(self.body, match.end() - 1, "[]")
            if close is None:
                continue
            reg = compact(self.body[match.end() : close])
            after = close + 1
            assign = re.match(r"\s*=(?!=)", self.body[after:])
            offset = match.start()
            base = dict(
                index=10**6 + offset,
                offset=offset,
                line=fn.line_at(offset),
                condition=fn.condition_at(offset) or "",
                thread=self._thread_at(offset),
            )
            if assign:
                end = scan.statement_end(self.body, after + assign.end())
                rhs = compact(self.body[after + assign.end() : end])
                events.append(_Event(kind="lreg_w", args=(reg, rhs), **base))
            else:
                events.append(_Event(kind="lreg_r", args=(reg,), **base))
        events.sort(key=lambda e: (e.offset, e.index))
        self._windows(events)
        return events

    def _insn_word(self, call) -> tuple[str, tuple] | None:
        """``TT_INSN(word)`` issuing a word built by ``TT_OP_SFP*(..)`` (directly or via a constexpr local).

        When the word is patched (``^ imm``, ``| bits``), operands that are not LREG indices become ``?``.
        """
        text = self.inline(call.args[0], call.offset)
        match = re.search(r"\bTT_OP_(SFP[A-Z0-9_]+)\s*\(", text)
        if match is None:
            return None
        close = scan.match_close(text, match.end() - 1, "()")
        if close is None:
            return None
        op = match.group(1)
        args = tuple(compact(a) for a in scan.split_args(text[match.end() : close]))
        if _strip_parens(text) != compact(text[match.start() : close + 1]):
            lregs = {position for position, _ in _roles(op, args, lambda t: None)}
            args = tuple(a if i in lregs else "?" for i, a in enumerate(args))
        return op, args

    def _thread_at(self, offset: int) -> str | None:
        spans = scan.thread_spans(self.body)
        found = scan._thread_at(spans, offset)
        if found is None:
            found = self.fn.pp_info_at_line(self.fn.line_at(offset))[0]
        return found

    @staticmethod
    def _exec_flag(template_args) -> bool:
        text = " ".join(template_args)
        return bool(re.search(r"(?<!No)Exec\b", text)) or text.strip() in ("true", "1")

    def _windows(self, events: list[_Event]) -> None:
        """Mark instructions that are only recorded into the replay buffer."""
        instructions = [
            e
            for e in events
            if e.kind in ("insn", "tinsn", "replay_insn", "replay")
            and e.form != "TT_OP"
        ]
        for event in events:
            if event.kind == "record" or (
                event.kind == "replay_insn"
                and len(event.args) >= 4
                and compact(event.args[3]) not in ("0", "false")
            ):
                start = self.ival(event.args[0], event.offset) if event.args else None
                length = (
                    self.ival(event.args[1], event.offset)
                    if len(event.args) > 1
                    else None
                )
                if event.kind == "record":
                    executes = self._exec_flag(event.call.template_args)
                else:
                    executes = compact(event.args[2]) not in ("0", "false")
                event.extra["recording"] = (start, length)
                following = [
                    e
                    for e in instructions
                    if (e.offset, e.index) > (event.offset, event.index)
                ]
                count = length if length is not None else len(following)
                for item in following[:count]:
                    item.rec = (start, length, event.index)
                    if not executes:
                        item.deferred = "deferred:replay"
            elif event.kind == "lrb":
                opening = self.body.find("(", event.offset)
                close = (
                    scan.match_close(self.body, opening, "()") if opening >= 0 else None
                )
                if close is None:
                    continue
                start = self.ival(event.args[0], event.offset) if event.args else None
                length = (
                    self.ival(event.args[1], event.offset)
                    if len(event.args) > 1
                    else None
                )
                executes = self._exec_flag(event.call.template_args)
                event.extra["recording"] = (start, length)
                for item in instructions:
                    if opening < item.offset < close:
                        item.rec = (start, length, event.index)
                        if not executes:
                            item.deferred = "deferred:replay"
        for event in events:
            if event.kind == "insn" and event.form == "TT_OP":
                event.deferred = "deferred"

    # ---- LREG flow --------------------------------------------------------- #
    def asserts(self) -> list[tuple[frozenset, frozenset]]:
        """(enclosing condition parts, asserted conjuncts) of the body's ``static_assert``s."""
        if self._asserts is None:
            found = []
            for match in re.finditer(r"\bstatic_assert\s*\(", self.body):
                close = scan.match_close(self.body, match.end() - 1, "()")
                if close is None:
                    continue
                args = scan.split_args(self.body[match.end() : close])
                if args:
                    found.append(
                        (_parts(self.fn.condition_at(match.start())), _parts(args[0]))
                    )
            self._asserts = found
        return self._asserts

    def with_asserts(self, parts: frozenset) -> frozenset:
        """``parts`` plus the static_assert conjuncts that hold wherever ``parts`` holds."""
        extra = set()
        for condition, asserted in self.asserts():
            if condition <= parts:
                extra |= asserted
        return parts | frozenset(extra) if extra else parts

    def lreg_keys(
        self, text: str, offset: int
    ) -> tuple[list[tuple[object, str]], bool]:
        """[(LREG index or symbolic operand text, resource expr)] and whether it is a may-access."""
        inlined = self.inline(text, offset)
        normalized = self.tables.normalize("lreg", inlined)
        if not normalized:
            return [], False
        match = re.fullmatch(r"SFPU\.LREG(\d+)", normalized[0].name)
        if match:
            return [(int(match.group(1)), inlined)], False
        values = self.possible(text, offset)
        if values is not None and len(values) <= 8:
            ordered = sorted(values)
            return [(value, str(value)) for value in ordered], len(ordered) > 1
        return [(compact(inlined), inlined)], False

    def read_reg(self, reg, expr: str, event: _Event, detail: str = "") -> None:
        if isinstance(reg, int):
            if 11 <= reg <= 14:
                self.emit(
                    "sfpu.const_read",
                    str(reg),
                    "lreg",
                    "read",
                    line=event.line,
                    condition=event.condition,
                    thread=event.thread,
                    detail=event.deferred or detail,
                    collapse=True,
                )
                return
            if not 0 <= reg <= 7:
                return
        if event.deferred:
            return
        parts = self.with_asserts(_parts(event.condition))
        residual = set()
        for defined in self.defined.get(reg, ()):
            if _implies(parts, defined):
                return
            rest = [x for x in defined if not _implies(parts, frozenset({x}))]
            if len(rest) == 1:
                residual.add(rest[0])
        if any(_negate(x) in residual for x in residual):
            return  # written on both arms of an if/else that encloses the read
        key = (reg, compact(event.condition), event.thread)
        if key not in self.live_in:
            self.live_in[key] = (expr, event.line, event.condition, event.thread)

    def write_reg(
        self,
        reg,
        expr: str,
        event: _Event,
        value: str,
        half: str | None,
        detail: str = "",
        may: bool = False,
    ) -> None:
        if isinstance(reg, int) and not 0 <= reg <= 7:
            return
        if event.deferred:
            return
        parts = _parts(event.condition)
        if may:
            self.writes.setdefault(reg, []).append(
                _Write(
                    reg=reg,
                    expr=expr,
                    offset=event.offset,
                    line=event.line,
                    condition=event.condition,
                    parts=parts,
                    thread=event.thread,
                    value=value,
                    half=None,
                    detail="may",
                )
            )
            return
        if half is None:
            hi, lo = _halves(value)
        else:
            hi, lo = (value, None) if half == "hi" else (None, value)
        new = _Write(
            reg=reg,
            expr=expr,
            offset=event.offset,
            line=event.line,
            condition=event.condition,
            parts=parts,
            thread=event.thread,
            value=value,
            half=half,
            lo=lo,
            hi=hi,
            detail=detail,
            lo_lines=(event.line,) if half in (None, "lo") else (),
            hi_lines=(event.line,) if half in (None, "hi") else (),
        )
        for old in self.writes.get(reg, ()):
            if old.killed or not _implies(old.parts, parts):
                continue
            if half is None or old.half == half:
                old.killed = True
            elif parts == old.parts:
                # complementary half under the same condition: the halves combine
                if new.hi is None:
                    new.hi, new.hi_lines = old.hi, old.hi_lines
                if new.lo is None:
                    new.lo, new.lo_lines = old.lo, old.lo_lines
                if new.hi is not None and new.lo is not None:
                    new.value = _combine(new.hi, new.lo)
                    new.half = None
                old.killed = True
        self.writes.setdefault(reg, []).append(new)
        if new.half is None:
            self.defined.setdefault(reg, []).append(parts)

    def staged(self, reg: int, event: _Event) -> tuple[str, tuple]:
        """(value, SFPLOADI lines) of LREG ``reg`` at ``event`` (the SFPCONFIG source).

        The value is ``?LREG<n>`` when no dominating write precedes ``event`` in the
        function, or when a later write under a non-exclusive condition makes it ambiguous.
        """
        parts = _parts(event.condition)
        candidates = [
            w
            for w in self.writes.get(reg, ())
            if not w.killed and w.offset <= event.offset
        ]
        chosen = None
        for write in reversed(candidates):
            if _implies(parts, write.parts):
                chosen = write
                break
            if not _exclusive(write.parts, parts):
                return f"?LREG{reg}", ()
        if chosen is None:
            return f"?LREG{reg}", ()
        if chosen.half is not None:
            return (
                _combine(
                    chosen.hi if chosen.hi is not None else "?",
                    chosen.lo if chosen.lo is not None else "?",
                ),
                chosen.lines,
            )
        return chosen.value, chosen.lines

    # ---- instruction handlers ---------------------------------------------- #
    def sfploadi_value(self, event: _Event) -> tuple[str, str | None]:
        args = event.args
        mod = self.ival(args[1], event.offset) if len(args) > 1 else None
        imm = self.inline(args[2], event.offset) if len(args) > 2 else "?"
        value = _int_value(self.tables, imm)
        if mod == 8:
            return (str(value & 0xFFFF) if value is not None else imm), "hi"
        if mod == 10:
            return (str(value & 0xFFFF) if value is not None else imm), "lo"
        if mod == 2:
            return (
                _hex(value & 0xFFFF) if value is not None else f"({imm}) & 0xffff"
            ), None
        if mod == 0:
            return (
                _hex((value & 0xFFFF) << 16) if value is not None else f"({imm}) << 16"
            ), None
        if mod == 4:
            if value is not None:
                value &= 0xFFFF
                return _hex(value - 0x10000 if value & 0x8000 else value), None
            return f"sext16({imm})", None
        if mod == 1:
            return f"fp16a({_hex(value) if value is not None else imm})", None
        return "?SFPLOADI", None

    def handle_sfpconfig(self, event: _Event) -> None:
        args = event.args
        if len(args) < 3:
            return
        imm_text = self.inline(args[0], event.offset)
        dest_text = self.inline(args[1], event.offset)
        mod = self.ival(args[2], event.offset)
        dest = _int_value(self.tables, dest_text)
        details = []
        if mod is None:
            value = self.canon(imm_text, event.offset)
            details.append(f"src=?;mod1={compact(args[2])}")
            source_lreg0 = False
        elif mod & 1:
            value = self.canon(imm_text, event.offset)
            details.append("src=imm16")
            source_lreg0 = False
        else:
            value, lines = (
                self.staged(0, event) if not event.deferred else ("?LREG0", ())
            )
            details.append(
                "src=LREG0"
                + (f";staged={','.join(str(n) for n in lines)}" if lines else "")
            )
            source_lreg0 = True
        operation = None
        if mod is not None and mod & 8:
            details.append(f"lane_mask={self.canon(imm_text, event.offset)}")
        if mod is not None and mod & 6:
            operation = {2: "and", 4: "or", 6: "xor"}[mod & 6]
            details.append(f"op={operation}")
        word = _maybe_int(value)
        if dest is not None and 4 <= dest <= 7 and word is not None:
            details.append(
                "uses="
                + _decode_sequence(word, imm16=(mod is not None and mod & 1 == 1))
            )
        if event.deferred:
            details.insert(0, event.deferred)
        detail = ";".join(details)
        self.emit(
            "sfpu.sfpconfig",
            dest_text,
            "sfpconfig_dest",
            "write",
            line=event.line,
            condition=event.condition,
            thread=event.thread,
            value=value,
            detail=detail,
        )
        if operation is not None:
            self.emit(
                "sfpu.sfpconfig.rmw",
                dest_text,
                "sfpconfig_dest",
                "read",
                line=event.line,
                condition=event.condition,
                thread=event.thread,
                detail=detail,
                collapse=True,
            )
        if source_lreg0:
            self.read_reg(0, "0", event)

    def loadmacro_reads(
        self, event: _Event, prefix: str, detail: str, confidence: str = "high"
    ) -> None:
        args = event.args
        if not args:
            return
        lreg_ind = self.inline(args[0], event.offset)
        common = dict(
            line=event.line,
            condition=event.condition,
            thread=event.thread,
            collapse=True,
        )
        sequences = self.possible(
            lreg_ind, event.offset, transform=lambda v: (v >> 2) & 3
        )
        if sequences is not None and len(sequences) <= 4:
            targets = [
                (str(seq << 2), confidence if len(sequences) == 1 else "low")
                for seq in sorted(sequences)
            ]
        else:
            targets = [(lreg_ind, confidence)]
        for target, level in targets:
            self.emit(
                f"{prefix}loadmacro.seq",
                target,
                "sfpu.lm_seq",
                "read",
                detail=detail,
                confidence=level,
                **common,
            )
        self.emit(
            f"{prefix}loadmacro.ctrl",
            "8",
            "sfpconfig_dest",
            "read",
            detail=detail,
            confidence=confidence,
            **common,
        )
        for slot in range(4):
            self.emit(
                f"{prefix}loadmacro.instr",
                f"SFPU.LOADMACRO.INSTR{slot}",
                "sfpu.slot",
                "read",
                detail=detail,
                confidence="medium" if confidence == "high" else confidence,
                **common,
            )

    def template_reads(self, event: _Event, detail_prefix: str = "") -> None:
        """Programmable constants read by the file's backdoor-installed templates a LOADMACRO runs."""
        templates = _file_templates(self.ctx, self.fn)
        if not templates:
            return
        writers = {t[0] for t in templates}
        inits = {w for w in writers if "init" in _short_name(w).lower()}
        chosen = inits or writers
        confidence = "low"
        for writer, slot, op, args in templates:
            if writer not in chosen:
                continue
            for reg_text in item_args_const_reads(op, args, self.tables, template=True):
                detail = f"via_template={_short_name(writer)}:INSTR{slot}"
                if detail_prefix:
                    detail = f"{detail_prefix};{detail}"
                self.emit(
                    "sfpu.loadmacro.template_const",
                    reg_text,
                    "lreg",
                    "read",
                    line=event.line,
                    condition=event.condition,
                    thread=event.thread,
                    detail=detail,
                    confidence=confidence,
                    collapse=True,
                )

    def handle_loadmacro(self, event: _Event) -> None:
        self.loadmacro_reads(event, "sfpu.", event.deferred)
        if not event.deferred:
            self.template_reads(event)
        args = event.args
        if len(args) < 4:
            return
        lreg_ind = self.inline(args[0], event.offset)
        address = self.inline(args[3], event.offset)
        expr = f"(({lreg_ind}) & 3) | ((({address}) & 1) << 2)"
        values = self.possible(expr, event.offset)
        if values is not None and len(values) <= 8:
            for vd in sorted(values):
                self.write_reg(
                    vd, str(vd), event, "?SFPLOADMACRO", None, may=len(values) > 1
                )
        else:
            self.write_reg(compact(expr), expr, event, "?SFPLOADMACRO", None)

    def handle_backdoor(self, event: _Event, slot: int) -> None:
        text = (
            f"{event.op}({', '.join(self.inline(a, event.offset) for a in event.args)})"
        )
        detail = ";".join(x for x in (event.deferred, "backdoor") if x)
        self.emit(
            "sfpu.backdoor",
            f"SFPU.LOADMACRO.INSTR{slot}",
            "sfpu.slot",
            "write",
            line=event.line,
            condition=event.condition,
            thread=event.thread,
            value=text,
            detail=detail,
        )
        self.emit(
            "sfpu.backdoor.ctrl",
            "15",
            "sfpconfig_dest",
            "read",
            line=event.line,
            condition=event.condition,
            thread=event.thread,
            detail="DISABLE_BACKDOOR_LOAD"
            + (f";{event.deferred}" if event.deferred else ""),
            collapse=True,
        )

    def handle_insn(self, event: _Event) -> None:
        op = event.op
        if op == "SFPCONFIG":
            self.handle_sfpconfig(event)
            return
        if op == "SFPLOADMACRO":
            self.handle_loadmacro(event)
            return
        if op == "SFPTRANSP":
            for reg in range(8):
                self.read_reg(reg, str(reg), event)
            for reg in range(8):
                self.write_reg(reg, str(reg), event, "?SFPTRANSP", None)
            return
        if op in _NO_LREG:
            return
        args = event.args
        position = _BACKDOOR_VD.get(op)
        if position is not None and position < len(args):
            vd = self.ival(args[position], event.offset)
            if vd is not None and 12 <= vd <= 15:
                self.handle_backdoor(event, vd - 12)
                return
        ival = lambda text: self.ival(text, event.offset)  # noqa: E731
        roles = _roles(op, args, ival)
        if op == "SFPSHFT2":
            mod = ival(args[-1]) if args else None
            if mod is not None and mod <= 4:
                for reg in range(4):
                    self.read_reg(reg, str(reg), event)
                for reg in range(4):
                    self.write_reg(reg, str(reg), event, "?SFPSHFT2", None)
        if op == "SFPSWAP":
            self.emit(
                "sfpu.swap.ctrl",
                "15",
                "sfpconfig_dest",
                "read",
                line=event.line,
                condition=event.condition,
                thread=event.thread,
                detail="swap_mode" + (f";{event.deferred}" if event.deferred else ""),
                collapse=True,
            )
        for position, role in roles:
            if "r" in role:
                for reg, expr in self.lreg_keys(args[position], event.offset)[0]:
                    self.read_reg(reg, expr, event)
        for position, role in roles:
            if "w" in role:
                keys, may = self.lreg_keys(args[position], event.offset)
                for reg, expr in keys:
                    if op == "SFPLOADI":
                        value, half = self.sfploadi_value(event)
                    else:
                        value, half = f"?{op}", None
                    self.write_reg(
                        reg, expr, event, value, None if may else half, may=may
                    )

    def handle_lreg(self, event: _Event) -> None:
        keys, may = self.lreg_keys(event.args[0], event.offset)
        for reg, expr in keys:
            if event.kind == "lreg_r":
                self.read_reg(reg, expr, event)
            else:
                rhs = self.inline(event.args[1], event.offset)
                self.write_reg(reg, expr, event, rhs, None, may=may)

    # ---- replay linking ---------------------------------------------------- #
    def handle_replay(self, event: _Event) -> None:
        if event.deferred:
            return
        args = event.args
        start = self.ival(args[0], event.offset) if args else None
        length = self.ival(args[1], event.offset) if len(args) > 1 else None
        recorded = _file_recordings(self.ctx, self.fn)
        hits: dict[str, list] = {}
        for recorder, window_start, items in recorded:
            for position, item in enumerate(items):
                slot = None if window_start is None else window_start + position
                if start is not None and length is not None and slot is not None:
                    if not start <= slot < start + length:
                        continue
                hits.setdefault(recorder, []).append(item)
        if not hits:
            return
        if self.fn.key in hits:
            # the function records the range itself (record + replay in one body)
            hits = {self.fn.key: hits[self.fn.key]}
        else:
            inits = {k: v for k, v in hits.items() if "init" in _short_name(k).lower()}
            if inits:
                hits = inits
        confidence = "medium" if len(hits) == 1 else "low"
        for recorder in sorted(hits):
            detail = f"via_replay={_short_name(recorder)}"
            for op, item_args, offset_text in hits[recorder]:
                fake = _Event(
                    index=event.index,
                    offset=event.offset,
                    kind="insn",
                    op=op,
                    args=item_args,
                    line=event.line,
                    condition=event.condition,
                    thread=event.thread,
                )
                if op == "SFPLOADMACRO":
                    self.loadmacro_reads(fake, "sfpu.replay.", detail, confidence)
                    self.template_reads(fake, detail)
                elif op == "SFPSWAP":
                    self.emit(
                        "sfpu.replay.swap.ctrl",
                        "15",
                        "sfpconfig_dest",
                        "read",
                        line=event.line,
                        condition=event.condition,
                        thread=event.thread,
                        detail=detail,
                        confidence=confidence,
                        collapse=True,
                    )
                for reg_text in item_args_const_reads(op, item_args, self.tables):
                    self.emit(
                        "sfpu.replay.const_read",
                        reg_text,
                        "lreg",
                        "read",
                        line=event.line,
                        condition=event.condition,
                        thread=event.thread,
                        detail=detail,
                        confidence=confidence,
                        collapse=True,
                    )

    # ---- sfpi -------------------------------------------------------------- #
    def sfpi_constants(self) -> None:
        """``sfpi::vConst{Float,Int}PrgmN = v`` -> SFPCONFIG of LREG11-14 with the inlined value."""
        for match in _SFPI_CONST_ASSIGN.finditer(self.body):
            end = scan.statement_end(self.body, match.end())
            value = self.inline(self.body[match.end() : end], match.start())
            number = _int_value(self.tables, value)
            offset = match.start()
            self.emit(
                "sfpu.sfpi_constant",
                match.group(1),
                "sfpi_const",
                "write",
                line=self.fn.line_at(offset),
                condition=self.fn.condition_at(offset) or "",
                thread=self._thread_at(offset),
                value=(
                    _hex(number)
                    if number is not None and not _FLOAT_LITERAL.search(value)
                    else value
                ),
                detail=match.group(1),
            )

    def sfpi(self) -> None:
        body = self.body
        pinned = []  # statements of explicit l_reg writes: the value goes to that LREG
        for match in _LREG_REF.finditer(body):
            close = scan.match_close(body, match.end() - 1, "[]")
            if close is not None and re.match(r"\s*=(?!=)", body[close + 1 :]):
                pinned.append((match.start(), scan.statement_end(body, close + 1)))
        markers = [
            m.start()
            for m in _SFPI_VECTOR.finditer(body)
            if not any(start <= m.start() <= end for start, end in pinned)
        ]
        dst = [m for m in _DST_REG.finditer(body)]
        for match in dst:
            close = scan.match_close(body, match.end() - 1, "[]")
            tail = body[close + 1 : close + 40] if close is not None else ""
            mode = re.match(r"\s*\.\s*mode\s*(?:<[^>]*>)?\s*\(\s*([^()]*?)\s*\)", tail)
            slot = "7"
            detail = "SFPLOAD/SFPSTORE_ADDR_MODE_NOINC"
            if mode:
                if not mode.group(1):
                    slot, detail = "7", "mode()"
                else:
                    slot, detail = mode.group(1), "mode"
            offset = match.start()
            self.emit(
                "sfpu.dst_reg.addr_mod",
                slot,
                "addr_mod",
                "read",
                line=self.fn.line_at(offset),
                condition=self.fn.condition_at(offset) or "",
                thread=self._thread_at(offset),
                value="sfpi::dst_reg",
                detail=detail,
                collapse=True,
                granularity=None,
            )
        if not markers:
            return
        common = None
        for offset in markers:
            parts = _parts(self.fn.condition_at(offset))
            common = parts if common is None else common & parts
        last = max(markers)
        first_text = self.fn.condition_at(min(markers)) or ""
        ordered = [
            p for p in _split_raw(first_text) if _canon_part(p) in (common or ())
        ]
        condition = " && ".join(ordered)
        first = min(markers)
        for reg in range(8):
            explicit = [
                w
                for w in self.writes.get(reg, ())
                if not w.killed
                and w.half is None
                and w.detail != "may"
                and w.offset > last
                and w.parts <= (common or frozenset())
            ]
            if explicit:
                continue
            self.emit(
                "sfpu.sfpi_clobber",
                str(reg),
                "lreg",
                "write",
                line=self.fn.line_at(first),
                condition=condition,
                thread=self._thread_at(first),
                value="?sfpi",
                detail="sfpi_regalloc",
                confidence="low",
                collapse=True,
            )

    # ---- driver ------------------------------------------------------------ #
    def run(self) -> list[Effect]:
        events = self.events()
        for event in events:
            if event.kind == "insn":
                self.handle_insn(event)
            elif event.kind in ("lreg_w", "lreg_r"):
                self.handle_lreg(event)
            elif event.kind == "replay" or (
                event.kind == "replay_insn"
                and (len(event.args) < 4 or compact(event.args[3]) in ("0", "false"))
            ):
                self.handle_replay(event)
        # A helper's live-in LREGs are normally loaded by its caller inside the same op: low.
        live_in_confidence = (
            "low" if effects_db.default_role(self.fn) == "helper" else "medium"
        )
        for key in sorted(
            self.live_in,
            key=lambda k: (self.live_in[k][1], str(k[0]), k[1], k[2] or ""),
        ):
            expr, line, condition, thread_name = self.live_in[key]
            self.emit(
                "sfpu.lreg.live_in",
                expr,
                "lreg",
                "read",
                line=line,
                condition=condition,
                thread=thread_name,
                detail="live_in",
                confidence=live_in_confidence,
            )
        for reg in sorted(self.writes, key=str):
            for write in self.writes[reg]:
                if write.killed:
                    continue
                value = (
                    write.value
                    if write.half is None
                    else _combine(
                        write.hi if write.hi is not None else "?",
                        write.lo if write.lo is not None else "?",
                    )
                )
                self.emit(
                    "sfpu.lreg.live_out",
                    write.expr,
                    "lreg",
                    "write",
                    line=write.line,
                    condition=write.condition,
                    thread=write.thread,
                    value=value,
                    detail="live_out"
                    + ("" if write.half is None else f";partial={write.half}")
                    + (";may" if write.detail == "may" else ""),
                    confidence="low" if write.detail == "may" else "high",
                    collapse=True,
                )
        self.sfpi_constants()
        self.sfpi()
        return self.out


def item_args_const_reads(
    op: str, args: tuple, tables, *, template: bool = False
) -> list[str]:
    """LREG11-14 operands read by one (recorded) instruction, or by a LOADMACRO template
    (``template=True``: the VD operand only selects the template slot)."""
    if op in _NO_LREG and op != "SFPLOADMACRO":
        return []
    found = []
    skip = _BACKDOOR_VD.get(op) if template else None
    for position, role in _roles(op, args, lambda text: _int_value(tables, text)):
        if "r" not in role or position == skip:
            continue
        value = _int_value(tables, args[position])
        if value is not None and 11 <= value <= 14:
            found.append(str(value))
    return sorted(set(found))


def _short_name(key: str) -> str:
    return key.split("(")[0].split("<")[0].rsplit("::", 1)[-1]


def _maybe_int(text: str | None) -> int | None:
    if text is None:
        return None
    text = compact(text)
    if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", text):
        return int(text, 0)
    return None


def _halves(value: str) -> tuple[str, str]:
    """(upper, lower) 16-bit pieces of a full 32-bit LREG value text."""
    number = _maybe_int(value)
    if number is not None:
        return str((number >> 16) & 0xFFFF), str(number & 0xFFFF)
    if value.startswith("?"):
        return "?", "?"
    return f"hi16w({value})", f"lo16w({value})"


def _call_arg(text: str, name: str) -> str | None:
    match = re.fullmatch(rf"{name}\s*\((.*)\)", text)
    if match and scan.match_close(text, text.index("("), "()") == len(text) - 1:
        return _strip_parens(match.group(1))
    return None


def _combine(hi: str | None, lo: str | None) -> str:
    """32-bit word from its halves: hex when constant, ``bits(X)`` for hi16(X)/lo16(X)."""
    hi = "?" if hi is None else compact(hi)
    lo = "?" if lo is None else compact(lo)
    hv, lv = _maybe_int(hi), _maybe_int(lo)
    if hv is not None and lv is not None:
        return _hex(((hv & 0xFFFF) << 16) | (lv & 0xFFFF))
    for upper, lower, wrap in (("hi16w", "lo16w", "{}"), ("hi16", "lo16", "bits({})")):
        x, y = _call_arg(hi, upper), _call_arg(lo, lower)
        if x is not None and y is not None and x == y:
            return wrap.format(x)
    if hi == "?" and lo == "?":
        return "?"
    return f"(({hi}) << 16) | ({lo})"


def _decode_sequence(word: int, *, imm16: bool = False) -> str:
    """LOADMACRO sequence word -> templates/units it uses (bytes: simple, mad, round, store)."""
    used = []
    names = ("simple", "mad", "round", "store")
    for unit in range(2 if imm16 else 4):
        byte = (word >> (8 * unit)) & 0xFF
        select = byte & 7
        if select >= 4:
            used.append(f"{names[unit]}:INSTR{select - 4}")
        elif select == 3:
            used.append(f"{names[unit]}:STORE")
        elif select == 2:
            used.append(f"{names[unit]}:NOP")
    return ",".join(used) if used else "none"


# --------------------------------------------------------------------------- #
# File-level replay recordings (for replay linking)
# --------------------------------------------------------------------------- #
_FILE_INFO: dict[tuple[str, str, str], dict] = {}


def _file_recordings(ctx, fn) -> list:
    """[(recorder key, window start, [(op, args, offset)])] for the file of ``fn``."""
    return _file_info(ctx, fn)["recordings"]


def _file_templates(ctx, fn) -> list:
    """[(writer key, template slot, op, args)] of backdoor-installed LOADMACRO templates in the file."""
    return _file_info(ctx, fn)["templates"]


def _file_info(ctx, fn) -> dict:
    """Replay recordings and LOADMACRO templates of the file holding ``fn`` (re-scanned once)."""
    root = str(getattr(ctx.tables, "repo_root", ""))
    path = Path(root) / fn.path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    key = (root, fn.path, hashlib.sha256(text.encode("utf-8")).hexdigest())
    cached = _FILE_INFO.get(key)
    if cached is not None:
        return cached
    info: dict = {"recordings": [], "templates": []}
    _FILE_INFO[key] = info
    if not text or not re.search(
        r"\blltt::record\b|\bload_replay_buf\b|TTI?_REPLAY\s*\(|TTI?_SFP", text
    ):
        return info
    source = scan.scan_text(text, fn.path, fn.layer)
    for other in source.functions:
        scan.attach_calls(other, {}, set())
        analysis = _Analysis(other, _NullContext(ctx.tables))
        events = analysis.events()
        windows: dict[int, tuple] = {}
        for event in events:
            if event.kind == "insn" and not event.deferred:
                position = _BACKDOOR_VD.get(event.op)
                if position is not None and position < len(event.args):
                    vd = analysis.ival(event.args[position], event.offset)
                    if vd is not None and 12 <= vd <= 15:
                        info["templates"].append(
                            (
                                other.key,
                                vd - 12,
                                event.op,
                                tuple(
                                    analysis.inline(a, event.offset) for a in event.args
                                ),
                            )
                        )
            if event.rec is not None and event.kind == "insn":
                start, _, origin = event.rec
                windows.setdefault(origin, (start, []))[1].append(
                    (
                        event.op,
                        tuple(analysis.inline(a, event.offset) for a in event.args),
                        event.offset,
                    )
                )
            elif event.rec is not None:
                start, _, origin = event.rec
                windows.setdefault(origin, (start, []))[1].append(
                    ("", (), event.offset)
                )
        for origin in sorted(windows):
            start, items = windows[origin]
            info["recordings"].append((other.key, start, items))
    return info


class _NullContext:
    """Minimal ctx for re-scanned recorder functions (no effects are emitted)."""

    def __init__(self, tables):
        self.tables = tables

    def effects(self, *args, **kwargs):
        return []


# --------------------------------------------------------------------------- #
# Call-graph repair: forwarded SFPU callables
# --------------------------------------------------------------------------- #
def _inject_forwarded_callables(fn) -> None:
    params = set(fn.param_names)
    if not params:
        return
    existing = {(c.name, c.line, tuple(c.args)) for c in fn.calls}
    added = []
    for call in fn.calls:
        spec = _FORWARDERS.get(call.name)
        if spec is None or call.is_member or not call.args:
            continue
        target = _unforward(call.args[0])
        if target not in params:
            continue
        fixed, tail = spec
        args = [_unforward(call.args[i]) for i in fixed if i < len(call.args)]
        args += [_unforward(a) for a in call.args[tail:]]
        site_key = (target, call.line, tuple(args))
        if site_key in existing:
            continue
        existing.add(site_key)
        added.append(
            scan.CallSite(
                name=target,
                callee=target,
                template_args=(),
                args=tuple(args),
                line=call.line,
                offset=call.offset,
                thread=call.thread,
                condition=call.condition,
                via_macro=call.via_macro,
            )
        )
    body = fn.masked_body or ""
    for match in re.finditer(
        r"std::forward\s*<[^<>()]*>\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\(", body
    ):
        target = match.group(1)
        if target not in params:
            continue
        opening = match.end() - 1
        close = scan.match_close(body, opening, "()")
        if close is None:
            continue
        args = tuple(_unforward(a) for a in scan.split_args(body[opening + 1 : close]))
        line = fn.line_at(match.start())
        site_key = (target, line, args)
        if site_key in existing:
            continue
        existing.add(site_key)
        added.append(
            scan.CallSite(
                name=target,
                callee=target,
                template_args=(),
                args=args,
                line=line,
                offset=match.start(),
                thread=None,
                condition=fn.condition_at(match.start()),
            )
        )
    if added:
        fn.calls = tuple(fn.calls) + tuple(added)


# --------------------------------------------------------------------------- #
# Hook
# --------------------------------------------------------------------------- #
def extract(fn, ctx):
    """Direct SFPU effects of one scanned function (see module docstring)."""
    global _TABLES
    _TABLES = ctx.tables
    body = fn.masked_body or ""
    if not body or not _QUICK.search(body):
        return []
    if INJECT_FORWARDED_CALLABLES:
        _inject_forwarded_callables(fn)
    if not re.search(
        r"SFP|l_reg|dst_reg|v_if|vFloat|vInt|vUInt|vConst|sfpi::|replay|record|__builtin_rvtt_sfp",
        body,
    ):
        return []
    return _Analysis(fn, ctx).run()


# --------------------------------------------------------------------------- #
# Compute-API init / execute pairing (for the checker; not used by build())
# --------------------------------------------------------------------------- #
def init_execute_pairs(db) -> list[dict]:
    """Pair each compute-API SFPU execute function with the init(s) of its file.

    Name rule: ``<api>`` pairs with ``<api>_init`` (``exp_tile`` / ``exp_tile_init``,
    ``add_binary_tile`` / ``add_binary_tile_init``); otherwise with every SFPU init of the
    same file. ``self_init`` is true when the execute reaches its paired init through the
    call graph (the folded tree's ``exp_tile`` calls ``exp_tile_init``); ``reaches_init`` lists
    every init/reconfig the execute reaches (also an inlined ``SFPU_*_INIT*`` macro). ``reads`` /
    ``writes`` are the SFPU and ADDR_MOD resources of the execute's transitive reads and of
    the inits' transitive writes, so a checker can see which state the pair hands over.
    Deterministic: sorted by execute key.
    """
    by_file: dict[str, list] = {}
    for key in db.keys():
        fn = db.function(key)
        if (
            fn.layer != "compute_api"
            or fn.definition is None
            or not in_scope(fn.definition)
        ):
            continue
        by_file.setdefault(fn.path, []).append(fn)
    pairs = []
    for path in sorted(by_file):
        functions = by_file[path]
        inits = [fn for fn in functions if fn.role == "init"]
        for fn in functions:
            if fn.role != "execute":
                continue
            named = [i for i in inits if i.name == f"{fn.name}_init"]
            chosen = named or inits
            if not chosen:
                continue
            reachable = {k for k, _, _ in db.reachable(fn.key)}
            reads = sorted(
                {
                    e.resource
                    for e in db.effects(fn.key)
                    if e.access == "read"
                    and e.resource.startswith(("SFPU.", "ADDR_MOD."))
                }
            )
            writes = sorted(
                {
                    e.resource
                    for init in chosen
                    for e in db.effects(init.key)
                    if e.access == "write"
                    and e.resource.startswith(("SFPU.", "ADDR_MOD."))
                }
            )
            pairs.append(
                {
                    "execute": fn.key,
                    "inits": sorted(i.key for i in chosen),
                    "rule": "name" if named else "file",
                    "path": path,
                    "self_init": any(i.key in reachable for i in chosen),
                    "reaches_init": sorted(
                        k
                        for k in reachable
                        if db.function(k) is not None
                        and db.function(k).role in ("init", "reconfig")
                    ),
                    "reads": reads,
                    "writes": writes,
                }
            )
    return sorted(pairs, key=lambda p: p["execute"])
