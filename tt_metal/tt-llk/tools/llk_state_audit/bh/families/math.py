# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole math (FPU, non-SFPU) family for the state audit.

Scope (path based, see :func:`in_scope`): ``llk_lib/llk_math_*.h`` and
``llk_lib/experimental/llk_math_*.h`` (non-sfpu), ``common/inc/cmath_common.h``
and the BH ``llk_api`` math wrappers (``llk_math_*.h``, ``experimental/`` and
``experimental/2_0/``).

The family owns ADDR_MOD semantics, RWC, the MOP program value, replay ranges and
the dest-offset state (SCHEMA.md, section 2), so it **replaces** eleven core
patterns (``DISABLE_CORE``) for *every* scanned function. Every family's code keeps
its coverage with the same resources, evidence, conditions and threads (checked
site by site against the core on both trees); only values and details get more
precise. Instruction macros without a callee are now matched in ``extract``
instead of by a stop pattern, so they no longer appear as callee-less stopped
``CallEdge`` rows (transitive effects are unchanged).

``math.addr_mod_set`` (replaces ``core.addr_mod_set``)
    ``addr_mod_t{..}.set(n)`` -> write ``ADDR_MOD.n``. The value is the canonical
    initializer: non-zero sub-fields in declaration order, e.g.
    ``{.srca.incr=16, .srcb.cr=1, .dest.incr=8}``. All ``addr_mod_t`` members
    default to 0 and ``set()`` writes the src, dest and bias words whole, so two
    textually different but equivalent initializers get the same value, and a
    fully constant one classifies as ``const``.
``math.addr_mod_use`` (replaces ``core.addr_mod_use``)
    ADDR_MOD operand of ELW*/MVMUL/MOV*/POOL/ZEROACC/SFPLOAD.. (read). ``detail``
    is ``deferred`` for ``TT_OP_*`` words *and* for instructions recorded into the
    replay buffer without executing (inside a ``load_replay_buf`` /
    ``code_sequence.load`` lambda, or the ``len`` instructions after
    ``lltt::record<NoExec>`` / ``TTI_REPLAY(.., load_mode=1)``).
``math.setrwc`` (replaces ``core.setrwc``)
    ``SETRWC(clear_ab, cr, d, b, a, BitMask)`` writes exactly the counters selected
    by ``BitMask`` (``RWC.SRCA/SRCB/DEST/FIDELITY``). A zero mask writes no counter
    (it only clears src valid), so no ``RWC.CR`` placeholder is emitted. Value
    ``.cr=<cr>;.d=<d>;.b=<b>;.a=<a>``.
``math.mop_program`` (replaces ``core.mop_program``)
    ``tmp.program()`` -> write ``MOP``. When ``tmp`` is a local
    ``ckernel_template`` / ``ckernel_unpack_template`` the value is a program
    signature ``mop::<function>#0x<hash of path:line>(<params>)`` whose arguments
    are the reporting function's parameters that flow (directly or through locals)
    into the constructor, the setters and the setters' extra conditions. After
    substitution two programs compare equal exactly when they come from the same
    program site with the same arguments. ``detail`` lists the instruction words
    (``MOP:ELWADD,INCRWC,SETRWC``; ``*`` is an opaque word). Other receivers keep
    the receiver text (core behaviour).

``math.setc16`` (replaces ``core.setc16``)
    Identical to the core pattern, except that a value that is a local reassigned
    in straight-line code is replaced by its composed value
    (``set_dst_write_addr``: ``(tile_index << DstTileSizeLog2[tile_shape]) +
    get_dest_buffer_base()`` instead of the callee-local name ``dst_index``, which
    the graph would otherwise leak into every caller as the same text).
``math.mailbox_write`` (replaces ``core.mailbox_write``)
    Same as the core pattern, with the same value composition as ``math.setc16``
    (``set_dst_write_addr<.., DestReg>`` sends the composed dest index).
``math.replay_*`` / ``math.load_replay_buf`` (replace the five core replay patterns)
    Same matches as the core; the ``math.replay`` normalizer keeps
    ``detail=start=<s>;len=<l>`` (namespace qualifiers dropped, integer arithmetic
    folded) plus ``;deferred`` for ``TT_OP_REPLAY`` words and
    ``lltt::replay_insn``. Record/load writes carry value ``exec``/``noexec``.

Additional effects:

``math.incrwc``          ``INCRWC(cr, d, b, a)`` -> write ``RWC.DEST/SRCB/SRCA`` for each
                         increment that is not literally 0 (value ``+<incr>``).
``math.mop_word``        ``TT_OP_MOP`` word (math thread) -> deferred read of ``MOP``.
``math.sw_read``         Reads of the math software shadows ``src_zero_flag_hw``,
                         ``src_zero_flag_srca_fmt/srcb_fmt``, ``dest_offset_id`` and
                         ``math_sync_tile_dst_index`` (core models the writes). Reads inside
                         ``LLK_ASSERT`` carry ``defined(ENABLE_LLK_ASSERT)``. In scope, plus the
                         ``ckernel.h`` helpers ``get_dest_buffer_base`` / ``update_dest_offset_id``.
``math.semaphore_read``  ``semaphore_read(semaphore::X)`` -> read ``SEM.X`` (in scope; disabled
                         automatically when a core pattern for it appears).
``math.cfg_mask_field``  ``cfg_reg_rmw_tensix<X_ADDR32, SHAMT, MASK>(v)`` whose mask spans several
                         fields of the word: one field write per covered field that the core field
                         effect does not already name (in scope).
``math.fpu_implicit``    Implicit config reads of immediate FPU instructions (in scope):
                         ``CFG.ALU_ACC_CTRL_Fp32_enabled/INT8_math_enabled/Zero_Flag_disabled_src``,
                         ``CFG.ALU_FORMAT_SPEC_REG0_SrcA``, ``CFG.DEST_ACCESS_CFG_zeroacc_absolute_tile_mode``
                         and ``THDCFG.DEST_TARGET_REG_CFG_MATH_Offset``, per instruction class, one
                         read per (field, condition, thread) per function, ``detail=implicit``,
                         ``confidence=medium``. MOP runs, replay runs and ``TT_INSN`` table issues
                         read the union of the FPU math fields with ``confidence=low`` and
                         ``detail=implicit:mop|implicit:replay|implicit:insn``.
``math.insn_word``       ``TT_INSN(word)`` (in scope) -> low-confidence reads of ``MOP`` and ``REPLAY``
                         (the face-compressed tables hold MOP/REPLAY/ZEROACC/SETRWC words).

Roles: functions in scope that the default rules call ``execute`` but that only
compute values (``get_*``, ``is_*``, ``*_validate_*``, ``*_supports_*``,
``*_func``, ``*_geom``, ``*_len``, ``*_tables``, ``run_throttled_sequence*``)
become ``helper``. Threads: functions in scope default to ``math``.

No BH math code uses ``p_gpr_math`` GPRs (both trees), so there is nothing to
model for them beyond the core SETDMAREG/WRCFG/regfile patterns.
"""

from __future__ import annotations

import hashlib
import re
import sys

from .. import effects_db, scan
from ..model import Pattern, PatternMatch, compact, identifiers
from ..resources import Normalized

FAMILY = "math"
ENABLED = True

DISABLE_CORE: tuple[str, ...] = (
    "core.addr_mod_set",
    "core.addr_mod_use",
    "core.load_replay_buf",
    "core.mailbox_write",
    "core.mop_program",
    "core.replay_insn.record",
    "core.replay_insn.replay",
    "core.replay_record",
    "core.replay_run",
    "core.setc16",
    "core.setrwc",
)

# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
_SCOPE_RE = (
    re.compile(
        r"^tt_metal/tt-llk/tt_llk_blackhole/llk_lib/(?:experimental/)?llk_math_[^/]*\.h$"
    ),
    re.compile(r"^tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common\.h$"),
    re.compile(
        r"^tt_metal/hw/ckernels/blackhole/metal/llk_api/(?:experimental/(?:2_0/)?)?llk_math_[^/]*\.h$"
    ),
)
#: common helpers (outside the scope files) whose software-shadow reads math models.
_SW_HELPERS = {
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h": (
        "get_dest_buffer_base",
        "update_dest_offset_id",
    )
}


def in_scope(path: str) -> bool:
    """True for the BH math (FPU, non-SFPU) headers this family covers."""
    if "sfpu" in path:
        return False
    return any(regex.match(path) for regex in _SCOPE_RE)


# --------------------------------------------------------------------------- #
# Instruction helpers
# --------------------------------------------------------------------------- #
_INSN = re.compile(r"TT(I|_OP)?_([A-Z0-9_]+)")
#: object-like instruction macros (no operands, written without parentheses).
_OBJECT_INSN = re.compile(
    r"(?<![\w.])TTI_(?:NOP|DMANOP|TRNSPSRCA|TRNSPSRCB)\b(?!\s*\()"
)
_ZERO = re.compile(r"\(?\s*(?:0+|0x0+|0[uU]|false)\s*\)?")
_INT = re.compile(r"(0x[0-9a-fA-F]+|\d+)[uUlL]*")


def _insn(name: str) -> tuple[str, str] | None:
    """``TTI_SETRWC`` -> ("TTI", "SETRWC"); ``TT_OP_MOP`` -> ("TT_OP", "MOP"); else None."""
    match = _INSN.fullmatch(name)
    if match is None:
        return None
    form = {"I": "TTI", "_OP": "TT_OP", None: "TT"}[match.group(1)]
    return form, match.group(2)


def _is_zero(text: str) -> bool:
    return bool(_ZERO.fullmatch(compact(text)))


def _int_literal(text: str) -> int | None:
    match = _INT.fullmatch(compact(text))
    return int(match.group(1), 0) if match else None


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
_RWC_UNITS = ((1, "SRCA"), (2, "SRCB"), (4, "DEST"), (8, "FIDELITY"))
_SET_TOKEN = re.compile(r"(?:(?:::)?ckernel::)?(?:p_setrwc::)?SET_([ABDF_]+)")


def _rwc_mask(text: str) -> int | None:
    """``p_setrwc::SET_ABD_F`` / ``SET_A | SET_D`` / ``0x7`` -> bitmask (None if unknown)."""
    value = 0
    for token in re.split(r"\s*[|+]\s*", compact(text).strip("() ")):
        literal = _int_literal(token)
        if literal is not None:
            value |= literal
            continue
        match = _SET_TOKEN.fullmatch(token.strip("() "))
        if match is None:
            return None
        letters = match.group(1).replace("_", "")
        value |= (
            (1 if "A" in letters else 0)
            | (2 if "B" in letters else 0)
            | (4 if "D" in letters else 0)
            | (8 if "F" in letters else 0)
        )
    return value


def _rwc_set(expr, thread=None):
    """SETRWC ``BitMask`` operand -> the RWC counters it writes ([] for a zero mask)."""
    text = compact(expr or "")
    if not text:
        return None
    mask = _rwc_mask(text)
    if mask is None:
        return [Normalized(f"RWC.?{text}", "slot", resolved=False)]
    return [Normalized(f"RWC.{unit}", "slot") for bit, unit in _RWC_UNITS if mask & bit]


def _rwc_name(expr, thread=None):
    """A literal ``RWC.<UNIT>`` name (INCRWC effects)."""
    text = compact(expr or "")
    return [Normalized(text, "slot")] if text else None


_REPLAY_SEP = "#"


def _replay_text(text: str) -> str:
    """Compact a replay start/len operand; fold pure integer arithmetic."""
    value = compact(text)
    # namespace qualifiers do not change the value (ckernel::math::replay_buf_offset)
    value = re.sub(
        r"(?<![\w:])(?:::)?(?:[A-Za-z_]\w*\s*::\s*)+(?=[A-Za-z_])", "", value
    )
    folded = scan.evaluate(value, symbolic_qualified=False) if value else None
    if isinstance(folded, int) and not isinstance(folded, bool):
        return str(folded)
    return value


def _replay(expr, thread=None):
    """``start#len[#tag..]`` -> ``REPLAY`` with ``detail=start=..;len=..[;tag..]``.

    Tags: ``deferred`` (a ``TT_OP_REPLAY`` word / ``lltt::replay_insn``) and
    ``insn-word`` (a ``TT_INSN`` table issue whose range is unknown, ``start=?``).

    (The core ``replay`` normalizer overwrites the pattern detail with the
    resource text, which loses both the range and the deferred marker.)
    """
    parts = (expr or "").split(_REPLAY_SEP)
    if len(parts) < 2:
        return [Normalized("REPLAY", "range")]
    detail = f"start={_replay_text(parts[0])};len={_replay_text(parts[1])}"
    for tag in parts[2:]:
        if tag.strip():
            detail += f";{tag.strip()}"
    return [Normalized("REPLAY", "range", detail=detail)]


NORMALIZERS = {
    "math.replay": _replay,
    "math.rwc_name": _rwc_name,
    "math.rwc_set": _rwc_set,
}

GROUPS = [
    ("math.alu_acc", r"^CFG\.ALU_ACC_CTRL_"),
    ("math.alu_format", r"^CFG\.ALU_FORMAT_SPEC_REG"),
    (
        "math.zero_flag",
        r"^CFG\.ALU_ACC_CTRL_Zero_Flag_disabled_src$|^SW\.src_zero_flag_",
    ),
    (
        "math.dest_section",
        r"^THDCFG\.DEST_TARGET_REG_CFG_MATH_|^SW\.dest_offset_id$|^SW\.math_sync_tile_dst_index$",
    ),
    ("math.dest_access", r"^CFG\.DEST_ACCESS_CFG_"),
    ("math.rwc", r"^RWC\."),
    ("math.fidelity", r"^RWC\.FIDELITY$"),
    ("math.sync", r"^SEM\.(?:MATH_PACK|MATH_DONE|UNPACK_TO_DEST)$"),
]


# --------------------------------------------------------------------------- #
# addr_mod_t canonical values
# --------------------------------------------------------------------------- #
_AM_FIELDS = ("srca", "srcb", "dest", "fidelity", "bias", "pack_ysrc", "pack_ydst")
_AM_SUBFIELDS = ("incr", "clr", "cr", "c_to_cr")
_AM_HEAD = re.compile(r"^(?:(?:::)?[A-Za-z_]\w*\s*::\s*)*addr_mod_t\s*\{")


def canonical_addr_mod(receiver: str) -> str | None:
    """``addr_mod_t {.srca = {.incr = 0}, .dest = {.incr = 8}}`` -> ``{.dest.incr=8}``.

    Returns None when the initializer is not a designated ``addr_mod_t{..}``.
    """
    text = compact(receiver)
    head = _AM_HEAD.match(text)
    if head is None:
        return None
    close = scan.match_close(text, head.end() - 1, "{}")
    if close is None or text[close + 1 :].strip():
        return None
    values: dict[tuple[str, str], str] = {}
    for item in scan.split_args(text[head.end() : close]):
        if not item:
            continue  # trailing comma
        match = re.fullmatch(r"\.(\w+)\s*=\s*\{(.*)\}", item)
        if match is None or match.group(1) not in _AM_FIELDS:
            return None
        for sub in scan.split_args(match.group(2)):
            if not sub:
                continue
            sub_match = re.fullmatch(r"\.(\w+)\s*=\s*(.+)", sub)
            if sub_match is None or sub_match.group(1) not in _AM_SUBFIELDS:
                return None
            values[(match.group(1), sub_match.group(1))] = compact(sub_match.group(2))
    parts = [
        f".{field}.{sub}={values[(field, sub)]}"
        for field in _AM_FIELDS
        for sub in _AM_SUBFIELDS
        if (field, sub) in values and not _is_zero(values[(field, sub)])
    ]
    return "{" + ", ".join(parts) + "}"


def _addr_mod_value(m: PatternMatch) -> str | None:
    return canonical_addr_mod(m.receiver) or compact(m.receiver) or None


# --------------------------------------------------------------------------- #
# MOP program signature values
# --------------------------------------------------------------------------- #
_MOP_TYPES = r"(?:(?:::)?ckernel::)?(?:ckernel_template|ckernel_unpack_template|auto)"
_MOP_SETTERS = frozenset(
    {
        "set_loop_op0",
        "set_loop_op1",
        "set_end_ops",
        "set_end_op",
        "set_start_op",
        "set_last_inner_loop_instr",
        "set_last_outer_loop_instr",
    }
)
_LOCAL_ASSIGN = re.compile(
    r"(?<![\w.>:])([A-Za-z_]\w*)(?:\s*\[[^\]]*\])?(?:\.[A-Za-z_]\w*)*\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)"
)


def _current_function():
    """The ``scan.FunctionDef`` whose patterns are being applied.

    ``PatternMatch`` has no reference to the enclosing function (see
    out/requests/math.md), so the MOP value spec finds it on the call stack of
    ``effects_db._direct_effects``. Returns None when not found (the value then
    falls back to the receiver text, the core behaviour).
    """
    frame = sys._getframe(2)
    for _ in range(16):
        if frame is None:
            return None
        candidate = frame.f_locals.get("fn")
        if (
            candidate is not None
            and hasattr(candidate, "masked_body")
            and hasattr(candidate, "calls")
            and hasattr(candidate, "param_names")
        ):
            return candidate
        frame = frame.f_back
    return None


def param_flow(fn) -> dict[str, set[str]]:
    """local name -> parameters of ``fn`` it is computed from (transitively)."""
    params = set(fn.param_names)
    flow: dict[str, set[str]] = {}
    text = fn.masked_body
    for _ in range(3):
        for match in _LOCAL_ASSIGN.finditer(text):
            name = match.group(1)
            if name in params or name in scan._KEYWORDS:
                continue
            end = scan.statement_end(text, match.end())
            sources: set[str] = set()
            for ident in identifiers(text[match.end() : end]):
                if ident in params:
                    sources.add(ident)
                elif ident in flow:
                    sources |= flow[ident]
            if sources:
                flow.setdefault(name, set()).update(sources)
    return flow


def _resolve_local(body: str, text: str, depth: int = 3) -> str:
    """Initializer text of a local named by ``text`` (``static constexpr x = TT_OP_..``)."""
    for _ in range(depth):
        name = compact(text)
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            break
        match = re.search(rf"(?<![\w.]){re.escape(name)}\s*=(?!=)\s*([^;]*);", body)
        if match is None:
            break
        text = compact(match.group(1))
    return text


def _word_names(text: str, body: str = "") -> list[str]:
    """Instruction mnemonics of one MOP operand text (``*`` for an opaque word)."""
    if body:
        text = _resolve_local(body, text)
    names = re.findall(r"\bTT_OP_([A-Z0-9_]+)", text)
    if re.search(r"\breplay_insn\s*\(", text):
        names.append("REPLAY")
    if not names and re.search(r"[A-Za-z_]\w*\s*(?:<[^;]*>)?\s*\(", text):
        names.append("*")
    return names


def mop_program_info(fn, receiver: str, offset: int):
    """(relevant params, op mnemonics) of the MOP programmed by ``receiver.program()``.

    ``receiver`` must be a local declared before ``offset`` as
    ``ckernel_template r(..)``, ``.. r{..}`` or ``.. r = ..``. Returns None otherwise.
    """
    body = fn.masked_body
    decls = [
        match
        for match in re.finditer(
            rf"(?<![\w.]){_MOP_TYPES}\s+{re.escape(receiver)}\s*(\(|\{{|=(?!=))", body
        )
        if match.start() < offset
    ]
    if not decls:
        return None
    decl = decls[-1]
    opener = decl.group(1)
    if opener in ("(", "{"):
        close = scan.match_close(body, decl.end() - 1, "()" if opener == "(" else "{}")
        if close is None:
            return None
        init_texts = list(scan.split_args(body[decl.end() : close]))
    else:
        end = scan.statement_end(body, decl.end())
        init_texts = [compact(body[decl.end() : end])]
    program_condition = ""
    for call in fn.calls:
        if call.offset == offset and call.is_member:
            program_condition = call.condition or ""
            break
    texts = list(init_texts)
    ops: list[str] = []
    word_texts = list(init_texts)
    if opener in ("(", "{"):
        word_texts = word_texts[2:]  # (outer/unpackB, inner/unpackHalo, words...)
    elif len(word_texts) == 1:
        copy = re.fullmatch(
            r"(?:(?:::)?ckernel::)?(?:ckernel_template|ckernel_unpack_template)\s*\((.*)\)",
            word_texts[0],
        )
        if copy is not None:
            word_texts = list(scan.split_args(copy.group(1)))[2:]
    for text in word_texts:
        ops.extend(_word_names(text, body))
    condition_ids: set[str] = set()
    for call in fn.calls:
        if (
            call.is_member
            and call.name in _MOP_SETTERS
            and compact(call.receiver or "") == receiver
            and decl.start() < call.offset < offset
        ):
            texts.extend(call.args)
            for arg in call.args:
                ops.extend(_word_names(arg, body))
            if (call.condition or "") != program_condition:
                condition_ids |= identifiers(call.condition) - identifiers(
                    program_condition
                )
    names: set[str] = set(condition_ids)
    for text in texts:
        names |= identifiers(text)
    params = set(fn.param_names)
    flow = param_flow(fn)
    relevant: set[str] = set()
    for name in names:
        if name in params:
            relevant.add(name)
        else:
            relevant |= flow.get(name, set())
    ordered = tuple(name for name in fn.param_names if name in relevant)
    return ordered, tuple(ops)


def mop_signature(fn, line: int, params: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{fn.path}:{line}".encode("utf-8")).hexdigest()[:8]
    return f"mop::{fn.name}#0x{digest}({', '.join(params)})"


def _mop_value(m: PatternMatch) -> str | None:
    receiver = compact(m.receiver)
    function = _current_function()
    if function is None or not re.fullmatch(r"[A-Za-z_]\w*", receiver):
        return receiver or None
    info = mop_program_info(function, receiver, m.offset)
    if info is None:
        return receiver or None
    return mop_signature(function, m.line, info[0])


def _mop_detail(m: PatternMatch) -> str | None:
    receiver = compact(m.receiver)
    function = _current_function()
    if function is None or not re.fullmatch(r"[A-Za-z_]\w*", receiver):
        return None
    info = mop_program_info(function, receiver, m.offset)
    if info is None or not info[1]:
        return None
    return "MOP:" + ",".join(info[1])


def compose_local(fn, name: str, offset: int, condition: str | None) -> str | None:
    """Value of local ``name`` at ``offset`` when it is reassigned in straight-line code.

    ``T name = a; name = name + b;`` -> ``(a) + b``. Every assignment before ``offset``
    must be unconditional or under the same condition as the use (so it dominates
    it); otherwise, or when ``name`` is assigned only once (the graph substitutes
    those itself), None.
    """
    body = fn.masked_body
    pattern = re.compile(
        rf"(?<![\w.>:]){re.escape(name)}\s*((?:[+\-*/%&|^]|<<|>>)?=)(?!=)\s*"
    )
    value = None
    count = 0
    first = pattern.search(body, 0, offset)
    if first is None or re.search(
        r"\b(?:for|while|do)\b", body[first.start() : offset]
    ):
        return None  # a loop between the definition and the use: not straight-line
    for match in pattern.finditer(body, 0, offset):
        where = fn.condition_at(match.start()) or ""
        if where and where != (condition or ""):
            return None
        end = scan.statement_end(body, match.end())
        rhs = compact(body[match.end() : end])
        if not rhs:
            return None
        operator = match.group(1)
        if value is None:
            if operator != "=":
                return None
            value = rhs
        else:
            previous = value if re.fullmatch(r"[\w:.]+", value) else f"({value})"
            if operator == "=":
                value = re.sub(rf"(?<![\w.>:]){re.escape(name)}\b", previous, rhs)
            else:
                value = f"{previous} {operator[:-1]} ({rhs})"
        count += 1
    return value if count > 1 else None


def _composed_arg1(m: PatternMatch) -> str | None:
    """Operand 1 (SETC16 / mailbox_write value); a reassigned local is composed."""
    if len(m.args) < 2:
        return None
    value = m.args[1]
    name = compact(value)
    if re.fullmatch(r"[A-Za-z_]\w*", name):
        function = _current_function()
        if function is not None and name not in function.param_names:
            composed = compose_local(function, name, m.offset, m.condition)
            if composed is not None:
                return composed
    return value


def _replay_expr(m: PatternMatch, deferred: bool | None = None) -> str | None:
    """``start#len[#deferred]`` operand text for the ``math.replay`` normalizer."""
    if len(m.args) < 2:
        return None
    if deferred is None:
        deferred = m.name.startswith("TT_OP_")
    parts = [m.args[0], m.args[1]] + (["deferred"] if deferred else [])
    return _REPLAY_SEP.join(parts)


PATTERNS = [
    Pattern(
        "math.addr_mod_set",
        "member",
        r"set",
        "arg:0",
        "addr_mod",
        value=_addr_mod_value,
        receiver_type=r"addr_mod_t",
        note="addr_mod_t{..}.set(n): canonical non-zero sub-fields as the value",
    ),
    Pattern(
        "math.mop_program",
        "member",
        r"program|program_and_run",
        "literal:MOP",
        "mop",
        value=_mop_value,
        detail=_mop_detail,
        receiver_type=r"ckernel_template|ckernel_unpack_template",
        note="MOP program; value is the program signature for local templates",
    ),
    # ---- per-thread config (replaces core.setc16; composes reassigned locals) ----
    Pattern(
        "math.setc16",
        "call",
        r"TT(?:I|_OP)?_SETC16",
        "arg:0",
        "thread_cfg",
        value=_composed_arg1,
        detail=effects_db.deferred,
        note="core.setc16 plus straight-line composition of a reassigned local value",
    ),
    Pattern(
        "math.mailbox_write",
        "call",
        r"mailbox_write",
        "arg:0",
        "mailbox",
        value=_composed_arg1,
        note="core.mailbox_write plus straight-line composition of a reassigned local value",
    ),
    # ---- replay buffer (replaces the core patterns: keeps start/len and deferral) ----
    Pattern(
        "math.replay_record",
        "call",
        r"record",
        lambda m: _replay_expr(m),
        "math.replay",
        value=lambda m: "exec" if _exec_while_loading(m.template_args) else "noexec",
        qualifier=r"(?:::)?lltt",
    ),
    Pattern(
        "math.load_replay_buf",
        "call",
        r"load_replay_buf",
        lambda m: _replay_expr(m),
        "math.replay",
        value=lambda m: "exec" if _exec_while_loading(m.template_args) else "noexec",
    ),
    Pattern(
        "math.replay_insn.record",
        "call",
        r"TT(?:I|_OP)?_REPLAY",
        lambda m: _replay_expr(m),
        "math.replay",
        value=lambda m: "noexec" if _is_zero(m.args[2]) else "exec",
        when=lambda m: len(m.args) >= 4 and not _is_zero(m.args[3]),
    ),
    Pattern(
        "math.replay_insn.replay",
        "call",
        r"TT(?:I|_OP)?_REPLAY",
        lambda m: _replay_expr(m),
        "math.replay",
        access="read",
        when=lambda m: len(m.args) >= 2 and (len(m.args) < 4 or _is_zero(m.args[3])),
    ),
    Pattern(
        "math.replay_run",
        "call",
        r"replay|replay_insn",
        lambda m: _replay_expr(m, deferred=m.name == "replay_insn"),
        "math.replay",
        access="read",
        qualifier=r"(?:::)?lltt",
    ),
]


# --------------------------------------------------------------------------- #
# Replay recording windows
# --------------------------------------------------------------------------- #
def _exec_while_loading(template_args) -> bool:
    text = " ".join(compact(t) for t in template_args)
    return bool(re.search(r"(?<!No)\bExec\b", text)) and "NoExec" not in text


def _lambda_body(body: str, start: int) -> tuple[int, int] | None:
    """Body span of the lambda whose capture list opens at ``body[start] == '['``."""
    close = scan.match_close(body, start, "[]")
    if close is None:
        return None
    position = close + 1
    while position < len(body) and body[position] in " \t\n":
        position += 1
    if position < len(body) and body[position] == "(":
        params_close = scan.match_close(body, position, "()")
        if params_close is None:
            return None
        position = params_close + 1
    brace = body.find("{", position)
    if brace < 0 or ";" in body[position:brace]:
        return None
    end = scan.match_close(body, brace, "{}")
    return (brace, end) if end is not None else None


def _named_lambdas(body: str) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for match in re.finditer(r"\bauto\s+([A-Za-z_]\w*)\s*=\s*\[", body):
        span = _lambda_body(body, match.end() - 1)
        if span is not None:
            found.setdefault(match.group(1), span)
    return found


def _call_arg_spans(body: str, offset: int) -> list[tuple[int, int]]:
    """Top-level argument spans of the call whose callee text starts at ``offset``."""
    match = re.compile(r"(?:(?:::)?[A-Za-z_]\w*\s*::\s*)*[A-Za-z_]\w*\s*").match(
        body, offset
    )
    if match is None:
        return []
    position = match.end()
    if position < len(body) and body[position] == "<":
        close = scan.match_angle(body, position)
        if close is None:
            return []
        position = close + 1
        while position < len(body) and body[position] in " \t\n":
            position += 1
    if position >= len(body) or body[position] != "(":
        return []
    close = scan.match_close(body, position, "()")
    if close is None:
        return []
    spans = []
    start = position + 1
    depth = 0
    for index in range(position + 1, close):
        char = body[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            spans.append((start, index))
            start = index + 1
    spans.append((start, close))
    return spans


def _callable_span(
    body: str, span: tuple[int, int], lambdas: dict[str, tuple[int, int]]
) -> tuple[int, int] | None:
    text = body[span[0] : span[1]]
    stripped = text.strip()
    if stripped.startswith("["):
        return _lambda_body(body, span[0] + text.index("["))
    name = compact(stripped)
    if name in lambdas:
        # a named lambda that is also invoked directly is ambiguous
        direct = [
            m
            for m in re.finditer(rf"(?<![\w.]){re.escape(name)}\s*\(", body)
            if not (lambdas[name][0] <= m.start() <= lambdas[name][1])
        ]
        return None if direct else lambdas[name]
    return None


def _enclosing_block_end(body: str, offset: int) -> int:
    depth = 0
    for index in range(offset, len(body)):
        char = body[index]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return index
            depth -= 1
    return len(body)


def recorded_calls(fn, tables=None) -> frozenset[int]:
    """Indices into ``fn.calls`` of instructions recorded into the replay buffer
    without executing (NoExec load_replay_buf / code_sequence.load lambdas and
    lltt::record / TTI_REPLAY(load) windows)."""
    body = fn.masked_body
    if not re.search(r"load_replay_buf|\brecord\b|\.load\s*\(|REPLAY", body):
        return frozenset()
    calls = fn.calls
    lambdas = _named_lambdas(body)
    spans: list[tuple[int, int]] = []
    for call in calls:
        if call.via_macro:
            continue
        if call.name == "load_replay_buf" and not call.is_member:
            if _exec_while_loading(call.template_args):
                continue
            args = _call_arg_spans(body, call.offset)
            if len(args) >= 3:
                span = _callable_span(body, args[2], lambdas)
                if span is not None:
                    spans.append(span)
        elif call.is_member and call.name == "load" and len(call.args) == 1:
            args = _call_arg_spans(body, call.offset)
            if args:
                span = _callable_span(body, args[0], lambdas)
                if span is not None:
                    spans.append(span)
    recorded: set[int] = set()
    for index, call in enumerate(calls):
        if any(start <= call.offset <= end for start, end in spans):
            recorded.add(index)
    # straight-line record windows
    events: list[tuple[int, int, int | None]] = []  # (offset, order, call index)
    for index, call in enumerate(calls):
        info = _insn(call.name) if not (call.is_member or call.is_ctor) else None
        if info is not None and info[0] != "TT_OP" and info[1] != "REPLAY":
            events.append((call.offset, index, index))
    for match in _OBJECT_INSN.finditer(body):
        events.append((match.start(), -1, None))
    events.sort()
    for index, call in enumerate(calls):
        if call.is_member or call.is_ctor:
            continue
        length = None
        if call.name == "record" and call.qualifier.endswith("lltt"):
            if _exec_while_loading(call.template_args) or len(call.args) < 2:
                continue
            length_text = call.args[1]
        else:
            info = _insn(call.name)
            if info is None or info[0] == "TT_OP" or info[1] != "REPLAY":
                continue
            if (
                len(call.args) < 4
                or _is_zero(call.args[3])
                or not _is_zero(call.args[2])
            ):
                continue  # not a NoExec load
            length_text = call.args[1]
        length = _int_literal(length_text)
        if length is None and tables is not None:
            length = tables.int_value(length_text)
        after = [event for event in events if event[0] > call.offset]
        if length is None:
            end = _enclosing_block_end(body, call.offset)
            window = [event for event in after if event[0] < end]
        else:
            window = after[:length]
        recorded.update(event[2] for event in window if event[2] is not None)
    return frozenset(recorded)


# --------------------------------------------------------------------------- #
# Implicit FPU configuration reads
# --------------------------------------------------------------------------- #
_FP32 = "ALU_ACC_CTRL_Fp32_enabled"
_INT8 = "ALU_ACC_CTRL_INT8_math_enabled"
_ZERO_FLAG = "ALU_ACC_CTRL_Zero_Flag_disabled_src"
_SRCA_FMT = "ALU_FORMAT_SPEC_REG0_SrcA"
_DEST_OFFSET = "DEST_TARGET_REG_CFG_MATH_Offset"
_ZEROACC_MODE = "DEST_ACCESS_CFG_zeroacc_absolute_tile_mode"
_MATH_FIELDS = (_FP32, _INT8, _ZERO_FLAG, _SRCA_FMT, _DEST_OFFSET)
#: instruction -> config fields its execution reads (FPU datapath on the math thread).
IMPLICIT_READS: dict[str, tuple[str, ...]] = {
    **{
        name: _MATH_FIELDS
        for name in (
            "ELWADD",
            "ELWSUB",
            "ELWMUL",
            "DOTPV",
            "MVMUL",
            "GAPOOL",
            "GMPOOL",
            "APOOL3S1",
            "APOOL3S2",
            "MPOOL3S1",
            "MPOOL3S2",
            "CONV3S1",
            "CONV3S2",
            "MFCONV3S1",
        )
    },
    "MOVA2D": (_FP32, _ZERO_FLAG, _SRCA_FMT, _DEST_OFFSET),
    "MOVB2D": (_FP32, _ZERO_FLAG, _DEST_OFFSET),
    "MOVB2A": (_ZERO_FLAG,),
    "MOVD2A": (_FP32, _DEST_OFFSET),
    "MOVD2B": (_FP32, _DEST_OFFSET),
    "ZEROACC": (_ZEROACC_MODE, _DEST_OFFSET),
}


def _mop_run_kind(call) -> str | None:
    """implicit:mop / implicit:replay / implicit:insn for run sites (immediate)."""
    if call.is_ctor:
        return None
    if call.is_member:
        if call.name in ("run", "program_and_run") and (call.receiver_type or "") in (
            "ckernel_template",
        ):
            return "implicit:mop"
        return None
    if call.name == "run" and re.search(r"(?:^|::)ckernel_template$", call.qualifier):
        return "implicit:mop"
    if call.name == "mop_run":
        return "implicit:mop"
    if call.name == "replay" and call.qualifier.endswith("lltt"):
        return "implicit:replay"
    info = _insn(call.name)
    if info is None or info[0] == "TT_OP":
        return None
    if info[1] == "MOP":
        return "implicit:mop"
    if info[1] == "REPLAY" and (len(call.args) < 4 or _is_zero(call.args[3])):
        return "implicit:replay"
    return None


# --------------------------------------------------------------------------- #
# Software shadows, asserts
# --------------------------------------------------------------------------- #
MATH_SW_GLOBALS = (
    "dest_offset_id",
    "math_sync_tile_dst_index",
    "src_zero_flag_hw",
    "src_zero_flag_srca_fmt",
    "src_zero_flag_srcb_fmt",
)
_SW_READ = re.compile(
    r"(?<![\w.>])(?<!->)(?:(?:::)?[A-Za-z_]\w*::)*(?P<g>"
    + "|".join(sorted(MATH_SW_GLOBALS, key=lambda n: (-len(n), n)))
    + r")\b"
)
_ASSERT_CONDITION = "defined(ENABLE_LLK_ASSERT)"


def _assert_spans(body: str) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"(?<![\w.])LLK_ASSERT\s*\(", body):
        close = scan.match_close(body, match.end() - 1, "()")
        if close is not None:
            spans.append((match.start(), close))
    return spans


def _with_assert(condition: str | None, offset: int, spans) -> str:
    parts = [condition] if condition else []
    if any(start <= offset <= end for start, end in spans):
        parts.append(_ASSERT_CONDITION)
    return " && ".join(parts)


def _semaphore_read_in_core() -> bool:
    return any(
        re.fullmatch(p.match, "semaphore_read")
        for p in effects_db.CORE_PATTERNS
        if p.kind == "call"
    )


_CORE_SEMAPHORE_READ = _semaphore_read_in_core()


# --------------------------------------------------------------------------- #
# extract hook
# --------------------------------------------------------------------------- #
def extract(fn, ctx):
    effects = []
    scoped = in_scope(fn.path)
    recorded = recorded_calls(fn, ctx.tables)
    implicit_seen: set[tuple[str, str, str]] = set()

    def add(**kwargs):
        effects.extend(ctx.effects(fn, **kwargs))

    def implicit(call, fields, detail, confidence):
        condition = call.condition or ""
        thread = call.thread or fn.thread_hint or "any"
        for field in fields:
            key = (field, condition, thread)
            if key in implicit_seen:
                continue
            implicit_seen.add(key)
            add(
                pattern_id="math.fpu_implicit",
                resource_expr=f"{field}_RMW",
                normalizer="cfg_rmw",
                access="read",
                value=call.name,
                line=call.line,
                condition=condition,
                thread=call.thread,
                detail=detail,
                confidence=confidence,
            )

    for index, call in enumerate(fn.calls):
        if call.is_member or call.is_ctor:
            if scoped:
                kind = _mop_run_kind(call)
                if kind is not None:
                    implicit(call, _MATH_FIELDS, kind, "low")
            continue
        if call.name == "TT_INSN":
            if scoped:
                for resource, normalizer in (
                    ("MOP", "mop"),
                    (_REPLAY_SEP.join(("?", "?", "insn-word")), "math.replay"),
                ):
                    add(
                        pattern_id="math.insn_word",
                        resource_expr=resource,
                        normalizer=normalizer,
                        access="read",
                        value=", ".join(call.args),
                        line=call.line,
                        condition=call.condition or "",
                        thread=call.thread,
                        detail="insn-word",
                        confidence="low",
                    )
                implicit(call, _MATH_FIELDS, "implicit:insn", "low")
            continue
        info = _insn(call.name)
        if info is None:
            if not scoped:
                continue
            if call.name == "semaphore_read" and call.args and not _CORE_SEMAPHORE_READ:
                add(
                    pattern_id="math.semaphore_read",
                    resource_expr=call.args[0],
                    normalizer="semaphore",
                    access="read",
                    value="",
                    line=call.line,
                    condition=call.condition or "",
                    thread=call.thread,
                )
            elif call.name == "cfg_reg_rmw_tensix" and len(call.template_args) == 3:
                effects.extend(_mask_field_effects(fn, ctx, call))
            kind = _mop_run_kind(call)
            if kind is not None:
                implicit(call, _MATH_FIELDS, kind, "low")
            continue
        form, base = info
        deferred = form == "TT_OP" or index in recorded
        detail = "deferred" if deferred else ""
        common = dict(
            line=call.line,
            condition=call.condition or "",
            thread=call.thread,
            detail=detail,
        )
        position = effects_db.ADDR_MOD_OPERANDS.get(base)
        if position is not None and position < len(call.args):
            add(
                pattern_id="math.addr_mod_use",
                resource_expr=call.args[position],
                normalizer="addr_mod",
                access="read",
                value=call.name,
                **common,
            )
        if base == "SETRWC" and len(call.args) == 6:
            a = call.args
            add(
                pattern_id="math.setrwc",
                resource_expr=a[5],
                normalizer="math.rwc_set",
                access="write",
                value=f".cr={a[1]};.d={a[2]};.b={a[3]};.a={a[4]}",
                **common,
            )
        elif base == "INCRWC" and len(call.args) == 4:
            a = call.args
            suffix = "" if _is_zero(a[0]) else f";.cr={a[0]}"
            for unit, amount in (("DEST", a[1]), ("SRCB", a[2]), ("SRCA", a[3])):
                if _is_zero(amount):
                    continue
                add(
                    pattern_id="math.incrwc",
                    resource_expr=f"RWC.{unit}",
                    normalizer="math.rwc_name",
                    access="write",
                    value=f"+{amount}{suffix}",
                    **common,
                )
        elif base == "MOP" and form == "TT_OP" and (scoped or fn.thread_hint == "math"):
            a = call.args
            add(
                pattern_id="math.mop_word",
                resource_expr="MOP",
                normalizer="mop",
                access="read",
                value=(
                    f"type={a[0]};count={a[1]};zmask={a[2]}"
                    if len(a) == 3
                    else ", ".join(a)
                ),
                **common,
            )
        if scoped and not deferred:
            fields = IMPLICIT_READS.get(base)
            if fields:
                implicit(call, fields, "implicit", "medium")
            else:
                kind = _mop_run_kind(call)
                if kind is not None:
                    implicit(call, _MATH_FIELDS, kind, "low")
    if scoped or fn.name in _SW_HELPERS.get(fn.path, ()):
        effects.extend(_sw_reads(fn, ctx))
    return effects


def _sw_reads(fn, ctx):
    body = fn.masked_body
    spans = _assert_spans(body)
    found = []
    for match in _SW_READ.finditer(body):
        rest = body[match.end() : match.end() + 3]
        if re.match(r"\s*=(?!=)", rest):
            continue  # plain assignment: a write (core.sw_global)
        before = body[max(0, match.start() - 40) : match.start()]
        if re.search(r"(?:std::)?u?int\w*_t\s*$|\bauto\s*$", before):
            continue  # a declaration of a same-named local
        offset = match.start()
        found.extend(
            ctx.effects(
                fn,
                pattern_id="math.sw_read",
                resource_expr=match.group("g"),
                normalizer="sw",
                access="read",
                value="",
                line=fn.line_at(offset),
                condition=_with_assert(fn.condition_at(offset), offset, spans),
            )
        )
    return found


def _mask_field_effects(fn, ctx, call):
    """Extra field writes for ``cfg_reg_rmw_tensix<X_ADDR32, SHAMT, MASK>(v)``."""
    addr_text, shamt_text, mask_text = (compact(t) for t in call.template_args)
    addr = re.fullmatch(r"(?:ckernel::)?([A-Za-z_]\w*?)_ADDR32", addr_text)
    if addr is None or not call.args:
        return []
    table = ctx.tables.cfg
    base = table.fields.get(addr.group(1))
    if base is None:
        return []
    mask_expr = mask_text
    if re.fullmatch(r"[A-Za-z_]\w*", mask_text) and not mask_text.endswith("_MASK"):
        local = re.search(
            rf"(?<![\w.]){re.escape(mask_text)}\s*=\s*([^;]*);", fn.masked_body
        )
        if local is None:
            return []
        mask_expr = compact(local.group(1))
    names = re.findall(r"(?:ckernel::)?([A-Za-z_]\w*?)_MASK\b", mask_expr)
    if not names or len(set(names)) < 2:
        return []
    core_field = addr.group(1)
    shamt_field = re.fullmatch(r"(?:ckernel::)?([A-Za-z_]\w*?)_SHAMT", shamt_text)
    if shamt_field is not None:
        core_field = shamt_field.group(1)
    result = []
    for name in sorted(set(names), key=names.index):
        item = table.fields.get(name)
        if item is None or name == core_field:
            continue
        if (item.space, item.addr32) != (base.space, base.addr32):
            continue
        result.extend(
            ctx.effects(
                fn,
                pattern_id="math.cfg_mask_field",
                resource_expr=f"{name}_ADDR32, {name}_SHAMT, {name}_MASK",
                normalizer="cfg_rmw",
                access="write",
                value=call.args[0],
                line=call.line,
                condition=call.condition or "",
                thread=call.thread,
            )
        )
    return result


# --------------------------------------------------------------------------- #
# Roles and threads
# --------------------------------------------------------------------------- #
_HELPER_NAME = re.compile(
    r"^(?:get|is|validate)_|_validate_|_supports_|_drives_|_geom$|_len$|_tables$|_func$|^run_throttled_sequence"
)


def role(fn, default: str) -> str | None:
    if default != "execute" or not in_scope(fn.path):
        return None
    if _HELPER_NAME.search(fn.name.strip("_").lower()):
        return "helper"
    return None


def thread(fn, default: str) -> str | None:
    if default == "any" and in_scope(fn.path):
        return "math"
    return None
