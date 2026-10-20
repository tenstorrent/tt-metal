# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the Blackhole effect DB for a ``--repo-root`` (``bh-effects``).

Pipeline: :func:`scan.scan_tree` -> per-function direct effects from the core
patterns below plus every family module in ``bh/families/`` -> call edges and
roles -> :class:`model.CallGraph` for transitive effects -> :class:`EffectDB`.

The DB object is what ``bh.hazards.run(db, repo_root)`` receives; ``to_json()``
is the sorted, deterministic ``bh-effects`` output.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

from . import resources, scan
from .model import (
    ROLES,
    THREAD_VALUES,
    CallEdge,
    CallGraph,
    Effect,
    FunctionModel,
    Pattern,
    PatternMatch,
    classify_value,
    combine_conditions,
    compact,
    dedupe_effects,
    identifiers,
    merge_detail,
    substitute,
)

SCHEMA_VERSION = 1
ARCH = "blackhole"
_FAMILY_DIR = Path(__file__).with_name("families")


# --------------------------------------------------------------------------- #
# Core patterns: generic Tensix state primitives shared by every family
# --------------------------------------------------------------------------- #
def _cfg_rmw_operands(m: PatternMatch) -> str | None:
    """cfg_rmw(X_RMW, v) / cfg_rmw(ADDR32, SHAMT, MASK, v): the field operands."""
    args = resources.expand_rmw_args(m.args)
    if len(args) >= 4:
        return ", ".join(args[:3])
    return args[0] if args else None


def _last_arg(m: PatternMatch) -> str | None:
    return m.args[-1] if m.args else None


def _deferred(m: PatternMatch) -> str:
    return "deferred" if m.name.startswith("TT_OP_") else ""


#: Public detail spec: ``"deferred"`` for ``TT_OP_*`` instruction words (MOP/replay), else "".
deferred = _deferred


def _ins(name: str) -> str:
    """Regex for an instruction macro in immediate (TTI_), runtime (TT_) or word (TT_OP_) form."""
    return rf"TT(?:I|_OP)?_{name}"


#: Tensix instructions whose operand at this position selects an ADDR_MOD slot.
ADDR_MOD_OPERANDS: dict[str, int] = {
    "ELWADD": 3,
    "ELWSUB": 3,
    "ELWMUL": 3,
    "DOTPV": 3,
    "MVMUL": 2,
    "GAPOOL": 2,
    "GMPOOL": 2,
    "APOOL3S1": 1,
    "APOOL3S2": 1,
    "MPOOL3S1": 1,
    "MPOOL3S2": 1,
    "CONV3S1": 2,
    "CONV3S2": 2,
    "MFCONV3S1": 2,
    "MOVA2D": 2,
    "MOVB2D": 2,
    "MOVD2A": 2,
    "MOVD2B": 2,
    "MOVDBGA2D": 2,
    "MOVDBGB2D": 2,
    "MOVB2A": 1,
    "SFPLOAD": 2,
    "SFPLOADMACRO": 2,
    "SFPSTORE": 2,
    "SHIFTXB": 0,
    "ZEROACC": 3,
}


#: DMA-register ALU instructions with an ``OpSel`` operand (5 operands:
#: ``OpBisConst, OpSel, Result, OpB, OpA``); the others have 4 (``OpBisConst, Result, OpB, OpA``).
_DMAREG_OPSEL = ("BITWOPDMAREG", "SHIFTDMAREG", "CMPDMAREG")


def _dmareg_operand(m: PatternMatch, which: str) -> str | None:
    base = re.sub(r"^TT(?:I|_OP)?_", "", m.name)
    first = 2 if base in _DMAREG_OPSEL else 1
    position = {"result": first, "b": first + 1, "a": first + 2}[which]
    return m.args[position] if position < len(m.args) else None


def _addr_mod_operand(m: PatternMatch) -> str | None:
    base = re.sub(r"^TT(?:I|_OP)?_", "", m.name)
    position = ADDR_MOD_OPERANDS.get(base)
    if position is None or position >= len(m.args):
        return None
    return m.args[position]


CORE_PATTERNS: tuple[Pattern, ...] = (
    # ---- shared config registers ----------------------------------------- #
    Pattern(
        "core.cfg_reg_rmw_tensix",
        "call",
        r"cfg_reg_rmw_tensix",
        "targs",
        "cfg_rmw",
        value="arg:0",
    ),
    Pattern(
        "core.cfg_rmw",
        "call",
        r"cfg_rmw",
        _cfg_rmw_operands,
        "cfg_rmw",
        value=_last_arg,
    ),
    Pattern(
        "core.cfg_rmw_gpr.cfg",
        "call",
        r"cfg_rmw_gpr",
        _cfg_rmw_operands,
        "cfg_rmw",
        value="fmt:GPR({arg3})",
    ),
    Pattern(
        "core.cfg_rmw_gpr.gpr", "call", r"cfg_rmw_gpr", _last_arg, "gpr", access="read"
    ),
    Pattern(
        "core.cfg_write", "call", r"cfg_write", "arg:0", "cfg_addr32", value="arg:1"
    ),
    Pattern(
        "core.cfg_store", "call", r"TT_CFG_STORE", "arg:1", "cfg_addr32", value="arg:2"
    ),
    Pattern(
        "core.cfg_assign",
        "assign",
        r"cfg|cfg_regs|cfg_ptr",
        "index",
        "cfg_addr32",
        value="rhs",
        when=lambda m: bool(m.index),
    ),
    Pattern(
        "core.wrcfg.cfg",
        "call",
        _ins("WRCFG"),
        "arg:2",
        "cfg_addr32",
        value="fmt:GPR({arg0})",
        detail=_deferred,
    ),
    Pattern(
        "core.wrcfg.gpr",
        "call",
        _ins("WRCFG"),
        "arg:0",
        "gpr",
        access="read",
        detail=_deferred,
    ),
    Pattern(
        "core.rdcfg.gpr",
        "call",
        _ins("RDCFG"),
        "arg:0",
        "gpr",
        value="fmt:CFG({arg1})",
        detail=_deferred,
    ),
    Pattern(
        "core.rdcfg.cfg",
        "call",
        _ins("RDCFG"),
        "arg:1",
        "cfg_addr32",
        access="read",
        detail=_deferred,
    ),
    Pattern(
        "core.rmwcib",
        "call",
        _ins(r"RMWCIB[0-3]"),
        "arg:2",
        "cfg_byte",
        value="fmt:mask={arg0};data={arg1}",
        detail=_deferred,
    ),
    Pattern(
        "core.cfgshiftmask",
        "call",
        _ins("CFGSHIFTMASK"),
        "arg:5",
        "cfg_byte",
        value="fmt:op={arg1};width={arg2};shift={arg3}",
        detail=_deferred,
    ),
    # ---- per-thread config registers (SETC16) ----------------------------- #
    Pattern(
        "core.setc16",
        "call",
        _ins("SETC16"),
        "arg:0",
        "thread_cfg",
        value="arg:1",
        detail=_deferred,
    ),
    # ---- GPRs ------------------------------------------------------------- #
    Pattern(
        "core.setdmareg",
        "call",
        _ins("SETDMAREG"),
        "arg:3",
        "gpr",
        value="arg:1",
        detail=_deferred,
    ),
    Pattern(
        "core.dmareg_alu.write",
        "call",
        _ins(r"(?:ADD|SUB|MUL|BITWOP|SHIFT|CMP)DMAREG"),
        lambda m: _dmareg_operand(m, "result"),
        "gpr",
        value=lambda m: f"{m.name}({_dmareg_operand(m, 'a')}, {_dmareg_operand(m, 'b')})",
        detail=_deferred,
    ),
    Pattern(
        "core.dmareg_alu.read",
        "call",
        _ins(r"(?:ADD|SUB|MUL|BITWOP|SHIFT|CMP)DMAREG"),
        lambda m: _dmareg_operand(m, "a"),
        "gpr",
        access="read",
        detail=_deferred,
    ),
    Pattern(
        "core.dmareg_alu.read_b",
        "call",
        _ins(r"(?:ADD|SUB|MUL|BITWOP|SHIFT|CMP)DMAREG"),
        lambda m: _dmareg_operand(m, "b"),
        "gpr",
        access="read",
        detail=_deferred,
        when=lambda m: compact(m.args[0] if m.args else "") not in ("1", "true"),
    ),
    Pattern(
        "core.regfile_assign",
        "assign",
        r"regfile",
        "index",
        "gpr",
        value="rhs",
        when=lambda m: bool(m.index),
    ),
    # ---- address modifiers ------------------------------------------------ #
    Pattern(
        "core.addr_mod_set",
        "member",
        r"set",
        "arg:0",
        "addr_mod",
        value="receiver",
        receiver_type=r"addr_mod_t",
    ),
    Pattern(
        "core.addr_mod_pack_set",
        "member",
        r"set",
        "arg:0",
        "addr_mod_pack",
        value="receiver",
        receiver_type=r"addr_mod_pack_t",
    ),
    Pattern(
        "core.addr_mod_use",
        "call",
        _ins("(?:" + "|".join(sorted(ADDR_MOD_OPERANDS)) + ")"),
        _addr_mod_operand,
        "addr_mod",
        access="read",
        value="name",
        detail=_deferred,
    ),
    Pattern(
        "core.addr_mod_pack_use",
        "call",
        _ins("PACR"),
        "arg:3",
        "addr_mod_pack",
        access="read",
        value="name",
        detail=_deferred,
    ),
    # ---- MOP --------------------------------------------------------------- #
    Pattern(
        "core.mop_program",
        "member",
        r"program|program_and_run",
        "literal:MOP",
        "mop",
        value="receiver",
        receiver_type=r"ckernel_template|ckernel_unpack_template",
    ),
    Pattern(
        "core.mop_cfg_assign",
        "assign",
        r"mop_cfg",
        "literal:MOP",
        "mop",
        value="rhs",
        detail="index",
        when=lambda m: bool(m.index),
    ),
    Pattern(
        "core.mop_run.member",
        "member",
        r"run|program_and_run",
        "literal:MOP",
        "mop",
        access="read",
        receiver_type=r"ckernel_template|ckernel_unpack_template",
    ),
    Pattern(
        "core.mop_run.insn",
        "call",
        r"TTI?_MOP",
        "literal:MOP",
        "mop",
        access="read",
        value="fmt:type={arg0};count={arg1};zmask={arg2}",
    ),
    # ---- replay buffer ---------------------------------------------------- #
    Pattern(
        "core.replay_record",
        "call",
        r"record",
        "literal:REPLAY",
        "replay",
        detail="fmt:start={arg0};len={arg1}",
        qualifier=r"(?:::)?lltt",
    ),
    Pattern(
        "core.load_replay_buf",
        "call",
        r"load_replay_buf",
        "literal:REPLAY",
        "replay",
        detail="fmt:start={arg0};len={arg1}",
    ),
    Pattern(
        "core.replay_insn.record",
        "call",
        _ins("REPLAY"),
        "literal:REPLAY",
        "replay",
        detail="fmt:start={arg0};len={arg1}",
        when=lambda m: len(m.args) >= 4 and compact(m.args[3]) not in ("0", "false"),
    ),
    Pattern(
        "core.replay_insn.replay",
        "call",
        _ins("REPLAY"),
        "literal:REPLAY",
        "replay",
        access="read",
        detail="fmt:start={arg0};len={arg1}",
        when=lambda m: len(m.args) < 4 or compact(m.args[3]) in ("0", "false"),
    ),
    Pattern(
        "core.replay_run",
        "call",
        r"replay|replay_insn",
        "literal:REPLAY",
        "replay",
        access="read",
        detail="fmt:start={arg0};len={arg1}",
        qualifier=r"(?:::)?lltt",
    ),
    # ---- address counters ------------------------------------------------- #
    Pattern(
        "core.setadc",
        "call",
        _ins("SETADC"),
        "fmt:{arg0}|{arg2}",
        "adc",
        value="arg:3",
        detail=_deferred,
    ),
    Pattern(
        "core.setadc_xy",
        "call",
        _ins(r"(?:SETADCXY|INCADCXY|ADDRCRXY)"),
        "fmt:{arg0}|XY",
        "adc",
        value="allargs",
        detail=_deferred,
    ),
    Pattern(
        "core.setadc_zw",
        "call",
        _ins(r"(?:SETADCZW|INCADCZW|ADDRCRZW)"),
        "fmt:{arg0}|ZW",
        "adc",
        value="allargs",
        detail=_deferred,
    ),
    Pattern(
        "core.setadc_xx",
        "call",
        _ins(r"SETADCXX"),
        "fmt:{arg0}|XX",
        "adc",
        value="allargs",
        detail=_deferred,
    ),
    # ---- read/write counters ---------------------------------------------- #
    Pattern(
        "core.setrwc",
        "call",
        _ins("SETRWC"),
        "arg:5",
        "rwc",
        value="allargs",
        detail=_deferred,
    ),
    # ---- SFPU ------------------------------------------------------------- #
    Pattern(
        "core.sfpconfig",
        "call",
        _ins("SFPCONFIG"),
        "arg:1",
        "sfpconfig_dest",
        value="fmt:imm={arg0};mod={arg2}",
        detail=_deferred,
    ),
    Pattern(
        "core.sfpi_constant",
        "assign",
        r"vConst(?:Float|Int)Prgm[0-3]",
        "name",
        "sfpi_const",
        value="rhs",
    ),
    Pattern(
        "core.sfpi_constant_read",
        "regex",
        r"(?<![\w.])(?:sfpi::)?(?P<c>vConst(?:Float|Int)Prgm[0-3])\b(?!\s*=(?!=))",
        "group:c",
        "sfpi_const",
        access="read",
    ),
    Pattern(
        "core.sfploadmacro",
        "call",
        _ins("SFPLOADMACRO"),
        "literal:8",
        "sfpconfig_dest",
        access="read",
        value="allargs",
        detail=_deferred,
    ),
    # ---- semaphores ------------------------------------------------------- #
    Pattern(
        "core.seminit",
        "call",
        _ins("SEMINIT"),
        "arg:2",
        "semaphore",
        value="fmt:max={arg0};init={arg1}",
        detail=_deferred,
    ),
    Pattern(
        "core.sempost_get",
        "call",
        _ins(r"SEM(?:POST|GET)"),
        "arg:0",
        "semaphore",
        value="name",
        detail=_deferred,
    ),
    Pattern(
        "core.semwait",
        "call",
        _ins("SEMWAIT"),
        "arg:1",
        "semaphore",
        access="read",
        value="arg:2",
        detail=_deferred,
    ),
    Pattern(
        "core.t6_semaphore_write",
        "call",
        r"t6_semaphore_(?:post|get|init)|semaphore_(?:post|get)",
        "arg:0",
        "semaphore",
        value="name",
    ),
    Pattern(
        "core.semaphore_read",
        "call",
        r"semaphore_read",
        "arg:0",
        "semaphore",
        access="read",
    ),
    Pattern(
        "core.t6_semaphore_wait",
        "call",
        r"t6_semaphore_wait_on_(?:max|zero)",
        "arg:0",
        "semaphore",
        access="read",
        value="name",
    ),
    # ---- mailbox ---------------------------------------------------------- #
    Pattern(
        "core.mailbox_write",
        "call",
        r"mailbox_write",
        "arg:0",
        "mailbox",
        value="arg:1",
    ),
    Pattern(
        "core.mailbox_read", "call", r"mailbox_read", "arg:0", "mailbox", access="read"
    ),
)

#: Software-shadow globals always tracked (in addition to scanned scalar globals).
SW_GLOBALS = (
    "cfg_state_id",
    "unp_cfg_context",
    "dest_offset_id",
    "pack_sync_tile_dst_ptr",
    "math_sync_tile_dst_index",
    "src_zero_flag_hw",
    "src_zero_flag_srca_fmt",
    "src_zero_flag_srcb_fmt",
)
_SW_EXCLUDE = {
    "rta_count",
    "crta_count",
    "dbg_event_index",
    "dbg_event_end",
    "mailbox_index",
}


def _sw_pattern(names: Iterable[str]) -> Pattern:
    alternation = "|".join(
        sorted({re.escape(n) for n in names}, key=lambda n: (-len(n), n))
    )
    return Pattern("core.sw_global", "assign", alternation, "name", "sw", value="rhs")


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
@dataclass
class Family:
    """A loaded family module (see SCHEMA.md for the module contract)."""

    name: str
    module: ModuleType
    patterns: tuple[Pattern, ...]
    disable_core: frozenset[str]
    groups: tuple[tuple[str, str], ...]
    normalizers: dict[str, Callable]
    extract: Callable | None
    role: Callable | None
    thread: Callable | None


class FamilyError(ValueError):
    pass


def load_families(names: Sequence[str] | None = None) -> list[Family]:
    """Import every ``bh/families/<name>.py`` (``_*.py`` are templates, skipped)."""
    stems = sorted(
        p.stem for p in _FAMILY_DIR.glob("*.py") if not p.stem.startswith("_")
    )
    if names is not None:
        stems = [stem for stem in stems if stem in names]
    families: list[Family] = []
    seen_ids: set[str] = {p.id for p in CORE_PATTERNS}
    for stem in stems:
        module = importlib.import_module(f"{__package__}.families.{stem}")
        family = _family_from_module(module, stem)
        for pattern in family.patterns:
            if pattern.id in seen_ids:
                raise FamilyError(
                    f"duplicate pattern id {pattern.id!r} in family {family.name}"
                )
            seen_ids.add(pattern.id)
        families.append(family)
    families.sort(key=lambda f: f.name)
    return families


def _family_from_module(module: ModuleType, stem: str) -> Family:
    name = getattr(module, "FAMILY", None)
    if not isinstance(name, str) or not name:
        raise FamilyError(f"family module {stem} must define FAMILY")
    if getattr(module, "ENABLED", True) is False:
        patterns: tuple[Pattern, ...] = ()
    else:
        patterns = tuple(getattr(module, "PATTERNS", ()) or ())
    for pattern in patterns:
        if not isinstance(pattern, Pattern):
            raise FamilyError(f"{stem}: PATTERNS must contain model.Pattern instances")
        if not pattern.id.startswith(f"{name}."):
            raise FamilyError(
                f"{stem}: pattern id {pattern.id!r} must start with '{name}.'"
            )
        if pattern.kind not in ("call", "member", "ctor", "regex", "assign"):
            raise FamilyError(
                f"{stem}: pattern {pattern.id} has unknown kind {pattern.kind!r}"
            )
        if pattern.access not in ("read", "write"):
            raise FamilyError(
                f"{stem}: pattern {pattern.id} has unknown access {pattern.access!r}"
            )
        if pattern.thread not in ("inherit",) + THREAD_VALUES:
            raise FamilyError(
                f"{stem}: pattern {pattern.id} has unknown thread {pattern.thread!r}"
            )
    normalizers = dict(getattr(module, "NORMALIZERS", {}) or {})
    for key in normalizers:
        if not key.startswith(f"{name}."):
            raise FamilyError(f"{stem}: normalizer {key!r} must start with '{name}.'")
    return Family(
        name=name,
        module=module,
        patterns=patterns,
        disable_core=frozenset(getattr(module, "DISABLE_CORE", ()) or ()),
        groups=tuple(getattr(module, "GROUPS", ()) or ()),
        normalizers=normalizers,
        extract=getattr(module, "extract", None),
        role=getattr(module, "role", None),
        thread=getattr(module, "thread", None),
    )


# --------------------------------------------------------------------------- #
# Pattern application
# --------------------------------------------------------------------------- #
_FMT = re.compile(
    r"\{(arg|targ|group|index|rhs|name|receiver|op)(\d+|:[A-Za-z_]\w*)?\}"
)


def eval_spec(spec: Any, m: PatternMatch) -> str | None:
    """Evaluate a resource/value/detail spec (see :class:`model.Pattern`)."""
    if spec is None:
        return None
    if callable(spec):
        value = spec(m)
        return None if value is None else str(value)
    if not isinstance(spec, str):
        return None
    if spec.startswith("literal:"):
        return spec[len("literal:") :]
    if spec.startswith("fmt:"):
        missing = False

        def fill(match: re.Match[str]) -> str:
            nonlocal missing
            value = _field(m, match.group(1), match.group(2))
            if value is None:
                missing = True
                return ""
            return value

        text = _FMT.sub(fill, spec[len("fmt:") :])
        return None if missing and not text.strip() else text
    kind, _, rest = spec.partition(":")
    if kind == "args" and rest:
        values = [
            m.args[int(i)]
            for i in rest.split(",")
            if i.strip().isdigit() and int(i) < len(m.args)
        ]
        return ", ".join(values) if values else None
    return _field(m, kind, (rest if rest else None))


def _field(m: PatternMatch, kind: str, index: str | None) -> str | None:
    if kind in ("arg", "targ"):
        values = m.args if kind == "arg" else m.template_args
        if index is None:
            return None
        position = int(index.lstrip(":"))
        return values[position] if 0 <= position < len(values) else None
    if kind == "group":
        return m.group(index.lstrip(":")) if index else None
    if kind == "targs":
        return ", ".join(m.template_args) if m.template_args else None
    if kind == "allargs":
        return ", ".join(m.args)
    if kind == "index":
        return m.index or None
    if kind == "rhs":
        return m.rhs or None
    if kind == "op":
        return m.op or None
    if kind == "name":
        return m.name
    if kind == "receiver":
        return m.receiver or None
    if kind == "text":
        return m.text or None
    return None


class FamilyContext:
    """Passed to family ``extract`` hooks: tables plus an effect factory."""

    def __init__(
        self,
        tables: resources.ResourceTables,
        family: str,
        normalizers: dict[str, Callable],
        index: scan.ScanIndex | None = None,
    ):
        self.tables = tables
        self.family = family
        self._normalizers = normalizers
        #: the tree's :class:`scan.ScanIndex` (callee lookup for argument mapping)
        self.index = index

    def normalize(self, normalizer: Any, expr: str | None, thread: str | None = None):
        if callable(normalizer):
            result = normalizer(expr, thread=thread)
        elif isinstance(normalizer, str) and normalizer in self._normalizers:
            result = self._normalizers[normalizer](expr, thread=thread)
        else:
            return self.tables.normalize(normalizer, expr, thread)
        if result is None:
            return []
        if isinstance(result, str):
            return [resources.Normalized(result)]
        if isinstance(result, resources.Normalized):
            return [result]
        return list(result)

    def effects(
        self,
        fn: scan.FunctionDef,
        *,
        pattern_id: str,
        resource_expr: str,
        normalizer: Any = "literal",
        access: str = "write",
        value: str | None = "",
        offset: int | None = None,
        line: int | None = None,
        condition: str | None = None,
        thread: str | None = None,
        detail: str = "",
        granularity: str | None = None,
        confidence: str = "high",
        evidence: str | None = None,
    ) -> list[Effect]:
        """Build effects for ``fn`` exactly like a declarative pattern would.

        ``evidence`` overrides the sink ``path:line`` (default: ``fn.path`` at ``line``).
        """
        if line is None:
            line = fn.line_at(offset or 0)
        if condition is None and offset is not None:
            condition = fn.condition_at(offset)
        effective_thread = thread or fn.thread_hint or "any"
        return _make_effects(
            family=self.family,
            pattern_id=pattern_id,
            normalizer_name=(
                normalizer if isinstance(normalizer, str) else f"{pattern_id}#norm"
            ),
            normalized=self.normalize(normalizer, resource_expr, effective_thread),
            resource_expr=resource_expr,
            access=access,
            value=value or "",
            detail=detail,
            condition=condition or "",
            evidence=evidence or f"{fn.path}:{line}",
            thread=effective_thread,
            granularity=granularity,
            confidence=confidence,
        )


def _make_effects(
    *,
    family: str,
    pattern_id: str,
    normalizer_name: str,
    normalized: list,
    resource_expr: str,
    access: str,
    value: str,
    detail: str,
    condition: str,
    evidence: str,
    thread: str,
    granularity: str | None,
    confidence: str,
) -> list[Effect]:
    result = []
    value = compact(value)
    if len(value) > 300:
        value = value[:297] + "..."
    for index, item in enumerate(normalized):
        result.append(
            Effect(
                resource=item.name,
                access=access,
                value_kind="unknown",
                value=value,
                condition=compact(condition),
                evidence=evidence,
                thread=thread,
                family=family,
                pattern=pattern_id,
                granularity=granularity or item.granularity,
                detail=compact(merge_detail(item.detail, detail)),
                resolved=item.resolved,
                confidence=confidence if item.resolved else "low",
                resource_expr=compact(resource_expr),
                normalizer=normalizer_name,
                norm_index=index,
            )
        )
    return result


def _call_matches(pattern: Pattern, call: scan.CallSite) -> bool:
    if pattern.kind == "call":
        if call.is_member or call.is_ctor:
            return False
        if not re.fullmatch(pattern.match, call.name):
            return False
        if pattern.qualifier is not None and not re.fullmatch(
            pattern.qualifier, call.qualifier
        ):
            return False
        return True
    if pattern.kind == "member":
        if not call.is_member or not re.fullmatch(pattern.match, call.name):
            return False
        if pattern.receiver_type is not None and not re.fullmatch(
            pattern.receiver_type, call.receiver_type or ""
        ):
            return False
        return True
    if pattern.kind == "ctor":
        return (
            call.is_ctor
            and re.fullmatch(pattern.match, call.receiver_type or "") is not None
        )
    return False


def _call_match(call: scan.CallSite) -> PatternMatch:
    return PatternMatch(
        name=call.name,
        callee=call.callee,
        template_args=call.template_args,
        args=call.args,
        receiver=call.receiver or "",
        receiver_type=call.receiver_type or "",
        text=call.callee,
        line=call.line,
        offset=call.offset,
        thread=call.thread,
        condition=call.condition,
        via_macro=call.via_macro,
    )


def _segments(fn: scan.FunctionDef):
    """(text, expansion or None) segments regex/assign patterns are applied to."""
    yield fn.masked_body, None
    for expansion in fn.expansions:
        yield expansion.text, expansion


_ASSIGN_OP = re.compile(r"\s*(<<=|>>=|\+\+|--|[+\-*/%&|^]?=)(?!=)")


def _assign_matches(
    pattern: Pattern, text: str
) -> Iterable[tuple[int, str, str, str, str]]:
    """Yield (offset, name, index, op, rhs) for ``NAME[idx] op= rhs`` assignments."""
    regex = re.compile(
        rf"(?<![\w.>])(?<!->)(?:(?:[A-Za-z_]\w*)::)*(?P<name>{pattern.match})\b"
    )
    for match in regex.finditer(text):
        position = match.end()
        length = len(text)
        while position < length and text[position] in " \t\n":
            position += 1
        index = ""
        if position < length and text[position] == "[":
            close = scan.match_close(text, position, "[]")
            if close is None:
                continue
            index = compact(text[position + 1 : close])
            position = close + 1
        operator = _ASSIGN_OP.match(text, position)
        if operator is None:
            prefix = text[max(0, match.start() - 2) : match.start()]
            if prefix in ("++", "--"):
                yield match.start(), match.group("name"), index, prefix, ""
            continue
        op = operator.group(1)
        if op == "=" and text[operator.end(1) : operator.end(1) + 1] == "=":
            continue
        rhs = ""
        if op not in ("++", "--"):
            end = scan.statement_end(text, operator.end())
            rhs = compact(text[operator.end() : end])
            if op != "=":
                rhs = f"{op} {rhs}"
        yield match.start(), match.group("name"), index, op, rhs


_INSN_MACRO = re.compile(r"TTI?_[A-Z][A-Z0-9_]*")
_EXEC_TARG = re.compile(r"(?<!No)Exec\b|^true$")


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


def replay_recorded(
    fn: scan.FunctionDef, tables: resources.ResourceTables | None = None
) -> tuple[list[tuple[int, int]], set[int]]:
    """Code that is recorded into the replay buffer instead of being issued.

    Returns (masked_body spans, ``fn.calls`` indices): the callable argument of
    ``load_replay_buf<NoExec>(..)`` (span), and the next ``len`` instruction macros after
    ``lltt::record<NoExec>(start, len)`` or ``TTI_REPLAY(start, len, 0 /*exec*/, 1 /*load*/)``
    (the rest of the enclosing block when ``len`` is not a constant).
    """
    masked = fn.masked_body
    calls = list(enumerate(fn.calls))
    spans: list[tuple[int, int]] = []

    def number(text: str) -> int | None:
        if tables is not None:
            return tables.int_value(text)
        value = scan.evaluate(text)
        return value if isinstance(value, int) else None

    for _, call in calls:
        if call.name == "load_replay_buf" and not any(
            _EXEC_TARG.search(compact(t)) for t in call.template_args
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
            and call.qualifier.lstrip(":").endswith("lltt")
            and not any(_EXEC_TARG.search(compact(t)) for t in call.template_args)
        ):
            window = number(call.args[1]) if len(call.args) > 1 else None
            window = -1 if window is None else window
        elif re.fullmatch(r"TTI?_REPLAY", call.name) and len(call.args) >= 4:
            load, execute = number(call.args[3]), number(call.args[2])
            if load and execute == 0:
                window = number(call.args[1])
                window = -1 if window is None else window
        if window is None:
            continue
        block_end = _block_end(masked, call.offset)
        remaining = window
        for later, other in calls[position + 1 :]:
            if other.offset > block_end or remaining == 0:
                break
            indices.add(later)
            if remaining > 0 and (
                _INSN_MACRO.fullmatch(other.name) or other.name == "TT_INSN"
            ):
                remaining -= 1
    for index, call in calls:
        if any(start < call.offset < end for start, end in spans):
            indices.add(index)
    return spans, indices


def _mark_deferred(effect: Effect) -> Effect:
    detail = effect.detail or ""
    if any(part.split(":", 1)[0] == "deferred" for part in detail.split(";")):
        return effect
    return replace(effect, detail=f"{detail};deferred" if detail else "deferred")


def _direct_effects(
    fn: scan.FunctionDef,
    patterns: Sequence[tuple[str, Pattern]],
    tables: resources.ResourceTables,
    normalizer_lookup: dict[str, Callable],
) -> tuple[list[Effect], dict[int, list[str]]]:
    """Apply patterns to one function. Returns effects and call-index -> pattern ids."""
    effects: list[Effect] = []
    consumed: dict[int, list[str]] = {}
    fn_thread = fn.thread_hint
    spans = scan.thread_spans(fn.masked_body)
    local_constants: dict[str, str] | None = None

    def normalize(pattern: Pattern, expr: str, thread: str):
        nonlocal local_constants
        if pattern.normalizer == "cfg_rmw" and fn.masked_body:
            # SHAMT/MASK operands are often constexpr locals (``alu_mask``): resolve them
            if local_constants is None:
                local_constants = {}
                for name, value in _local_constants(fn):
                    local_constants[name] = substitute(value, local_constants)
            if local_constants and identifiers(expr) & set(local_constants):
                expr = substitute(expr, local_constants)
        if callable(pattern.normalizer):
            result = pattern.normalizer(expr, thread=thread)
        elif pattern.normalizer in normalizer_lookup:
            result = normalizer_lookup[pattern.normalizer](expr, thread=thread)
        else:
            return tables.normalize(pattern.normalizer, expr, thread)
        if result is None:
            return []
        if isinstance(result, str):
            return [resources.Normalized(result)]
        if isinstance(result, resources.Normalized):
            return [result]
        return list(result)

    def emit(
        family: str, pattern: Pattern, m: PatternMatch, evidence_line: int
    ) -> bool:
        if pattern.when is not None and not pattern.when(m):
            return False
        expr = eval_spec(pattern.resource, m)
        if expr is None or not compact(expr):
            return False
        thread = (
            pattern.thread
            if pattern.thread != "inherit"
            else (m.thread or fn_thread or "any")
        )
        normalized = normalize(pattern, expr, thread)
        if not normalized:
            return False
        effects.extend(
            _make_effects(
                family=family,
                pattern_id=pattern.id,
                normalizer_name=(
                    pattern.normalizer
                    if isinstance(pattern.normalizer, str)
                    else f"{pattern.id}#norm"
                ),
                normalized=normalized,
                resource_expr=expr,
                access=pattern.access,
                value=eval_spec(pattern.value, m) or "",
                detail=eval_spec(pattern.detail, m) or "",
                condition=m.condition or "",
                evidence=f"{fn.path}:{evidence_line}",
                thread=thread,
                granularity=pattern.granularity,
                confidence=pattern.confidence,
            )
        )
        return True

    recorded_spans, recorded_calls = (
        replay_recorded(fn, tables)
        if (
            "load_replay_buf" in fn.masked_body
            or "record" in fn.masked_body
            or "REPLAY" in fn.masked_body
        )
        else ([], set())
    )
    for family, pattern in patterns:
        if pattern.kind in ("call", "member", "ctor"):
            for index, call in enumerate(fn.calls):
                before = len(effects)
                if _call_matches(pattern, call) and emit(
                    family, pattern, _call_match(call), call.line
                ):
                    consumed.setdefault(index, []).append(pattern.id)
                    if index in recorded_calls:
                        # recorded into the replay buffer: runs when the replay is issued
                        effects[before:] = [_mark_deferred(e) for e in effects[before:]]
            continue
        for text, expansion in _segments(fn):
            if pattern.kind == "regex":
                found = (
                    (
                        match.start(),
                        match.group(0),
                        "",
                        "",
                        "",
                        tuple(
                            sorted(
                                (k, v)
                                for k, v in match.groupdict().items()
                                if v is not None
                            )
                        ),
                    )
                    for match in re.finditer(pattern.match, text)
                )
            else:
                found = (
                    (offset, name, index, op, rhs, ())
                    for offset, name, index, op, rhs in _assign_matches(pattern, text)
                )
            for offset, name, index, op, rhs, groups in found:
                if expansion is None:
                    line = fn.line_at(offset)
                    condition = fn.condition_at(offset)
                    thread = scan._thread_at(spans, offset)
                    if thread is None:
                        thread = fn.pp_info_at_line(line)[0]
                    via_macro = None
                else:
                    line, condition, thread, via_macro = (
                        expansion.line,
                        expansion.condition,
                        expansion.thread,
                        expansion.name,
                    )
                m = PatternMatch(
                    name=name,
                    groups=groups,
                    index=index,
                    rhs=rhs,
                    op=op,
                    text=name,
                    line=line,
                    offset=offset if expansion is None else expansion.offset,
                    thread=thread,
                    condition=condition,
                    via_macro=via_macro,
                )
                before = len(effects)
                if emit(family, pattern, m, line) and recorded_spans:
                    at = offset if expansion is None else expansion.offset
                    if any(start < at < end for start, end in recorded_spans):
                        effects[before:] = [_mark_deferred(e) for e in effects[before:]]
    return effects, consumed


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #
def default_role(fn: scan.FunctionDef) -> str:
    """Name/layer based role (families may override; see SCHEMA.md)."""
    name = fn.name.strip("_").lower()
    if fn.is_ctor:
        return "helper"
    if re.search(r"hw_configure|hw_startup|_init_common$|^init_common$", name):
        return "hw_startup"
    if "uninit" in name:
        return "uninit"
    if re.search(r"reconfig|reconfigure", name):
        return "reconfig"
    if re.search(r"(?:^|_)init(?:$|_)|init$", name) and not re.search(
        r"(?:^|_)(?:is|get)_", name
    ):
        return "init"
    if re.search(
        r"(?:^|_)(?:config|configure|cfg|setup|program)(?:$|_)", name
    ) or re.match(r"(?:set|clear|reset|enable|disable)_", name):
        return "config"
    if fn.layer in ("compute_api", "llk_api", "llk_lib"):
        return "execute"
    if fn.layer == "sfpu" and re.match(r"(?:calculate|calc)_", name):
        return "execute"
    return "helper"


# --------------------------------------------------------------------------- #
# The DB
# --------------------------------------------------------------------------- #
class EffectDB:
    """Everything the checker needs: functions, edges, roles, effects, tables."""

    def __init__(
        self,
        repo_root: Path,
        index: scan.ScanIndex,
        tables: resources.ResourceTables,
        functions: dict[str, FunctionModel],
        graph: CallGraph,
        families: list[Family],
        patterns: list[tuple[str, Pattern]],
        tree_fingerprint: str,
    ):
        self.repo_root = repo_root
        self.scan = index
        self.tables = tables
        self.functions = functions
        self.graph = graph
        self.families = families
        self.patterns = patterns
        self.tree_fingerprint = tree_fingerprint
        self._transitive: dict[str, tuple[Effect, ...]] = {}
        self._normalizer_lookup = graph.custom_normalizers
        self._stop_ids: set[str] = {
            pattern.id for _, pattern in patterns if pattern.stop
        }

    # ---- lookup ----------------------------------------------------------- #
    def function(self, key: str) -> FunctionModel | None:
        return self.functions.get(key)

    def by_name(self, name: str) -> list[FunctionModel]:
        return self.graph.by_name(name)

    def keys(self) -> list[str]:
        return sorted(self.functions)

    def layer_counts(self) -> dict[str, int]:
        counts = {layer: 0 for layer in scan.LAYERS}
        for fn in self.functions.values():
            if fn.layer in counts:
                counts[fn.layer] += 1
        return counts

    def role_counts(self) -> dict[str, int]:
        counts = {role: 0 for role in ROLES}
        for fn in self.functions.values():
            if fn.layer in scan.LAYERS:
                counts[fn.role] = counts.get(fn.role, 0) + 1
        return counts

    # ---- effects ---------------------------------------------------------- #
    def effects(self, key: str) -> tuple[Effect, ...]:
        """Transitive effects of a function in its own scope (cached)."""
        cached = self._transitive.get(key)
        if cached is None:
            cached = self.graph.effects(key)
            self._transitive[key] = cached
        return cached

    def direct_effects(self, key: str) -> tuple[Effect, ...]:
        fn = self.functions.get(key)
        return fn.effects if fn else ()

    def resolve(
        self,
        name: str,
        *,
        qualifier: str = "",
        template_args: Sequence[str] = (),
        args: Sequence[str] = (),
        is_member: bool = False,
        receiver_type: str | None = None,
        caller_path: str | None = None,
    ) -> tuple[list[FunctionModel], str]:
        caller = None
        if caller_path is not None:
            caller = FunctionModel(
                key="<caller>",
                name="<caller>",
                qualified_name="<caller>",
                path=caller_path,
                line=0,
                end_line=0,
                layer="kernel",
                role="execute",
                thread="any",
                params=(),
                template_params=(),
                signature="",
            )
        return self.graph.candidates(
            name,
            qualifier=qualifier,
            template_args=template_args,
            args=args,
            is_member=is_member,
            receiver_type=receiver_type,
            caller=caller,
        )

    def call_effects(
        self,
        name: str,
        *,
        qualifier: str = "",
        template_args: Sequence[str] = (),
        args: Sequence[str] = (),
        thread: str | None = None,
        condition: str | None = None,
        evidence: str = "",
        caller_path: str | None = None,
        is_member: bool = False,
        receiver_type: str | None = None,
    ) -> tuple[Effect, ...]:
        """Effects of a call written in some other code (e.g. a kernel), with
        the call's arguments substituted. Pattern-only primitives (TTI_SETC16,
        cfg_reg_rmw_tensix, ...) are matched too. ``evidence`` (path:line of the
        call) is prepended to ``via``.
        """
        site = scan.CallSite(
            name=name,
            callee=(f"{qualifier}::{name}" if qualifier else name),
            template_args=tuple(template_args),
            args=tuple(args),
            line=(
                int(evidence.rpartition(":")[2])
                if evidence.rpartition(":")[2].isdigit()
                else 0
            ),
            offset=0,
            thread=thread,
            condition=condition,
            is_member=is_member,
            receiver_type=receiver_type,
        )
        result: list[Effect] = []
        stopped = False
        for family, pattern in self.patterns:
            if pattern.kind not in ("call", "member", "ctor") or not _call_matches(
                pattern, site
            ):
                continue
            fake = scan.FunctionDef(
                name="<call>",
                qualified_name="<call>",
                namespace="",
                owner="",
                key="<call>",
                path=evidence.rpartition(":")[0] or (caller_path or ""),
                layer="kernel",
                line=site.line,
                start_line=site.line,
                end_line=site.line,
                body_line=site.line,
                signature="",
                params=(),
                template_params=(),
                specialization_args=(),
                attributes=(),
                body="",
                masked_body="",
                pp_condition=None,
                pp_thread=None,
                thread_hint=thread,
                fingerprint="",
                calls=(site,),
            )
            effects, consumed = _direct_effects(
                fake, [(family, pattern)], self.tables, self._normalizer_lookup
            )
            result.extend(effects)
            if consumed and pattern.stop:
                stopped = True
        if not stopped:
            callees, resolution = self.resolve(
                name,
                qualifier=qualifier,
                template_args=template_args,
                args=args,
                is_member=is_member,
                receiver_type=receiver_type,
                caller_path=caller_path,
            )
            for callee in callees:
                bindings = self.graph.bind(callee, template_args, args)
                for effect in self.graph._resolve(callee, bindings, 1):
                    if effect.condition:
                        verdict, simplified = scan.simplify_condition(effect.condition)
                        if verdict is False:
                            continue
                        if simplified != effect.condition:
                            effect = replace(effect, condition=simplified)
                    confidence = (
                        "low"
                        if resolution == "ambiguous" or not effect.resolved
                        else (
                            "medium"
                            if effect.confidence == "high"
                            else effect.confidence
                        )
                    )
                    result.append(
                        replace(
                            effect,
                            confidence=confidence,
                            via=((callee.key, evidence),) + effect.via,
                        )
                    )
        finished = []
        for effect in result:
            effect_thread = (
                effect.thread if effect.thread != "any" else (thread or "any")
            )
            finished.append(
                replace(
                    effect,
                    thread=effect_thread,
                    condition=combine_conditions(condition, effect.condition),
                    value_kind=classify_value(
                        effect.value, effect.params, self.graph.constants
                    ),
                )
            )
        return tuple(dedupe_effects(finished))

    def reachable(
        self, key: str, *, include_stopped: bool = False
    ) -> list[tuple[str, int, str]]:
        return self.graph.reachable(key, include_stopped=include_stopped)

    def add_sources(
        self, sources: Iterable[tuple[str, str]], *, layer: str = "kernel"
    ) -> list[FunctionModel]:
        """Scan extra files (kernels, kernel-local helper headers) into the call graph.

        ``sources``: (repo-relative path, text) pairs. Their functions get keys
        ``<key>@<path>`` (so every ``MAIN`` is distinct), the same patterns, roles
        and call resolution as the library, and are then usable with
        :meth:`effects` / :meth:`reachable`. Library functions are unaffected.
        Returns the new models in (path, line) order.
        """
        sources = sorted(sources)
        scanned: list[scan.FunctionDef] = []
        for path, text in sources:
            source = scan.scan_source(text, path, layer, index=self.scan)
            scanned.extend(source.functions)
        added: list[tuple[FunctionModel, dict[int, list[str]]]] = []
        families = self.families
        for fn in scanned:
            fn.key = f"{fn.key}@{fn.path}"
            model, consumed = _function_model(
                fn,
                self.patterns,
                self.tables,
                self._normalizer_lookup,
                families,
                self.scan,
            )
            if model.layer == "kernel" and model.name in ("MAIN", "kernel_main"):
                model.role = "execute"
            self.functions[model.key] = model
            self.graph.functions[model.key] = model
            self.graph._by_name.setdefault(model.name, []).append(model)
            self.graph._by_name[model.name].sort(key=lambda m: m.key)
            added.append((model, consumed))
        for model, consumed in added:
            model.calls = _call_edges(model, consumed, self.graph, self._stop_ids)
        self._transitive.clear()
        self.graph._memo.clear()
        return [
            model
            for model, _ in sorted(
                added, key=lambda item: (item[0].path, item[0].line, item[0].key)
            )
        ]

    # ---- output ----------------------------------------------------------- #
    def to_json(self, *, transitive: bool = True) -> dict:
        functions = []
        total_direct = 0
        total_transitive = 0
        for key in self.keys():
            fn = self.functions[key]
            if fn.layer not in scan.LAYERS:
                continue  # kernel sources added by the checker are not part of bh-effects
            effects = self.effects(key) if transitive else None
            total_direct += len(fn.effects)
            total_transitive += len(effects or ())

            functions.append(fn.to_json(effects))
        return {
            "arch": ARCH,
            "counts": {
                "effects_direct": total_direct,
                "effects_transitive": total_transitive,
                "functions": sum(self.layer_counts().values()),
                "functions_per_layer": self.layer_counts(),
                "roles": self.role_counts(),
            },
            "families": [
                {
                    "disable_core": sorted(family.disable_core),
                    "name": family.name,
                    "patterns": sorted(p.id for p in family.patterns),
                }
                for family in self.families
            ],
            "functions": functions,
            "patterns": sorted(pattern.id for _, pattern in self.patterns),
            "resource_classes": [
                resource.to_json() for resource in resources.taxonomy()
            ],
            "schema_version": SCHEMA_VERSION,
            "tree_fingerprint": self.tree_fingerprint,
        }


def build(
    repo_root: Path | str,
    *,
    families: Sequence[str] | None = None,
    modules: Sequence[ModuleType] = (),
) -> EffectDB:
    """Scan ``repo_root`` and build the effect DB (no git, no compiler).

    ``families`` restricts which ``bh/families/*.py`` load (default: all);
    ``modules`` injects extra family modules (used by tests and ``_example``).
    """
    root = Path(repo_root).resolve()
    index = scan.scan_tree(root)
    tables = resources.ResourceTables(root, index.constants)
    loaded = load_families(families)
    for module in modules:
        loaded.append(
            _family_from_module(
                module, getattr(module, "__name__", "module").rsplit(".", 1)[-1]
            )
        )
    loaded.sort(key=lambda f: f.name)
    for family in loaded:
        for group, regex in family.groups:
            tables.add_group(group, regex)
        for name, function in sorted(family.normalizers.items()):
            tables.register(name, function)
    disabled = (
        set().union(*(family.disable_core for family in loaded)) if loaded else set()
    )
    sw_names = (set(index.globals) | set(SW_GLOBALS)) - _SW_EXCLUDE
    core = [pattern for pattern in CORE_PATTERNS if pattern.id not in disabled]
    if "core.sw_global" not in disabled:
        core.append(_sw_pattern(sw_names))
    patterns: list[tuple[str, Pattern]] = [("core", p) for p in core]
    for family in loaded:
        patterns.extend((family.name, p) for p in family.patterns)
    custom_normalizers: dict[str, Callable] = {}
    for _, pattern in patterns:
        if callable(pattern.normalizer):
            custom_normalizers[f"{pattern.id}#norm"] = pattern.normalizer
    for family in loaded:
        custom_normalizers.update(family.normalizers)

    functions: dict[str, FunctionModel] = {}
    consumed_by_key: dict[str, dict[int, list[str]]] = {}
    for fn in index.functions:
        model, consumed = _function_model(
            fn, patterns, tables, custom_normalizers, loaded, index
        )
        functions[model.key] = model
        consumed_by_key[model.key] = consumed
    graph = CallGraph(
        functions,
        normalize=tables.normalize,
        truth=lambda text: scan.truth(text),
        arg_expanders=(resources.expand_rmw_args,),
        constants=set(index.constants),
        custom_normalizers=custom_normalizers,
        simplify=scan.simplify_condition,
    )
    stop_ids = {pattern.id for _, pattern in patterns if pattern.stop}
    for key in sorted(functions):
        functions[key].calls = _call_edges(
            functions[key], consumed_by_key[key], graph, stop_ids
        )

    digest = hashlib.sha256()
    digest.update(index.fingerprint.encode("ascii"))
    cfg_path = root / resources.CFG_DEFINES
    if cfg_path.is_file():
        digest.update(
            b"\0"
            + resources.CFG_DEFINES.encode("utf-8")
            + b"\0"
            + cfg_path.read_bytes()
        )
    db = EffectDB(
        repo_root=root,
        index=index,
        tables=tables,
        functions=functions,
        graph=graph,
        families=loaded,
        patterns=patterns,
        tree_fingerprint=digest.hexdigest(),
    )
    db._stop_ids = stop_ids
    return db


def _content(effect: Effect) -> tuple:
    return (
        effect.resource,
        effect.access,
        effect.value,
        effect.condition,
        effect.evidence,
        effect.thread,
        effect.granularity,
        effect.detail,
    )


def _supersede(effects: list[Effect], families: Sequence[Family]) -> list[Effect]:
    """Drop coarse effects that a family effect at the same sink refines.

    A family module may export ``SUPERSEDES = {"<family pattern>": ("<pattern>", ..)}``:
    an effect of a listed pattern is dropped when an effect of the refining pattern
    exists at the same evidence with the same access.
    """
    # a family companion identical to a core effect (same sink, resource, value,
    # condition, thread): the family's workaround and the core now agree, keep the core one
    core_items = {_content(e) for e in effects if e.family == "core"}
    if core_items:
        effects = [
            e for e in effects if e.family == "core" or _content(e) not in core_items
        ]
    rules: dict[str, tuple[str, ...]] = {}
    for family in families:
        for refined, coarse in sorted(
            (getattr(family.module, "SUPERSEDES", None) or {}).items()
        ):
            rules[refined] = tuple(coarse)
    if not rules:
        return effects
    covered: set[tuple[str, str, str]] = set()
    for effect in effects:
        for coarse in rules.get(effect.pattern, ()):
            covered.add((coarse, effect.evidence, effect.access))
    if not covered:
        return effects
    return [e for e in effects if (e.pattern, e.evidence, e.access) not in covered]


def _function_model(
    fn: scan.FunctionDef,
    patterns: Sequence[tuple[str, Pattern]],
    tables: resources.ResourceTables,
    custom_normalizers: dict[str, Callable],
    families: Sequence[Family],
    index: scan.ScanIndex | None = None,
) -> tuple[FunctionModel, dict[int, list[str]]]:
    """Direct effects, role and thread of one scanned function."""
    effects, consumed = _direct_effects(fn, patterns, tables, custom_normalizers)
    for family in families:
        if family.extract is None:
            continue
        extra = family.extract(
            fn, FamilyContext(tables, family.name, custom_normalizers, index)
        )
        if extra:
            effects.extend(extra)
    effects = _supersede(effects, families)
    role = default_role(fn)
    thread = fn.thread_hint or "any"
    for family in families:
        if family.role is not None:
            override = family.role(fn, role)
            if override is not None:
                if override not in ROLES:
                    raise FamilyError(
                        f"{family.name}.role returned unknown role {override!r}"
                    )
                role = override
                break
    for family in families:
        if family.thread is not None:
            override = family.thread(fn, thread)
            if override is not None:
                if override not in THREAD_VALUES:
                    raise FamilyError(
                        f"{family.name}.thread returned unknown thread {override!r}"
                    )
                thread = override
                break
    if thread != "any":
        effects = [
            replace(e, thread=thread) if e.thread == "any" else e for e in effects
        ]
    flow = _local_flow(fn)
    own = set(fn.param_names)
    effects = [
        replace(
            e, params=tuple(sorted(_flowing_params(_effect_identifiers(e), flow, own)))
        )
        for e in effects
    ]
    constants = tables.constant_names()
    effects = [
        (
            replace(e, value_kind=classify_value(e.value, e.params, constants))
            if e.value_kind == "unknown"
            else e
        )
        for e in effects
    ]
    model = FunctionModel(
        key=fn.key,
        name=fn.name,
        qualified_name=fn.qualified_name,
        path=fn.path,
        line=fn.line,
        end_line=fn.end_line,
        layer=fn.layer,
        role=role,
        thread=thread,
        params=fn.params,
        template_params=fn.template_params,
        signature=fn.signature,
        effects=tuple(dedupe_effects(effects)),
        calls=(),
        attributes=fn.attributes,
        pp_condition=fn.pp_condition,
        fingerprint=fn.fingerprint,
        owner=fn.owner,
        local_constants=_local_constants(fn),
        definition=fn,
    )
    return model, consumed


def _call_edges(
    model: FunctionModel,
    consumed: dict[int, list[str]],
    graph: CallGraph,
    stop_ids: set[str],
) -> tuple[CallEdge, ...]:
    """Statically resolved call edges of one function."""
    fn = model.definition
    edges: list[CallEdge] = []
    for position, call in enumerate(fn.calls):
        matched = tuple(sorted(consumed.get(position, ())))
        stop = any(pid in stop_ids for pid in matched)
        if (
            call.name in model.param_names
            and not call.is_member
            and not call.is_ctor
            and "::" not in call.callee
        ):
            keys: tuple[str, ...] = ()
            resolution = "param_callable"
        else:
            candidates, resolution = graph.candidates(
                call.name,
                qualifier=call.qualifier,
                template_args=call.template_args,
                args=call.args,
                is_member=call.is_member,
                receiver_type=call.receiver_type,
                is_ctor=call.is_ctor,
                caller=model,
            )
            keys = tuple(c.key for c in candidates if c.key != model.key)
            if not keys and not matched:
                continue
            if stop:
                resolution = "stopped"
        edges.append(
            CallEdge(
                name=call.name,
                callee=call.callee,
                callee_keys=keys,
                resolution=(
                    resolution
                    if keys or resolution == "param_callable"
                    else ("stopped" if stop else "unresolved")
                ),
                line=call.line,
                evidence=f"{fn.path}:{call.line}",
                thread=call.thread,
                condition=call.condition,
                template_args=call.template_args,
                args=call.args,
                via_macro=call.via_macro,
                is_member=call.is_member,
                receiver_type=call.receiver_type,
                is_ctor=call.is_ctor,
                stop=stop,
                patterns=matched,
                qualifier=call.qualifier,
            )
        )
    graph._relevant.pop(model.key, None)
    return tuple(edges)


_LOCAL_CONST = re.compile(
    r"(?<![\w.>:])((?:(?:static|constexpr|const|volatile)\s+)*)((?:::)?[A-Za-z_][\w:]*)(?:\s*<[^;<>]*>)?\s+([A-Za-z_]\w*)\s*=(?!=)\s*"
)


def _local_constants(fn: scan.FunctionDef) -> tuple[tuple[str, str], ...]:
    """``constexpr``/``const`` locals assigned exactly once, with a simple initializer."""
    text = fn.masked_body
    found: list[tuple[str, str]] = []
    params = set(fn.param_names)
    for match in _LOCAL_CONST.finditer(text):
        type_name, name = match.group(2), match.group(3)
        if (
            name in params
            or name in scan._KEYWORDS
            or type_name in scan._KEYWORDS
            or type_name in scan._DECL_PREFIX_KEYWORDS
        ):
            continue
        if re.search(
            rf"(?:\+\+|--)\s*{re.escape(name)}\b|\b{re.escape(name)}\s*(?:\+\+|--)|(?<![&\w)\]\s])&(?!&)\s*{re.escape(name)}\b",
            text,
        ):
            continue
        end = scan.statement_end(text, match.end())
        expr = compact(text[match.end() : end])
        if not expr or len(expr) > 200 or "{" in expr or "[" in expr:
            continue
        assignments = len(
            re.findall(
                rf"(?<![\w.>:]){re.escape(name)}\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)", text
            )
        )
        if assignments != 1 or any(name == existing for existing, _ in found):
            continue
        found.append((name, expr))
    return tuple(found)


_LOCAL_ASSIGN = re.compile(
    r"(?<![\w.>:])([A-Za-z_]\w*)(?:\s*\[[^\]]*\])?(?:\.[A-Za-z_]\w*)*\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)"
)


def _local_flow(fn: scan.FunctionDef) -> dict[str, set[str]]:
    """local variable -> parameters it is (transitively) computed from."""
    params = set(fn.param_names)
    flow: dict[str, set[str]] = {}
    text = fn.masked_body
    for _ in range(2):  # two passes reach locals assigned from locals declared later
        for match in _LOCAL_ASSIGN.finditer(text):
            name = match.group(1)
            if name in params or name in scan._KEYWORDS:
                continue
            end = scan.statement_end(text, match.end())
            sources: set[str] = set()
            for identifier in identifiers(text[match.end() : end]):
                if identifier in params:
                    sources.add(identifier)
                elif identifier in flow:
                    sources |= flow[identifier]
            if sources:
                flow.setdefault(name, set()).update(sources)
    return flow


def _flowing_params(
    names: set[str], flow: dict[str, set[str]], params: set[str]
) -> set[str]:
    result = names & params
    for name in names:
        result |= flow.get(name, set())
    return result


def _effect_identifiers(effect: Effect) -> set[str]:
    return (
        identifiers(effect.resource_expr)
        | identifiers(effect.value)
        | identifiers(effect.condition)
        | identifiers(effect.detail)
    )


def dumps(data: dict) -> str:
    """Canonical JSON text used for every bh-* output file."""
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def dumps_effects(data: dict) -> str:
    """Canonical bh-effects text: sorted keys, one compact line per function."""
    keys = sorted(data)
    lines = ["{"]
    for index, key in enumerate(keys):
        comma = "," if index < len(keys) - 1 else ""
        if key == "functions":
            lines.append(' "functions": [')
            functions = data["functions"]
            for position, item in enumerate(functions):
                tail = "," if position < len(functions) - 1 else ""
                lines.append(
                    "  "
                    + json.dumps(
                        item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    )
                    + tail
                )
            lines.append(" ]" + comma)
        else:
            lines.append(
                f" {json.dumps(key)}: {json.dumps(data[key], sort_keys=True, ensure_ascii=False)}{comma}"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_effects(
    repo_root: Path | str, out: Path | str, *, families: Sequence[str] | None = None
) -> dict:
    db = build(repo_root, families=families)
    data = db.to_json()
    Path(out).write_text(dumps_effects(data), encoding="utf-8", newline="\n")
    return data


# --------------------------------------------------------------------------- #
# Findings (bh-hazards output)
# --------------------------------------------------------------------------- #
FINDING_KINDS = ("resource_conflict", "fold_drop", "ordering")
FINDING_CONFIDENCE = ("high", "medium", "low")
MAX_KERNELS = 20


def normalize_findings(findings: Iterable[dict]) -> list[dict]:
    """Validate, canonicalize, sort and number findings (see SCHEMA.md).

    Sorted by (rank, kind, resource, ops) -- then summary/kernels/evidence as
    tie-breakers -- and numbered ``H0001``... Only the documented keys are kept.
    """
    items: list[dict] = []
    for raw in findings:
        if not isinstance(raw, dict):
            raise ValueError("each finding must be a dict")
        kind = raw.get("kind")
        if kind not in FINDING_KINDS:
            raise ValueError(f"finding kind must be one of {FINDING_KINDS}: {kind!r}")
        confidence = raw.get("confidence")
        if confidence not in FINDING_CONFIDENCE:
            raise ValueError(
                f"finding confidence must be one of {FINDING_CONFIDENCE}: {confidence!r}"
            )
        rank = raw.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"finding rank must be an int: {rank!r}")
        resource = raw.get("resource")
        if not isinstance(resource, str) or not resource:
            raise ValueError("finding resource must be a non-empty canonical name")
        ops: list[str] = []
        for op in raw.get("ops", []):
            if isinstance(op, str) and op and op not in ops:
                ops.append(op)
        all_kernels = sorted(
            {k for k in raw.get("kernels", []) if isinstance(k, str) and k}
        )
        kernel_count = raw.get("kernel_count", len(all_kernels))
        if not isinstance(kernel_count, int) or isinstance(kernel_count, bool):
            raise ValueError("finding kernel_count must be an int")
        evidence = sorted(
            {e for e in raw.get("evidence", []) if isinstance(e, str) and e},
            key=_evidence_sort,
        )
        items.append(
            {
                "confidence": confidence,
                "evidence": evidence,
                "kernel_count": kernel_count,
                "kernels": all_kernels[:MAX_KERNELS],
                "kind": kind,
                "ops": ops,
                "rank": rank,
                "resource": resource,
                "summary": compact(str(raw.get("summary", ""))),
            }
        )
    items.sort(
        key=lambda f: (
            f["rank"],
            f["kind"],
            f["resource"],
            f["ops"],
            f["summary"],
            f["kernels"],
            f["evidence"],
        )
    )
    for number, item in enumerate(items, 1):
        item["id"] = f"H{number:04d}"
    return items


def _evidence_sort(evidence: str) -> tuple:
    path, _, line = evidence.rpartition(":")
    return (path, int(line) if line.isdigit() else 0, evidence)


def write_hazards(
    repo_root: Path | str, out: Path | str, *, families: Sequence[str] | None = None
) -> dict:
    """Run ``bh.hazards.run(db, repo_root)`` when that module exists (else no findings)."""
    db = build(repo_root, families=families)
    try:
        hazards = importlib.import_module(f"{__package__}.hazards")
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}.hazards":
            raise
        hazards = None
    fingerprint = db.tree_fingerprint
    findings: Iterable[dict] = []
    if hazards is not None:
        result = hazards.run(db, Path(repo_root))
        if isinstance(result, dict):
            fingerprint = result.get("tree_fingerprint", fingerprint)
            findings = result.get("findings", [])
        else:
            findings = result or []
    data = {"findings": normalize_findings(findings), "tree_fingerprint": fingerprint}
    Path(out).write_text(dumps(data), encoding="utf-8", newline="\n")
    return data
