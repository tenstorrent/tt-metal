#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static checker for compute-kernel API ordering (init / uninit / reconfig).

The compute API is a thin layer over the stateful LLK API.  An operation works
only after a matching init configured its Tensix engine; the init's compile-time
arguments must still match at execute time; a different init on the same engine
replaces the seated operation; format reconfiguration invalidates inits that
snapshotted the changed fields; and a curated set of operations must be
uninitialized before the kernel returns.

The rules live in compute_api_order_rules.yaml (LLK level) and
compute_api_order_sfpu.yaml (SFPU init/execute pairs).  This checker reads a
kernel's real compiled translation unit, so preprocessor and `if constexpr`
variants are the ones the device build sees, follows calls through helper
headers, and applies the rules per thread (UNPACK / MATH / PACK).

Examples
--------
Check a kernel source directly (synthesizes the JIT compile flags)::

    python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
        --kernel tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary.cpp \
        --arch wormhole_b0 --thread MATH

Check a captured JIT build log (the same log the kernel clang-tidy flow uses)::

    TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_LOGGER_LEVEL=info <workload> 2>&1 | tee build.log
    python3 tt_metal/tt-llk/infra/check_compute_api_order.py --log build.log

Check an existing compile_commands.json::

    python3 tt_metal/tt-llk/infra/check_compute_api_order.py --compile-db compile_commands.json

The checker is advisory: it exits 0 unless --fail-on-findings is passed.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not a finding
    sys.stderr.write("error: PyYAML is required (python3 -m pip install pyyaml)\n")
    raise SystemExit(2)

try:
    import clang.cindex as ci
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: the clang python bindings are required. Install them with "
        "`python3 -m pip install clang` (version >= 17).\n"
    )
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RULES_PATH = HERE / "compute_api_order_rules.yaml"
SFPU_RULES_PATH = HERE / "compute_api_order_sfpu.yaml"
PRELUDE_PATH = HERE / "compute_api_order_prelude.h"

THREADS = {
    "UNPACK": {"define": "TRISC_UNPACK", "trisc": 0, "processor_index": 1},
    "MATH": {"define": "TRISC_MATH", "trisc": 1, "processor_index": 2},
    "PACK": {"define": "TRISC_PACK", "trisc": 2, "processor_index": 3},
}

ARCHES = {
    "wormhole_b0": {
        "define": "ARCH_WORMHOLE",
        "prefix": "wormhole",
        "dram_banks": 12,
        "l1_banks": 64,
        "mcpu": "tt-wh",
        "mcpu_tensix": "tt-wh-tensix",
    },
    "blackhole": {
        "define": "ARCH_BLACKHOLE",
        "prefix": "blackhole",
        "dram_banks": 8,
        "l1_banks": 140,
        "mcpu": "tt-bh",
        "mcpu_tensix": "tt-bh-tensix",
    },
}

# Calls that never touch compute/LLK state.  Everything else that cannot be
# resolved inside the device headers is reported as an analysis hole instead of
# guessed at.
TRANSPARENT_PREFIXES = (
    "cb_",
    "noc_",
    "get_arg_",
    "get_common_arg_",
    "get_compile_time_arg_",
    "get_named_compile_time_arg_",
    "tile_regs_",
    "dprint",
    "DPRINT",
    "waypoint",
    "WAYPOINT",
    "DeviceZone",
    "DeviceTimestamped",
    "SCOPED_TRACE",
    "llk_assert",
    "LLK_ASSERT",
    "ASSERT",
    "experimental::",
)
TRANSPARENT_EXACT = {
    "get_arg_val",
    "get_arg_addr",
    "get_common_arg_val",
    "get_compile_time_arg_val",
    "get_named_compile_time_arg_val",
    "tile_regs_acquire",
    "tile_regs_commit",
    "tile_regs_wait",
    "tile_regs_release",
    "tile_regs_release",
    "tile_regs_dummy",
    "cb_wait_front",
    "cb_pop_front",
    "cb_reserve_back",
    "cb_push_back",
}

# Paths whose definitions are in the analysis surface.  The checker follows
# calls into these and stops at a rule-table function.
SURFACE_MARKERS = (
    "/tt_metal/hw/inc/api/compute/",
    "/tt_metal/hw/ckernels/",
    "/tt_metal/tt-llk/",
    "/ttnn/cpp/ttnn/kernel_lib/",
)

# These names are LLK API entry points, not implementation helpers.  We stop at
# them whether or not a rule-table entry exists, so an unmodelled LLK call is an
# explicit hole instead of silently walking into the hardware layer.
LLK_API_PREFIX = "llk_"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MatchSpec:
    template: tuple[tuple[int, int], ...] = ()
    args: tuple[tuple[int, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class Transition:
    init_fn: str
    execute_fns: tuple[str, ...]
    uninit_fns: tuple[str, ...]
    match: MatchSpec
    architectures: tuple[str, ...] = ()


@dataclasses.dataclass
class OperationRule:
    id: str
    thread: str
    exu: str
    transitions: tuple[Transition, ...]
    requires_uninit: bool = False
    invalidated_by: tuple[str, ...] = ()
    sanitizer_operation: Optional[str] = None
    citation: str = ""
    additive: bool = False  # SFPU state accumulates instead of seating one op
    descend: bool = False  # compute-level rule: also follow the body


@dataclasses.dataclass(frozen=True)
class ExtraEvent:
    function: str
    kind: str
    op_id: str
    match: MatchSpec
    architectures: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SequenceRule:
    """A postcondition between a trigger init and another operation's init.

    The checker validates the condition after the trigger's body has been
    followed, so it sees the initializations the compute/LLK helper performs
    on the caller's behalf.  ``require_args`` maps a trigger argument index to
    a required-init argument index; both are compared as canonical values.
    """

    id: str
    trigger: str
    thread: str
    require_op: str
    require_args: tuple[tuple[int, int], ...] = ()
    message: str = ""
    citation: str = ""
    architectures: tuple[str, ...] = ()


@dataclasses.dataclass
class RuleTable:
    operations: dict[str, OperationRule]
    init_by_function: dict[str, list[tuple[OperationRule, Transition]]]
    execute_by_function: dict[str, list[tuple[OperationRule, Transition]]]
    uninit_by_function: dict[str, list[tuple[OperationRule, Transition]]]
    extra_events: dict[str, list[ExtraEvent]]
    invalidators: dict[str, set[str]]  # function -> exu names
    deprecated_init_apis: set[str]
    sfpu_groups: dict[str, OperationRule]
    sequence_rules: dict[str, list[SequenceRule]] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, arch: Optional[str]) -> "RuleTable":
        raw = yaml.safe_load(RULES_PATH.read_text())
        sfpu_raw = yaml.safe_load(SFPU_RULES_PATH.read_text()) if SFPU_RULES_PATH.exists() else {}

        ops: dict[str, OperationRule] = {}
        init_by: dict[str, list] = collections.defaultdict(list)
        execute_by: dict[str, list] = collections.defaultdict(list)
        uninit_by: dict[str, list] = collections.defaultdict(list)
        extra_events: dict[str, list[ExtraEvent]] = collections.defaultdict(list)
        invalidators: dict[str, set[str]] = collections.defaultdict(set)

        for entry in raw.get("operations", []):
            architectures = set(entry.get("architectures", []))
            if arch and architectures and arch not in architectures:
                continue
            transitions = []
            for tr in entry.get("transitions", []):
                tr_arches = tuple(tr.get("architectures", ()))
                if arch and tr_arches and arch not in tr_arches:
                    continue
                match_raw = tr.get("match") or {}
                match = MatchSpec(
                    template=tuple(tuple(x) for x in match_raw.get("template", ())),
                    args=tuple(tuple(x) for x in match_raw.get("args", ())),
                )
                transitions.append(
                    Transition(
                        init_fn=tr["init"],
                        execute_fns=tuple(tr.get("execute", ())),
                        uninit_fns=tuple(tr.get("uninit", ())),
                        match=match,
                        architectures=tr_arches,
                    )
                )
            if not transitions:
                continue
            op = OperationRule(
                id=entry["id"],
                thread=entry["thread"],
                exu=entry["exu"],
                transitions=tuple(transitions),
                requires_uninit=bool(entry.get("requires_uninit", False)),
                invalidated_by=tuple(entry.get("invalidated_by", ())),
                sanitizer_operation=entry.get("sanitizer_operation"),
                citation=entry.get("citation", ""),
                descend=bool(entry.get("descend", False)),
                additive=bool(entry.get("additive", False)),
            )
            ops[op.id] = op
            for tr in op.transitions:
                init_by[tr.init_fn].append((op, tr))
                for fn in tr.execute_fns:
                    execute_by[fn].append((op, tr))
                for fn in tr.uninit_fns:
                    uninit_by[fn].append((op, tr))

        for ev in raw.get("extra_events", []):
            ev_arches = tuple(ev.get("architectures", ()))
            if arch and ev_arches and arch not in ev_arches:
                continue
            match_raw = ev.get("match") or {}
            extra_events[ev["function"]].append(
                ExtraEvent(
                    function=ev["function"],
                    kind=ev["kind"],
                    op_id=ev["op"],
                    match=MatchSpec(
                        template=tuple(tuple(x) for x in match_raw.get("template", ())),
                        args=tuple(tuple(x) for x in match_raw.get("args", ())),
                    ),
                    architectures=ev_arches,
                )
            )

        for inv in raw.get("invalidators", []):
            for fn in inv.get("functions", ()):
                invalidators[fn].add(inv["exu"])

        sequence_rules: dict[str, list[SequenceRule]] = collections.defaultdict(list)
        for entry in raw.get("sequence_rules", []):
            arches = tuple(entry.get("architectures", ()))
            if arch and arches and arch not in arches:
                continue
            rule = SequenceRule(
                id=entry["id"],
                trigger=entry["trigger"],
                thread=entry["thread"],
                require_op=entry["require_init"],
                require_args=tuple(tuple(x) for x in entry.get("require_args", ())),
                message=entry.get("message", ""),
                citation=entry.get("citation", ""),
                architectures=arches,
            )
            sequence_rules[rule.trigger].append(rule)

        sfpu_groups: dict[str, OperationRule] = {}
        sfpu_entries = list(sfpu_raw.get("sfpu_groups", ())) + list(raw.get("sfpu_extra_groups", ()))
        for idx, entry in enumerate(sfpu_entries):
            op = OperationRule(
                id=entry["id"],
                thread="MATH",
                exu="Sfpu",
                transitions=(
                    Transition(
                        init_fn=init,
                        execute_fns=tuple(entry.get("execute", ())),
                        uninit_fns=tuple(entry.get("uninit", ())),
                        match=MatchSpec(),
                    )
                    for init in entry.get("init", ())
                ),
                requires_uninit=False,
                citation=entry.get("citation", ""),
                additive=True,
            )
            # One rule per execute function so execute lookup is unambiguous even
            # when two groups share an init (they do not today, but keep it safe).
            for tr in op.transitions:
                init_by[tr.init_fn].append((op, tr))
                for fn in tr.execute_fns:
                    execute_by[fn].append((op, tr))
                for fn in tr.uninit_fns:
                    uninit_by[fn].append((op, tr))
            sfpu_groups[op.id] = op

        return cls(
            operations=ops,
            init_by_function=dict(init_by),
            execute_by_function=dict(execute_by),
            uninit_by_function=dict(uninit_by),
            extra_events=dict(extra_events),
            invalidators=dict(invalidators),
            deprecated_init_apis=set(raw.get("deprecated_init_apis", ()) or ()),
            sfpu_groups=sfpu_groups,
            sequence_rules=dict(sequence_rules),
        )


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Finding:
    rule: str
    severity: str
    file: str
    line: int
    column: int
    thread: str
    arch: str
    message: str
    op: str = ""
    earlier: Optional[str] = None
    citation: str = ""
    detail: dict = dataclasses.field(default_factory=dict)

    def as_json(self) -> dict:
        d = dataclasses.asdict(self)
        for key in ("earlier", "citation"):
            if not d[key]:
                d.pop(key)
        if not d["detail"]:
            d.pop("detail")
        return d

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}:{self.column}"
        extra = f" [{self.thread}/{self.arch}]" if self.thread or self.arch else ""
        msg = self.message
        if self.earlier:
            msg += f" (earlier call: {self.earlier})"
        return f"{loc}: {self.severity}: {msg} [{self.rule}]{extra}"


# ---------------------------------------------------------------------------
# Lightweight constant evaluation / canonicalisation over libclang cursors
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)([uUlL]*)$")

_BINARY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b else None,
    "%": lambda a, b: a % b if b else None,
    "<<": lambda a, b: a << b if 0 <= b < 64 else None,
    ">>": lambda a, b: a >> b if 0 <= b < 64 else None,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "&&": lambda a, b: int(bool(a) and bool(b)),
    "||": lambda a, b: int(bool(a) or bool(b)),
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_primary(text: str) -> bool:
    """True for a canonical expression that needs no outer parentheses."""
    if text.startswith("#"):
        return True
    if re.fullmatch(r"[A-Za-z_]\w*", text):
        return True
    if re.fullmatch(r"[A-Za-z_]\w*\(.*\)", text):
        return True
    if re.fullmatch(r"[0-9]+", text):
        return True
    return False


class ExprEval:
    """A small integer constant evaluator for kernel argument expressions."""

    def __init__(self):
        self._const_cache: dict[str, Optional[int]] = {}
        self._active: set[str] = set()
        # Bindings for the constexpr function currently being folded.  This is
        # deliberately separate from Canonicalizer's text bindings: these are
        # integer values, not canonical source text.
        self._fn_frames: list[dict[str, Optional[int]]] = []
        self._fn_active: set = set()
        self.lookup = None  # set to Canonicalizer.lookup

    def eval(self, cursor: "ci.Cursor") -> Optional[int]:
        kind = cursor.kind
        try:
            if kind == ci.CursorKind.INTEGER_LITERAL:
                m = _INT_RE.match(cursor.spelling or "".join(t.spelling for t in cursor.get_tokens()))
                if not m:
                    return None
                return int(m.group(1), 0)
            if kind == ci.CursorKind.CHARACTER_LITERAL:
                tokens = "".join(t.spelling for t in cursor.get_tokens())
                try:
                    return ord(tokens.strip("'"))
                except Exception:
                    return None
            if kind == ci.CursorKind.CXX_BOOL_LITERAL_EXPR:
                tokens = "".join(t.spelling for t in cursor.get_tokens())
                return 1 if tokens == "true" else 0
            if kind == ci.CursorKind.DECL_REF_EXPR:
                ref_name = cursor.spelling or "".join(t.spelling for t in cursor.get_tokens())
                for frame in reversed(self._fn_frames):
                    if ref_name in frame:
                        return frame[ref_name]
                decl = cursor.referenced
                if self.lookup is not None:
                    bound = self.lookup(ref_name)
                    if bound and bound.startswith("#"):
                        try:
                            return int(bound[1:])
                        except ValueError:
                            return None
                return self._eval_decl(decl)
            if kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                return self._eval_subscript(cursor)
            if kind == ci.CursorKind.CALL_EXPR:
                return self._eval_call(cursor)
            if kind in (
                ci.CursorKind.PAREN_EXPR,
                ci.CursorKind.CXX_STATIC_CAST_EXPR,
                ci.CursorKind.CXX_REINTERPRET_CAST_EXPR,
                ci.CursorKind.CXX_CONST_CAST_EXPR,
                ci.CursorKind.CXX_DYNAMIC_CAST_EXPR,
                ci.CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
                ci.CursorKind.CSTYLE_CAST_EXPR,
                ci.CursorKind.UNEXPOSED_EXPR,
            ):
                for child in cursor.get_children():
                    value = self.eval(child)
                    if value is not None:
                        return value
                return None
            if kind == ci.CursorKind.UNARY_OPERATOR:
                tokens = [t.spelling for t in cursor.get_tokens()]
                children = list(cursor.get_children())
                if not children:
                    return None
                value = self.eval(children[0])
                if value is None:
                    return None
                if "-" in tokens:
                    return -value
                if "~" in tokens:
                    return ~value
                if "!" in tokens:
                    return int(not value)
                if "+" in tokens:
                    return value
                return None
            if kind == ci.CursorKind.BINARY_OPERATOR:
                children = list(cursor.get_children())
                if len(children) < 2:
                    return None
                left = self.eval(children[0])
                right = self.eval(children[1])
                if left is None or right is None:
                    return None
                op = self._operator_token(cursor, children)
                fn = _BINARY_OPS.get(op)
                if fn is None:
                    return None
                return fn(left, right)
            if kind == ci.CursorKind.CONDITIONAL_OPERATOR:
                children = list(cursor.get_children())
                if len(children) != 3:
                    return None
                cond = self.eval(children[0])
                if cond is None:
                    return None
                return self.eval(children[1] if cond else children[2])
        except Exception:
            return None
        return None

    def _operator_token(self, cursor, children) -> str:
        try:
            lhs_end = children[0].extent.end.offset
            rhs_start = children[1].extent.start.offset
            text = ""
            for tok in cursor.get_tokens():
                off = tok.extent.start.offset
                if lhs_end <= off <= rhs_start:
                    text += tok.spelling
            for op in ("<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "%", "&", "|", "^", "<", ">"):
                if op in text:
                    return op
        except Exception:
            pass
        return ""

    def _eval_decl(self, decl) -> Optional[int]:
        if decl is None:
            return None
        if decl.kind == ci.CursorKind.ENUM_CONSTANT_DECL:
            try:
                return int(decl.enum_value)
            except Exception:
                return None
        if decl.kind != ci.CursorKind.VAR_DECL:
            return None
        key = decl.get_usr() or f"{decl.location.file}:{decl.location.line}:{decl.spelling}"
        local = self._is_function_local(decl)
        if not local and key in self._const_cache:
            return self._const_cache[key]
        if key in self._active:
            return None
        self._active.add(key)
        value = None
        try:
            tokens = "".join(t.spelling for t in decl.get_tokens())
            is_const = "constexpr" in tokens or re.search(r"\bconst\b", tokens) is not None
            if is_const:
                for child in decl.get_children():
                    if child.kind in (
                        ci.CursorKind.TYPE_REF,
                        ci.CursorKind.TEMPLATE_REF,
                        ci.CursorKind.NAMESPACE_REF,
                    ):
                        continue
                    value = self.eval(child)
                    if value is not None:
                        break
        finally:
            self._active.discard(key)
        # A local constexpr can depend on template parameters, so its value is
        # per-instantiation and must not be cached by declaration identity.
        if not local:
            self._const_cache[key] = value
        return value

    @staticmethod
    def _is_function_local(decl) -> bool:
        function_kinds = {
            getattr(ci.CursorKind, name)
            for name in (
                "FUNCTION_DECL",
                "FUNCTION_TEMPLATE",
                "CXX_METHOD",
                "CONSTRUCTOR",
                "DESTRUCTOR",
                "LAMBDA_EXPR",
            )
            if hasattr(ci.CursorKind, name)
        }
        parent = getattr(decl, "semantic_parent", None)
        while parent is not None:
            if parent.kind in function_kinds:
                return True
            parent = getattr(parent, "semantic_parent", None)
        return False

    def _eval_subscript(self, cursor) -> Optional[int]:
        children = list(cursor.get_children())
        if len(children) < 2:
            return None
        index = self.eval(children[1])
        if index is None:
            return None
        base = children[0]
        while base is not None and base.kind in (
            ci.CursorKind.PAREN_EXPR,
            ci.CursorKind.UNEXPOSED_EXPR,
        ):
            base = next(iter(base.get_children()), None)
        decl = base.referenced if base is not None and base.kind == ci.CursorKind.DECL_REF_EXPR else None
        if decl is None or decl.kind != ci.CursorKind.VAR_DECL:
            return None
        initializer = next(
            (c for c in decl.get_children() if c.kind == ci.CursorKind.INIT_LIST_EXPR),
            None,
        )
        if initializer is None:
            return None
        elements = list(initializer.get_children())
        if 0 <= index < len(elements):
            return self.eval(elements[index])
        return None

    def _eval_call(self, cursor) -> Optional[int]:
        """Fold a call to a constexpr function with compile-time-known arguments.

        The checker analyses template definitions, where libclang cannot fold an
        expression itself (the type is dependent).  A small statement
        interpreter over the function body is enough for the helper predicates
        used by the compute kernel library (``if constexpr`` plus ``return``).
        """
        ref = cursor.referenced
        callee_expr = next(
            (c for c in cursor.get_children() if c.kind == ci.CursorKind.DECL_REF_EXPR),
            None,
        )
        if ref is None and callee_expr is not None:
            ref = callee_expr.referenced
        if ref is None or ref.kind not in (
            ci.CursorKind.FUNCTION_DECL,
            ci.CursorKind.FUNCTION_TEMPLATE,
            ci.CursorKind.CXX_METHOD,
        ):
            return None
        head = []
        for token in ref.get_tokens():
            if token.spelling == "{":
                break
            head.append(token.spelling)
        if "constexpr" not in head:
            return None
        body = next((c for c in ref.get_children() if c.kind == ci.CursorKind.COMPOUND_STMT), None)
        if body is None:
            return None
        tparams = [
            c.spelling
            for c in ref.get_children()
            if c.kind
            in (
                ci.CursorKind.TEMPLATE_TYPE_PARAMETER,
                ci.CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
                getattr(ci.CursorKind, "TEMPLATE_TEMPLATE_PARAMETER", ci.CursorKind.UNEXPOSED_DECL),
            )
        ]
        targs = []
        if callee_expr is not None:
            targs = [
                c
                for c in callee_expr.get_children()
                if c.kind != getattr(ci.CursorKind, "OVERLOADED_DECL_REF", ci.CursorKind.UNEXPOSED_DECL)
            ]
        bindings: dict[str, Optional[int]] = {}
        for name, arg in zip(tparams, targs):
            value = self.eval(arg)
            if value is None:
                return None
            bindings[name] = value
        params = [c for c in ref.get_children() if c.kind == ci.CursorKind.PARM_DECL]
        for param, arg in zip(params, cursor.get_arguments()):
            value = self.eval(arg)
            if value is None:
                return None
            bindings[param.spelling] = value
        key = (ref.get_usr(), tuple(bindings.get(name) for name in tparams))
        if key in self._fn_active:
            return None
        self._fn_active.add(key)
        self._fn_frames.append(bindings)
        try:
            value, returned = self._eval_constexpr_stmt(body)
            return value if returned else None
        finally:
            self._fn_frames.pop()
            self._fn_active.discard(key)

    def _eval_constexpr_stmt(self, cursor) -> tuple[Optional[int], bool]:
        kind = cursor.kind
        if kind == ci.CursorKind.COMPOUND_STMT:
            for child in cursor.get_children():
                value, returned = self._eval_constexpr_stmt(child)
                if returned:
                    return value, True
            return None, False
        if kind == ci.CursorKind.DECL_STMT:
            for child in cursor.get_children():
                if child.kind != ci.CursorKind.VAR_DECL:
                    continue
                tokens = "".join(t.spelling for t in child.get_tokens())
                if "constexpr" not in tokens and re.search(r"\bconst\b", tokens) is None:
                    continue
                initializer = next(
                    (
                        c
                        for c in child.get_children()
                        if c.kind
                        not in (
                            ci.CursorKind.TYPE_REF,
                            ci.CursorKind.TEMPLATE_REF,
                            ci.CursorKind.NAMESPACE_REF,
                        )
                    ),
                    None,
                )
                if initializer is not None and self._fn_frames:
                    self._fn_frames[-1][child.spelling] = self.eval(initializer)
            return None, False
        if kind == ci.CursorKind.IF_STMT:
            children = list(cursor.get_children())
            if not children:
                return None, False
            condition = self.eval(children[0])
            if condition is None:
                return None, False
            branch = children[1] if condition else (children[2] if len(children) > 2 else None)
            if branch is None:
                return None, False
            return self._eval_constexpr_stmt(branch)
        if kind == ci.CursorKind.RETURN_STMT:
            expression = next(iter(cursor.get_children()), None)
            return (self.eval(expression) if expression is not None else None), True
        return None, False


class Canonicalizer:
    """Canonical expression text plus a stack of parameter/local bindings."""

    def __init__(self, evaluator: ExprEval):
        self.eval = evaluator
        self.frames: list[dict[str, Optional[str]]] = []
        self._text_cache: dict[tuple, str] = {}

    def push(self, bindings: Optional[dict[str, Optional[str]]] = None):
        self.frames.append(bindings or {})

    def pop(self):
        self.frames.pop()

    def bind(self, name: str, value: Optional[str]):
        if self.frames:
            self.frames[-1][name] = value

    def lookup(self, name: str) -> Optional[str]:
        for frame in reversed(self.frames):
            if name in frame:
                return frame[name]
        return None

    def text(self, cursor: "ci.Cursor") -> str:
        key = (
            cursor.location.file.name if cursor.location.file else "",
            cursor.extent.start.offset,
            cursor.extent.end.offset,
        )
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        raw = " ".join(t.spelling for t in cursor.get_tokens())
        raw = _norm(raw)
        # Substitute bound parameter/local names token-wise.  The substitution
        # values are themselves canonical text, so a helper called from two
        # sites compares equal exactly when the caller expressions are equal.
        tokens = re.findall(r"[A-Za-z_]\w*|[0-9]+|[^A-Za-z_0-9]+", raw)
        out = []
        for tok in tokens:
            if re.fullmatch(r"[A-Za-z_]\w*", tok):
                repl = self.lookup(tok)
                if repl is not None:
                    out.append(repl if _is_primary(repl) else f"({repl})")
                    continue
            out.append(tok)
        text = _norm("".join(out))
        self._text_cache[key] = text
        return text

    def canonical(self, cursor: "ci.Cursor") -> tuple[str, Optional[int]]:
        value = self.eval.eval(cursor)
        if value is not None:
            return f"#{value}", value
        return self.text(cursor), None


# ---------------------------------------------------------------------------
# Analyzer state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InitRecord:
    op_id: str
    init_fn: str
    file: str
    line: int
    column: int
    thread: str
    template: tuple
    args: tuple[str, ...]
    arg_values: tuple[Optional[int], ...] = ()
    unknown: bool = False
    invalidated_by: Optional[tuple[str, int]] = None
    generation: int = 0

    @property
    def site(self) -> str:
        return f"{self.file}:{self.line}"

    def clone(self) -> "InitRecord":
        return dataclasses.replace(self)


@dataclasses.dataclass
class State:
    seated: dict[str, Optional[InitRecord]] = dataclasses.field(default_factory=dict)
    armed: dict[str, InitRecord] = dataclasses.field(default_factory=dict)
    sfpu: dict[str, list[InitRecord]] = dataclasses.field(default_factory=dict)
    unknown: bool = False

    def clone(self) -> "State":
        return State(
            seated={k: (v.clone() if v else None) for k, v in self.seated.items()},
            armed={k: v.clone() for k, v in self.armed.items()},
            sfpu={k: [r.clone() for r in v] for k, v in self.sfpu.items()},
            unknown=self.unknown,
        )


@dataclasses.dataclass
class LoopInfo:
    variables: set[str]
    call_depth: int
    conditional_depth: int = 0
    warned: set = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class Flow:
    NORMAL = "normal"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


class Scope:
    """A function invocation in the inlined call stack."""

    def __init__(self, cursor, name: str):
        self.cursor = cursor
        self.name = name
        self.locals: dict[str, Optional[str]] = {}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class Analyzer:
    def __init__(self, args: list[str], thread: str, arch: str, table: RuleTable, source: str):
        self.thread = thread
        self.arch = arch
        self.table = table
        self.source = source
        self.evaluator = ExprEval()
        self.canon = Canonicalizer(self.evaluator)
        self.evaluator.lookup = self.canon.lookup
        self.findings: list[Finding] = []
        self._finding_keys: set = set()
        self._function_cache: dict[tuple, object] = {}
        self._call_stack: list[tuple[str, tuple]] = []
        self._template_params_cache: dict[str, list[str]] = {}
        self._loop_stack: list[LoopInfo] = []
        self._exit_states: list[State] = []
        self._entry_name = ""
        self._diagnostic_count = 0
        self._forced_conditions: dict[str, bool] = {}
        self._discovered_conditions: set[str] = set()
        self._condition_files: dict[str, str] = {}
        self._site_stack: list[tuple[str, int, int]] = []
        self._init_generation = 0

        index = ci.Index.create()
        self.tu = index.parse(
            source,
            args=args,
            options=ci.TranslationUnit.PARSE_NONE,
        )
        self.parse_errors = [d for d in self.tu.diagnostics if d.severity >= ci.Diagnostic.Error]

    # -- diagnostics -----------------------------------------------------

    def _add(self, finding: Finding):
        key = (finding.rule, finding.file, finding.line, finding.thread, finding.op, finding.message)
        if key in self._finding_keys:
            return
        self._finding_keys.add(key)
        self.findings.append(finding)

    def _loc(self, cursor) -> tuple[str, int, int]:
        loc = cursor.location
        file = loc.file.name if loc.file else self.source
        try:
            file = str(Path(file).resolve())
        except Exception:
            pass
        return file, loc.line, loc.column

    # -- entry points ----------------------------------------------------

    def find_entry(self):
        candidates = []
        tt_kernel_names = self._tt_kernel_names()

        def visit(cursor):
            if cursor.kind == ci.CursorKind.FUNCTION_DECL and cursor.is_definition():
                name = cursor.spelling
                if name == "kernel_main":
                    candidates.append((0, cursor))
                elif name in tt_kernel_names:
                    candidates.append((1, cursor))
            for child in cursor.get_children():
                visit(child)

        visit(self.tu.cursor)
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _tt_kernel_names(self) -> set[str]:
        try:
            text = Path(self.source).read_text(errors="replace")
        except Exception:
            return set()
        names = set()
        for m in re.finditer(r"\bTT_KERNEL\b", text):
            tail = text[m.end() : m.end() + 500]
            m2 = re.search(r"\b([A-Za-z_]\w*)\s*\(", tail)
            if m2:
                names.add(m2.group(1))
        return names

    def analyze_paths(self, max_paths: int = 24) -> None:
        """Enumerate correlated runtime conditions (bounded) and union findings.

        A single merge at an `if` loses the correlation between two `if (x == 0)`
        statements: the first initializes an engine on one path, the second runs the
        operation on the same path.  Rather than a full path-sensitive CFG, rerun the
        walk with each discovered condition forced both ways.  The condition text is
        the key, so repeated `if (step == 0)` sites stay consistent.  The number of
        runs is capped; paths beyond the cap are reported as a coverage note.
        """
        max_paths = max(1, max_paths)
        assignments = [{}]
        seen = set()
        runs = 0
        while assignments and runs < max_paths:
            forced = assignments.pop(0)
            key = tuple(sorted(forced.items()))
            if key in seen:
                continue
            seen.add(key)
            self._forced_conditions = forced
            self._discovered_conditions = set()
            self._exit_states = []
            self.analyze_one()
            runs += 1
            new_conditions = [c for c in self._discovered_conditions if c not in forced]
            # Prefer conditions written in the kernel source: library-header
            # conditions are usually independent configuration branches, while
            # the kernel's own conditions correlate init/operation pairs.
            new_conditions.sort(key=lambda c: (self._condition_files.get(c, "") != self.source, c))
            for cond in new_conditions:
                new_forced = dict(forced)
                new_forced[cond] = False
                assignments.append(new_forced)
        self._forced_conditions = {}

    def analyze_one(self) -> None:
        entry = self.find_entry()
        if entry is None:
            return
        self._entry_name = entry.spelling
        state = State()
        state = self._run_function(entry, state, top_level=True)
        for exit_state in self._exit_states or [state]:
            if exit_state.unknown:
                continue
            for op_id, record in exit_state.armed.items():
                op = self.table.operations.get(op_id)
                if op is None or not op.requires_uninit:
                    continue
                file, line, col = self._loc(entry)
                self._add(
                    Finding(
                        rule="missing-uninit",
                        severity="error",
                        file=record.file,
                        line=record.line,
                        column=record.column,
                        thread=self.thread,
                        arch=self.arch,
                        op=op_id,
                        message=(
                            f"{op_id} init is never paired with a required uninit "
                            f"({', '.join(self._uninit_names(op)) or 'no uninit API'})"
                        ),
                        earlier=f"{record.site} ({record.init_fn})",
                        citation=op.citation,
                    )
                )

    def _uninit_names(self, op: OperationRule) -> list[str]:
        return sorted({fn for tr in op.transitions for fn in tr.uninit_fns})

    # -- call graph / interpretation ------------------------------------

    def _surface_file(self, path: str) -> bool:
        p = path.replace(os.sep, "/")
        if any(marker in p for marker in SURFACE_MARKERS):
            return True
        try:
            return Path(path).resolve().parent == Path(self.source).resolve().parent
        except Exception:
            return False

    def _is_rule_function(self, name: str) -> bool:
        return (
            name in self.table.init_by_function
            or name in self.table.execute_by_function
            or name in self.table.uninit_by_function
            or name in self.table.extra_events
            or name in self.table.invalidators
        )

    def _resolve_body(self, ref):
        if ref is None:
            return None
        if ref.is_definition():
            return ref
        try:
            d = ref.get_definition()
        except Exception:
            d = None
        return d if d is not None and d.is_definition() else None

    def _call_key(self, ref) -> tuple:
        vals = []
        try:
            n = ref.get_num_template_arguments()
        except Exception:
            n = -1
        if n and n > 0:
            for i in range(n):
                kind = ref.get_template_argument_kind(i)
                if kind == ci.TemplateArgumentKind.INTEGRAL:
                    vals.append(("int", ref.get_template_argument_unsigned_value(i)))
                elif kind == ci.TemplateArgumentKind.TYPE:
                    vals.append(("type", ref.get_template_argument_type(i).spelling))
                else:
                    vals.append(("other", str(kind)))
        return (ref.get_usr() or ref.spelling, tuple(vals))

    def _template_param_names(self, body) -> list[str]:
        usr = body.get_usr() or f"{body.location.file}:{body.location.line}"
        if usr in self._template_params_cache:
            return self._template_params_cache[usr]
        names: list[str] = []
        try:
            path = Path(body.location.file.name)
            text = path.read_text(errors="replace")
            start = max(0, body.location.offset - 3000)
            head = text[start : body.location.offset]
            idx = head.rfind("template")
            if idx >= 0:
                lt = head.find("<", idx)
                if lt >= 0:
                    depth = 0
                    end = None
                    for j in range(lt, len(head)):
                        c = head[j]
                        if c == "<":
                            depth += 1
                        elif c == ">":
                            depth -= 1
                            if depth == 0:
                                end = j
                                break
                    if end is not None:
                        raw = head[lt + 1 : end]
                        depth = 0
                        piece = ""
                        for c in raw + ",":
                            if c in "<([":
                                depth += 1
                            elif c in ">)]":
                                depth -= 1
                            if c == "," and depth == 0:
                                m = re.findall(r"[A-Za-z_]\w*", piece.split("=")[0])
                                if m:
                                    names.append(m[-1])
                                piece = ""
                            else:
                                piece += c
        except Exception:
            names = []
        self._template_params_cache[usr] = names
        return names

    def _run_function(self, body, state: State, top_level: bool = False, bindings: Optional[dict] = None) -> State:
        key = self._call_key(body)
        if key in self._call_stack:
            state.unknown = True
            return state
        self._call_stack.append(key)
        self.canon.push(dict(bindings or {}))
        try:
            compound = next(
                (c for c in body.get_children() if c.kind == ci.CursorKind.COMPOUND_STMT), None
            )
            if compound is not None:
                exit_state, flow = self._walk_stmt(compound, state)
                if flow == Flow.RETURN:
                    state = exit_state
                else:
                    state = exit_state
            return state
        finally:
            self.canon.pop()
            self._call_stack.pop()

    def _walk_expr(self, cursor, state: State) -> State:
        kind = cursor.kind
        if kind in (ci.CursorKind.CALL_EXPR, ci.CursorKind.CALL_EXPR):
            args = list(cursor.get_arguments())
            arg_set = {id(a) for a in args}
            for child in cursor.get_children():
                if id(child) in arg_set:
                    continue
                if child.kind in (
                    ci.CursorKind.DECL_REF_EXPR,
                    ci.CursorKind.MEMBER_REF_EXPR,
                    ci.CursorKind.UNEXPOSED_EXPR,
                ):
                    # Callee expression: no ordering side effects worth modelling.
                    if child.kind != ci.CursorKind.UNEXPOSED_EXPR:
                        continue
                state = self._walk_expr(child, state)
            for a in args:
                state = self._walk_expr(a, state)
            return self._handle_call(cursor, state)
        # Do not descend into lambda bodies as part of expression evaluation;
        # a lambda is a separate function and is handled when called.
        if kind == ci.CursorKind.LAMBDA_EXPR:
            return state
        for child in cursor.get_children():
            if child.kind in (
                ci.CursorKind.COMPOUND_STMT,
                ci.CursorKind.IF_STMT,
                ci.CursorKind.FOR_STMT,
                ci.CursorKind.WHILE_STMT,
                ci.CursorKind.DO_STMT,
                ci.CursorKind.SWITCH_STMT,
            ):
                state = self._walk_stmt(child, state)[0]
            elif child.kind in (ci.CursorKind.CALL_EXPR, ci.CursorKind.CALL_EXPR):
                state = self._walk_expr(child, state)
            else:
                state = self._walk_expr(child, state)
        return state

    def _constant_condition(self, cursor) -> Optional[int]:
        value = self.evaluator.eval(cursor)
        if value is not None:
            return value
        # Clang wraps an `if constexpr` condition in a ConstantExpr whose
        # libclang spelling is the underlying expression; the evaluated value is
        # the single bool child.
        for child in cursor.get_children():
            if child.kind == ci.CursorKind.CXX_BOOL_LITERAL_EXPR:
                return self.evaluator.eval(child)
        return None

    def _walk_stmt(self, cursor, state: State) -> tuple[State, str]:
        kind = cursor.kind

        if kind == ci.CursorKind.COMPOUND_STMT:
            for child in cursor.get_children():
                state, flow = self._walk_stmt(child, state)
                if flow != Flow.NORMAL:
                    return state, flow
            return state, Flow.NORMAL

        if kind == ci.CursorKind.DECL_STMT:
            for child in cursor.get_children():
                if child.kind == ci.CursorKind.VAR_DECL:
                    init = next(
                        (
                            c
                            for c in child.get_children()
                            if c.kind
                            not in (
                                ci.CursorKind.TYPE_REF,
                                ci.CursorKind.TEMPLATE_REF,
                                ci.CursorKind.NAMESPACE_REF,
                            )
                        ),
                        None,
                    )
                    if init is not None:
                        state = self._walk_expr(init, state)
                        canon, value = self.canon.canonical(init)
                        tokens = "".join(t.spelling for t in child.get_tokens())
                        is_const = "constexpr" in tokens or re.search(r"\bconst\b", tokens) is not None
                        self.canon.bind(child.spelling, canon if is_const else None)
                    else:
                        self.canon.bind(child.spelling, None)
                else:
                    state = self._walk_expr(child, state)
            return state, Flow.NORMAL

        if kind == ci.CursorKind.IF_STMT:
            children = list(cursor.get_children())
            if not children:
                return state, Flow.NORMAL
            cond = children[0]
            state = self._walk_expr(cond, state)
            cond_value = self._constant_condition(cond)
            cond_key = None
            if cond_value is None:
                cond_key = self.canon.text(cond)
                if cond_key in self._forced_conditions:
                    cond_value = 1 if self._forced_conditions[cond_key] else 0
                else:
                    self._discovered_conditions.add(cond_key)
                    self._condition_files.setdefault(cond_key, self._record_site(cond)[0])
                    # Explore one path per run; analyze_paths() reruns with the
                    # condition forced false to cover the other branch.
                    cond_value = 1
            for info in self._loop_stack:
                info.conditional_depth += 1
            try:
                if cond_value is not None:
                    branch = children[1] if cond_value else (children[2] if len(children) > 2 else None)
                    if branch is None:
                        return state, Flow.NORMAL
                    return self._walk_stmt(branch, state)
                then_state = state.clone()
                else_state = state.clone()
                flow_then = Flow.NORMAL
                flow_else = Flow.NORMAL
                if len(children) > 1:
                    then_state, flow_then = self._walk_stmt(children[1], then_state)
                if len(children) > 2:
                    else_state, flow_else = self._walk_stmt(children[2], else_state)
                merged = self._merge(then_state, else_state)
                if flow_then != Flow.NORMAL and flow_then == flow_else:
                    return merged, flow_then
                return merged, Flow.NORMAL
            finally:
                for info in self._loop_stack:
                    info.conditional_depth -= 1

        if kind in (ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT, ci.CursorKind.DO_STMT, ci.CursorKind.CXX_FOR_RANGE_STMT):
            return self._walk_loop(cursor, state)

        if kind == ci.CursorKind.SWITCH_STMT:
            children = list(cursor.get_children())
            if not children:
                return state, Flow.NORMAL
            state = self._walk_expr(children[0], state)
            body = children[1] if len(children) > 1 else None
            if body is None:
                return state, Flow.NORMAL
            cases = [c for c in body.get_children() if c.kind in (ci.CursorKind.CASE_STMT, ci.CursorKind.DEFAULT_STMT)]
            if not cases:
                return self._walk_stmt(body, state)
            merged = None
            for case in cases:
                case_state = state.clone()
                card = list(case.get_children())
                if card:
                    case_state = self._walk_expr(card[0], case_state)
                for child in card[1:]:
                    case_state, _ = self._walk_stmt(child, case_state)
                merged = case_state if merged is None else self._merge(merged, case_state)
            return (merged or state), Flow.NORMAL

        if kind == ci.CursorKind.RETURN_STMT:
            for child in cursor.get_children():
                state = self._walk_expr(child, state)
            if len(self._call_stack) == 1:
                self._exit_states.append(state.clone())
            return state, Flow.RETURN

        if kind == ci.CursorKind.BREAK_STMT:
            return state, Flow.BREAK
        if kind == ci.CursorKind.CONTINUE_STMT:
            return state, Flow.CONTINUE

        if kind in (
            ci.CursorKind.CASE_STMT,
            ci.CursorKind.DEFAULT_STMT,
            ci.CursorKind.LABEL_STMT,
            ci.CursorKind.CXX_TRY_STMT,
            ci.CursorKind.CXX_CATCH_STMT,
            ci.CursorKind.NULL_STMT,
        ):
            for child in cursor.get_children():
                kind_c = child.kind
                if kind_c in (
                    ci.CursorKind.COMPOUND_STMT,
                    ci.CursorKind.IF_STMT,
                    ci.CursorKind.FOR_STMT,
                    ci.CursorKind.WHILE_STMT,
                    ci.CursorKind.DO_STMT,
                    ci.CursorKind.SWITCH_STMT,
                    ci.CursorKind.RETURN_STMT,
                    ci.CursorKind.DECL_STMT,
                ):
                    state, flow = self._walk_stmt(child, state)
                    if flow != Flow.NORMAL:
                        return state, flow
                else:
                    state = self._walk_expr(child, state)
            return state, Flow.NORMAL

        # Expression statement and anything else with children.
        if kind in (ci.CursorKind.GOTO_STMT,):
            return state, Flow.NORMAL
        return self._walk_expr(cursor, state), Flow.NORMAL

    def _walk_loop(self, cursor, state: State) -> tuple[State, str]:
        children = list(cursor.get_children())
        kind = cursor.kind
        loop_vars: set[str] = set()
        init_child = None
        cond_child = None
        inc_child = None
        body = None

        if kind == ci.CursorKind.FOR_STMT:
            rest = list(children)
            if rest and rest[0].kind == ci.CursorKind.DECL_STMT:
                init_child = rest.pop(0)
                for d in init_child.get_children():
                    if d.kind == ci.CursorKind.VAR_DECL:
                        loop_vars.add(d.spelling)
            if rest and rest[0].kind not in (ci.CursorKind.COMPOUND_STMT,):
                cond_child = rest.pop(0)
            if len(rest) > 1:
                inc_child = rest.pop(0)
            body = rest[-1] if rest else None
        elif kind == ci.CursorKind.WHILE_STMT:
            if children:
                cond_child = children[0]
            body = children[1] if len(children) > 1 else None
        elif kind == ci.CursorKind.DO_STMT:
            body = children[0] if children else None
            cond_child = children[1] if len(children) > 1 else None
        elif kind == ci.CursorKind.CXX_FOR_RANGE_STMT:
            if children:
                loop_vars.add(children[0].spelling)
            body = children[-1] if children else None

        entry = state.clone()
        if init_child is not None:
            state, _ = self._walk_stmt(init_child, state)
            # The induction variable's first-iteration value is known even
            # though the variable is mutable; bind it so `if (x == 0)` resolves.
            for decl in init_child.get_children():
                if decl.kind != ci.CursorKind.VAR_DECL:
                    continue
                initializer = next(
                    (
                        c
                        for c in decl.get_children()
                        if c.kind
                        not in (
                            ci.CursorKind.TYPE_REF,
                            ci.CursorKind.TEMPLATE_REF,
                            ci.CursorKind.NAMESPACE_REF,
                        )
                    ),
                    None,
                )
                if initializer is not None:
                    canon, _ = self.canon.canonical(initializer)
                    self.canon.bind(decl.spelling, canon)
        cond_value = None
        if cond_child is not None:
            state = self._walk_expr(cond_child, state)
            cond_value = self._constant_condition(cond_child)

        info = LoopInfo(variables=loop_vars, call_depth=len(self._call_stack))
        self._loop_stack.append(info)
        try:
            if body is not None:
                body_state, _ = self._walk_stmt(body, state.clone())
                if inc_child is not None:
                    body_state = self._walk_expr(inc_child, body_state)
        finally:
            self._loop_stack.pop()

        if cond_child is not None:
            state = self._walk_expr(cond_child, state)

        body_state = body_state if body is not None else state
        if kind == ci.CursorKind.DO_STMT:
            merged = body_state
        elif cond_value is not None and cond_value != 0:
            # A compile-time-true loop condition cannot take the zero-iteration
            # path; keeping the entry state would drop an init that is required
            # before a later operation (e.g. pack init before a fixed-trip loop).
            merged = body_state
        else:
            merged = self._merge(entry, body_state)
        return merged, Flow.NORMAL

    def _merge(self, a: State, b: State) -> State:
        out = State()
        out.unknown = a.unknown or b.unknown
        for exu in ("Unpack", "Fpu", "Pack"):
            ra = a.seated.get(exu)
            rb = b.seated.get(exu)
            if ra is None or rb is None:
                out.seated[exu] = None
            elif ra.op_id != rb.op_id:
                out.seated[exu] = None
            else:
                rec = ra.clone()
                if ra.template != rb.template or ra.args != rb.args:
                    rec.unknown = True
                out.seated[exu] = rec
        out.armed = dict(a.armed)
        for op_id, rec in b.armed.items():
            if op_id in out.armed:
                merged = out.armed[op_id]
                if merged.template != rec.template or merged.args != rec.args:
                    merged = merged.clone()
                    merged.unknown = True
                    out.armed[op_id] = merged
            else:
                out.armed[op_id] = rec.clone()
        sfpu_ids = set(a.sfpu) & set(b.sfpu)
        for op_id in sfpu_ids:
            same = []
            for ra in a.sfpu[op_id]:
                for rb in b.sfpu[op_id]:
                    if ra.template == rb.template and ra.args == rb.args:
                        same.append(ra.clone())
                        break
            if same:
                out.sfpu[op_id] = same
        return out

    # -- rule application ------------------------------------------------

    def _template_args(self, ref) -> tuple:
        args = []
        try:
            n = ref.get_num_template_arguments()
        except Exception:
            n = -1
        for i in range(max(0, n)):
            kind = ref.get_template_argument_kind(i)
            if kind == ci.TemplateArgumentKind.INTEGRAL:
                value = ref.get_template_argument_unsigned_value(i)
                if value is None:
                    value = ref.get_template_argument_value(i)
                args.append(("int", value))
            elif kind == ci.TemplateArgumentKind.TYPE:
                args.append(("type", ref.get_template_argument_type(i).spelling))
            else:
                args.append(("other", None))
        return tuple(args)

    def _canon_args(self, call, params) -> tuple[tuple[str, ...], tuple[Optional[int], ...]]:
        args = list(call.get_arguments())
        texts = []
        values = []
        for i, a in enumerate(args):
            canon, value = self.canon.canonical(a)
            if params is not None and i < len(params):
                texts.append(canon)
            else:
                texts.append(canon)
            values.append(value)
        return tuple(texts), tuple(values)

    def _is_kernel_file(self, file: str) -> bool:
        try:
            p = Path(file).resolve()
            src = Path(self.source).resolve()
        except Exception:
            return False
        return (
            p == src
            or p.parent == src.parent
            or "/kernels/" in p.as_posix()
            or "_kernels/" in p.as_posix()
        )

    def _handle_call(self, call, state: State) -> State:
        loc = call.location
        loc_file = loc.file.name if loc.file else ""
        pushed = False
        if len(self._call_stack) == 1 and loc_file and self._is_kernel_file(loc_file):
            self._site_stack.append((self._loc(call)[0], loc.line, loc.column, call))
            pushed = True
        try:
            return self._handle_call_inner(call, state)
        finally:
            if pushed:
                self._site_stack.pop()

    def _handle_call_inner(self, call, state: State) -> State:
        ref = call.referenced
        if ref is None:
            return state
        name = ref.spelling
        if not name:
            return state

        # Deprecated functions are reported even when we stop at the rule table.
        # Init-style deprecations keep the historical ``deprecated-init`` rule;
        # every other deprecated call (e.g. acquire_dst/release_dst in
        # api/compute/reg_api.h) is a ``deprecated-api`` finding.  Both come
        # from the C++ declaration, so a new [[deprecated]] annotation is
        # covered without editing the rule table.
        if name in self.table.deprecated_init_apis or self._is_deprecated(ref):
            is_init = "_init" in name or name in self.table.init_by_function
            file, line, col = self._record_site(call)
            self._add(
                Finding(
                    rule="deprecated-init" if is_init else "deprecated-api",
                    severity="error",
                    file=file,
                    line=line,
                    column=col,
                    thread=self.thread,
                    arch=self.arch,
                    op=name,
                    message=(
                        f"deprecated init API '{name}' is used"
                        if is_init
                        else f"deprecated API '{name}' is used"
                    ),
                    citation="[[deprecated]] declaration or .github/deprecations.json",
                )
            )

        handled = False
        should_descend = False
        for op, _tr in self.table.init_by_function.get(name, ()):
            if op.descend:
                should_descend = True
        for op, _tr in self.table.execute_by_function.get(name, ()):
            if op.descend:
                should_descend = True
        for op, _tr in self.table.uninit_by_function.get(name, ()):
            if op.descend:
                should_descend = True
        if name in self.table.init_by_function:
            self._apply_init(name, call, ref, state)
            handled = True
        if name in self.table.execute_by_function:
            self._apply_execute(name, call, ref, state)
            handled = True
        if name in self.table.uninit_by_function:
            self._apply_uninit(name, call, ref, state)
            handled = True
        for ev in self.table.extra_events.get(name, ()):
            if ev.kind == "init":
                self._apply_extra_init(ev, call, ref, state)
            handled = True

        if name in self.table.invalidators:
            for exu in self.table.invalidators[name]:
                self._invalidate(exu, name, self._record_site(call), state)
            handled = True
            should_descend = True

        if handled and not should_descend:
            return state

        # Not a rule function.  Decide whether to follow it.
        if name in TRANSPARENT_EXACT or any(name.startswith(p) for p in TRANSPARENT_PREFIXES):
            return state

        body = self._resolve_body(ref)
        loc_file = ref.location.file.name if ref.location.file else ""
        in_surface = bool(loc_file) and self._surface_file(loc_file)
        is_llk_api = name.startswith(LLK_API_PREFIX) and in_surface

        if body is not None and in_surface:
            params = [c for c in body.get_children() if c.kind == ci.CursorKind.PARM_DECL]
            args = list(call.get_arguments())
            bindings: dict[str, Optional[str]] = {}

            # Non-type template parameters are compile-time values; binding them
            # makes `if constexpr (N == 1)` resolve to one branch instead of both.
            tparam_names = self._template_param_names(body)
            try:
                nta = ref.get_num_template_arguments()
            except Exception:
                nta = -1
            for i, tp_name in enumerate(tparam_names):
                if i >= nta or nta < 0:
                    break
                kind = ref.get_template_argument_kind(i)
                if kind == ci.TemplateArgumentKind.INTEGRAL:
                    bindings[tp_name] = "#" + str(ref.get_template_argument_unsigned_value(i))
                elif kind == ci.TemplateArgumentKind.TYPE:
                    bindings[tp_name] = ref.get_template_argument_type(i).spelling
            for i, p in enumerate(params):
                if i < len(args):
                    canon, _ = self.canon.canonical(args[i])
                    bindings[p.spelling] = canon
                else:
                    bindings[p.spelling] = None
            generation = self._init_generation
            state = self._run_function(body, state, bindings=bindings)
            rules = self.table.sequence_rules.get(name)
            if rules:
                self._check_sequence_rules(rules, call, state, generation)
            return state

        # Implementation-level LLK names (_llk_*) are below the rule table's
        # abstraction: they do not change the API-level seated operation.
        if name.startswith("_llk_") or name.startswith("ckernel::"):
            return state
        if in_surface and body is None and name.startswith(LLK_API_PREFIX):
            state.unknown = True
            return state
        if in_surface and body is None:
            return state
        # Host/stdlib/unknown helper: assumed not to mutate Tensix state.
        return state

    def _is_deprecated(self, ref) -> bool:
        deprecated_kind = getattr(ci.CursorKind, "DEPRECATED_ATTR", None)
        try:
            for child in ref.get_children():
                if deprecated_kind is not None and child.kind == deprecated_kind:
                    return True
                if child.kind == ci.CursorKind.UNEXPOSED_ATTR:
                    tokens = [t.spelling for t in child.get_tokens()]
                    if child.spelling == "deprecated" or (tokens and tokens[0] == "deprecated"):
                        return True
        except Exception:
            pass
        return False

    def _event_match(
        self,
        record: InitRecord,
        transition: Transition,
        exec_template: tuple,
        exec_args: tuple[str, ...],
        exec_values: tuple[Optional[int], ...],
    ) -> tuple[bool, str]:
        """Return (matches, detail). Unknown comparisons are treated as matching."""
        if record.unknown:
            return True, "unknown"
        details = []
        for init_idx, exec_idx in transition.match.template:
            if init_idx >= len(record.template) or exec_idx >= len(exec_template):
                return True, "unknown"
            # The sanitizer stores get_effective_math_fidelity(): ELWADD/ELWSUB
            # ignore the configured fidelity and always run as LoFi.
            if (
                transition.init_fn.startswith("llk_math_eltwise_binary_init")
                and (init_idx, exec_idx) == (2, 3)
                and record.template
                and record.template[0][0] == "int"
                and record.template[0][1] != 0
            ):
                continue
            a = record.template[init_idx]
            b = exec_template[exec_idx]
            if a[0] == "int" and b[0] == "int":
                if a[1] == 255 or b[1] == 255:
                    return True, "unknown"
                if a[1] != b[1]:
                    details.append(f"template[{init_idx}]={a[1]} != template[{exec_idx}]={b[1]}")
            elif a[0] == "type" and b[0] == "type":
                if a[1] != b[1]:
                    details.append(f"template[{init_idx}]={a[1]} != template[{exec_idx}]={b[1]}")
            else:
                return True, "unknown"
        for init_idx, exec_idx in transition.match.args:
            if init_idx >= len(record.args) or exec_idx >= len(exec_args):
                return True, "unknown"
            va = record.arg_values[init_idx] if init_idx < len(record.arg_values) else None
            vb = exec_values[exec_idx] if exec_idx < len(exec_values) else None
            if va is not None and vb is not None:
                if va != vb:
                    details.append(f"arg[{init_idx}]={va} != arg[{exec_idx}]={vb}")
            elif record.args[init_idx] == exec_args[exec_idx]:
                continue
            else:
                return True, "unknown"
        if details:
            return False, "; ".join(details)
        return True, "match"

    def _record_site(self, call) -> tuple[str, int, int]:
        # Attribute a diagnostic to the outermost call the kernel made, not to the
        # inlined LLK/compute-API header that happens to contain the hook.
        if self._site_stack:
            return self._site_stack[-1][:3]
        return self._loc(call)

    def _entries_for(self, name: str, table: dict):
        return [e for e in table.get(name, ()) if e[0].thread == self.thread]

    def _apply_init(self, name: str, call, ref, state: State):
        entries = self._entries_for(name, self.table.init_by_function)
        if not entries:
            return
        op, transition = entries[0]
        template = self._template_args(ref)
        args, values = self._canon_args(call, None)
        file, line, col = self._record_site(call)
        self._init_generation += 1
        record = InitRecord(
            op_id=op.id,
            init_fn=name,
            file=file,
            line=line,
            column=col,
            thread=self.thread,
            template=template,
            args=args,
            arg_values=values,
            generation=self._init_generation,
        )
        if op.additive:
            state.sfpu.setdefault(op.id, []).append(record)
        else:
            state.seated[op.exu] = record
        if op.requires_uninit:
            state.armed[op.id] = record
        self._maybe_loop_invariant_init(call, op, record)

    def _apply_extra_init(self, ev: ExtraEvent, call, ref, state: State):
        op = self.table.operations.get(ev.op_id)
        if op is None or op.thread != self.thread:
            return
        template = self._template_args(ref)
        args, values = self._canon_args(call, None)
        file, line, col = self._record_site(call)
        self._init_generation += 1
        record = InitRecord(
            op_id=op.id,
            init_fn=ev.function,
            file=file,
            line=line,
            column=col,
            thread=self.thread,
            template=template,
            args=args,
            arg_values=values,
            generation=self._init_generation,
        )
        state.seated[op.exu] = record
        if op.requires_uninit:
            state.armed[op.id] = record

    def _apply_execute(self, name: str, call, ref, state: State):
        entries = self._entries_for(name, self.table.execute_by_function)
        if not entries:
            return
        op, transition = entries[0]
        template = self._template_args(ref)
        args, values = self._canon_args(call, None)
        file, line, col = self._record_site(call)

        if op.additive:
            records = state.sfpu.get(op.id, [])
            invalid = next((r for r in records if r.invalidated_by is not None), None)
            if invalid is not None and not state.unknown:
                inv_fn, inv_line = invalid.invalidated_by
                self._add(
                    Finding(
                        rule="wrong-init",
                        severity="error",
                        file=file,
                        line=line,
                        column=col,
                        thread=self.thread,
                        arch=self.arch,
                        op=op.id,
                        message=(
                            f"'{name}' uses an init invalidated by '{inv_fn}' at line {inv_line}; "
                            f"re-run '{transition.init_fn}' after the reconfiguration"
                        ),
                        earlier=f"{invalid.site} ({invalid.init_fn})",
                        citation=op.citation,
                        detail={"invalidated_by": inv_fn, "invalidator_line": inv_line},
                    )
                )
                return
            if not records:
                if not state.unknown:
                    self._add(
                        Finding(
                            rule="missing-init",
                            severity="error",
                            file=file,
                            line=line,
                            column=col,
                            thread=self.thread,
                            arch=self.arch,
                            op=op.id,
                            message=f"'{name}' has no preceding '{transition.init_fn}' on this path",
                            citation=op.citation,
                        )
                    )
                return
            # Any record matching the execute's arguments is sufficient.
            best = records[-1]
            for rec in reversed(records):
                ok, detail = self._event_match(rec, transition, template, args, values)
                if ok:
                    return
                best = rec
            if not state.unknown:
                ok_unknown = any(
                    self._event_match(rec, transition, template, args, values)[1] == "unknown"
                    for rec in records
                )
                if ok_unknown:
                    return
                self._add(
                    Finding(
                        rule="init-arg-mismatch",
                        severity="error",
                        file=file,
                        line=line,
                        column=col,
                        thread=self.thread,
                        arch=self.arch,
                        op=op.id,
                        message=f"'{name}' does not match the arguments of its init '{best.init_fn}'",
                        earlier=f"{best.site} ({best.init_fn})",
                        citation=op.citation,
                        detail={"comparison": detail},
                    )
                )
            return

        record = state.seated.get(op.exu)
        if state.unknown:
            return
        if record is None:
            # Distinguish "never initialized" from "initialized for a different
            # operation" by asking whether any init for this exu has been seen.
            other = self._last_unseated(op.exu, state)
            if other is not None:
                self._add(
                    Finding(
                        rule="wrong-init",
                        severity="error",
                        file=file,
                        line=line,
                        column=col,
                        thread=self.thread,
                        arch=self.arch,
                        op=op.id,
                        message=(
                            f"'{name}' is for {op.id}, but the {op.exu} engine was last "
                            f"initialized for {other.op_id}"
                        ),
                        earlier=f"{other.site} ({other.init_fn})",
                        citation=op.citation,
                    )
                )
            else:
                self._add(
                    Finding(
                        rule="missing-init",
                        severity="error",
                        file=file,
                        line=line,
                        column=col,
                        thread=self.thread,
                        arch=self.arch,
                        op=op.id,
                        message=f"'{name}' has no preceding '{transition.init_fn}' on this path",
                        citation=op.citation,
                    )
                )
            return
        if record.op_id != op.id:
            self._add(
                Finding(
                    rule="wrong-init",
                    severity="error",
                    file=file,
                    line=line,
                    column=col,
                    thread=self.thread,
                    arch=self.arch,
                    op=op.id,
                    message=(
                        f"'{name}' is for {op.id}, but the {op.exu} engine was last "
                        f"initialized for {record.op_id}"
                    ),
                    earlier=f"{record.site} ({record.init_fn})",
                    citation=op.citation,
                )
            )
            return
        if record.invalidated_by is not None:
            inv_fn, inv_line = record.invalidated_by
            self._add(
                Finding(
                    rule="wrong-init",
                    severity="error",
                    file=file,
                    line=line,
                    column=col,
                    thread=self.thread,
                    arch=self.arch,
                    op=op.id,
                    message=(
                        f"'{name}' uses an init invalidated by '{inv_fn}' at line {inv_line}; "
                        f"re-run '{transition.init_fn}' after the reconfiguration"
                    ),
                    earlier=f"{record.site} ({record.init_fn})",
                    citation=op.citation,
                    detail={"invalidated_by": inv_fn, "invalidator_line": inv_line},
                )
            )
            return
        ok, detail = self._event_match(record, transition, template, args, values)
        if not ok:
            self._add(
                Finding(
                    rule="init-arg-mismatch",
                    severity="error",
                    file=file,
                    line=line,
                    column=col,
                    thread=self.thread,
                    arch=self.arch,
                    op=op.id,
                    message=f"'{name}' does not match the arguments of its init '{record.init_fn}'",
                    earlier=f"{record.site} ({record.init_fn})",
                    citation=op.citation,
                    detail={"comparison": detail},
                )
            )

    def _last_unseated(self, exu: str, state: State) -> Optional[InitRecord]:
        # State only keeps the seated record.  When it is None because a
        # different init replaced it, the merge/init path already forgot it; the
        # more useful message is produced by the initialised-other-op case below.
        return None

    def _apply_uninit(self, name: str, call, ref, state: State):
        entries = self._entries_for(name, self.table.uninit_by_function)
        if not entries:
            return
        op, transition = entries[0]
        if op.additive:
            state.sfpu.pop(op.id, None)
        else:
            record = state.seated.get(op.exu)
            if record is not None and record.op_id == op.id:
                state.seated[op.exu] = None
        state.armed.pop(op.id, None)

    def _invalidate(self, exu: str, fn: str, loc: tuple[str, int, int], state: State):
        candidates = []
        record = state.seated.get(exu)
        if record is not None:
            candidates.append(record)
        for records in state.sfpu.values():
            candidates.extend(records)
        for record in candidates:
            op = self.table.operations.get(record.op_id) or self.table.sfpu_groups.get(record.op_id)
            if op is None:
                continue
            if fn in op.invalidated_by:
                record.invalidated_by = (fn, loc[1])

    def _check_sequence_rules(self, rules, call, state: State, generation: int):
        """Postconditions expressed by a trigger init's effect on another engine.

        Example (#52175): ``tilize_init(icb, block, ocb)`` must re-point the
        packer at ``ocb``.  A tilize init that leaves a previous operation's
        packer geometry in place silently packs the wrong tile shape.
        """
        args, values = self._canon_args(call, None)
        for rule in rules:
            if rule.thread != self.thread:
                continue
            required = self.table.operations.get(rule.require_op)
            if required is None or required.thread != self.thread:
                continue
            records = (
                state.sfpu.get(required.id, ())
                if required.additive
                else [state.seated.get(required.exu)]
            )
            record = next(
                (r for r in reversed(records) if r is not None and r.op_id == required.id),
                None,
            )
            bad = None
            if record is None or record.generation <= generation:
                # Presence alone is not enough: the packer may already have been
                # seated before the trigger and simply survive the body.
                bad = f"no fresh '{rule.require_op}' init followed it"
            else:
                for trigger_idx, required_idx in rule.require_args:
                    if trigger_idx >= len(args) or required_idx >= len(record.args):
                        continue
                    if args[trigger_idx] != record.args[required_idx]:
                        bad = (
                            f"the required init uses '{record.args[required_idx]}', "
                            f"not the operation's '{args[trigger_idx]}' operand"
                        )
                        break
            if bad is None:
                continue
            file, line, col = self._record_site(call)
            self._add(
                Finding(
                    rule="wrong-init",
                    severity="error",
                    file=file,
                    line=line,
                    column=col,
                    thread=self.thread,
                    arch=self.arch,
                    op=rule.id,
                    message=rule.message or f"'{rule.trigger}' is not self-contained: {bad}",
                    earlier=record.site if record is not None else None,
                    citation=rule.citation,
                    detail={"sequence": rule.id, "reason": bad},
                )
            )

    # -- loop-invariant init --------------------------------------------

    def _maybe_loop_invariant_init(self, call, op: OperationRule, record: InitRecord):
        if not self._loop_stack:
            return
        info = self._loop_stack[-1]
        if info.conditional_depth > 0:
            return
        outer = self._site_stack[-1][3] if self._site_stack else call
        tokens = {t.spelling for t in outer.get_tokens()}
        if tokens & info.variables:
            return
        key = (record.file, record.line, record.column, op.id)
        if key in info.warned:
            return
        info.warned.add(key)
        self._add(
            Finding(
                rule="loop-invariant-init",
                severity="warning",
                file=record.file,
                line=record.line,
                column=record.column,
                thread=self.thread,
                arch=self.arch,
                op=op.id,
                message=(
                    f"'{record.init_fn}' is loop-invariant but is re-run on every "
                    f"iteration; hoist it if the operation does not change the "
                    f"configuration between iterations"
                ),
                citation=op.citation,
            )
        )


# ---------------------------------------------------------------------------
# Compile-command handling
# ---------------------------------------------------------------------------


def _find_sfpi(explicit: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for env in ("TT_METAL_SFPI_ROOT", "SFPI_ROOT", "TT_METAL_SFPI_PATH"):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]))
    candidates += [
        REPO_ROOT / "runtime" / "sfpi",
        Path("/opt/tenstorrent/sfpi"),
        Path("/opt/sfpi"),
    ]
    for c in candidates:
        if (c / "include" / "sfpi.h").is_file():
            return c
    # Search a couple of common install roots for a pinned release.
    for base in (Path("/home"), Path("/opt"), Path("/usr/local")):
        if not base.is_dir():
            continue
        try:
            for p in base.glob("*/sfpi*/sfpi/include/sfpi.h"):
                return p.parent.parent
        except Exception:
            continue
    return None


def _sfpi_include_flags(sfpi: Path) -> list[str]:
    compiler = sfpi / "compiler"
    versions = []
    gcc_dir = compiler / "lib" / "gcc" / "riscv-tt-elf"
    if gcc_dir.is_dir():
        versions = sorted(
            (d.name for d in gcc_dir.iterdir() if d.is_dir()),
            key=lambda v: tuple(int(x) for x in re.findall(r"\d+", v)) or (0,),
        )
    ver = versions[-1] if versions else ""
    cxx = compiler / "riscv-tt-elf" / "include" / "c++" / ver if ver else None
    flags = ["-nostdinc++", "-nostdinc"]
    for path in (
        gcc_dir / ver / "include" if ver else None,
        compiler / "riscv-tt-elf" / "include",
        cxx,
        cxx / "riscv-tt-elf" if cxx else None,
        cxx / "riscv-tt-elf" / "wh-ilp32" if cxx else None,
        sfpi / "include",
    ):
        if path is not None and path.is_dir():
            flags += ["-isystem", str(path)]
    return flags


def _arch_include_flags(arch: str) -> list[str]:
    info = ARCHES[arch]
    prefix = info["prefix"]
    root = str(REPO_ROOT)
    dirs = [
        root,
        root + "/ttnn",
        root + "/ttnn/cpp",
        root + "/tt_metal/hw/inc",
        root + "/tt_metal/hostdevcommon/api",
        root + "/tt_metal/hw/inc/debug",
        root + "/tt_metal/api/",
        root + "/tt_metal/api/tt-metalium/",
        root + "/tt_metal/hw/firmware/src",
        root + "/tt_metal/hw/firmware/src/tt-1xx",
        root + "/tt_metal/hw/inc/" + prefix,
        root + "/tt_metal/hw/inc/" + prefix + "/noc",
        root + "/tt_metal/hw/inc/" + prefix + "/" + arch + "_defines",
        root + "/tt_metal/hw/ckernels/" + arch + "/metal/common",
        root + "/tt_metal/tt-llk/common",
        root + "/tt_metal/tt-llk/tools/include",
        root + "/tt_metal/tt-llk/tt_llk_" + arch + "/common/inc",
        root + "/tt_metal/tt-llk/tt_llk_" + arch + "/llk_lib",
        root + "/tt_metal/tt-llk/tt_llk_" + arch + "/llk_lib/hal",
        root + "/tt_metal/hw/ckernels/" + arch + "/metal/llk_api",
        root + "/tt_metal/hw/ckernels/" + arch + "/metal/llk_api/llk_sfpu",
        root + "/tt_metal/hw/ckernels/" + arch + "/metal/llk_io",
        root + "/tt_metal/tt-llk/tt_llk_" + arch + "/common/inc/sfpu",
        root + "/tt_metal/hw/inc/internal/tt-1xx",
        root + "/tt_metal/hw/inc/internal/tt-1xx/" + prefix,
        root + "/tt_metal/hw/inc/internal/tt-1xx/" + prefix + "/" + arch + "_defines",
        root + "/tt_metal/hw/inc/internal/tt-1xx/" + prefix + "/noc",
    ]
    flags = []
    for d in dirs:
        if Path(d).is_dir():
            flags += ["-I", d]
    return flags


def synth_args(source: str, arch: str, thread: str, sfpi: Path, extra_defines: Iterable[str] = ()) -> list[str]:
    info = ARCHES[arch]
    th = THREADS[thread]
    args = [
        "-x",
        "c++",
        "--target=riscv32-unknown-elf",
        "-std=c++17",
        "-w",
        "-ferror-limit=0",
        "-Wno-unknown-attributes",
        "-Wno-missing-template-arg-list-after-template-kw",
        "-U__INT32_TYPE__",
        "-D__INT32_TYPE__=long int",
        "-U__UINT32_TYPE__",
        "-D__UINT32_TYPE__=long unsigned int",
        "-U__INT_LEAST32_TYPE__",
        "-D__INT_LEAST32_TYPE__=long int",
        "-U__UINT_LEAST32_TYPE__",
        "-D__UINT_LEAST32_TYPE__=long unsigned int",
    ]
    args += _sfpi_include_flags(sfpi)
    args += _arch_include_flags(arch)
    args += [
        "-D" + info["define"],
        "-DNUM_DRAM_BANKS=" + str(info["dram_banks"]),
        "-DNUM_L1_BANKS=" + str(info["l1_banks"]),
        "-DPCIE_NOC_X=0",
        "-DPCIE_NOC_Y=3",
        "-D" + th["define"] + "=1",
        "-DUCK_CHLKC_" + thread,
        "-DCOMPILE_FOR_TRISC=" + str(th["trisc"]),
        "-DPROCESSOR_INDEX=" + str(th["processor_index"]),
        "-DPROGRAMMABLE_CORE_TYPE=1",
        "-DNAMESPACE=chlkc_" + thread.lower(),
        "-DTENSIX_FIRMWARE",
        "-DLOCAL_MEM_EN=0",
        "-DROUTING_FW_ENABLED",
        "-DKERNEL_BUILD",
        "-DNOC_INDEX=0",
        "-DNOC_MODE=0",
        "-DDATA_FORMATS_DEFINED=1",
        "-DKERNEL_COMPILE_TIME_ARGS=1,1,1,1,1,1,1,1",
        "-include",
        str(PRELUDE_PATH),
    ]
    for d in extra_defines:
        args.append(d if d.startswith("-D") else "-D" + d)
    return args


def detect_from_args(args: list[str], source: str = "") -> tuple[Optional[str], Optional[str]]:
    arch = None
    thread = None
    joined = " ".join(args)
    if "ARCH_WORMHOLE" in joined or "-mcpu=tt-wh" in joined or "/wormhole_b0/" in joined:
        arch = "wormhole_b0"
    if "ARCH_BLACKHOLE" in joined or "-mcpu=tt-bh" in joined or "/blackhole/" in joined:
        arch = "blackhole"
    if "TRISC_UNPACK" in joined or "UCK_CHLKC_UNPACK" in joined or "COMPILE_FOR_TRISC=0" in joined:
        thread = "UNPACK"
    if "TRISC_MATH" in joined or "UCK_CHLKC_MATH" in joined or "COMPILE_FOR_TRISC=1" in joined:
        thread = "MATH"
    if "TRISC_PACK" in joined or "UCK_CHLKC_PACK" in joined or "COMPILE_FOR_TRISC=2" in joined:
        thread = "PACK"
    if arch is None:
        m = re.search(r"/(wormhole_b0|blackhole)/", source)
        if m:
            arch = m.group(1)
    return arch, thread


def translate_captured_argv(argv: list[str]) -> Optional[list[str]]:
    """Translate a captured SFPI g++ command to the clang analysis command."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import build_kernel_clang_tidy_commands as bkc  # type: ignore
    except Exception:
        return None
    translated = bkc.transform(argv, "clang++")
    if translated is None:
        return None
    out = translated[1:]  # drop the clang++ program name
    # libclang does not need these, and a PCH/nodefault built by another
    # compiler can make it fail before it reaches the kernel source.
    cleaned = []
    i = 0
    while i < len(out):
        a = out[i]
        if a in ("-c", "-fsyntax-only"):
            i += 1
            continue
        if a in ("-o", "-MF", "-MT", "-MQ"):
            i += 2
            continue
        if a in ("-MMD", "-MP", "-MD"):
            i += 1
            continue
        if a == "-include" and i + 1 < len(out):
            inc = out[i + 1]
            if "pch" in os.path.basename(inc).lower() or inc.endswith(".pch"):
                # SFPI-GCC PCHs are not readable by clang; let clang parse the
                # headers textually instead.
                i += 2
                continue
        cleaned.append(a)
        i += 1
    return cleaned


def entries_from_log(path: Path) -> list[dict]:
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import build_kernel_clang_tidy_commands as bkc  # type: ignore
    except Exception:
        eprint("error: cannot import scripts/build_kernel_clang_tidy_commands.py")
        return []
    entries = []
    seen = set()
    for entry in bkc.entries_from_log(str(path)):
        src = entry["file"]
        argv = bkc.entry_argv(entry)
        translated = translate_captured_argv(argv)
        if translated is None:
            continue
        joined = " ".join(translated)
        # Keep the compute-thread kernel translation units.  Firmware and
        # dataflow-only TUs cannot contain compute API calls.
        if "FW_BUILD" in joined or "UCK_CHLKC_" not in joined:
            continue
        key = (entry.get("directory", ""), src, tuple(translated))
        if key in seen:
            continue
        seen.add(key)
        entries.append({"directory": entry.get("directory", str(REPO_ROOT)), "file": src, "arguments": translated})
    return entries


def entries_from_compile_db(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    entries = []
    for entry in data:
        argv = list(entry.get("arguments") or [])
        if not argv and entry.get("command"):
            import shlex

            argv = shlex.split(entry["command"])
        if not argv:
            continue
        # A captured SFPI command needs translation; a clang command is ready.
        if any("riscv-tt-elf-g++" in os.path.basename(a) for a in argv[:2]):
            argv = translate_captured_argv(argv)
            if argv is None:
                continue
        entries.append(
            {
                "directory": entry.get("directory", str(REPO_ROOT)),
                "file": entry.get("file", ""),
                "arguments": argv,
            }
        )
    return entries


def _entry_source(entry: dict) -> str:
    src = entry.get("file", "")
    if not src:
        args = entry.get("arguments", [])
        for a in reversed(args):
            if a.endswith((".cpp", ".cc", ".cxx")):
                src = a
                break
    if not os.path.isabs(src):
        src = os.path.join(entry.get("directory", str(REPO_ROOT)), src)
    return os.path.abspath(src)


def _absolutize_args(args: list[str], cwd: str) -> list[str]:
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-I" and i + 1 < len(args):
            inc = args[i + 1]
            out += ["-I", inc if os.path.isabs(inc) else os.path.abspath(os.path.join(cwd, inc))]
            i += 2
            continue
        if a.startswith("-I") and len(a) > 2:
            inc = a[2:]
            out.append("-I" + (inc if os.path.isabs(inc) else os.path.abspath(os.path.join(cwd, inc))))
            i += 1
            continue
        if a == "-include" and i + 1 < len(args):
            inc = args[i + 1]
            out += ["-include", inc if os.path.isabs(inc) else os.path.abspath(os.path.join(cwd, inc))]
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def analyze_entry(entry: dict, table: RuleTable, cfg) -> tuple[Optional[Analyzer], Optional[str]]:
    src = _entry_source(entry)
    if not os.path.isfile(src):
        return None, f"source not found: {src}"
    args = list(entry.get("arguments") or [])
    if args and not args[0].startswith("-"):
        args = args[1:]
    args = _absolutize_args(args, entry.get("directory", str(REPO_ROOT)))
    # libclang takes the main file as an argument; a second copy in `arguments`
    # (captured commands end with the source) is parsed as another input.
    src_abs = os.path.abspath(src)
    args = [a for a in args if os.path.abspath(a) != src_abs]
    # Prefer the analysis SFPI headers over the ones baked into the captured
    # command: SFPI 7.94 has an analysis-only header regression (vNarrow is
    # instantiated before vUInt is complete), while the device build itself is
    # unaffected.  The first -isystem wins.
    analysis_sfpi = _find_sfpi(cfg.sfpi)
    if analysis_sfpi is not None:
        args = _sfpi_include_flags(analysis_sfpi) + args
    # Clang diagnostics are not the product here; keep parsing past SFPI's
    # analysis-only header quirks and suppress GCC-only warning spellings.
    args = ["-w", "-ferror-limit=0", "-include", str(HERE / "compute_api_order_sfpi_compat.h")] + args
    arch, thread = detect_from_args(args, src)
    arch = cfg.arch or arch
    thread = cfg.thread or thread
    if arch is None or thread is None:
        return None, "cannot determine arch/thread from compile flags"
    if arch not in ARCHES:
        return None, f"unsupported arch: {arch}"
    # Make the flags self-consistent with the selected thread/arch.
    clean = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-DARCH_WORMHOLE", "-DARCH_BLACKHOLE") or a.startswith("-DTRISC_") or a.startswith("-DUCK_CHLKC_"):
            i += 1
            continue
        if a.startswith("-DCOMPILE_FOR_TRISC="):
            i += 1
            continue
        clean.append(a)
        i += 1
    args = clean + [
        "-D" + ARCHES[arch]["define"],
        "-D" + THREADS[thread]["define"] + "=1",
        "-DUCK_CHLKC_" + thread,
        "-DCOMPILE_FOR_TRISC=" + str(THREADS[thread]["trisc"]),
        "-DPROCESSOR_INDEX=" + str(THREADS[thread]["processor_index"]),
        "-DPROGRAMMABLE_CORE_TYPE=1",
        "-DNAMESPACE=chlkc_" + thread.lower(),
    ]
    if not any(a == "-include" for a in args):
        args += ["-include", str(PRELUDE_PATH)]
    try:
        analyzer = Analyzer(args, thread, arch, table, src)
        analyzer.analyze_paths()
    except Exception as exc:  # noqa: BLE001 - report a hole, keep the corpus going
        return None, f"{type(exc).__name__}: {exc}"
    return analyzer, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_libclang(explicit: Optional[str]):
    # libclang cannot be re-pointed after it is loaded (e.g. calling main() twice
    # in the same process); the first configuration wins.
    if ci.Config.loaded:
        return
    if explicit:
        ci.Config.set_library_file(explicit)
        return
    if os.environ.get("LIBCLANG_PATH"):
        ci.Config.set_library_path(os.environ["LIBCLANG_PATH"])
        return
    candidates = [
        "/usr/lib/llvm-21/lib/libclang.so",
        "/usr/lib/llvm-20/lib/libclang.so",
        "/usr/lib/llvm-19/lib/libclang.so",
        "/usr/lib/llvm-18/lib/libclang.so",
        "/usr/lib/libclang.so",
    ]
    for c in candidates:
        if Path(c).is_file():
            ci.Config.set_library_file(c)
            return


def _format_finding(f: Finding) -> list[str]:
    lines = [f"{f.file}:{f.line}:{f.column}: {f.severity}: {f.message} [{f.rule}]"]
    if f.earlier:
        lines.append(f"  earlier call: {f.earlier}")
    if f.citation:
        lines.append(f"  source: {f.citation}")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    global RULES_PATH
    parser = argparse.ArgumentParser(
        description="Static checker for compute-kernel init/uninit/reconfig ordering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Exit status is 0 for findings (advisory) unless --fail-on-findings is used.
            A run that cannot analyse an input exits 2.
            """
        ),
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--kernel", help="kernel source file (synthesizes JIT-like flags)")
    src.add_argument("--compile-db", help="compile_commands.json")
    src.add_argument("--log", help="tt-metal JIT log captured with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1")
    parser.add_argument("--arch", choices=sorted(ARCHES), help="override architecture")
    parser.add_argument("--thread", choices=sorted(THREADS), help="override Tensix thread")
    parser.add_argument("--define", action="append", default=[], help="extra -D for --kernel")
    parser.add_argument("--include", action="append", default=[], help="extra -I for --kernel")
    parser.add_argument("--sfpi", help="SFPI install root (contains include/sfpi.h)")
    parser.add_argument("--libclang", help="path to libclang.so")
    parser.add_argument("--rules", default=str(RULES_PATH), help=argparse.SUPPRESS)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", help="write the report to this file")
    parser.add_argument("--fail-on-findings", action="store_true", help="exit 1 when anything is found")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument("--list-rules", action="store_true", help="print the rule table and exit")
    parser.add_argument("--reproduce", action="store_true", help="print the compile command used")
    args = parser.parse_args(argv)

    _configure_libclang(args.libclang)

    RULES_PATH = Path(args.rules)
    table = RuleTable.load(args.arch)
    if args.list_rules:
        for op in list(table.operations.values()) + list(table.sfpu_groups.values()):
            print(f"{op.id:32s} {op.thread:6s} {op.exu:7s} {op.citation}")
        return 0

    if not (args.kernel or args.compile_db or args.log):
        parser.error("one of --kernel, --compile-db or --log is required")

    if args.kernel:
        sfpi = _find_sfpi(args.sfpi)
        if sfpi is None:
            eprint(
                "error: SFPI headers not found. Pass --sfpi <root> or set TT_METAL_SFPI_ROOT "
                "(the tt-metal build installs them under runtime/sfpi)."
            )
            return 2
        if args.thread is None:
            eprint("error: --thread is required with --kernel (UNPACK, MATH or PACK)")
            return 2
        arch = args.arch or "wormhole_b0"
        extra = list(args.define)
        for inc in args.include:
            extra.append("-I" + inc)
        cmd_args = synth_args(args.kernel, arch, args.thread, sfpi, extra)
        if args.reproduce:
            print("clang++ " + " ".join(cmd_args + [args.kernel]))
        entry = {"directory": str(REPO_ROOT), "file": args.kernel, "arguments": cmd_args}
        entries = [entry]
    elif args.compile_db:
        entries = entries_from_compile_db(Path(args.compile_db))
    else:
        entries = entries_from_log(Path(args.log))

    if not entries:
        eprint("error: no kernel translation units found in the input")
        return 2

    all_findings: list[Finding] = []
    reports = []
    holes = []
    for entry in entries:
        analyzer, error = analyze_entry(entry, table, args)
        if analyzer is None:
            holes.append({"file": _entry_source(entry), "error": error})
            continue
        reports.append(
            {
                "file": _entry_source(entry),
                "thread": analyzer.thread,
                "arch": analyzer.arch,
                "entry": analyzer._entry_name,
                "parse_errors": len(analyzer.parse_errors),
                "findings": len(analyzer.findings),
            }
        )
        all_findings.extend(analyzer.findings)

    all_findings.sort(key=lambda f: (f.file, f.line, f.column, f.rule))
    if args.format == "json":
        payload = {
            "version": 1,
            "findings": [f.as_json() for f in all_findings],
            "units": reports,
            "holes": holes,
            "summary": _summary(all_findings),
        }
        text = json.dumps(payload, indent=2)
    else:
        lines = []
        for f in all_findings:
            if args.quiet:
                break
            lines.extend(_format_finding(f))
        if lines:
            lines.append("")
        counts = _summary(all_findings)
        lines.append(
            f"{counts['total']} finding(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts["by_rule"].items()))
            if counts["by_rule"]
            else "0 findings"
        )
        if holes:
            lines.append(f"{len(holes)} translation unit(s) could not be analysed:")
            for h in holes[:10]:
                lines.append(f"  {h['file']}: {h['error']}")
            if len(holes) > 10:
                lines.append(f"  ... and {len(holes) - 10} more")
        text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)

    if args.fail_on_findings and all_findings:
        return 1
    return 0


def _summary(findings: list[Finding]) -> dict:
    by_rule = collections.Counter(f.rule for f in findings)
    by_sev = collections.Counter(f.severity for f in findings)
    return {"total": len(findings), "by_rule": dict(by_rule), "by_severity": dict(by_sev)}


if __name__ == "__main__":
    raise SystemExit(main())
