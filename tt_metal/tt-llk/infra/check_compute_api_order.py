#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Static checker for compute-kernel API ordering rules.

Compute kernels drive the unpacker, math engine and packer through a stateful API: most operations
only work after a matching init, a few init/operation pairs must be closed by an uninit, and a
data-format reconfig silently drops a configuration. Those rules exist only as prose and doc
comments, so breaking them compiles fine and shows up later as a device hang or a wrong result.

This checker reads `compute_api_rules.yaml` and interprets the *per-thread translation unit* the JIT
compiler builds for a kernel. UNPACK, MATH and PACK keep separate hardware state, so each thread is
checked on its own. Calls are followed into helper headers and template instantiations, so wrappers
such as `compute_kernel_lib::tilize<...>` are expanded to the LLK-level calls the rule table
describes.

Getting a compile database
--------------------------
The weekly kernel clang-tidy job captures the SFPI compiler invocations:

    TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 pytest <test>
    scripts/build_kernel_clang_tidy_commands.py --log build.log --output compile_commands.json

The `kernel-clang-tidy-*` CI artifacts contain that database. Point the checker at it, at the
checked-out tree, and (if the generated kernel sources came from the artifact) at their extraction
directory:

    python3 tt_metal/tt-llk/infra/check_compute_api_order.py \\
        --compile-db /tmp/tidy/compile_commands.json --repo-root . \\
        --gensrc-root /tmp/tidy/generated-sources

Only entries compiled for a TRISC thread (`-DUCK_CHLKC_{UNPACK,MATH,PACK}`) are analysed. Dataflow
kernels, folded-execution kernels and Quasar are out of scope, as the issue says.

Design notes
------------
* Parsing uses libclang (`pip install libclang`, or the distro's `python3-clang`). The SFPI
  toolchain's newlib/libstdc++ headers and the SFPI -> clang flag translation come from
  `scripts/build_kernel_clang_tidy_commands.py`, so this tool stays in step with the kernel
  clang-tidy flow instead of duplicating it.
* `sfpi_analysis_shim.h` is force-included only when SFPI's own analysis-tool fallback is missing
  `__xtt_vector8`. That is a gap in the pinned SFPI release, not in the kernels.
* Branches are explored on both sides: a missed init on either path is reported. After a branch the
  state keeps only configurations that hold on every path, so a later operation is still checked
  against what is guaranteed.
* Arguments are compared as symbols, never evaluated at runtime. Two identical expressions are
  equal, two different compile-time constants are unequal, and a value that cannot be compared is
  reported as unknown instead of guessed, so the checker never claims a mismatch it cannot prove.
* The check is non-blocking by default: findings print and the process exits 0. `--strict` makes
  rule findings fatal for a job that is ready to gate.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib.util
import json
import os
import re
import shlex
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES = os.path.join(HERE, "compute_api_rules.yaml")
SHIM = os.path.join(HERE, "sfpi_analysis_shim.h")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TIDY_SCRIPT = os.path.join(REPO, "scripts", "build_kernel_clang_tidy_commands.py")

THREAD_DEFINES = {
    "UCK_CHLKC_UNPACK": "UNPACK",
    "UCK_CHLKC_MATH": "MATH",
    "UCK_CHLKC_PACK": "PACK",
}
ARCH_DEFINES = {
    "ARCH_WORMHOLE": "wormhole_b0",
    "ARCH_BLACKHOLE": "blackhole",
}

# Functions that never carry compute-API state. Inlining them only costs time; naming them also
# keeps the call chain readable.
NO_INLINE = {
    "get_arg",
    "get_arg_addr",
    "get_common_arg_addr",
    "get_arg_val",
    "get_common_arg_val",
    "printf",
    "ASSERT",
    "assert",
    "min",
    "max",
}

# Calls through these are still given a symbolic identity, but their bodies are not followed.
PURE_ACCESSORS = {"get_id", "get_face_r_dim", "get_face_c_dim", "get_num_faces", "value"}


# --------------------------------------------------------------------------------------------------
# Values, sites, findings
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Val:
    """A symbolic argument.

    `key` is a stable identity rendered from the AST (a DFB binding, a kernel-argument accessor, a
    local constant, ...). `const` is set when the expression folds. `unknown` marks a value the
    checker refuses to compare. `stable` means the key names something that is fixed for the whole
    kernel (a `constexpr` object, a compile-time accessor, a template argument), so two different
    keys are provably different; a plain runtime variable is not stable, and two different runtime
    variables may well hold the same value.
    """

    key: Optional[str] = None
    const: Optional[object] = None
    unknown: bool = False
    stable: bool = False

    def render(self) -> str:
        if self.const is not None:
            return repr(self.const)
        return self.key or "<unknown>"


UNKNOWN = Val(None, None, True)


def compare(a: Val, b: Val) -> Optional[bool]:
    """True/False when the relation is provable, None when it is not.

    Identical keys are the same operand. Two different keys are only known to be different when
    both are compile-time identities; two runtime variables may hold the same value, so a
    difference there is not a finding.
    """
    if a.key is not None and a.key == b.key:
        # The same named operand is the same operand even when its value is not known: `cb` at the
        # init and `cb` at the operation cannot disagree.
        return True
    if a.unknown or b.unknown:
        return None
    if a.const is not None and b.const is not None:
        return a.const == b.const
    if a.const is not None or b.const is not None:
        return None
    if a.key is None or b.key is None:
        return None
    if a.stable and b.stable:
        return False
    return None


@dataclasses.dataclass(frozen=True)
class Site:
    """A source location plus the inlined call chain that reached it."""

    file: str
    line: int
    col: int = 0
    chain: Tuple[str, ...] = ()

    def render(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"

    def as_dict(self) -> dict:
        out = {"file": self.file, "line": self.line, "column": self.col}
        if self.chain:
            out["inlined_from"] = list(self.chain)
        return out


@dataclasses.dataclass
class Finding:
    rule: str
    kind: str
    site: Site
    message: str
    related: Optional[Site] = None
    note: str = ""
    source: str = ""

    def sort_key(self):
        return (self.site.file, self.site.line, self.kind, self.rule)

    def as_dict(self) -> dict:
        out = {
            "rule": self.rule,
            "kind": self.kind,
            "message": self.message,
            "location": self.site.as_dict(),
        }
        if self.related:
            out["related"] = self.related.as_dict()
        if self.note:
            out["note"] = self.note
        if self.source:
            out["rule_source"] = self.source
        return out

    def render(self) -> str:
        text = f"{self.site.render()}: {self.kind} [{self.rule}]: {self.message}"
        if self.related:
            text += f"\n    conflicts with {self.related.render()}"
        if self.site.chain:
            text += f"\n    reached via {' -> '.join(self.site.chain)}"
        if self.note:
            text += f"\n    {self.note}"
        if self.source:
            text += f"\n    rule source: {self.source}"
        return text


# --------------------------------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Rule:
    id: str
    op: str
    init: str
    thread: str
    arg_match: List[Tuple[int, int]]
    uninit: Optional[str]
    requires_uninit: bool
    invalidated_by: List[str]
    source: str
    note: str
    arches: Optional[List[str]]
    configured_at_entry: bool

    def active_on(self, arch: Optional[str]) -> bool:
        return arch is None or self.arches is None or arch in self.arches


@dataclasses.dataclass
class RuleTable:
    operations: List[Rule]
    deprecated: Dict[str, str]
    architectures: List[str]

    @classmethod
    def load(cls, path: str) -> "RuleTable":
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ops = []
        for raw in data.get("operations", []):
            arg_match = [tuple(pair) for pair in raw.get("arg_match") or []]
            for a, b in arg_match:
                if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b < 0:
                    raise ValueError(f"{raw['id']}: bad arg_match pair {a},{b}")
            ops.append(
                Rule(
                    id=raw["id"],
                    op=raw["op"],
                    init=raw["init"],
                    thread=raw["thread"].upper(),
                    arg_match=arg_match,
                    uninit=raw.get("uninit"),
                    requires_uninit=bool(raw.get("requires_uninit")),
                    invalidated_by=list(raw.get("invalidated_by") or []),
                    source=raw.get("source", ""),
                    note=" ".join((raw.get("note") or "").split()),
                    arches=list(raw["arch"]) if raw.get("arch") else None,
                    configured_at_entry=bool(raw.get("configured_at_entry")),
                )
            )
        deprecated = {}
        for entry in data.get("deprecated_inits", []):
            for name in entry["inits"]:
                deprecated[name] = entry["id"]
        return cls(ops, deprecated, list(data.get("architectures", [])))


# --------------------------------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------------------------------


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    pass


@dataclasses.dataclass
class Config:
    """One active init: the rule it configures, its arguments and the call site."""

    rule: Rule
    args: List[Val]
    site: Site
    op_seen: bool = False
    clobbered_by: Optional[Site] = None
    clobbered_name: Optional[str] = None


class State:
    def __init__(
        self,
        configs: Optional[Dict[str, Config]] = None,
        dropped: Optional[Dict[str, Tuple[str, Site]]] = None,
    ) -> None:
        self.configs: Dict[str, Config] = dict(configs or {})
        # rule id -> (call that invalidated it, its site). Kept so a later operation can say why the
        # configuration it expected is gone instead of only that it is missing.
        self.dropped: Dict[str, Tuple[str, Site]] = dict(dropped or {})

    def clone(self) -> "State":
        return State(self.configs, self.dropped)

    def intersect(self, other: "State") -> "State":
        """Keep only what is configured identically on both paths."""
        out: Dict[str, Config] = {}
        for key, mine in self.configs.items():
            theirs = other.configs.get(key)
            if theirs is None:
                continue
            if [v.key or v.const for v in mine.args] != [v.key or v.const for v in theirs.args]:
                continue
            out[key] = mine
        dropped = {
            key: value
            for key, value in self.dropped.items()
            if other.dropped.get(key) == value
        }
        return State(out, dropped)


class Analyzer:
    """Path-aware interpreter over one thread's translation unit."""

    MAX_INLINE_DEPTH = 8

    def __init__(
        self,
        ci,
        tu,
        rules: RuleTable,
        thread: str,
        arch: Optional[str],
        src: str,
    ) -> None:
        self.ci = ci
        self.tu = tu
        self.rules = rules
        self.thread = thread
        self.arch = arch
        self.src = os.path.abspath(src)

        self.active_rules = [r for r in rules.operations if r.thread == thread and r.active_on(arch)]
        self.by_op: Dict[str, List[Rule]] = collections.defaultdict(list)
        self.by_init: Dict[str, List[Rule]] = collections.defaultdict(list)
        self.by_uninit: Dict[str, List[Rule]] = collections.defaultdict(list)
        for rule in self.active_rules:
            self.by_op[rule.op].append(rule)
            self.by_init[rule.init].append(rule)
            if rule.uninit:
                self.by_uninit[rule.uninit].append(rule)
        self.invalidators = {name for r in self.active_rules for name in r.invalidated_by}
        self.relevant_names = (
            set(self.by_op) | set(self.by_init) | set(self.by_uninit) | self.invalidators
            | set(rules.deprecated)
        )

        self.state = State()
        self.findings: List[Finding] = []
        self._seen: Set[tuple] = set()
        self.bindings: List[Dict[str, Val]] = [{}]
        self.call_stack: List[str] = []
        self.loop_frames: List[Set[str]] = []
        self._inline_depth = 0
        self._inline_stack: Set[Tuple[str, int]] = set()
        self._reach_cache: Dict[Tuple[str, int], bool] = {}
        self._var_cache: Dict[Tuple[str, int], Optional[object]] = {}
        self._template_cache: Dict[str, object] = {}
        self.unverified_args = 0

    # -- cursor helpers ----------------------------------------------------------------------

    def render(self, cursor) -> str:
        """Source text of a cursor, with bound parameters substituted by their identity."""
        out = []
        try:
            tokens = list(cursor.get_tokens())
        except Exception:
            return ""
        for token in tokens:
            spelling = token.spelling
            if re.fullmatch(r"[A-Za-z_]\w*", spelling):
                bound = self.lookup(spelling)
                if bound is not None and bound.key:
                    spelling = bound.key
            out.append(spelling)
        return "".join(out)

    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r"\.get_id\(\)", "", text)
        text = re.sub(r"\bthis->", "", text)
        return text

    def lookup(self, name: str) -> Optional[Val]:
        for scope in reversed(self.bindings):
            if name in scope:
                return scope[name]
        return None

    def _stable_identity(self, cursor) -> bool:
        """True when a name refers to something fixed for the whole kernel (a `constexpr` object).

        DFB tokens (`in`, `out`) and other `constexpr` handles are distinct objects with distinct
        identities, so two different ones are provably different operands. A plain `uint32_t`
        local is not stable: two different locals may hold the same CB id at runtime.
        """
        decl = cursor.referenced
        if decl is None:
            return False
        if decl.kind not in (self.ci.CursorKind.VAR_DECL, self.ci.CursorKind.PARM_DECL):
            return False
        try:
            return "constexpr" in self.render(decl)
        except Exception:
            return False

    def value(self, cursor) -> Val:
        ci = self.ci
        kind = cursor.kind
        text = self.normalize(self.render(cursor))

        folded = self.const_eval(cursor)
        if folded is not None and kind != ci.CursorKind.CALL_EXPR:
            return Val(text, folded)

        if kind == ci.CursorKind.DECL_REF_EXPR:
            bound = self.lookup(cursor.spelling)
            if bound is not None:
                return bound
            return Val(text, stable=self._stable_identity(cursor))

        if kind in (ci.CursorKind.INTEGER_LITERAL, ci.CursorKind.CHARACTER_LITERAL,
                    ci.CursorKind.CXX_BOOL_LITERAL_EXPR):
            return Val(text, folded) if folded is not None else UNKNOWN

        if kind == ci.CursorKind.CALL_EXPR:
            callee = cursor.referenced
            if callee is not None and callee.spelling in PURE_ACCESSORS:
                return Val(text, stable=True)
            return Val(text) if text else UNKNOWN

        if kind in (
            ci.CursorKind.MEMBER_REF_EXPR,
            ci.CursorKind.UNARY_OPERATOR,
            ci.CursorKind.ARRAY_SUBSCRIPT_EXPR,
        ):
            return Val(text) if text else UNKNOWN

        if kind in (
            ci.CursorKind.PAREN_EXPR,
            ci.CursorKind.UNEXPOSED_EXPR,
        ) or kind.name.endswith("CAST_EXPR"):
            children = [c for c in cursor.get_children()]
            if children:
                return self.value(children[-1])
            return UNKNOWN

        if kind == ci.CursorKind.CONDITIONAL_OPERATOR:
            return UNKNOWN

        return Val(text) if text else UNKNOWN

    # -- constant folding --------------------------------------------------------------------

    def const_eval(self, cursor) -> Optional[object]:
        ci = self.ci
        kind = cursor.kind

        if kind == ci.CursorKind.INTEGER_LITERAL:
            return _parse_int(self.render(cursor))
        if kind == ci.CursorKind.CHARACTER_LITERAL:
            text = self.render(cursor).strip("'")
            return ord(text) if len(text) == 1 else None
        if kind == ci.CursorKind.CXX_BOOL_LITERAL_EXPR:
            return self.render(cursor) == "true"

        if kind == ci.CursorKind.DECL_REF_EXPR:
            bound = self.lookup(cursor.spelling)
            if bound is not None:
                return bound.const
            decl = cursor.referenced
            if decl is None:
                return None
            if decl.kind == ci.CursorKind.ENUM_CONSTANT_DECL:
                try:
                    return decl.enum_value
                except Exception:
                    return None
            if decl.kind == ci.CursorKind.VAR_DECL:
                return self.const_of_var(decl)
            return None

        if kind == ci.CursorKind.UNARY_OPERATOR:
            children = list(cursor.get_children())
            if not children:
                return None
            operand = self.const_eval(children[0])
            if operand is None:
                return None
            op = self.render(cursor)[: -len(self.render(children[0]))]
            if op == "!":
                return not operand
            if op == "-":
                return -operand
            if op == "~":
                return ~operand
            if op == "+":
                return +operand
            return None

        if kind == ci.CursorKind.BINARY_OPERATOR:
            children = list(cursor.get_children())
            if len(children) != 2:
                return None
            left = self.const_eval(children[0])
            right = self.const_eval(children[1])
            if left is None or right is None:
                return None
            op = self._operator_between(children[0], children[1])
            return _apply_binary(op, left, right)

        if kind == ci.CursorKind.CONDITIONAL_OPERATOR:
            children = list(cursor.get_children())
            if len(children) != 3:
                return None
            cond = self.const_eval(children[0])
            if cond is None:
                return None
            return self.const_eval(children[1] if cond else children[2])

        if kind == ci.CursorKind.PAREN_EXPR or kind == ci.CursorKind.UNEXPOSED_EXPR:
            children = list(cursor.get_children())
            return self.const_eval(children[-1]) if children else None

        if kind.name.endswith("CAST_EXPR"):
            children = list(cursor.get_children())
            return self.const_eval(children[-1]) if children else None

        return None

    def _operator_between(self, left, right) -> str:
        try:
            left_end = left.extent.end.offset
            right_start = right.extent.start.offset
            path = left.extent.end.file.name
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            return text[left_end:right_start].strip()
        except Exception:
            return ""

    def const_of_var(self, decl) -> Optional[object]:
        key = (decl.location.file.name if decl.location.file else "", decl.location.offset)
        if key in self._var_cache:
            return self._var_cache[key]
        self._var_cache[key] = None
        declaration = self.render(decl)
        if "constexpr" not in declaration and "const " not in declaration:
            return None
        children = [c for c in decl.get_children() if c.kind not in (
            self.ci.CursorKind.TYPE_REF,
            self.ci.CursorKind.TEMPLATE_REF,
            self.ci.CursorKind.NAMESPACE_REF,
            self.ci.CursorKind.PARM_DECL,
        )]
        if not children:
            return None
        value = self.const_eval(children[-1])
        self._var_cache[key] = value
        return value

    # -- reporting ---------------------------------------------------------------------------

    def site(self, cursor) -> Site:
        location = cursor.location
        path = location.file.name if location.file else self.src
        if path and not os.path.isabs(path):
            path = os.path.abspath(path)
        return Site(path or self.src, location.line or 0, location.column or 0, tuple(self.call_stack))

    def report(self, rule: Rule, kind: str, cursor, message: str,
               related: Optional[Site] = None) -> None:
        site = self.site(cursor)
        key = (kind, rule.id, site.file, site.line, related.render() if related else "")
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(
            Finding(rule.id, kind, site, message, related, rule.note, rule.source)
        )

    # -- entry -------------------------------------------------------------------------------

    def entry_points(self):
        main = []
        fallback = []
        for cursor in self.tu.cursor.get_children():
            if cursor.kind not in (
                self.ci.CursorKind.FUNCTION_DECL,
                self.ci.CursorKind.FUNCTION_TEMPLATE,
            ):
                continue
            if not cursor.is_definition() or cursor.location.file is None:
                continue
            path = os.path.abspath(cursor.location.file.name)
            if cursor.spelling == "kernel_main":
                main.append(cursor)
            elif path == self.src and self._reaches_rules(cursor):
                fallback.append(cursor)
        return main or fallback

    def run(self) -> List[Finding]:
        entries = self.entry_points()
        for entry in entries:
            self.state = State()
            for rule in self.active_rules:
                if rule.configured_at_entry:
                    self.state.configs[rule.id] = Config(
                        rule, [UNKNOWN], Site(self.src, 0, 0)
                    )
            self.bindings = [{}]
            self.call_stack = []
            self.loop_frames = []
            self._inline_depth = 0
            self._inline_stack = set()
            try:
                self.exec_cursor(entry)
            except (_Return, _Break, _Continue):
                pass
            self._check_unclosed(entry)
        return self.findings

    def _check_unclosed(self, entry) -> None:
        end_line = entry.extent.end.line if entry.extent.end.line else 0
        for config in list(self.state.configs.values()):
            if config.rule.requires_uninit and config.op_seen:
                finding_site = Site(self.src, end_line, 0)
                key = ("missing-uninit", config.rule.id, finding_site.file, end_line, "")
                if key in self._seen:
                    continue
                self._seen.add(key)
                self.findings.append(
                    Finding(
                        config.rule.id,
                        "missing-uninit",
                        finding_site,
                        f"{config.rule.init}() opens a region for {config.rule.op} and the kernel "
                        f"ends without {config.rule.uninit}()",
                        config.site,
                        config.rule.note,
                        config.rule.source,
                    )
                )

    # -- statement dispatch ------------------------------------------------------------------

    def exec_cursor(self, cursor) -> None:
        ci = self.ci
        kind = cursor.kind

        if kind in (ci.CursorKind.COMPOUND_STMT,):
            for child in cursor.get_children():
                self.exec_cursor(child)
        elif kind == ci.CursorKind.DECL_STMT:
            for child in cursor.get_children():
                if child.kind == ci.CursorKind.VAR_DECL:
                    self.declare(child)
                else:
                    self.exec_expr(child)
        elif kind == ci.CursorKind.IF_STMT:
            self.exec_if(cursor)
        elif kind in (ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT,
                      ci.CursorKind.DO_STMT, ci.CursorKind.CXX_FOR_RANGE_STMT):
            self.exec_loop(cursor)
        elif kind == ci.CursorKind.SWITCH_STMT:
            self.exec_switch(cursor)
        elif kind in (ci.CursorKind.CASE_STMT, ci.CursorKind.DEFAULT_STMT,
                      ci.CursorKind.LABEL_STMT):
            for child in cursor.get_children():
                self.exec_cursor(child)
        elif kind == ci.CursorKind.BREAK_STMT:
            raise _Break()
        elif kind == ci.CursorKind.CONTINUE_STMT:
            raise _Continue()
        elif kind == ci.CursorKind.RETURN_STMT:
            for child in cursor.get_children():
                self.exec_expr(child)
            raise _Return()
        elif kind in (ci.CursorKind.NULL_STMT, ci.CursorKind.UNEXPOSED_STMT):
            for child in cursor.get_children():
                self.exec_cursor(child)
        elif kind in (
            ci.CursorKind.FUNCTION_DECL,
            ci.CursorKind.FUNCTION_TEMPLATE,
            ci.CursorKind.CXX_METHOD,
            ci.CursorKind.CONSTRUCTOR,
            ci.CursorKind.DESTRUCTOR,
            ci.CursorKind.LAMBDA_EXPR,
        ):
            # Entry points and inlined helpers are both function cursors. Statement dispatch must
            # start at their body, otherwise `if`/`for` would be walked as plain expressions and
            # every branch would look like straight-line code.
            body = self.body_of(cursor)
            if body is not None:
                self.exec_cursor(body)
        else:
            self.exec_expr(cursor)

    def body_of(self, cursor):
        return next(
            (child for child in cursor.get_children()
             if child.kind == self.ci.CursorKind.COMPOUND_STMT),
            None,
        )

    def declare(self, var) -> None:
        name = var.spelling
        init = None
        for child in var.get_children():
            if child.kind in (self.ci.CursorKind.TYPE_REF, self.ci.CursorKind.TEMPLATE_REF,
                              self.ci.CursorKind.NAMESPACE_REF):
                continue
            init = child
        if init is not None:
            self.exec_expr(init)
            value = self.value(init)
        else:
            value = UNKNOWN
        if value.unknown and name:
            # The value is not knowable, but the variable itself is still a stable identity: `cb` at
            # the init and `cb` at the operation cannot disagree, so keep the name for comparison.
            value = Val(name, None, unknown=True)
        if self.bindings:
            self.bindings[-1][name] = value

    def exec_expr(self, cursor) -> None:
        ci = self.ci
        if cursor.kind == ci.CursorKind.CALL_EXPR:
            for argument in cursor.get_arguments():
                self.exec_expr(argument)
            self.handle_call(cursor)
            return
        if cursor.kind == ci.CursorKind.LAMBDA_EXPR:
            return
        # A translation unit that does not parse cleanly (an optional header, a compiler builtin
        # clang does not model) leaves calls as an unresolved overload set: an UNEXPOSED_EXPR whose
        # first child is a DECL_REF_EXPR wrapping an OVERLOADED_DECL_REF. The rule table matches on
        # names, so the call is still checkable; without this the whole rest of the body would be
        # skipped and the kernel would look clean.
        unresolved = self.unresolved_call(cursor)
        if unresolved is not None:
            name, arguments = unresolved
            for argument in arguments:
                self.exec_expr(argument)
            self.handle_call(cursor, name=name, args=arguments)
            return
        for child in cursor.get_children():
            self.exec_expr(child)

    def explicit_arguments(self, call) -> list:
        """The arguments written at the call site.

        libclang reports a `CXXDefaultArgExpr` for every omitted default argument as a cursor with
        an empty extent. Those carry no value and are not part of what the kernel author wrote, so
        they must not be fed to the rule table (a defaulted argument would otherwise look like an
        unknown value and hide a real finding).
        """
        return [
            arg
            for arg in call.get_arguments()
            if arg.extent.start.offset != arg.extent.end.offset or len(list(arg.get_tokens()))
        ]

    def unresolved_call(self, cursor):
        """(name, argument cursors) for a call clang left unresolved, else None."""
        if cursor.kind.name != "UNEXPOSED_EXPR":
            return None
        children = list(cursor.get_children())
        if not children:
            return None
        callee = children[0]
        if callee.kind != self.ci.CursorKind.DECL_REF_EXPR:
            return None
        refs = [c for c in callee.get_children() if c.kind.name == "OVERLOADED_DECL_REF"]
        if not refs:
            return None
        name = refs[0].spelling or callee.spelling
        if not name:
            return None
        return name, children[1:]

    # -- control flow ------------------------------------------------------------------------

    def exec_if(self, cursor) -> None:
        children = list(cursor.get_children())
        if not children:
            return
        condition = children[0]
        then = children[1] if len(children) > 1 else None
        other = children[2] if len(children) > 2 else None

        folded = self.const_eval(condition)
        if isinstance(folded, (bool, int)):
            branch = then if folded else other
            if branch is not None:
                self.exec_cursor(branch)
            return

        entry = self.state.clone()
        terminated = 0
        self.state = entry.clone()
        if then is not None:
            try:
                self.exec_cursor(then)
            except _Return:
                terminated += 1
        left = self.state

        self.state = entry.clone()
        if other is not None:
            try:
                self.exec_cursor(other)
            except _Return:
                terminated += 1
        right = self.state

        if terminated == 2:
            raise _Return()
        self.state = left.intersect(right)

    def exec_switch(self, cursor) -> None:
        entry = self.state.clone()
        body = None
        for child in cursor.get_children():
            if child.kind == self.ci.CursorKind.COMPOUND_STMT:
                body = child
        if body is None:
            return
        exits = []
        cases = list(body.get_children())
        for index in range(len(cases)):
            self.state = entry.clone()
            try:
                for case in cases[index:]:
                    self.exec_cursor(case)
            except (_Break, _Return):
                pass
            exits.append(self.state)
        if not exits:
            self.state = entry
            return
        merged = exits[0]
        for other in exits[1:]:
            merged = merged.intersect(other)
        self.state = merged

    def exec_loop(self, cursor) -> None:
        children = list(cursor.get_children())
        body = children[-1] if children else None
        frame: Set[str] = set()

        if cursor.kind == self.ci.CursorKind.FOR_STMT and children:
            init = children[0]
            if init.kind == self.ci.CursorKind.DECL_STMT:
                for decl in init.get_children():
                    if decl.kind == self.ci.CursorKind.VAR_DECL:
                        self.declare(decl)
                        frame.add(decl.spelling)
                        # The body sees every value the induction variable takes, not its
                        # initialiser, so keep the name but forget the constant.
                        if self.bindings:
                            self.bindings[-1][decl.spelling] = Val(decl.spelling, None, True)
        elif cursor.kind == self.ci.CursorKind.CXX_FOR_RANGE_STMT and children:
            decl = children[0]
            if decl.kind == self.ci.CursorKind.VAR_DECL:
                self.declare(decl)
                frame.add(decl.spelling)
                if self.bindings:
                    self.bindings[-1][decl.spelling] = Val(decl.spelling, None, True)

        if body is not None:
            frame |= _declared_names(body, self.ci)

        entry = self.state.clone()
        self.loop_frames.append(frame)
        try:
            if body is not None:
                self.exec_cursor(body)
        except (_Break, _Continue):
            pass
        finally:
            self.loop_frames.pop()
        self.state = entry.intersect(self.state)

    # -- calls -------------------------------------------------------------------------------

    def handle_call(self, call, name=None, args=None) -> None:
        ci = self.ci
        callee = call.referenced
        if name is None:
            if callee is None:
                return
            name = callee.spelling
        if not name:
            return

        args = list(args) if args is not None else self.explicit_arguments(call)
        values = [self.value(a) for a in args]
        site = self.site(call)

        rules = self.by_op.get(name)
        if rules:
            for rule in rules:
                self.on_op(rule, call, site, values)
            return

        inits = self.by_init.get(name)
        if inits:
            for rule in inits:
                self.on_init(rule, call, site, values)
            self.on_deprecated(name, call)
            return

        if name in self.by_uninit:
            self.on_uninit(name, call)
            return

        if name in self.invalidators:
            self.on_invalidate(name, call, site)
            return

        if name in self.rules.deprecated:
            self.on_deprecated(name, call)
            return

        if callee is not None:
            self.maybe_inline(call, callee, values)

    def on_init(self, rule: Rule, call, site: Site, values: List[Val]) -> None:
        previous = self.state.configs.get(rule.id)
        if previous is not None and previous.op_seen:
            self.report(
                rule,
                "missing-uninit",
                call,
                f"{rule.init}() is called again while the region opened for {rule.op} is still open; "
                f"{rule.uninit}() is missing",
                previous.site,
            )
        for other in list(self.state.configs.values()):
            if other.rule.requires_uninit and other.op_seen and other.rule.id != rule.id:
                self.report(
                    other.rule,
                    "missing-uninit",
                    call,
                    f"{other.rule.uninit}() is missing before {rule.init}()",
                    other.site,
                )

        self.check_loop_invariant(rule, call, values)

        for key, config in self.state.configs.items():
            if key != rule.id:
                config.clobbered_by = site
                config.clobbered_name = rule.init

        self.state.configs[rule.id] = Config(rule, values, site)
        self.state.dropped.pop(rule.id, None)

    def check_loop_invariant(self, rule: Rule, call, values: List[Val]) -> None:
        if not self.loop_frames:
            return
        declared = self.loop_frames[-1]
        for value in values:
            if value.unknown:
                return
            if value.const is None and value.key:
                if any(re.search(rf"\b{re.escape(name)}\b", value.key) for name in declared):
                    return
        self.report(
            rule,
            "loop-invariant-init",
            call,
            f"{rule.init}() is loop-invariant: its arguments do not depend on the loop, so it can "
            f"be hoisted (warning only)",
        )

    def on_op(self, rule: Rule, call, site: Site, values: List[Val]) -> None:
        config = self.state.configs.get(rule.id)
        if config is None:
            message = f"{rule.op}() has no preceding {rule.init}() on this path"
            dropped = self.state.dropped.get(rule.id)
            if dropped is not None:
                name, where = dropped
                message += (
                    f": {name}() dropped the configuration "
                    f"({os.path.basename(where.file)}:{where.line})"
                )
            self.report(rule, "missing-init", call, message)
            return

        mismatches = []
        unknown = 0
        for init_index, op_index in rule.arg_match:
            left = config.args[init_index] if init_index < len(config.args) else UNKNOWN
            right = values[op_index] if op_index < len(values) else UNKNOWN
            relation = compare(left, right)
            if relation is False:
                mismatches.append((init_index, op_index, left, right))
            elif relation is None:
                unknown += 1
        if unknown:
            self.unverified_args += 1
        if mismatches:
            detail = ", ".join(
                f"arg {op_index} is {right.render()} but {rule.init}() arg {init_index} was "
                f"{left.render()}"
                for init_index, op_index, left, right in mismatches
            )
            self.report(
                rule,
                "arg-mismatch",
                call,
                f"{rule.op}() does not match the preceding {rule.init}(): {detail}",
                config.site,
            )

        if config.clobbered_by is not None:
            self.report(
                rule,
                "init-conflict",
                call,
                f"{config.clobbered_name}() runs between {rule.init}() and this {rule.op}(), so the "
                f"configuration may have been replaced",
                config.clobbered_by,
            )

        config.op_seen = True

    def on_uninit(self, name: str, call) -> None:
        for rule in self.by_uninit[name]:
            self.state.configs.pop(rule.id, None)

    def on_invalidate(self, name: str, call, site: Site) -> None:
        for key, config in list(self.state.configs.items()):
            if name not in config.rule.invalidated_by:
                continue
            if config.rule.requires_uninit and config.op_seen:
                self.report(
                    config.rule,
                    "missing-uninit",
                    call,
                    f"{name}() runs inside the region opened by {config.rule.init}(); the region must "
                    f"be closed by {config.rule.uninit}() first",
                    config.site,
                )
            self.state.dropped[key] = (name, site)
            del self.state.configs[key]

    def on_deprecated(self, name: str, call) -> None:
        entry = self.rules.deprecated.get(name)
        if entry is None:
            return
        rule = Rule(
            id=entry,
            op=name,
            init=name,
            thread=self.thread,
            arg_match=[],
            uninit=None,
            requires_uninit=False,
            invalidated_by=[],
            source=".github/deprecations.json",
            note="Scheduled for removal; see the deprecation entry for the replacement.",
            arches=None,
            configured_at_entry=False,
        )
        self.report(rule, "deprecated-init", call, f"{name}() is a deprecated init API")

    # -- inlining ----------------------------------------------------------------------------

    def maybe_inline(self, call, callee, values: List[Val]) -> None:
        ci = self.ci
        if callee is None:
            return
        if self._inline_depth >= self.MAX_INLINE_DEPTH:
            return
        if callee.spelling in NO_INLINE:
            return
        if callee.kind not in (
            ci.CursorKind.FUNCTION_DECL,
            ci.CursorKind.FUNCTION_TEMPLATE,
            ci.CursorKind.CXX_METHOD,
        ):
            return
        if not callee.is_definition():
            return
        location = callee.location
        if location.file is None:
            return
        path = location.file.name.replace(os.sep, "/")
        if "/usr/" in path or "/sfpi/" in path or "/include/c++/" in path:
            return
        if not self._reaches_rules(callee):
            return
        key = (path, location.offset)
        if key in self._inline_stack:
            return

        scope: Dict[str, Val] = {}
        params = list(callee.get_arguments())
        for param, value in zip(params, values):
            if param.spelling:
                scope[param.spelling] = value

        for name, value in self.template_bindings(call, callee):
            scope.setdefault(name, value)

        body = next(
            (child for child in callee.get_children() if child.kind == ci.CursorKind.COMPOUND_STMT),
            None,
        )
        if body is None:
            return

        self.bindings.append(scope)
        self.call_stack.append(f"{os.path.basename(path)}:{location.line}")
        self._inline_depth += 1
        self._inline_stack.add(key)
        try:
            self.exec_cursor(body)
        except (_Break, _Continue, _Return):
            pass
        finally:
            self._inline_stack.discard(key)
            self._inline_depth -= 1
            self.call_stack.pop()
            self.bindings.pop()

    def template_bindings(self, call, callee) -> List[Tuple[str, Val]]:
        """Bind a template helper's parameters from the call's explicit template arguments.

        Kernel helpers pass DFBs and block sizes as template parameters, not function parameters
        (`tilize<per_core_block_tile_cnt, dfb::in, dfb::out, ...>()`). Without this, the arguments
        inside the helper would lose their identity and every comparison would be unknown.
        """
        args = _split_template_arguments(call, self.ci)
        if not args:
            return []
        params = self.template_parameters(callee)
        if not params:
            return []
        out = []
        for name, tokens in zip(params, args):
            rendered = self._render_tokens(tokens)
            const = _parse_int(rendered)
            if const is None and rendered in ("true", "false"):
                const = rendered == "true"
            out.append((name, Val(rendered or None, const, stable=True)))
        return out

    def _render_tokens(self, tokens) -> str:
        out = []
        for token in tokens:
            spelling = token.spelling
            if re.fullmatch(r"[A-Za-z_]\w*", spelling):
                bound = self.lookup(spelling)
                if bound is not None and bound.key:
                    spelling = bound.key
            out.append(spelling)
        return self.normalize("".join(out))

    def template_parameters(self, callee) -> List[str]:
        name = callee.spelling
        if name in self._template_cache:
            cached = self._template_cache[name]
            return cached or []
        params: List[str] = []
        candidates = []
        parent = callee.semantic_parent
        if parent is not None and parent.kind == self.ci.CursorKind.FUNCTION_TEMPLATE:
            candidates.append(parent)
        for cursor in self.tu.cursor.walk_preorder():
            if cursor.kind == self.ci.CursorKind.FUNCTION_TEMPLATE and cursor.spelling == name:
                candidates.append(cursor)
                break
        for template in candidates:
            names = [
                child.spelling
                for child in template.get_children()
                if child.kind in (
                    self.ci.CursorKind.TEMPLATE_TYPE_PARAMETER,
                    self.ci.CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
                    self.ci.CursorKind.TEMPLATE_TEMPLATE_PARAMETER,
                )
            ]
            if names:
                params = names
                break
        self._template_cache[name] = params
        return params

    def _reaches_rules(self, callee) -> bool:
        """Does this function transitively call any function the rule table cares about?"""
        location = callee.location
        if location.file is None:
            return False
        key = (location.file.name, location.offset)
        if key in self._reach_cache:
            return self._reach_cache[key]
        self._reach_cache[key] = False  # break recursion

        body = next(
            (child for child in callee.get_children()
             if child.kind == self.ci.CursorKind.COMPOUND_STMT),
            None,
        )
        if body is None:
            return False

        found = False
        stack = [body]
        seen = set()
        while stack and not found:
            node = stack.pop()
            if node.kind == self.ci.CursorKind.CALL_EXPR:
                target = node.referenced
                if target is not None:
                    if target.spelling in self.relevant_names:
                        found = True
                        break
                    if target.kind in (self.ci.CursorKind.FUNCTION_DECL,
                                       self.ci.CursorKind.FUNCTION_TEMPLATE,
                                       self.ci.CursorKind.CXX_METHOD):
                        target_location = target.location
                        if target_location.file is not None:
                            target_key = (target_location.file.name, target_location.offset)
                            if target_key not in seen:
                                seen.add(target_key)
                                if not any(
                                    part in target_location.file.name
                                    for part in ("/usr/", "/sfpi/", "/include/c++/")
                                ):
                                    target_body = next(
                                        (child for child in target.get_children()
                                         if child.kind == self.ci.CursorKind.COMPOUND_STMT),
                                        None,
                                    )
                                    if target_body is not None:
                                        stack.append(target_body)
            if node.kind == self.ci.CursorKind.LAMBDA_EXPR:
                continue
            stack.extend(node.get_children())
        self._reach_cache[key] = found
        return found


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def _parse_int(text: str) -> Optional[int]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    match = re.fullmatch(r"(0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*", text)
    if not match:
        return None
    body = match.group(1)
    try:
        return int(body, 0) if body.lower().startswith(("0x", "0b")) else int(body, 10)
    except ValueError:
        return None


def _apply_binary(op: str, left, right) -> Optional[object]:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return None if right == 0 else left // right
        if op == "%":
            return None if right == 0 else left % right
        if op == "<<":
            return left << right
        if op == ">>":
            return left >> right
        if op == "&":
            return left & right
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "&&":
            return bool(left) and bool(right)
        if op == "||":
            return bool(left) or bool(right)
    except Exception:
        return None
    return None


def _declared_names(cursor, ci) -> Set[str]:
    names: Set[str] = set()
    stack = [cursor]
    while stack:
        node = stack.pop()
        if node.kind == ci.CursorKind.LAMBDA_EXPR:
            continue
        if node.kind == ci.CursorKind.VAR_DECL and node.spelling:
            names.add(node.spelling)
        stack.extend(node.get_children())
    return names


def _split_template_arguments(call, ci) -> List[List]:
    """Split `f<a, b<c, d>>()` into the top-level template argument token lists."""
    try:
        tokens = list(call.get_tokens())
    except Exception:
        return []
    # Find the '(' of the call at depth 0, then the matching '<' just before it.
    depth = 0
    paren = None
    for index, token in enumerate(tokens):
        if token.spelling == "(":
            if depth == 0:
                paren = index
                break
            depth += 1
        elif token.spelling == ")":
            depth -= 1
    if paren is None or paren == 0 or tokens[paren - 1].spelling != ">":
        return []

    depth = 0
    start = None
    for index in range(paren - 1, -1, -1):
        spelling = tokens[index].spelling
        if spelling == ">":
            depth += 1
        elif spelling == "<":
            depth -= 1
            if depth == 0:
                start = index
                break
    if start is None:
        return []

    args: List[List] = []
    current: List = []
    depth = 0
    for token in tokens[start + 1: paren - 1]:
        if token.spelling in ("<", "(", "[", "{"):
            depth += 1
        elif token.spelling in (">", ")", "]", "}"):
            depth -= 1
        if token.spelling == "," and depth == 0:
            args.append(current)
            current = []
            continue
        current.append(token)
    if current:
        args.append(current)
    return args


# --------------------------------------------------------------------------------------------------
# Compile database handling
# --------------------------------------------------------------------------------------------------


def _load_tidy_transform():
    if not os.path.isfile(TIDY_SCRIPT):
        return None
    spec = importlib.util.spec_from_file_location("build_kernel_clang_tidy_commands", TIDY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "transform", None)


def _entry_argv(entry) -> List[str]:
    if entry.get("arguments"):
        return list(entry["arguments"])
    if entry.get("command"):
        return shlex.split(entry["command"])
    return []


def _thread_of(argv: Sequence[str]) -> Optional[str]:
    for arg in argv:
        if arg.startswith("-D"):
            name = arg[2:].split("=", 1)[0]
            if name in THREAD_DEFINES:
                return THREAD_DEFINES[name]
    return None


def _arch_of(argv: Sequence[str]) -> Optional[str]:
    for arg in argv:
        if arg.startswith("-D"):
            name = arg[2:].split("=", 1)[0]
            if name in ARCH_DEFINES:
                return ARCH_DEFINES[name]
    return None


def _rewrite(path: str, mappings: Sequence[Tuple[str, str]]) -> str:
    for old, new in mappings:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _prepare_argv(entry, args, mappings, sfpi_root, extra_args) -> Optional[List[str]]:
    argv = _entry_argv(entry)
    if not argv:
        return None

    source = entry.get("file", "")
    program = os.path.basename(argv[0])
    if re.search(r"riscv(32)?-tt-elf-g\+\+$", program):
        transform = _load_tidy_transform()
        if transform is None:
            return None
        if sfpi_root:
            argv = [os.path.join(sfpi_root, "compiler", "bin", program)] + argv[1:]
        translated = transform(argv, "clang++")
        if translated is None:
            return None
        args = list(translated[1:])
    elif program.startswith("clang"):
        args = list(argv[1:])
    else:
        args = list(argv[1:])

    out: List[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-o", "-MF", "-MT", "-MQ"):
            skip_next = True
            continue
        if arg in ("-c", "-fsyntax-only", "-MD", "-MMD", "-MP", "-Winvalid-pch"):
            continue
        if arg.startswith(("-Wall", "-Werror", "-Wno-error", "-Winvalid-pch", "-Wmissing-")):
            continue
        if arg.endswith((".cpp", ".cc", ".c", ".o", ".d")):
            continue
        if arg == "-I" or arg == "-isystem":
            out.append(arg)
            continue
        if arg.startswith("-I") and not os.path.isabs(arg[2:]):
            out.append("-I" + os.path.abspath(os.path.join(entry.get("directory", "."), arg[2:])))
            continue
        out.append(_rewrite(arg, mappings))

    if source:
        source = _rewrite(source, mappings)
    out += list(extra_args)
    out += ["-Wno-unknown-attributes", "-Wno-unknown-warning-option", "-w"]
    if os.path.isfile(SHIM):
        out += ["-include", SHIM]
    out += ["-fsyntax-only"]
    return out


def _resolve_source(entry, mappings, gensrc_root) -> Tuple[str, str]:
    """Return (existing TU path, working directory)."""
    source = _rewrite(entry.get("file", ""), mappings)
    directory = _rewrite(entry.get("directory", "") or "", mappings)

    def existing(path: str) -> bool:
        return bool(path) and os.path.isfile(path)

    if existing(source):
        return source, directory if os.path.isdir(directory) else os.path.dirname(source)

    parent = os.path.dirname(source)
    basename = os.path.basename(source)
    for candidate in (
        os.path.join(parent, basename),
        os.path.join(os.path.dirname(parent), basename),
    ):
        if existing(candidate):
            return candidate, os.path.dirname(candidate)

    if gensrc_root and os.path.isdir(gensrc_root):
        for root, _, files in os.walk(gensrc_root):
            if basename in files:
                candidate = os.path.join(root, basename)
                if _same_kernel(candidate, source):
                    return candidate, root
    return source, directory or "."


def _same_kernel(candidate: str, wanted: str) -> bool:
    """Match `<name>/<hash>/chlkc_x.cpp` between the artifact and an extracted tree."""
    parts = wanted.replace(os.sep, "/").split("/")
    tail = "/".join(parts[-3:])
    return candidate.replace(os.sep, "/").endswith(tail) or (
        "/".join(candidate.replace(os.sep, "/").split("/")[-2:])
        == "/".join(parts[-2:])
    )


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------


def analyze_entry(ci, index, entry, rules, mappings, sfpi_root, extra_args, inline_all=False):
    argv = _prepare_argv(entry, [], mappings, sfpi_root, extra_args)
    if argv is None:
        return None, "could not translate the compiler command"
    thread = _thread_of(argv)
    if thread is None:
        return None, "not a TRISC thread"
    arch = _arch_of(argv)
    source, directory = _resolve_source(entry, mappings, None)
    if not os.path.isfile(source):
        return None, f"translation unit not found: {source}"

    previous = os.getcwd()
    try:
        os.chdir(directory)
        tu = index.parse(
            source,
            args=argv,
            options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
    finally:
        os.chdir(previous)

    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    analyzer = Analyzer(ci, tu, rules, thread, arch, source)
    findings = analyzer.run()
    if not analyzer.entry_points():
        return None, "no kernel entry point found"
    return (findings, thread, source, len(errors), analyzer.unverified_args), None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check compute-kernel init/op/uninit ordering against compute_api_rules.yaml."
    )
    parser.add_argument("--compile-db", required=True, help="compile_commands.json for kernel TUs")
    parser.add_argument("--rules", default=DEFAULT_RULES, help="rule table (default: alongside this script)")
    parser.add_argument("--repo-root", default=REPO, help="tt-metal checkout the database refers to")
    parser.add_argument("--gensrc-root", default=None, help="extracted generated-sources tree, if any")
    parser.add_argument("--sfpi-root", default=None, help="SFPI install root (contains compiler/)")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="rewrite a path prefix from the captured build environment")
    parser.add_argument("--thread", choices=("UNPACK", "MATH", "PACK"), help="only this thread")
    parser.add_argument("--kernel", default=None, help="only kernels whose path matches this regex")
    parser.add_argument("--clang-arg", action="append", default=[], help="extra argument for clang")
    parser.add_argument("--json", action="store_true", help="one JSON object per finding")
    parser.add_argument("--strict", action="store_true", help="exit 1 on findings")
    parser.add_argument("--max-findings", type=int, default=0, help="stop after N findings (0 = all)")
    parser.add_argument("--list-skipped", action="store_true", help="print why TUs were skipped")
    args = parser.parse_args(argv)

    try:
        import clang.cindex as ci
    except ImportError:
        print(
            "check_compute_api_order: the libclang Python bindings are required "
            "(`pip install libclang`, or install python3-clang).",
            file=sys.stderr,
        )
        return 2

    rules = RuleTable.load(args.rules)
    mappings = []
    for mapping in args.path_map:
        if "=" not in mapping:
            parser.error(f"--path-map expects OLD=NEW, got {mapping!r}")
        old, new = mapping.split("=", 1)
        mappings.append((old, new))
    if args.sfpi_root:
        mappings.append(("/opt/tenstorrent/sfpi", args.sfpi_root))

    try:
        with open(args.compile_db, encoding="utf-8") as f:
            database = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        print(f"check_compute_api_order: cannot read {args.compile_db}: {error}", file=sys.stderr)
        return 2

    index = ci.Index.create()
    findings: List[Finding] = []
    analyzed = 0
    skipped: List[Tuple[str, str]] = []
    unverified = 0
    kernel_re = re.compile(args.kernel) if args.kernel else None

    for entry in database:
        source = _rewrite(entry.get("file", ""), mappings)
        if kernel_re and not kernel_re.search(source):
            continue
        argv_probe = _entry_argv(entry)
        thread = _thread_of(argv_probe)
        if thread is None:
            continue
        if args.thread and thread != args.thread:
            continue

        result, reason = analyze_entry(
            ci, index, entry, rules, mappings, args.sfpi_root, args.clang_arg
        )
        if result is None:
            skipped.append((source, reason))
            continue
        entry_findings, thread, source, error_count, entry_unverified = result
        analyzed += 1
        unverified += entry_unverified
        findings.extend(entry_findings)
        if error_count and args.list_skipped:
            skipped.append((source, f"{error_count} parse error(s); analysis may be partial"))
        if args.max_findings and len(findings) >= args.max_findings:
            findings = findings[: args.max_findings]
            break

    findings.sort(key=lambda f: f.sort_key())
    for finding in findings:
        if args.json:
            print(json.dumps(finding.as_dict()))
        else:
            print(finding.render())

    counts = collections.Counter(f.kind for f in findings)
    if not args.json:
        print(
            f"compute-api-order: {len(findings)} finding(s) in {analyzed} translation unit(s): "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
            if findings
            else f"compute-api-order: no findings in {analyzed} translation unit(s)"
        )
        if unverified:
            print(
                f"compute-api-order: {unverified} operation(s) had arguments that are not "
                "comparable at compile time; they were reported as unknown, not guessed"
            )
        if skipped and args.list_skipped:
            for source, reason in skipped:
                print(f"compute-api-order: skipped {source}: {reason}")
    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
