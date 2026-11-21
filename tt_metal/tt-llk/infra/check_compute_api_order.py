#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Static typestate and dataflow checker for compute-kernel API ordering rules.

Reads the per-thread (UNPACK, MATH, PACK) and per-execution-unit (Unpack, Fpu, Sfpu, Pack)
rule table in `tt_metal/tt-llk/infra/compute_api_rules.yaml` and checks compute kernels across
Wormhole (`wormhole_b0`) and Blackhole (`blackhole`) for six categories of API ordering defects:

  1. `missing-init` (error):
     An operation is executed with no preceding matching initialization call on some control-flow
     path reaching it on that Tensix thread.
  2. `arg-mismatch` (error):
     An init call's compile-time arguments or template parameters differ from those of the
     operation it precedes (e.g., #59276 `AccumulateToDest` mismatch, `BroadcastType` mismatch,
     `EltwiseBinaryType` mismatch, `PoolType`/`ReduceDim` mismatch, or #39945 `reconfig_data_format`
     operand-order reversal before `matmul_tiles`/`matmul_block`). Arguments not known at compile
     time are recorded as `unknown`, never guessed.
  3. `init-conflict` (error):
     An init (or state-clobbering reconfiguration) for a different operation on the same execution
     unit intervenes between an init and its operation without re-initialization (e.g., #61146
     `pack_reconfig_data_format` after `tilize_init` on Blackhole, #39184 unpacker mode overwrite
     before `llk_unpack_tilizeA_B`, or #50381 SFPU constant overwrite between `exp_tile_init`
     and `exp_tile`).
  4. `missing-uninit` (error):
     An operation from the curated set requiring uninitialization (`tilize`, `untilize`,
     `pack_untilize`, `fast_tilize`, `reduce`, `reduce_block_max_row`) transitions to another
     operation or exits the kernel without its required `*_uninit` call (#52175).
  5. `deprecated-init` (error):
     Use of a deprecated compute or LLK init/execution API listed in `compute_api_rules.yaml`
     and cross-referenced with `.github/deprecations.json`.
  6. `loop-invariant-init` (warning):
     A hoistable init with loop-invariant arguments is called inside a loop body that contains no
     competing init/uninit on the same execution unit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_RULES_PATH = os.path.join(HERE, "compute_api_rules.yaml")

sys.path.insert(0, HERE)
from check_mutex_balance import _blank_noncode  # noqa: E402

THREADS: Tuple[str, ...] = ("UNPACK", "MATH", "PACK")
ARCHES: Tuple[str, ...] = ("wormhole_b0", "blackhole")

THREAD_MACROS: Dict[str, Set[str]] = {
    "UNPACK": {"TRISC_UNPACK", "UCK_CHLKC_UNPACK", "LLK_TRISC_UNPACK"},
    "MATH": {"TRISC_MATH", "UCK_CHLKC_MATH", "LLK_TRISC_MATH"},
    "PACK": {"TRISC_PACK", "UCK_CHLKC_PACK", "LLK_TRISC_PACK"},
}
ALL_THREAD_MACROS: Set[str] = set().union(*THREAD_MACROS.values())

ARCH_MACROS: Dict[str, Set[str]] = {
    "wormhole_b0": {"ARCH_WORMHOLE", "ARCH_WORMHOLE_B0"},
    "blackhole": {"ARCH_BLACKHOLE"},
}
ALL_ARCH_MACROS: Set[str] = {
    "ARCH_WORMHOLE",
    "ARCH_WORMHOLE_B0",
    "ARCH_BLACKHOLE",
    "ARCH_QUASAR",
}

# Known compile-time constants and enum values in compute/LLK headers.
KNOWN_ENUM_TOKENS: Dict[str, str] = {
    "NONE": "BroadcastType::NONE",
    "BroadcastType::NONE": "BroadcastType::NONE",
    "ckernel::BroadcastType::NONE": "BroadcastType::NONE",
    "ROW": "BroadcastType::ROW",
    "BroadcastType::ROW": "BroadcastType::ROW",
    "ckernel::BroadcastType::ROW": "BroadcastType::ROW",
    "COL": "BroadcastType::COL",
    "BroadcastType::COL": "BroadcastType::COL",
    "ckernel::BroadcastType::COL": "BroadcastType::COL",
    "SCALAR": "BroadcastType::SCALAR",
    "BroadcastType::SCALAR": "BroadcastType::SCALAR",
    "ckernel::BroadcastType::SCALAR": "BroadcastType::SCALAR",
    "ELWADD": "ELWADD",
    "EltwiseBinaryType::ELWADD": "ELWADD",
    "ckernel::EltwiseBinaryType::ELWADD": "ELWADD",
    "ELWSUB": "ELWSUB",
    "EltwiseBinaryType::ELWSUB": "ELWSUB",
    "ckernel::EltwiseBinaryType::ELWSUB": "ELWSUB",
    "ELWMUL": "ELWMUL",
    "EltwiseBinaryType::ELWMUL": "ELWMUL",
    "ckernel::EltwiseBinaryType::ELWMUL": "ELWMUL",
    "EltwiseBinaryReuseDestType::NONE": "EltwiseBinaryReuseDestType::NONE",
    "ckernel::EltwiseBinaryReuseDestType::NONE": "EltwiseBinaryReuseDestType::NONE",
    "DEST_TO_SRCA": "EltwiseBinaryReuseDestType::DEST_TO_SRCA",
    "EltwiseBinaryReuseDestType::DEST_TO_SRCA": "EltwiseBinaryReuseDestType::DEST_TO_SRCA",
    "ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA": "EltwiseBinaryReuseDestType::DEST_TO_SRCA",
    "DEST_TO_SRCB": "EltwiseBinaryReuseDestType::DEST_TO_SRCB",
    "EltwiseBinaryReuseDestType::DEST_TO_SRCB": "EltwiseBinaryReuseDestType::DEST_TO_SRCB",
    "ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB": "EltwiseBinaryReuseDestType::DEST_TO_SRCB",
    "SUM": "PoolType::SUM",
    "PoolType::SUM": "PoolType::SUM",
    "ckernel::PoolType::SUM": "PoolType::SUM",
    "AVG": "PoolType::AVG",
    "PoolType::AVG": "PoolType::AVG",
    "ckernel::PoolType::AVG": "PoolType::AVG",
    "MAX": "PoolType::MAX",
    "PoolType::MAX": "PoolType::MAX",
    "ckernel::PoolType::MAX": "PoolType::MAX",
    "REDUCE_ROW": "ReduceDim::REDUCE_ROW",
    "ReduceDim::REDUCE_ROW": "ReduceDim::REDUCE_ROW",
    "ckernel::ReduceDim::REDUCE_ROW": "ReduceDim::REDUCE_ROW",
    "REDUCE_COL": "ReduceDim::REDUCE_COL",
    "ReduceDim::REDUCE_COL": "ReduceDim::REDUCE_COL",
    "ckernel::ReduceDim::REDUCE_COL": "ReduceDim::REDUCE_COL",
    "REDUCE_SCALAR": "ReduceDim::REDUCE_SCALAR",
    "ReduceDim::REDUCE_SCALAR": "ReduceDim::REDUCE_SCALAR",
    "ckernel::ReduceDim::REDUCE_SCALAR": "ReduceDim::REDUCE_SCALAR",
    "A2D": "A2D",
    "DataCopyType::A2D": "A2D",
    "ckernel::DataCopyType::A2D": "A2D",
    "B2D": "B2D",
    "DataCopyType::B2D": "B2D",
    "ckernel::DataCopyType::B2D": "B2D",
    "LoFi": "MathFidelity::LoFi",
    "MathFidelity::LoFi": "MathFidelity::LoFi",
    "HiFi2": "MathFidelity::HiFi2",
    "MathFidelity::HiFi2": "MathFidelity::HiFi2",
    "HiFi3": "MathFidelity::HiFi3",
    "MathFidelity::HiFi3": "MathFidelity::HiFi3",
    "HiFi4": "MathFidelity::HiFi4",
    "MathFidelity::HiFi4": "MathFidelity::HiFi4",
    "DST_ACCUM_MODE": "DST_ACCUM_MODE",
    "MATH_FIDELITY": "MATH_FIDELITY",
    "REDUCE_OP": "REDUCE_OP",
    "REDUCE_DIM": "REDUCE_DIM",
    "BCAST_LLKOP": "BCAST_LLKOP",
    "BCAST_DIM": "BCAST_DIM",
    "true": "true",
    "false": "false",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Value:
    """Represents either a resolved compile-time constant or an unknown runtime value."""

    raw: str
    resolved: Optional[str]  # None means not known at compile time (`unknown`)

    @property
    def is_known(self) -> bool:
        return self.resolved is not None

    def format_for_diag(self) -> str:
        if self.resolved is not None:
            return self.resolved
        return f"unknown({self.raw})"


@dataclasses.dataclass
class Diagnostic:
    rule: str
    severity: str  # "error" or "warning"
    file: str
    line: int
    thread: str
    exu: str
    arch: str
    call: str
    earlier_call: Optional[str]
    earlier_line: Optional[int]
    message: str
    source_ref: str
    unknown_args: List[str] = dataclasses.field(default_factory=list)

    def key(self) -> Tuple[Any, ...]:
        return (
            self.rule,
            self.file,
            self.line,
            self.thread,
            self.call,
            self.earlier_call,
            self.earlier_line,
            self.message,
        )

    def format_text(self) -> str:
        loc = f"{self.file}:{self.line}"
        earlier = ""
        if self.earlier_call and self.earlier_line:
            earlier = (
                f" (conflicts with `{self.earlier_call}` at line {self.earlier_line})"
            )
        elif self.earlier_call:
            earlier = f" (conflicts with `{self.earlier_call}`)"
        unk = ""
        if self.unknown_args:
            unk = f" [compile-time unknown args: {', '.join(self.unknown_args)}]"
        return (
            f"{loc}: {self.severity} [{self.rule}] ({self.thread}/{self.exu}, arch={self.arch}): "
            f"{self.message}{earlier}{unk} [source: {self.source_ref}]"
        )


@dataclasses.dataclass(frozen=True)
class ActiveInit:
    op_family: str
    init_call: str
    user_call: str
    line: int
    template_args: Tuple[Value, ...]
    args: Tuple[Value, ...]
    init_mode: str = "default"
    overwrote_call: Optional[str] = None
    overwrote_line: Optional[int] = None
    invalidated_by_call: Optional[str] = None
    invalidated_at_line: Optional[int] = None
    invalidated_reason: Optional[str] = None
    invalidated_source: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PendingUninit:
    op_family: str
    init_call: str
    user_call: str
    line: int
    uninit_calls: Tuple[str, ...]
    source_ref: str


@dataclasses.dataclass(frozen=True)
class ReconfigState:
    call_name: str
    line: int
    srca: Value
    srcb: Value


@dataclasses.dataclass
class PathState:
    """Per-path typestate for a single Tensix thread across its execution units."""

    # Map from EXU ("Unpack", "Fpu", "Sfpu", "Pack") -> Optional[ActiveInit]
    exu_init: Dict[str, Optional[ActiveInit]] = dataclasses.field(
        default_factory=lambda: {
            "Unpack": None,
            "Fpu": None,
            "Sfpu": None,
            "Pack": None,
        }
    )
    # Map from EXU -> Optional[PendingUninit]
    pending_uninit: Dict[str, Optional[PendingUninit]] = dataclasses.field(
        default_factory=lambda: {
            "Unpack": None,
            "Fpu": None,
            "Sfpu": None,
            "Pack": None,
        }
    )
    # Last reconfig_data_format on UNPACK/MATH
    reconfig: Optional[ReconfigState] = None
    # Correlated compile-time branch assumptions along this path (e.g., `if constexpr (is_sfpu)`)
    branch_facts: Dict[str, str] = dataclasses.field(default_factory=dict)

    def clone(self) -> "PathState":
        return PathState(
            exu_init=dict(self.exu_init),
            pending_uninit=dict(self.pending_uninit),
            reconfig=self.reconfig,
            branch_facts=dict(self.branch_facts),
        )

    def freeze_key(self) -> Tuple[Any, ...]:
        return (
            tuple(sorted(self.exu_init.items())),
            tuple(sorted(self.pending_uninit.items())),
            self.reconfig,
            tuple(sorted(self.branch_facts.items())),
        )


# ---------------------------------------------------------------------------
# AST Nodes for Function Bodies
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Stmt:
    line: int


@dataclasses.dataclass
class CallStmt(Stmt):
    func_name: str
    template_args: List[str]
    args: List[str]
    raw: str


@dataclasses.dataclass
class VarDeclStmt(Stmt):
    name: str
    is_constexpr: bool
    expr: str


@dataclasses.dataclass
class AssignStmt(Stmt):
    name: str
    expr: str


@dataclasses.dataclass
class ThreadScopeStmt(Stmt):
    thread: str  # "UNPACK", "MATH", "PACK"
    body: List[Stmt]


@dataclasses.dataclass
class IfStmt(Stmt):
    cond: str
    is_constexpr: bool
    then_body: List[Stmt]
    else_body: List[Stmt]


@dataclasses.dataclass
class PpBranchStmt(Stmt):
    branches: List[Tuple[str, List[Stmt]]]  # list of (cond_expr, body)
    has_else: bool


@dataclasses.dataclass
class LoopStmt(Stmt):
    kind: str  # "for", "while", "do"
    header: str
    loop_vars: Set[str]
    body: List[Stmt]


@dataclasses.dataclass
class ReturnStmt(Stmt):
    pass


@dataclasses.dataclass
class FunctionDef:
    name: str
    template_params: List[Tuple[str, Optional[str]]]  # (param_name, default_val)
    params: List[Tuple[str, str, Optional[str]]]  # (type_str, param_name, default_val)
    body_src: str
    start_line: int
    file_path: str
    is_entry: bool = False
    stmts: Optional[List[Stmt]] = None


# ---------------------------------------------------------------------------
# Rule Table Loader
# ---------------------------------------------------------------------------


_RULE_CACHE: Dict[str, Dict[str, Any]] = {}


class RuleTable:
    def __init__(self, path: str = DEFAULT_RULES_PATH):
        self.path = path
        norm = os.path.abspath(path)
        if norm in _RULE_CACHE:
            self.raw = _RULE_CACHE[norm]
        else:
            loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            with open(path, "r", encoding="utf-8") as f:
                self.raw = yaml.load(f, Loader=loader)
            _RULE_CACHE[norm] = self.raw

        self.sanitizer_operations: Dict[str, Dict[str, Any]] = self.raw.get(
            "sanitizer_operations", {}
        )
        self.llk_operations: Dict[str, Dict[str, Any]] = self.raw.get(
            "llk_operations", {}
        )
        self.sfpu_operations: Dict[str, Dict[str, Any]] = self.raw.get(
            "sfpu_operations", {}
        )
        self.reconfig_operations: Dict[str, Dict[str, Any]] = self.raw.get(
            "reconfig_operations", {}
        )
        self.compute_api_wrappers: Dict[str, Dict[str, Any]] = self.raw.get(
            "compute_api_wrappers", {}
        )
        self.deprecated_apis: Dict[str, Dict[str, Any]] = self.raw.get(
            "deprecated_apis", {}
        )

        # Build fast lookup maps per thread
        # exec_call -> (op_family, rule_dict)
        self.exec_by_thread: Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]] = {
            t: {} for t in THREADS
        }
        # init_call -> list of (op_family, rule_dict)
        self.init_by_thread: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {
            t: {} for t in THREADS
        }
        # uninit_call -> list of (op_family, rule_dict)
        self.uninit_by_thread: Dict[
            str, Dict[str, List[Tuple[str, Dict[str, Any]]]]
        ] = {t: {} for t in THREADS}

        for family, spec in self.llk_operations.items():
            t = spec["thread"]
            for ex in spec.get("exec_calls", []):
                self.exec_by_thread[t][ex] = (family, spec)
            for init_c in spec.get("required_inits", []):
                self.init_by_thread[t].setdefault(init_c, []).append((family, spec))
            for uninit_c in spec.get("uninit_calls", []):
                self.uninit_by_thread[t].setdefault(uninit_c, []).append((family, spec))

        for family, spec in self.sfpu_operations.items():
            t = spec["thread"]
            for ex in spec.get("exec_calls", []):
                self.exec_by_thread[t][ex] = (family, spec)
            for init_c in spec.get("required_inits", []):
                self.init_by_thread[t].setdefault(init_c, []).append((family, spec))


# ---------------------------------------------------------------------------
# C++ / Kernel Lightweight Parser
# ---------------------------------------------------------------------------


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split `text` on `sep` at top-level (outside `()`, `<>`, `[]`, `{}`)."""
    parts: List[str] = []
    cur: List[str] = []
    paren = 0
    angle = 0
    brack = 0
    brace = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(paren - 1, 0)
        elif c == "[":
            brack += 1
        elif c == "]":
            brack = max(brack - 1, 0)
        elif c == "{":
            brace += 1
        elif c == "}":
            brace = max(brace - 1, 0)
        elif c == "<" and paren == 0 and brack == 0 and brace == 0:
            # Avoid treating `<=` or `<<` as template angle bracket
            if i + 1 < n and text[i + 1] in ("=", "<"):
                cur.append(c)
                cur.append(text[i + 1])
                i += 2
                continue
            angle += 1
        elif c == ">" and paren == 0 and brack == 0 and brace == 0 and angle > 0:
            if i + 1 < n and text[i + 1] == ">":
                angle = max(angle - 1, 0)
            else:
                angle = max(angle - 1, 0)
        if c == sep and paren == 0 and angle == 0 and brack == 0 and brace == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    n = len(text)
    i = start
    while i < n:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


_CALL_HEAD_RE = re.compile(
    r"(?<![\w:])(?:ckernel::|tt::tt_metal::|experimental::)?"
    r"([A-Za-z_]\w*)\s*(?:<([^{};()]*)>)?\s*\("
)


def _extract_calls_from_expr(expr: str, line: int) -> List[CallStmt]:
    """Extract function/macro calls in evaluation order from a single statement expression."""
    calls: List[CallStmt] = []
    i = 0
    n = len(expr)
    while i < n:
        m = _CALL_HEAD_RE.search(expr, i)
        if not m:
            break
        fn_name = m.group(1)
        tpl_raw = m.group(2)
        open_paren = m.end() - 1
        close_paren = _find_matching(expr, open_paren, "(", ")")
        if close_paren < 0:
            break
        if fn_name in (
            "if",
            "for",
            "while",
            "switch",
            "sizeof",
            "alignof",
            "static_cast",
            "reinterpret_cast",
            "bit_cast",
            "defined",
        ):
            i = m.end()
            continue
        tpl_args = (
            _split_top_level(tpl_raw) if tpl_raw is not None and tpl_raw.strip() else []
        )
        arg_str = expr[open_paren + 1 : close_paren]
        args = _split_top_level(arg_str) if arg_str.strip() else []
        calls.append(
            CallStmt(
                line=line,
                func_name=fn_name,
                template_args=tpl_args,
                args=args,
                raw=expr[m.start() : close_paren + 1].strip(),
            )
        )
        i = close_paren + 1
    return calls


_VAR_DECL_RE = re.compile(
    r"^\s*(constexpr\s+)?(?:const\s+)?(?:auto|uint32_t|int32_t|int|uint16_t|uint8_t|bool|std::uint32_t|"
    r"tt::CBIndex|DataFormat|BroadcastType|EltwiseBinaryType|PoolType|ReduceDim|MathFidelity)\s+"
    r"([A-Za-z_]\w*)\s*=\s*(.+)$"
)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([^=].*)$")
_FOR_VAR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*[^;]+")
_INC_VAR_RE = re.compile(
    r"(?:\+\+|--)\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?:\+\+|--|\+=|-=)"
)


def _parse_block(code: str, base_line: int) -> List[Stmt]:
    """Parse a C++ compound statement body into structured Stmt nodes."""
    stmts: List[Stmt] = []
    n = len(code)
    i = 0

    def line_at(pos: int) -> int:
        return base_line + code.count("\n", 0, pos)

    while i < n:
        # Skip whitespace and stray semicolons
        while i < n and code[i] in " \t\r\n;":
            i += 1
        if i >= n:
            break

        # Preprocessor directive at start of line
        line_start = code.rfind("\n", 0, i) + 1
        if code[line_start:i].strip() == "" and code[i] == "#":
            pp_end = code.find("\n", i)
            while pp_end >= 0 and pp_end > 0 and code[pp_end - 1] == "\\":
                pp_end = code.find("\n", pp_end + 1)
            if pp_end < 0:
                pp_end = n
            directive = code[i:pp_end].strip()
            ln = line_at(i)
            m_pp = re.match(r"^#\s*(ifdef|ifndef|if)\b(.*)$", directive)
            if m_pp:
                kind_pp, cond_pp = m_pp.group(1), m_pp.group(2).strip()
                if kind_pp == "ifdef":
                    cond_pp = f"defined({cond_pp})"
                elif kind_pp == "ifndef":
                    cond_pp = f"!defined({cond_pp})"
                branches: List[Tuple[str, List[Stmt]]] = []
                has_else = False
                cur_cond = cond_pp
                branch_start = pp_end + 1
                branch_line = line_at(branch_start)
                depth = 1
                scan = branch_start
                while scan < n and depth > 0:
                    nl = code.find("\n", scan)
                    if nl < 0:
                        nl = n
                    raw_line = code[scan:nl].strip()
                    if raw_line.startswith("#"):
                        m_sub = re.match(
                            r"^#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$", raw_line
                        )
                        if m_sub:
                            kw = m_sub.group(1)
                            rest = m_sub.group(2).strip()
                            if kw in ("if", "ifdef", "ifndef"):
                                depth += 1
                            elif kw == "endif":
                                depth -= 1
                                if depth == 0:
                                    branches.append(
                                        (
                                            cur_cond,
                                            _parse_block(
                                                code[branch_start:scan], branch_line
                                            ),
                                        )
                                    )
                                    i = nl + 1
                                    break
                            elif kw == "elif" and depth == 1:
                                branches.append(
                                    (
                                        cur_cond,
                                        _parse_block(
                                            code[branch_start:scan], branch_line
                                        ),
                                    )
                                )
                                cur_cond = rest
                                branch_start = nl + 1
                                branch_line = line_at(branch_start)
                            elif kw == "else" and depth == 1:
                                branches.append(
                                    (
                                        cur_cond,
                                        _parse_block(
                                            code[branch_start:scan], branch_line
                                        ),
                                    )
                                )
                                cur_cond = "1"
                                has_else = True
                                branch_start = nl + 1
                                branch_line = line_at(branch_start)
                    scan = nl + 1
                else:
                    i = n
                stmts.append(
                    PpBranchStmt(line=ln, branches=branches, has_else=has_else)
                )
                continue
            else:
                # #define or #pragma inside function
                m_def = re.match(r"^#\s*define\s+([A-Za-z_]\w*)\s+(.+)$", directive)
                if m_def:
                    stmts.append(
                        VarDeclStmt(
                            line=ln,
                            name=m_def.group(1),
                            is_constexpr=True,
                            expr=m_def.group(2).strip(),
                        )
                    )
                i = pp_end + 1
                continue

        # Nested block `{ ... }`
        if code[i] == "{":
            close_b = _find_matching(code, i, "{", "}")
            if close_b < 0:
                break
            stmts.extend(_parse_block(code[i + 1 : close_b], line_at(i)))
            i = close_b + 1
            continue

        # Thread macros UNPACK(( ... )), MATH(( ... )), PACK(( ... ))
        m_thr = re.match(r"^(UNPACK|MATH|PACK)\s*\(", code[i:])
        if m_thr:
            t_name = m_thr.group(1)
            open_p = i + m_thr.end() - 1
            close_p = _find_matching(code, open_p, "(", ")")
            if close_p > open_p:
                inner = code[open_p + 1 : close_p].strip()
                while (
                    inner.startswith("(")
                    and inner.endswith(")")
                    and _find_matching(inner, 0, "(", ")") == len(inner) - 1
                ):
                    inner = inner[1:-1].strip()
                inner_stmts = _parse_block(inner + ";", line_at(open_p))
                stmts.append(
                    ThreadScopeStmt(line=line_at(i), thread=t_name, body=inner_stmts)
                )
                i = close_p + 1
                continue

        # `if` or `if constexpr`
        m_if = re.match(r"^if\s*(constexpr\b)?\s*\(", code[i:])
        if m_if:
            is_ce = bool(m_if.group(1))
            ln = line_at(i)
            open_p = i + m_if.end() - 1
            close_p = _find_matching(code, open_p, "(", ")")
            if close_p < 0:
                break
            cond_str = code[open_p + 1 : close_p].strip()
            j = close_p + 1
            while j < n and code[j] in " \t\r\n":
                j += 1
            if j < n and code[j] == "{":
                close_then = _find_matching(code, j, "{", "}")
                if close_then < 0:
                    break
                then_stmts = _parse_block(code[j + 1 : close_then], line_at(j))
                j = close_then + 1
            else:
                semi = code.find(";", j)
                if semi < 0:
                    semi = n
                then_stmts = _parse_block(code[j : semi + 1], line_at(j))
                j = semi + 1

            else_stmts: List[Stmt] = []
            while j < n and code[j] in " \t\r\n":
                j += 1
            m_else = re.match(r"^else\b", code[j:])
            if m_else:
                j += m_else.end()
                while j < n and code[j] in " \t\r\n":
                    j += 1
                if j < n and code[j] == "{":
                    close_else = _find_matching(code, j, "{", "}")
                    if close_else < 0:
                        break
                    else_stmts = _parse_block(code[j + 1 : close_else], line_at(j))
                    j = close_else + 1
                else:
                    # `else if (...)` or single-statement else
                    if code[j:].startswith("if"):
                        # Parse single `if` construct recursively
                        sub_end = j
                        # Find end of else-if chain
                        else_stmts = _parse_single_if_chain(code, j, line_at(j))
                        j = _advance_past_if_chain(code, j)
                    else:
                        semi = code.find(";", j)
                        if semi < 0:
                            semi = n
                        else_stmts = _parse_block(code[j : semi + 1], line_at(j))
                        j = semi + 1

            stmts.append(
                IfStmt(
                    line=ln,
                    cond=cond_str,
                    is_constexpr=is_ce,
                    then_body=then_stmts,
                    else_body=else_stmts,
                )
            )
            i = j
            continue

        # Loops: `for (...)`, `while (...)`, `do { ... } while (...);`
        m_loop = re.match(r"^(for|while)\s*\(", code[i:])
        if m_loop:
            l_kind = m_loop.group(1)
            ln = line_at(i)
            open_p = i + m_loop.end() - 1
            close_p = _find_matching(code, open_p, "(", ")")
            if close_p < 0:
                break
            hdr_str = code[open_p + 1 : close_p].strip()
            loop_vars: Set[str] = set()
            for mv in _FOR_VAR_RE.finditer(hdr_str):
                loop_vars.add(mv.group(1))
            for mv in _INC_VAR_RE.finditer(hdr_str):
                v = mv.group(1) or mv.group(2)
                if v:
                    loop_vars.add(v)
            j = close_p + 1
            while j < n and code[j] in " \t\r\n":
                j += 1
            if j < n and code[j] == "{":
                close_b = _find_matching(code, j, "{", "}")
                if close_b < 0:
                    break
                body_src = code[j + 1 : close_b]
                for mv in _INC_VAR_RE.finditer(body_src):
                    v = mv.group(1) or mv.group(2)
                    if v:
                        loop_vars.add(v)
                body_stmts = _parse_block(body_src, line_at(j))
                j = close_b + 1
            else:
                semi = code.find(";", j)
                if semi < 0:
                    semi = n
                body_stmts = _parse_block(code[j : semi + 1], line_at(j))
                j = semi + 1
            stmts.append(
                LoopStmt(
                    line=ln,
                    kind=l_kind,
                    header=hdr_str,
                    loop_vars=loop_vars,
                    body=body_stmts,
                )
            )
            i = j
            continue

        m_do = re.match(r"^do\b", code[i:])
        if m_do and (i + 2 >= n or not (code[i + 2].isalnum() or code[i + 2] == "_")):
            ln = line_at(i)
            j = i + 2
            while j < n and code[j] in " \t\r\n":
                j += 1
            if j < n and code[j] == "{":
                close_b = _find_matching(code, j, "{", "}")
                if close_b < 0:
                    break
                body_src = code[j + 1 : close_b]
                body_stmts = _parse_block(body_src, line_at(j))
                j = close_b + 1
                while j < n and code[j] in " \t\r\n":
                    j += 1
                m_w = re.match(r"^while\s*\(", code[j:])
                hdr_str = ""
                if m_w:
                    op_w = j + m_w.end() - 1
                    cp_w = _find_matching(code, op_w, "(", ")")
                    if cp_w > op_w:
                        hdr_str = code[op_w + 1 : cp_w].strip()
                        j = cp_w + 1
                stmts.append(
                    LoopStmt(
                        line=ln,
                        kind="do",
                        header=hdr_str,
                        loop_vars=set(),
                        body=body_stmts,
                    )
                )
                i = j
                continue

        # Ordinary statement up to `;` (respecting balanced parens/braces)
        stmt_start = i
        ln = line_at(i)
        paren = 0
        while i < n:
            c = code[i]
            if c == "(":
                paren += 1
            elif c == ")":
                paren = max(paren - 1, 0)
            elif c == ";" and paren == 0:
                break
            elif c == "{" and paren == 0:
                close_b = _find_matching(code, i, "{", "}")
                if close_b < 0:
                    i = n
                    break
                i = close_b + 1
                continue
            i += 1
        stmt_str = code[stmt_start:i].strip()
        if i < n and code[i] == ";":
            i += 1
        if not stmt_str:
            continue

        if re.match(r"^return\b", stmt_str):
            stmts.append(ReturnStmt(line=ln))
            continue

        m_vd = _VAR_DECL_RE.match(stmt_str)
        if m_vd:
            is_ce = bool(m_vd.group(1))
            v_name = m_vd.group(2)
            v_expr = m_vd.group(3).strip()
            stmts.append(
                VarDeclStmt(line=ln, name=v_name, is_constexpr=is_ce, expr=v_expr)
            )
            for call_s in _extract_calls_from_expr(v_expr, ln):
                stmts.append(call_s)
            continue

        m_as = _ASSIGN_RE.match(stmt_str)
        if m_as:
            a_name = m_as.group(1)
            a_expr = m_as.group(2).strip()
            stmts.append(AssignStmt(line=ln, name=a_name, expr=a_expr))
            for call_s in _extract_calls_from_expr(a_expr, ln):
                stmts.append(call_s)
            continue

        for call_s in _extract_calls_from_expr(stmt_str, ln):
            stmts.append(call_s)

    return stmts


def _advance_past_if_chain(code: str, start: int) -> int:
    n = len(code)
    m_if = re.match(r"^if\s*(?:constexpr\b)?\s*\(", code[start:])
    if not m_if:
        return start
    open_p = start + m_if.end() - 1
    close_p = _find_matching(code, open_p, "(", ")")
    if close_p < 0:
        return n
    j = close_p + 1
    while j < n and code[j] in " \t\r\n":
        j += 1
    if j < n and code[j] == "{":
        close_b = _find_matching(code, j, "{", "}")
        j = (close_b + 1) if close_b >= 0 else n
    else:
        semi = code.find(";", j)
        j = (semi + 1) if semi >= 0 else n
    while j < n and code[j] in " \t\r\n":
        j += 1
    m_else = re.match(r"^else\b", code[j:])
    if m_else:
        j += m_else.end()
        while j < n and code[j] in " \t\r\n":
            j += 1
        if j < n and code[j] == "{":
            close_b = _find_matching(code, j, "{", "}")
            j = (close_b + 1) if close_b >= 0 else n
        elif code[j:].startswith("if"):
            j = _advance_past_if_chain(code, j)
        else:
            semi = code.find(";", j)
            j = (semi + 1) if semi >= 0 else n
    return j


def _parse_single_if_chain(code: str, start: int, base_line: int) -> List[Stmt]:
    end = _advance_past_if_chain(code, start)
    return _parse_block(code[start:end], base_line)


# ---------------------------------------------------------------------------
# Translation Unit Function & Header Extractor
# ---------------------------------------------------------------------------

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_DEFINE_CONST_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([^\n\\]+)$", re.MULTILINE
)


def _parse_template_param_list(tpl_str: str) -> List[Tuple[str, Optional[str]]]:
    params: List[Tuple[str, Optional[str]]] = []
    for item in _split_top_level(tpl_str):
        if not item:
            continue
        eq = item.find("=")
        default_val: Optional[str] = None
        if eq >= 0:
            default_val = item[eq + 1 :].strip()
            item = item[:eq].strip()
        tokens = re.findall(r"[A-Za-z_]\w*", item)
        if tokens:
            params.append((tokens[-1], default_val))
    return params


def _parse_func_param_list(param_str: str) -> List[Tuple[str, str, Optional[str]]]:
    params: List[Tuple[str, str, Optional[str]]] = []
    for item in _split_top_level(param_str):
        if not item or item == "void":
            continue
        eq = item.find("=")
        default_val: Optional[str] = None
        if eq >= 0:
            default_val = item[eq + 1 :].strip()
            item = item[:eq].strip()
        # Strip attributes like [[maybe_unused]]
        item = re.sub(r"\[\[[^\]]*\]\]", "", item).strip()
        tokens = re.findall(r"[A-Za-z_]\w*", item)
        if len(tokens) >= 2:
            p_name = tokens[-1]
            p_type = item[: item.rfind(p_name)].strip()
            params.append((p_type, p_name, default_val))
    return params


_MAIN_ENTRY_RE = re.compile(
    r"\b(?:void\s+MAIN|MAIN)\s*(?:\(\s*\))?\s*\{|\bTT_KERNEL\s*\(([^{};]*)\)\s*\{"
)
_FN_DEF_RE = re.compile(
    r"\b(?:void|bool|int|uint32_t|std::uint32_t|auto)\s+([A-Za-z_]\w*)\s*\(([^{};()]*)\)\s*\{"
)
_TPL_PREFIX_RE = re.compile(
    r"template\s*<([^{};>]*)>\s*(?:(?:ALWI|inline|static|constexpr|FORCE_INLINE|TT_KERNEL)\s+)*$"
)


def extract_functions_and_consts(
    src: str, file_path: str
) -> Tuple[List[FunctionDef], Dict[str, str]]:
    """Extract all function definitions (including `MAIN`, `kernel_main`, `TT_KERNEL`, templates) and top-level constants."""
    clean = _blank_noncode(src)
    consts: Dict[str, str] = {}

    for m in _DEFINE_CONST_RE.finditer(clean):
        name, val = m.group(1), m.group(2).strip()
        if "(" not in name and val and not val.startswith("{"):
            consts[name] = val

    # Top-level constexpr declarations
    for m in re.finditer(
        r"(?m)^\s*constexpr\s+(?:auto|uint32_t|int32_t|int|bool|std::uint32_t|tt::CBIndex|DataFormat|BroadcastType|"
        r"EltwiseBinaryType|PoolType|ReduceDim|MathFidelity)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);",
        clean,
    ):
        consts[m.group(1)] = m.group(2).strip()

    funcs: List[FunctionDef] = []
    n = len(clean)
    i = 0

    def line_at(pos: int) -> int:
        return 1 + clean.count("\n", 0, pos)

    while i < n:
        m_main = _MAIN_ENTRY_RE.search(clean, i)
        m_fn = _FN_DEF_RE.search(clean, i)

        cand_main = (m_main.start(), m_main, "main") if m_main else None
        cand_fn = (m_fn.start(), m_fn, "fn") if m_fn else None

        chosen = None
        if cand_main and cand_fn:
            chosen = cand_main if cand_main[0] <= cand_fn[0] else cand_fn
        else:
            chosen = cand_main or cand_fn

        if not chosen:
            break

        pos, m_obj, kind = chosen
        open_brace = m_obj.end() - 1
        close_brace = _find_matching(clean, open_brace, "{", "}")
        if close_brace < 0:
            break

        body_src = clean[open_brace + 1 : close_brace]
        start_ln = line_at(open_brace)

        if kind == "main":
            tt_k_args = m_obj.group(1)
            fn_name = "MAIN"
            f_params: List[Tuple[str, str, Optional[str]]] = []
            if tt_k_args is not None:
                parts = _split_top_level(tt_k_args)
                if len(parts) >= 2:
                    fn_name = parts[1].strip()
                if len(parts) >= 3:
                    p_raw = parts[2].strip()
                    if p_raw.startswith("(") and p_raw.endswith(")"):
                        p_raw = p_raw[1:-1]
                    f_params = _parse_func_param_list(p_raw)
            funcs.append(
                FunctionDef(
                    name=fn_name,
                    template_params=[],
                    params=f_params,
                    body_src=body_src,
                    start_line=start_ln,
                    file_path=file_path,
                    is_entry=True,
                )
            )
        else:
            prefix_window = clean[max(0, pos - 300) : pos]
            m_tpl = _TPL_PREFIX_RE.search(prefix_window)
            tpl_str = m_tpl.group(1) if m_tpl else ""
            fn_name = m_obj.group(1)
            param_str = m_obj.group(2) or ""
            is_tt_kernel = bool(re.search(r"\bTT_KERNEL\s+$", prefix_window))
            is_entry = fn_name in ("kernel_main", "MAIN") or is_tt_kernel
            funcs.append(
                FunctionDef(
                    name=fn_name,
                    template_params=(
                        _parse_template_param_list(tpl_str) if tpl_str else []
                    ),
                    params=_parse_func_param_list(param_str),
                    body_src=body_src,
                    start_line=start_ln,
                    file_path=file_path,
                    is_entry=is_entry,
                )
            )

        i = close_brace + 1

    return funcs, consts


# ---------------------------------------------------------------------------
# Header Resolution
# ---------------------------------------------------------------------------


def _resolve_include(
    inc_path: str,
    including_file: str,
    extra_include_dirs: Sequence[str] = (),
) -> Optional[str]:
    """Locate an `#include "..."` header across the kernel directory and repository include roots."""
    # Skip standard compute/LLK implementation headers that are already modeled in compute_api_rules.yaml
    if inc_path.startswith(
        ("api/compute/", "llk_", "ckernel", "sanitizer/", "internal/")
    ):
        return None
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(including_file)), inc_path),
        *[os.path.join(d, inc_path) for d in extra_include_dirs],
        os.path.join(REPO_ROOT, "ttnn", "cpp", inc_path),
        os.path.join(REPO_ROOT, inc_path),
        os.path.join(REPO_ROOT, "tt_metal", "hw", "inc", inc_path),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


_FILE_CACHE: Dict[str, Tuple[List[FunctionDef], Dict[str, str], List[str]]] = {}


def _load_and_parse_file(
    norm_path: str, display_path: str
) -> Tuple[List[FunctionDef], Dict[str, str], List[str]]:
    cached = _FILE_CACHE.get(norm_path)
    if cached is not None:
        return cached
    try:
        src = open(norm_path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        res: Tuple[List[FunctionDef], Dict[str, str], List[str]] = ([], {}, [])
        _FILE_CACHE[norm_path] = res
        return res
    includes = [m.group(1) for m in _INCLUDE_RE.finditer(src)]
    funcs, consts = extract_functions_and_consts(src, display_path)
    for fn in funcs:
        if fn.stmts is None:
            fn.stmts = _parse_block(fn.body_src, fn.start_line)
    res = (funcs, consts, includes)
    _FILE_CACHE[norm_path] = res
    return res


def collect_translation_unit(
    kernel_path: str,
    extra_include_dirs: Sequence[str] = (),
) -> Tuple[List[FunctionDef], Dict[str, FunctionDef], Dict[str, str]]:
    """Load `kernel_path` and recursively collect helper functions and constants from included headers."""
    visited: Set[str] = set()
    all_helpers: Dict[str, FunctionDef] = {}
    all_consts: Dict[str, str] = {}
    main_funcs: List[FunctionDef] = []

    def visit(path: str, is_root: bool) -> None:
        norm = os.path.normpath(os.path.abspath(path))
        if norm in visited or not os.path.isfile(norm):
            return
        visited.add(norm)
        disp = path if is_root else os.path.relpath(norm, REPO_ROOT)
        funcs, consts, includes = _load_and_parse_file(norm, disp)
        for inc in includes:
            resolved = _resolve_include(inc, norm, extra_include_dirs)
            if resolved:
                visit(resolved, is_root=False)
        all_consts.update(consts)
        for fn in funcs:
            all_helpers[fn.name] = fn
            if is_root:
                main_funcs.append(fn)

    visit(kernel_path, is_root=True)
    return main_funcs, all_helpers, all_consts


# ---------------------------------------------------------------------------
# Compile-Time Expression & Preprocessor Evaluator
# ---------------------------------------------------------------------------


def normalize_const(expr: str, env: Dict[str, Value]) -> Value:
    """Resolve `expr` against compile-time environment `env`.

    Never guesses runtime values: if `expr` is not a compile-time literal, enum constant,
    or bound `constexpr`/template parameter, returns `Value(raw=expr, resolved=None)`.
    """
    s = expr.strip()
    while (
        s.startswith("(")
        and s.endswith(")")
        and _find_matching(s, 0, "(", ")") == len(s) - 1
    ):
        s = s[1:-1].strip()

    if not s:
        return Value(raw="", resolved="")

    if s in env:
        return env[s]

    if s in KNOWN_ENUM_TOKENS:
        return Value(raw=s, resolved=KNOWN_ENUM_TOKENS[s])

    # Numeric literal (e.g. `0`, `1`, `1u`, `0x10`)
    m_num = re.match(r"^(0x[0-9a-fA-F]+|\d+)[uUlL]*$", s)
    if m_num:
        val_int = int(m_num.group(1), 0)
        return Value(raw=s, resolved=str(val_int))

    # `tt::CBIndex::c_0` or `CBIndex::c_16`
    m_cb = re.match(r"^(?:tt::)?CBIndex::(c_\d+)$", s)
    if m_cb:
        return Value(raw=s, resolved=f"tt::CBIndex::{m_cb.group(1)}")

    # `DataFormat::Float16_b` etc.
    m_df = re.match(r"^(?:ckernel::)?DataFormat::(\w+)$", s)
    if m_df:
        return Value(raw=s, resolved=f"DataFormat::{m_df.group(1)}")

    # Boolean negation `!x`
    if s.startswith("!") and not s.startswith("!="):
        sub = normalize_const(s[1:], env)
        if sub.is_known:
            if sub.resolved in ("true", "1"):
                return Value(raw=s, resolved="false")
            if sub.resolved in ("false", "0"):
                return Value(raw=s, resolved="true")
        return Value(raw=s, resolved=None)

    # Simple equality / inequality `a == b` or `a != b`
    for op_tok, is_eq in (("==", True), ("!=", False)):
        if op_tok in s and "&&" not in s and "||" not in s:
            lhs_s, rhs_s = s.split(op_tok, 1)
            lhs = normalize_const(lhs_s, env)
            rhs = normalize_const(rhs_s, env)
            if lhs.is_known and rhs.is_known:
                res = (
                    (lhs.resolved == rhs.resolved)
                    if is_eq
                    else (lhs.resolved != rhs.resolved)
                )
                return Value(raw=s, resolved="true" if res else "false")
            return Value(raw=s, resolved=None)

    return Value(raw=s, resolved=None)


def eval_pp_condition(
    cond: str, active_macros: Set[str], const_env: Dict[str, Value]
) -> Optional[bool]:
    """Evaluate a `#if` / `#ifdef` condition for the active `(arch, thread)`.

    Returns `True`, `False`, or `None` if the condition depends on an unknown external macro.
    """
    s = cond.strip()
    if s == "1":
        return True
    if s == "0":
        return False

    # Check if any unknown identifier appears in `defined(X)`
    unknown_defined = False

    def repl_defined(m: re.Match[str]) -> str:
        nonlocal unknown_defined
        ident = m.group(1) or m.group(2)
        if ident in active_macros or ident in const_env:
            return "1"
        if ident in ALL_THREAD_MACROS or ident in ALL_ARCH_MACROS:
            return "0"
        unknown_defined = True
        return "0"

    expr = re.sub(
        r"defined\s*\(\s*([A-Za-z_]\w*)\s*\)|defined\s+([A-Za-z_]\w*)", repl_defined, s
    )
    if unknown_defined:
        return None

    expr = expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")

    # Replace remaining identifiers
    def repl_ident(m: re.Match[str]) -> str:
        nonlocal unknown_defined
        tok = m.group(0)
        if tok in ("and", "or", "not", "True", "False"):
            return tok
        if tok in active_macros:
            return "1"
        if tok in ALL_THREAD_MACROS or tok in ALL_ARCH_MACROS:
            return "0"
        if tok in const_env and const_env[tok].is_known:
            v = const_env[tok].resolved
            if v in ("true", "1"):
                return "1"
            if v in ("false", "0"):
                return "0"
        unknown_defined = True
        return "0"

    expr = re.sub(r"\b[A-Za-z_]\w*\b", repl_ident, expr)
    if unknown_defined:
        return None

    try:
        return bool(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def values_compatible(init_val: Value, exec_val: Value) -> Tuple[bool, Optional[str]]:
    """Compare an init compile-time value and an exec compile-time value.

    Returns `(is_compatible, unknown_note)`.
    Never guesses when either argument is unknown at compile time.
    """
    if not init_val.is_known or not exec_val.is_known:
        unk_parts = []
        if not init_val.is_known:
            unk_parts.append(f"init={init_val.format_for_diag()}")
        if not exec_val.is_known:
            unk_parts.append(f"exec={exec_val.format_for_diag()}")
        return True, ", ".join(unk_parts)

    a = init_val.resolved
    b = exec_val.resolved
    # Boolean / integer equivalence (`false` == `0`, `true` == `1`)
    bool_norm = {"0": "false", "1": "true", "false": "false", "true": "true"}
    if a in bool_norm and b in bool_norm:
        return (bool_norm[a] == bool_norm[b]), None

    return (a == b), None


# ---------------------------------------------------------------------------
# Per-Thread Typestate & Dataflow Analyzer
# ---------------------------------------------------------------------------


class KernelOrderChecker:
    def __init__(self, rules: RuleTable):
        self.rules = rules

    def check_file(
        self,
        kernel_path: str,
        arches: Sequence[str] = ARCHES,
        extra_include_dirs: Sequence[str] = (),
        extra_defines: Optional[Dict[str, str]] = None,
    ) -> List[Diagnostic]:
        main_funcs, helpers, file_consts = collect_translation_unit(
            kernel_path, extra_include_dirs
        )
        if extra_defines:
            file_consts = {**file_consts, **extra_defines}

        # Select entry functions (`MAIN`, `kernel_main`, `TT_KERNEL`), or top-level functions if none
        entries = [fn for fn in main_funcs if fn.is_entry]
        if not entries:
            entries = main_funcs

        all_diags: List[Diagnostic] = []
        seen_keys: Set[Tuple[Any, ...]] = set()

        for arch in arches:
            for entry in entries:
                for thread in THREADS:
                    diags = self._analyze_entry(
                        entry=entry,
                        helpers=helpers,
                        file_consts=file_consts,
                        arch=arch,
                        thread=thread,
                        kernel_path=kernel_path,
                    )
                    for d in diags:
                        k = d.key()
                        if k not in seen_keys:
                            seen_keys.add(k)
                            all_diags.append(d)

        return all_diags

    def _analyze_entry(
        self,
        entry: FunctionDef,
        helpers: Dict[str, FunctionDef],
        file_consts: Dict[str, str],
        arch: str,
        thread: str,
        kernel_path: str,
    ) -> List[Diagnostic]:
        active_macros = set(THREAD_MACROS[thread]) | set(ARCH_MACROS.get(arch, set()))
        env: Dict[str, Value] = {}
        for k, v in file_consts.items():
            env[k] = normalize_const(v, env)
        for _, p_name, p_def in entry.params:
            if p_def is not None:
                env[p_name] = normalize_const(p_def, env)
            else:
                env[p_name] = Value(raw=p_name, resolved=None)

        if entry.stmts is None:
            entry.stmts = _parse_block(entry.body_src, entry.start_line)

        diags: List[Diagnostic] = []
        initial_states = [PathState()]

        out_states = self._analyze_stmt_list(
            stmts=entry.stmts,
            states=initial_states,
            env=env,
            active_macros=active_macros,
            helpers=helpers,
            arch=arch,
            thread=thread,
            file_path=kernel_path,
            call_stack=[(entry.name, entry.start_line)],
            loop_depth=0,
            loop_vars=set(),
            loop_competing_exu=set(),
            diags=diags,
        )

        # At kernel exit, check if any path still holds a `pending_uninit`
        for st in out_states:
            for exu, pend in st.pending_uninit.items():
                if pend is not None:
                    diags.append(
                        Diagnostic(
                            rule="missing-uninit",
                            severity="error",
                            file=kernel_path,
                            line=pend.line,
                            thread=thread,
                            exu=exu,
                            arch=arch,
                            call=pend.user_call,
                            earlier_call=pend.init_call,
                            earlier_line=pend.line,
                            message=(
                                f"`{pend.user_call}` (configuring `{pend.op_family}`) requires an uninit call "
                                f"({', '.join(f'`{u}`' for u in pend.uninit_calls)}) before kernel exit"
                            ),
                            source_ref=pend.source_ref,
                        )
                    )

        return diags

    def _dedup_states(self, states: Iterable[PathState]) -> List[PathState]:
        out: List[PathState] = []
        seen: Set[Tuple[Any, ...]] = set()
        for s in states:
            k = s.freeze_key()
            if k not in seen:
                seen.add(k)
                out.append(s)
                if len(out) >= 8:
                    break
        return out

    def _collect_loop_exu_transitions(
        self,
        stmts: List[Stmt],
        thread: str,
        helpers: Dict[str, FunctionDef],
        visited_fns: Optional[Set[str]] = None,
    ) -> Dict[str, Set[str]]:
        """Collect which operation families or uninits are initialized per EXU inside a loop body."""
        if visited_fns is None:
            visited_fns = set()
        exu_ops: Dict[str, Set[str]] = {
            "Unpack": set(),
            "Fpu": set(),
            "Sfpu": set(),
            "Pack": set(),
        }

        def record_llk_call(c_name: str, user_fn: str, tpl_tag: str = "") -> None:
            init_matches = self.rules.init_by_thread[thread].get(c_name, [])
            if init_matches:
                for fam, spec in init_matches:
                    exu_ops[spec["exu"]].add(f"init:{user_fn}:{c_name}:{tpl_tag}")
                return
            for fam, spec in self.rules.uninit_by_thread[thread].get(c_name, []):
                exu_ops[spec["exu"]].add(f"uninit:{user_fn}:{c_name}")
            if c_name in (
                "llk_unpack_reconfig_data_format",
                "llk_math_reconfig_data_format",
            ):
                exu_ops["Unpack"].add("reconfig")
                exu_ops["Fpu"].add("reconfig")
            if c_name == "llk_pack_reconfig_data_format":
                exu_ops["Pack"].add("reconfig")

        def walk(s_list: List[Stmt]) -> None:
            for s in s_list:
                if isinstance(s, ThreadScopeStmt):
                    if s.thread == thread:
                        walk(s.body)
                elif isinstance(s, IfStmt):
                    walk(s.then_body)
                    walk(s.else_body)
                elif isinstance(s, PpBranchStmt):
                    for _, b_body in s.branches:
                        walk(b_body)
                elif isinstance(s, LoopStmt):
                    walk(s.body)
                elif isinstance(s, CallStmt):
                    fn = s.func_name
                    tpl_tag = ",".join(s.template_args)
                    if fn in self.rules.compute_api_wrappers:
                        w_spec = self.rules.compute_api_wrappers[fn]
                        for item in w_spec.get(thread, []):
                            item_tpl = ",".join(item.get("template_args", []))
                            record_llk_call(item["call"], fn, f"{tpl_tag}|{item_tpl}")
                    else:
                        record_llk_call(fn, fn, tpl_tag)
                        if fn in helpers and fn not in visited_fns:
                            visited_fns.add(fn)
                            h = helpers[fn]
                            if h.stmts is None:
                                h.stmts = _parse_block(h.body_src, h.start_line)
                            sub = self._collect_loop_exu_transitions(
                                h.stmts, thread, helpers, visited_fns
                            )
                            for k, v in sub.items():
                                exu_ops[k].update(v)

        walk(stmts)
        return exu_ops

    def _analyze_stmt_list(
        self,
        stmts: List[Stmt],
        states: List[PathState],
        env: Dict[str, Value],
        active_macros: Set[str],
        helpers: Dict[str, FunctionDef],
        arch: str,
        thread: str,
        file_path: str,
        call_stack: List[Tuple[str, int]],
        loop_depth: int,
        loop_vars: Set[str],
        loop_competing_exu: Set[str],
        diags: List[Diagnostic],
    ) -> List[PathState]:
        cur_states = [s.clone() for s in states]

        for stmt in stmts:
            if not cur_states:
                break

            if isinstance(stmt, ReturnStmt):
                return cur_states

            if isinstance(stmt, VarDeclStmt):
                if stmt.is_constexpr:
                    env[stmt.name] = normalize_const(stmt.expr, env)
                else:
                    # Non-constexpr variable is treated as runtime unknown unless assigned a literal
                    env[stmt.name] = Value(raw=stmt.name, resolved=None)
                continue

            if isinstance(stmt, AssignStmt):
                env[stmt.name] = Value(raw=stmt.name, resolved=None)
                continue

            if isinstance(stmt, ThreadScopeStmt):
                if stmt.thread == thread:
                    cur_states = self._analyze_stmt_list(
                        stmts=stmt.body,
                        states=cur_states,
                        env=env,
                        active_macros=active_macros,
                        helpers=helpers,
                        arch=arch,
                        thread=thread,
                        file_path=file_path,
                        call_stack=call_stack,
                        loop_depth=loop_depth,
                        loop_vars=loop_vars,
                        loop_competing_exu=loop_competing_exu,
                        diags=diags,
                    )
                continue

            if isinstance(stmt, PpBranchStmt):
                known_taken: Optional[List[Stmt]] = None
                any_unknown = False
                for cond_str, b_body in stmt.branches:
                    ev = eval_pp_condition(cond_str, active_macros, env)
                    if ev is True:
                        known_taken = b_body
                        break
                    if ev is None:
                        any_unknown = True

                if known_taken is not None and not any_unknown:
                    cur_states = self._analyze_stmt_list(
                        stmts=known_taken,
                        states=cur_states,
                        env=env,
                        active_macros=active_macros,
                        helpers=helpers,
                        arch=arch,
                        thread=thread,
                        file_path=file_path,
                        call_stack=call_stack,
                        loop_depth=loop_depth,
                        loop_vars=loop_vars,
                        loop_competing_exu=loop_competing_exu,
                        diags=diags,
                    )
                elif any_unknown:
                    joined: List[PathState] = []
                    for cond_str, b_body in stmt.branches:
                        ev = eval_pp_condition(cond_str, active_macros, env)
                        if ev is False:
                            continue
                        branch_env = dict(env)
                        joined.extend(
                            self._analyze_stmt_list(
                                stmts=b_body,
                                states=cur_states,
                                env=branch_env,
                                active_macros=active_macros,
                                helpers=helpers,
                                arch=arch,
                                thread=thread,
                                file_path=file_path,
                                call_stack=call_stack,
                                loop_depth=loop_depth,
                                loop_vars=loop_vars,
                                loop_competing_exu=loop_competing_exu,
                                diags=diags,
                            )
                        )
                    if not stmt.has_else:
                        joined.extend([s.clone() for s in cur_states])
                    cur_states = self._dedup_states(joined)
                continue

            if isinstance(stmt, IfStmt):
                # Evaluate `if constexpr` or first-iteration loop guard when condition is known
                cond_key = stmt.cond.strip()
                cond_val = normalize_const(cond_key, env)
                is_loop_zero_guard = bool(
                    re.match(
                        r"^([A-Za-z_]\w*)\s*==\s*0$|^0\s*==\s*([A-Za-z_]\w*)$", cond_key
                    )
                )
                if (stmt.is_constexpr or is_loop_zero_guard) and cond_val.is_known:
                    chosen_body = (
                        stmt.then_body
                        if cond_val.resolved in ("true", "1")
                        else stmt.else_body
                    )
                    cur_states = self._analyze_stmt_list(
                        stmts=chosen_body,
                        states=cur_states,
                        env=env,
                        active_macros=active_macros,
                        helpers=helpers,
                        arch=arch,
                        thread=thread,
                        file_path=file_path,
                        call_stack=call_stack,
                        loop_depth=loop_depth,
                        loop_vars=loop_vars,
                        loop_competing_exu=loop_competing_exu,
                        diags=diags,
                    )
                else:
                    # Partition states by correlated branch facts (e.g. repeated `if constexpr (is_sfpu)` or `!is_sfpu`)
                    neg_key = (
                        cond_key[1:].strip()
                        if (cond_key.startswith("!") and not cond_key.startswith("!="))
                        else f"!{cond_key}"
                    )
                    then_input: List[PathState] = []
                    else_input: List[PathState] = []
                    for st in cur_states:
                        known_fact = st.branch_facts.get(cond_key)
                        if known_fact is None and neg_key in st.branch_facts:
                            known_fact = (
                                "false"
                                if st.branch_facts[neg_key] == "true"
                                else "true"
                            )
                        if known_fact == "true":
                            then_input.append(st.clone())
                        elif known_fact == "false":
                            else_input.append(st.clone())
                        else:
                            t_st = st.clone()
                            e_st = st.clone()
                            if stmt.is_constexpr and len(t_st.branch_facts) < 8:
                                t_st.branch_facts[cond_key] = "true"
                                e_st.branch_facts[cond_key] = "false"
                            then_input.append(t_st)
                            else_input.append(e_st)

                    then_states = (
                        self._analyze_stmt_list(
                            stmts=stmt.then_body,
                            states=then_input,
                            env=dict(env),
                            active_macros=active_macros,
                            helpers=helpers,
                            arch=arch,
                            thread=thread,
                            file_path=file_path,
                            call_stack=call_stack,
                            loop_depth=loop_depth,
                            loop_vars=loop_vars,
                            loop_competing_exu=loop_competing_exu,
                            diags=diags,
                        )
                        if then_input
                        else []
                    )
                    else_states = (
                        self._analyze_stmt_list(
                            stmts=stmt.else_body,
                            states=else_input,
                            env=dict(env),
                            active_macros=active_macros,
                            helpers=helpers,
                            arch=arch,
                            thread=thread,
                            file_path=file_path,
                            call_stack=call_stack,
                            loop_depth=loop_depth,
                            loop_vars=loop_vars,
                            loop_competing_exu=loop_competing_exu,
                            diags=diags,
                        )
                        if else_input
                        else []
                    )
                    cur_states = self._dedup_states(then_states + else_states)
                continue

            if isinstance(stmt, LoopStmt):
                new_loop_vars = loop_vars | stmt.loop_vars
                iter1_env = dict(env)
                iter2_env = dict(env)
                for lv in stmt.loop_vars:
                    # On iteration 1, `lv == 0` is true; on subsequent iterations, `lv == 0` is false
                    iter1_env[lv] = Value(raw=lv, resolved="0")
                    iter2_env[lv] = Value(raw=lv, resolved="1")
                    env[lv] = Value(raw=lv, resolved=None)
                exu_transitions = self._collect_loop_exu_transitions(
                    stmt.body, thread, helpers
                )
                competing = set(loop_competing_exu)
                for exu_k, ops_set in exu_transitions.items():
                    if len(ops_set) > 1:
                        competing.add(exu_k)

                # Analyze first iteration
                pre_keys = {s.freeze_key() for s in cur_states}
                iter1_states = self._analyze_stmt_list(
                    stmts=stmt.body,
                    states=cur_states,
                    env=iter1_env,
                    active_macros=active_macros,
                    helpers=helpers,
                    arch=arch,
                    thread=thread,
                    file_path=file_path,
                    call_stack=call_stack,
                    loop_depth=loop_depth + 1,
                    loop_vars=new_loop_vars,
                    loop_competing_exu=competing,
                    diags=diags,
                )
                # Only analyze back-edge states that actually changed during iteration 1 at outermost loop
                delta_states = [
                    s for s in iter1_states if s.freeze_key() not in pre_keys
                ]
                if delta_states and loop_depth == 0:
                    iter2_states = self._analyze_stmt_list(
                        stmts=stmt.body,
                        states=delta_states,
                        env=iter2_env,
                        active_macros=active_macros,
                        helpers=helpers,
                        arch=arch,
                        thread=thread,
                        file_path=file_path,
                        call_stack=call_stack,
                        loop_depth=loop_depth + 1,
                        loop_vars=new_loop_vars,
                        loop_competing_exu=competing,
                        diags=diags,
                    )
                    cur_states = self._dedup_states(iter1_states + iter2_states)
                else:
                    cur_states = iter1_states
                continue

            if isinstance(stmt, CallStmt):
                cur_states = self._handle_call(
                    call=stmt,
                    states=cur_states,
                    env=env,
                    active_macros=active_macros,
                    helpers=helpers,
                    arch=arch,
                    thread=thread,
                    file_path=file_path,
                    call_stack=call_stack,
                    loop_depth=loop_depth,
                    loop_vars=loop_vars,
                    loop_competing_exu=loop_competing_exu,
                    diags=diags,
                )

        return cur_states

    def _handle_call(
        self,
        call: CallStmt,
        states: List[PathState],
        env: Dict[str, Value],
        active_macros: Set[str],
        helpers: Dict[str, FunctionDef],
        arch: str,
        thread: str,
        file_path: str,
        call_stack: List[Tuple[str, int]],
        loop_depth: int,
        loop_vars: Set[str],
        loop_competing_exu: Set[str],
        diags: List[Diagnostic],
    ) -> List[PathState]:
        fn = call.func_name
        report_line = call.line

        # 1. Check deprecated APIs first (report on UNPACK thread to avoid 3x duplicate for thread-agnostic wrappers)
        if fn in self.rules.deprecated_apis:
            dep_spec = self.rules.deprecated_apis[fn]
            allowed_threads = dep_spec.get("threads", list(THREADS))
            if thread == allowed_threads[0]:
                diags.append(
                    Diagnostic(
                        rule="deprecated-init",
                        severity="error",
                        file=file_path,
                        line=report_line,
                        thread=thread,
                        exu="All",
                        arch=arch,
                        call=fn,
                        earlier_call=None,
                        earlier_line=None,
                        message=f"`{fn}` is deprecated; use `{dep_spec['replacement']}` instead ({dep_spec['reason']})",
                        source_ref=dep_spec["source"],
                    )
                )

        # Resolve call arguments and template arguments in current env
        resolved_tpl = [normalize_const(t, env) for t in call.template_args]
        resolved_args = [normalize_const(a, env) for a in call.args]

        # 2. If `fn` is a user/library helper function (and not a primitive LLK or modeled compute wrapper),
        #    inline/expand its body with bound template & value parameters!
        if (
            fn in helpers
            and fn not in self.rules.compute_api_wrappers
            and fn not in self.rules.exec_by_thread[thread]
            and fn not in self.rules.init_by_thread[thread]
            and fn not in self.rules.uninit_by_thread[thread]
            and len(call_stack) < 12
            and not any(cs_name == fn for cs_name, _ in call_stack)
        ):
            helper_def = helpers[fn]
            sub_env = dict(env)
            for idx, (tp_name, tp_def) in enumerate(helper_def.template_params):
                if idx < len(resolved_tpl):
                    sub_env[tp_name] = resolved_tpl[idx]
                elif tp_def is not None:
                    sub_env[tp_name] = normalize_const(tp_def, sub_env)
                else:
                    sub_env[tp_name] = Value(raw=tp_name, resolved=None)
            for idx, (_, p_name, p_def) in enumerate(helper_def.params):
                if idx < len(resolved_args):
                    sub_env[p_name] = resolved_args[idx]
                elif p_def is not None:
                    sub_env[p_name] = normalize_const(p_def, sub_env)
                else:
                    sub_env[p_name] = Value(raw=p_name, resolved=None)

            if helper_def.stmts is None:
                helper_def.stmts = _parse_block(
                    helper_def.body_src, helper_def.start_line
                )

            return self._analyze_stmt_list(
                stmts=helper_def.stmts,
                states=states,
                env=sub_env,
                active_macros=active_macros,
                helpers=helpers,
                arch=arch,
                thread=thread,
                file_path=file_path,
                call_stack=call_stack + [(fn, report_line)],
                loop_depth=loop_depth,
                loop_vars=loop_vars,
                loop_competing_exu=loop_competing_exu,
                diags=diags,
            )

        # 3. Expand compute-API wrapper to per-thread LLK operations if applicable
        if fn in self.rules.compute_api_wrappers:
            w_spec = self.rules.compute_api_wrappers[fn]
            llk_items = w_spec.get(thread, [])
            cur_states = states
            for item in llk_items:
                llk_call_name = item["call"]
                init_mode = item.get("init_mode", "default")
                mapped_tpl = [
                    self._bind_placeholder(p, resolved_tpl, resolved_args, env)
                    for p in item.get("template_args", [])
                ]
                mapped_args = [
                    self._bind_placeholder(p, resolved_tpl, resolved_args, env)
                    for p in item.get("args", [])
                ]
                cur_states = self._apply_llk_operation(
                    llk_name=llk_call_name,
                    user_call=fn,
                    line=report_line,
                    tpl_vals=mapped_tpl,
                    arg_vals=mapped_args,
                    raw_call_args=call.args + call.template_args,
                    init_mode=init_mode,
                    states=cur_states,
                    arch=arch,
                    thread=thread,
                    file_path=file_path,
                    loop_depth=loop_depth,
                    loop_vars=loop_vars,
                    loop_competing_exu=loop_competing_exu,
                    diags=diags,
                )
            return cur_states

        # 4. Direct LLK or SFPU call
        return self._apply_llk_operation(
            llk_name=fn,
            user_call=fn,
            line=report_line,
            tpl_vals=resolved_tpl,
            arg_vals=resolved_args,
            raw_call_args=call.args + call.template_args,
            init_mode=(
                "tilize"
                if (
                    fn == "llk_pack_init"
                    and len(resolved_tpl) >= 3
                    and resolved_tpl[2].resolved in ("true", "1")
                )
                else "default"
            ),
            states=states,
            arch=arch,
            thread=thread,
            file_path=file_path,
            loop_depth=loop_depth,
            loop_vars=loop_vars,
            loop_competing_exu=loop_competing_exu,
            diags=diags,
        )

    def _bind_placeholder(
        self,
        pattern: str,
        tpl_vals: Sequence[Value],
        arg_vals: Sequence[Value],
        env: Dict[str, Value],
    ) -> Value:
        if pattern.startswith("$tpl"):
            idx = int(pattern[4:])
            if idx < len(tpl_vals):
                return tpl_vals[idx]
            return Value(raw=pattern, resolved=None)
        if pattern.startswith("$arg"):
            idx = int(pattern[4:])
            if idx < len(arg_vals):
                return arg_vals[idx]
            return Value(raw=pattern, resolved=None)
        return normalize_const(pattern, env)

    def _apply_llk_operation(
        self,
        llk_name: str,
        user_call: str,
        line: int,
        tpl_vals: List[Value],
        arg_vals: List[Value],
        raw_call_args: List[str],
        init_mode: str,
        states: List[PathState],
        arch: str,
        thread: str,
        file_path: str,
        loop_depth: int,
        loop_vars: Set[str],
        loop_competing_exu: Set[str],
        diags: List[Diagnostic],
    ) -> List[PathState]:
        next_states: List[PathState] = []

        # Check data-format reconfig calls
        if llk_name in (
            "llk_unpack_reconfig_data_format",
            "llk_math_reconfig_data_format",
        ):
            is_reverse = any("Reverse" in (t.raw or "") for t in tpl_vals)
            for st in states:
                ns = st.clone()
                if len(arg_vals) >= 2:
                    ns.reconfig = ReconfigState(
                        call_name=user_call,
                        line=line,
                        srca=arg_vals[-1] if is_reverse else arg_vals[-2],
                        srcb=arg_vals[-2] if is_reverse else arg_vals[-1],
                    )
                next_states.append(ns)
            return self._dedup_states(next_states)

        if llk_name == "llk_pack_reconfig_data_format" and thread == "PACK":
            for st in states:
                ns = st.clone()
                active = ns.exu_init.get("Pack")
                if (
                    arch == "blackhole"
                    and active is not None
                    and active.init_mode == "tilize"
                ):
                    ns.exu_init["Pack"] = dataclasses.replace(
                        active,
                        invalidated_by_call=user_call,
                        invalidated_at_line=line,
                        invalidated_reason=(
                            f"`{user_call}` after `{active.user_call}` on Blackhole overwrites the packer's "
                            f"Tilize Z-stride (`PCK_EDGE_OFFSET`), corrupting half of every output tile (#61146)"
                        ),
                        invalidated_source="tt_metal/hw/inc/api/compute/tilize.h:35 (#61146)",
                    )
                next_states.append(ns)
            return self._dedup_states(next_states)

        # Check if `llk_name` is an uninit call on this thread
        uninit_specs = self.rules.uninit_by_thread[thread].get(llk_name, [])
        is_also_init = llk_name in self.rules.init_by_thread[thread]
        if uninit_specs and not is_also_init:
            for st in states:
                ns = st.clone()
                for fam, spec in uninit_specs:
                    exu = spec["exu"]
                    pend = ns.pending_uninit.get(exu)
                    if pend is not None and llk_name in pend.uninit_calls:
                        ns.pending_uninit[exu] = None
                    active = ns.exu_init.get(exu)
                    if active is not None and active.op_family == fam:
                        ns.exu_init[exu] = None
                next_states.append(ns)
            return self._dedup_states(next_states)

        # Check if `llk_name` is an init call on this thread
        init_specs = self.rules.init_by_thread[thread].get(llk_name, [])
        if init_specs:
            primary_fam, primary_spec = init_specs[0]
            exu = primary_spec["exu"]
            hoistable = primary_spec.get("hoistable", True)

            # Rule 6: `loop-invariant-init` (warning only)
            if (
                loop_depth > 0
                and hoistable
                and exu not in loop_competing_exu
                and not self._depends_on_vars(raw_call_args, loop_vars)
            ):
                diags.append(
                    Diagnostic(
                        rule="loop-invariant-init",
                        severity="warning",
                        file=file_path,
                        line=line,
                        thread=thread,
                        exu=exu,
                        arch=arch,
                        call=user_call,
                        earlier_call=None,
                        earlier_line=None,
                        message=(
                            f"`{user_call}` (`{llk_name}`) is loop-invariant inside a loop and can be hoisted "
                            f"outside the loop"
                        ),
                        source_ref=primary_spec["source"],
                    )
                )

            for st in states:
                ns = st.clone()
                # Check if a pending uninit on this EXU is satisfied by this call or violated!
                pend = ns.pending_uninit.get(exu)
                if pend is not None:
                    if llk_name in pend.uninit_calls:
                        ns.pending_uninit[exu] = None
                    elif pend.op_family != primary_fam:
                        diags.append(
                            Diagnostic(
                                rule="missing-uninit",
                                severity="error",
                                file=file_path,
                                line=line,
                                thread=thread,
                                exu=exu,
                                arch=arch,
                                call=user_call,
                                earlier_call=pend.user_call,
                                earlier_line=pend.line,
                                message=(
                                    f"`{user_call}` (`{llk_name}`) is called while `{pend.user_call}` "
                                    f"still requires an uninit call ({', '.join(f'`{u}`' for u in pend.uninit_calls)})"
                                ),
                                source_ref=pend.source_ref,
                            )
                        )
                        ns.pending_uninit[exu] = None

                prev_active = ns.exu_init.get(exu)
                new_active = ActiveInit(
                    op_family=primary_fam,
                    init_call=llk_name,
                    user_call=user_call,
                    line=line,
                    template_args=tuple(tpl_vals),
                    args=tuple(arg_vals),
                    init_mode=init_mode,
                    overwrote_call=(
                        prev_active.user_call
                        if (prev_active and prev_active.op_family != primary_fam)
                        else None
                    ),
                    overwrote_line=(
                        prev_active.line
                        if (prev_active and prev_active.op_family != primary_fam)
                        else None
                    ),
                )
                ns.exu_init[exu] = new_active

                if primary_spec.get("requires_uninit", False):
                    ns.pending_uninit[exu] = PendingUninit(
                        op_family=primary_fam,
                        init_call=llk_name,
                        user_call=user_call,
                        line=line,
                        uninit_calls=tuple(primary_spec.get("uninit_calls", [])),
                        source_ref=primary_spec["source"],
                    )

                # Clear reconfig state when a fresh matmul/binary init configures operands
                if exu in ("Unpack", "Fpu"):
                    ns.reconfig = None

                next_states.append(ns)
            return self._dedup_states(next_states)

        # Check if `llk_name` is an execute call on this thread
        exec_entry = self.rules.exec_by_thread[thread].get(llk_name)
        if exec_entry is not None:
            fam, spec = exec_entry
            if arch not in spec.get("arches", ARCHES):
                return states
            exu = spec["exu"]
            req_inits = set(spec.get("required_inits", []))

            for st in states:
                ns = st.clone()
                active = ns.exu_init.get(exu)

                # Check if `llk_pack_reduce_mask_config` (which is modeled as both an init-like mask config
                # and requires `llk_pack_reduce_mask_clear` uninit) is called
                if fam == "llk_pack_reduce_mask":
                    if ns.exu_init.get(exu) is None:
                        ns.exu_init[exu] = ActiveInit(
                            op_family="llk_pack",
                            init_call="llk_pack_init",
                            user_call=user_call,
                            line=line,
                            template_args=(),
                            args=(),
                        )
                    if spec.get("requires_uninit", False):
                        ns.pending_uninit[exu] = PendingUninit(
                            op_family=fam,
                            init_call=llk_name,
                            user_call=user_call,
                            line=line,
                            uninit_calls=tuple(spec.get("uninit_calls", [])),
                            source_ref=spec["source"],
                        )
                    next_states.append(ns)
                    continue

                # Check if an earlier uninit-requiring op on this EXU was never uninitialized
                # Note: `llk_pack` (`pack_tile` / `pack_block`) is legitimately called while
                # `llk_pack_reduce_mask` is active to pack the reduced tile out of DST;
                # `reduce_uninit` (`llk_pack_reduce_mask_clear`) is checked on subsequent
                # non-reduce init or at kernel exit.
                pend = ns.pending_uninit.get(exu)
                if (
                    pend is not None
                    and pend.op_family != fam
                    and not (
                        fam == "llk_pack" and pend.op_family == "llk_pack_reduce_mask"
                    )
                ):
                    diags.append(
                        Diagnostic(
                            rule="missing-uninit",
                            severity="error",
                            file=file_path,
                            line=line,
                            thread=thread,
                            exu=exu,
                            arch=arch,
                            call=user_call,
                            earlier_call=pend.user_call,
                            earlier_line=pend.line,
                            message=(
                                f"`{user_call}` (`{llk_name}`) is executed before `{pend.user_call}` "
                                f"was uninitialized with {', '.join(f'`{u}`' for u in pend.uninit_calls)}"
                            ),
                            source_ref=pend.source_ref,
                        )
                    )
                    ns.pending_uninit[exu] = None

                # Rule 1: `missing-init`
                if active is None:
                    diags.append(
                        Diagnostic(
                            rule="missing-init",
                            severity="error",
                            file=file_path,
                            line=line,
                            thread=thread,
                            exu=exu,
                            arch=arch,
                            call=user_call,
                            earlier_call=None,
                            earlier_line=None,
                            message=(
                                f"`{user_call}` (`{llk_name}`) executed on {thread}/{exu} with no preceding "
                                f"matching init ({', '.join(f'`{r}`' for r in sorted(req_inits))}) on some path"
                            ),
                            source_ref=spec["source"],
                        )
                    )
                    next_states.append(ns)
                    continue

                # Rule 3: `init-conflict` (explicit arch invalidation, e.g. #61146 on Blackhole)
                if active.invalidated_by_call and active.init_call in req_inits:
                    diags.append(
                        Diagnostic(
                            rule="init-conflict",
                            severity="error",
                            file=file_path,
                            line=line,
                            thread=thread,
                            exu=exu,
                            arch=arch,
                            call=user_call,
                            earlier_call=active.invalidated_by_call,
                            earlier_line=active.invalidated_at_line,
                            message=active.invalidated_reason
                            or (
                                f"`{active.user_call}` at line {active.line} was invalidated by "
                                f"`{active.invalidated_by_call}` at line {active.invalidated_at_line} before `{user_call}`"
                            ),
                            source_ref=active.invalidated_source or spec["source"],
                        )
                    )
                    next_states.append(ns)
                    continue

                # Rule 3: `init-conflict` (active init belongs to a different operation family)
                if active.init_call not in req_inits:
                    earlier_note = (
                        f" (which overwrote `{active.overwrote_call}` from line {active.overwrote_line})"
                        if active.overwrote_call
                        else ""
                    )
                    diags.append(
                        Diagnostic(
                            rule="init-conflict",
                            severity="error",
                            file=file_path,
                            line=line,
                            thread=thread,
                            exu=exu,
                            arch=arch,
                            call=user_call,
                            earlier_call=active.user_call,
                            earlier_line=active.line,
                            message=(
                                f"`{user_call}` (`{llk_name}`) requires {', '.join(f'`{r}`' for r in sorted(req_inits))}, "
                                f"but {thread}/{exu} is currently configured by `{active.user_call}` (`{active.init_call}`) "
                                f"at line {active.line}{earlier_note}"
                            ),
                            source_ref=spec["source"],
                        )
                    )
                    next_states.append(ns)
                    continue

                # Rule 2: `arg-mismatch` (check `reconfig_data_format` operand reversal before `matmul`, #39945)
                if (
                    fam == "llk_unpack_AB_matmul"
                    and ns.reconfig is not None
                    and len(arg_vals) >= 2
                ):
                    in0 = arg_vals[0]
                    in1 = arg_vals[1]
                    rc = ns.reconfig
                    if (
                        rc.srca.raw == in0.raw
                        and rc.srcb.raw == in1.raw
                        and in0.raw != in1.raw
                    ):
                        diags.append(
                            Diagnostic(
                                rule="arg-mismatch",
                                severity="error",
                                file=file_path,
                                line=line,
                                thread=thread,
                                exu=exu,
                                arch=arch,
                                call=user_call,
                                earlier_call=rc.call_name,
                                earlier_line=rc.line,
                                message=(
                                    f"`{rc.call_name}({rc.srca.raw}, {rc.srcb.raw})` configures SrcA={rc.srca.raw}, "
                                    f"SrcB={rc.srcb.raw}, but `{user_call}({in0.raw}, {in1.raw}, ...)` unpacks "
                                    f"SrcA={in1.raw}, SrcB={in0.raw} (reversed matmul operand order, #39945)"
                                ),
                                source_ref="docs/source/tt-metalium/tools/llk_asserts.rst:152 (#39945)",
                            )
                        )

                # Rule 2: `arg-mismatch` on `matching_args`
                self._check_matching_args(
                    spec=spec,
                    active=active,
                    exec_call=user_call,
                    llk_name=llk_name,
                    line=line,
                    tpl_vals=tpl_vals,
                    arg_vals=arg_vals,
                    arch=arch,
                    thread=thread,
                    exu=exu,
                    file_path=file_path,
                    diags=diags,
                )

                next_states.append(ns)

            return self._dedup_states(next_states)

        return states

    def _check_matching_args(
        self,
        spec: Dict[str, Any],
        active: ActiveInit,
        exec_call: str,
        llk_name: str,
        line: int,
        tpl_vals: List[Value],
        arg_vals: List[Value],
        arch: str,
        thread: str,
        exu: str,
        file_path: str,
        diags: List[Diagnostic],
    ) -> None:
        empty_env: Dict[str, Value] = {}
        for m_rule in spec.get("matching_args", []):
            field_name = m_rule["field"]
            init_val = self._extract_rule_arg(
                m_rule,
                prefix="init",
                tpl_vals=active.template_args,
                arg_vals=active.args,
                env=empty_env,
            )
            exec_val = self._extract_rule_arg(
                m_rule,
                prefix="exec",
                tpl_vals=tpl_vals,
                arg_vals=arg_vals,
                env=empty_env,
            )
            if init_val is None or exec_val is None:
                continue

            # Resolve `FullCtDim` default referencing `BlockCtDim`
            if init_val.resolved == "BlockCtDim" and len(active.template_args) > 0:
                init_val = active.template_args[0]
            if exec_val.resolved == "BlockCtDim" and len(tpl_vals) > 0:
                exec_val = tpl_vals[0]

            ok, unk_note = values_compatible(init_val, exec_val)
            if not ok:
                diags.append(
                    Diagnostic(
                        rule="arg-mismatch",
                        severity="error",
                        file=file_path,
                        line=line,
                        thread=thread,
                        exu=exu,
                        arch=arch,
                        call=exec_call,
                        earlier_call=active.user_call,
                        earlier_line=active.line,
                        message=(
                            f"compile-time argument `{field_name}` mismatch between `{active.user_call}` "
                            f"(`{field_name}={init_val.format_for_diag()}`) at line {active.line} and "
                            f"`{exec_call}` (`{field_name}={exec_val.format_for_diag()}`)"
                        ),
                        source_ref=m_rule.get("source", spec["source"]),
                        unknown_args=[unk_note] if unk_note else [],
                    )
                )

    def _extract_rule_arg(
        self,
        m_rule: Dict[str, Any],
        prefix: str,
        tpl_vals: Sequence[Value],
        arg_vals: Sequence[Value],
        env: Dict[str, Value],
    ) -> Optional[Value]:
        t_idx = m_rule.get(f"{prefix}_template_index")
        a_idx = m_rule.get(f"{prefix}_arg_index")
        default_str = m_rule.get(f"{prefix}_default")

        if t_idx is not None:
            if t_idx < len(tpl_vals):
                return tpl_vals[t_idx]
            if default_str is not None:
                return normalize_const(default_str, env)
            return None

        if a_idx is not None:
            if a_idx < len(arg_vals):
                return arg_vals[a_idx]
            if default_str is not None:
                return normalize_const(default_str, env)
            return None

        return None

    def _depends_on_vars(self, raw_args: Sequence[str], loop_vars: Set[str]) -> bool:
        if not loop_vars:
            return False
        for expr in raw_args:
            tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
            if tokens & loop_vars:
                return True
        return False


# ---------------------------------------------------------------------------
# Corpus Discovery (`ttnn/`, `models/`, `tests/`, `tt-train/`)
# ---------------------------------------------------------------------------

COMPUTE_INCLUDE_RE = re.compile(
    r'#\s*include\s+"(?:api/compute/|compute_kernel_api|llk_(?:unpack|math|pack)_)'
)
LLK_2_0_RE = re.compile(r'#\s*include\s+"api/compute/experimental/2_0/')
NOC_KERNEL_RE = re.compile(
    r'#\s*include\s+"(?:api/dataflow/dataflow_api\.h|dataflow_api\.h)"'
)


def discover_corpus_kernels(repo_root: str = REPO_ROOT) -> List[str]:
    """Find all compute kernels in `ttnn/`, `models/`, `tests/`, and `tt-train/`."""
    roots = ["ttnn", "models", "tests", "tt-train"]
    kernels: List[str] = []
    for sub in roots:
        base = os.path.join(repo_root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # Skip third_party or build directories
            dirnames[:] = [
                d
                for d in dirnames
                if d not in ("third_party", "build", ".git", "__pycache__", "quasar")
            ]
            for fn in sorted(filenames):
                if not fn.endswith(".cpp"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    head = open(full, "r", encoding="utf-8", errors="ignore").read(8192)
                except OSError:
                    continue
                if not COMPUTE_INCLUDE_RE.search(head):
                    continue
                if LLK_2_0_RE.search(head) or NOC_KERNEL_RE.search(head):
                    continue
                if "ARCH_QUASAR" in head and "TRISC_" not in head:
                    continue
                kernels.append(os.path.relpath(full, repo_root))
    return sorted(kernels)


def load_compile_commands(path: str) -> Dict[str, Tuple[List[str], Dict[str, str]]]:
    """Parse `compile_commands.json` into `{file_path: (include_dirs, defines)}`."""
    db = json.load(open(path, "r", encoding="utf-8"))
    result: Dict[str, Tuple[List[str], Dict[str, str]]] = {}
    for entry in db:
        f_path = entry.get("file", "")
        cmd = entry.get("command", "") or " ".join(entry.get("arguments", []))
        inc_dirs = re.findall(r"-I\s*([^\s]+)", cmd)
        defs: Dict[str, str] = {}
        for m in re.finditer(r"-D([A-Za-z_]\w*)(?:=([^\s]+))?", cmd):
            defs[m.group(1)] = m.group(2) if m.group(2) is not None else "1"
        result[os.path.normpath(f_path)] = (inc_dirs, defs)
    return result


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="*", help="Kernel `.cpp`/`.h` files to check.")
    ap.add_argument(
        "--rules",
        default=DEFAULT_RULES_PATH,
        help="Path to `compute_api_rules.yaml` (default: %(default)s).",
    )
    ap.add_argument(
        "--arch",
        choices=("wormhole_b0", "blackhole", "all"),
        default="all",
        help="Target architecture(s) to check (default: all).",
    )
    ap.add_argument(
        "--corpus",
        action="store_true",
        help="Scan all compute kernels in `ttnn/`, `models/`, `tests/`, and `tt-train/`.",
    )
    ap.add_argument(
        "--compile-commands",
        help="Optional path to `compile_commands.json` for `-I` and `-D` extraction.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON report instead of plain text.",
    )
    ap.add_argument(
        "--advisory",
        action="store_true",
        help="Always exit 0 (non-blocking CI / pre-commit mode).",
    )
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    rules = RuleTable(args.rules)
    checker = KernelOrderChecker(rules)
    arches = ARCHES if args.arch == "all" else (args.arch,)

    cc_map: Dict[str, Tuple[List[str], Dict[str, str]]] = {}
    if args.compile_commands and os.path.isfile(args.compile_commands):
        cc_map = load_compile_commands(args.compile_commands)

    target_files: List[str] = list(args.files)
    if args.corpus:
        target_files.extend(
            os.path.join(REPO_ROOT, rel) for rel in discover_corpus_kernels(REPO_ROOT)
        )

    # Filter out non-kernel files when invoked from pre-commit
    all_diags: List[Diagnostic] = []
    scanned_count = 0
    for f_path in target_files:
        if not os.path.isfile(f_path):
            continue
        scanned_count += 1
        norm_p = os.path.normpath(os.path.abspath(f_path))
        inc_dirs, defs = cc_map.get(norm_p, ([], {}))
        file_diags = checker.check_file(
            kernel_path=f_path,
            arches=arches,
            extra_include_dirs=inc_dirs,
            extra_defines=defs,
        )
        all_diags.extend(file_diags)

    elapsed = time.monotonic() - t0

    counts_by_rule: Dict[str, int] = {
        "missing-init": 0,
        "arg-mismatch": 0,
        "init-conflict": 0,
        "missing-uninit": 0,
        "deprecated-init": 0,
        "loop-invariant-init": 0,
    }
    for d in all_diags:
        counts_by_rule[d.rule] = counts_by_rule.get(d.rule, 0) + 1

    if args.json:
        payload = {
            "files_scanned": scanned_count,
            "arches": list(arches),
            "runtime_seconds": round(elapsed, 3),
            "counts_by_rule": counts_by_rule,
            "total_findings": len(all_diags),
            "findings": [dataclasses.asdict(d) for d in all_diags],
        }
        print(json.dumps(payload, indent=2))
    else:
        for d in all_diags:
            print(d.format_text())
        if args.corpus or all_diags:
            err_count = sum(1 for d in all_diags if d.severity == "error")
            warn_count = sum(1 for d in all_diags if d.severity == "warning")
            print(
                f"\nSummary: scanned {scanned_count} kernel(s) across {','.join(arches)} in {elapsed:.2f}s — "
                f"{err_count} error(s), {warn_count} warning(s) "
                f"({', '.join(f'{k}={v}' for k, v in counts_by_rule.items() if v > 0) or 'clean'})."
            )

    if args.advisory:
        return 0
    has_errors = any(d.severity == "error" for d in all_diags)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
