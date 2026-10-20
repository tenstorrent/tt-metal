# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, source-only C++ function extractor for the Blackhole LLK stack.

No compiler is involved. Each header is processed in four passes that keep byte
offsets (and therefore line numbers) intact:

1. comments and string/char literals are blanked;
2. preprocessor conditionals are evaluated with a tri-state evaluator that knows
   the Blackhole build (``ARCH_BLACKHOLE`` defined, ``ARCH_WORMHOLE*`` /
   ``ARCH_QUASAR`` undefined, ``__has_builtin(__builtin_rvtt_*)`` true). Branches
   that are decidably inactive are blanked, undecidable ones are kept and their
   condition text is remembered, and ``#ifdef TRISC_{UNPACK,MATH,PACK}`` regions
   are tagged with their thread. Directive lines are blanked; ``#define`` lines are
   collected as :class:`MacroDef`;
3. a brace-scope walk finds every function definition at namespace or class
   scope (``_llk_*``, ``llk_*``, ``ALWI``, ``inline``, ``sfpi_inline``,
   templates, specializations, overloads, out-of-class methods, constructors);
4. call sites are extracted from every body, with function-like macros defined
   in the scanned tree expanded (``SFPU_UNARY_CALL`` and friends), the thread
   implied by ``UNPACK((..))`` / ``MATH((..))`` / ``PACK((..))`` wrappers or by
   ``#ifdef TRISC_*``, the enclosing ``if`` condition, and member-call receivers.

Everything returned is ordered deterministically and carries repo-relative paths.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Tree layout
# --------------------------------------------------------------------------- #
LAYERS = ("compute_api", "llk_api", "llk_lib", "common", "sfpu")
THREADS = ("unpack", "math", "pack")

#: (repo-relative directory, layer) roots analysed for the Blackhole stack.
SOURCE_ROOTS: tuple[tuple[str, str], ...] = (
    ("tt_metal/hw/inc/api/compute", "compute_api"),
    ("tt_metal/hw/ckernels/blackhole/metal/llk_api", "llk_api"),
    ("tt_metal/hw/ckernels/blackhole/metal/llk_io", "llk_api"),
    ("tt_metal/tt-llk/tt_llk_blackhole/llk_lib", "llk_lib"),
    ("tt_metal/tt-llk/tt_llk_blackhole/common/inc", "common"),
)
HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".inc")

#: Files whose function-like macros are instruction encoders. Their macros are
#: never expanded: families match ``TTI_*`` / ``TT_*`` / ``TT_OP_*`` by name.
INSTRUCTION_MACRO_FILES = ("tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h",)

#: Thread wrapper macros of the compute API (handled as thread scopes).
THREAD_WRAPPERS = {"UNPACK": "unpack", "MATH": "math", "PACK": "pack"}
#: Preprocessor symbols that select a TRISC thread.
THREAD_DEFINES = {
    "TRISC_UNPACK": "unpack",
    "TRISC_MATH": "math",
    "TRISC_PACK": "pack",
    "UCK_CHLKC_UNPACK": "unpack",
    "UCK_CHLKC_MATH": "math",
    "UCK_CHLKC_PACK": "pack",
}
#: Macros never expanded at call sites (noise or handled elsewhere).
NO_EXPAND_MACROS = frozenset(
    {
        "ALWI",
        "UNPACK",
        "MATH",
        "PACK",
        "LLK_ASSERT",
        "LLK_SAN_FUNCTION",
        "SAN_HOOK",
        "DeviceZoneScopedN",
        "DeviceZoneScopedSumN1",
        "DeviceZoneScopedSumN2",
        "SYNC_WAIT",
        "WAYPOINT",
        "DPRINT",
        "static_assert",
    }
)

#: Default Blackhole preprocessor knowledge. ``True``/``int`` = defined with that
#: value, ``False`` = known undefined; anything absent is unknown.
BH_DEFINES: dict[str, bool | int] = {
    "ARCH_BLACKHOLE": 1,
    "ARCH_WORMHOLE": False,
    "ARCH_WORMHOLE_B0": False,
    "ARCH_GRAYSKULL": False,
    "ARCH_QUASAR": False,
}

_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "alignof",
        "alignas",
        "decltype",
        "static_assert",
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "catch",
        "defined",
        "__attribute__",
        "__has_builtin",
        "new",
        "delete",
        "throw",
        "typeid",
        "noexcept",
        "requires",
        "operator",
        "else",
        "do",
        "case",
        "goto",
        "typename",
        "template",
        "using",
        "namespace",
        "struct",
        "class",
        "union",
        "enum",
        "constexpr",
        "consteval",
        "inline",
        "static",
        "volatile",
        "const",
        "extern",
        "explicit",
        "virtual",
        "asm",
        "__asm__",
        "__volatile__",
    }
)
_PRIMITIVE_TYPES = frozenset(
    {
        "void",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "unsigned",
        "signed",
        "auto",
        "size_t",
        "uint",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uintptr_t",
        "std::size_t",
        "std::uint8_t",
        "std::uint16_t",
        "std::uint32_t",
        "std::uint64_t",
        "std::int8_t",
        "std::int16_t",
        "std::int32_t",
        "std::int64_t",
        "std::uintptr_t",
    }
)
_DECL_PREFIX_KEYWORDS = frozenset(
    {
        "return",
        "else",
        "case",
        "do",
        "throw",
        "new",
        "delete",
        "goto",
        "co_return",
        "co_yield",
    }
)
_IDENT = re.compile(r"[A-Za-z_]\w*")
_QUALIFIED_IDENT = re.compile(
    r"(?<![\w$])(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*~?[A-Za-z_]\w*"
)
_SPACE = re.compile(r"\s+")


def compact(value: str) -> str:
    """Collapse whitespace runs to one space and strip."""
    return _SPACE.sub(" ", value).strip()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Param:
    """One template or runtime parameter of a definition."""

    name: str
    type: str
    default: str | None
    kind: str  # "template" | "runtime"
    position: int
    variadic: bool = False

    def to_json(self) -> dict:
        return {
            "default": self.default,
            "kind": self.kind,
            "name": self.name,
            "position": self.position,
            "type": self.type,
            "variadic": self.variadic,
        }


@dataclass(frozen=True)
class CallSite:
    """A call (function, macro invocation, member call or constructor) in a body."""

    name: str  # final identifier, e.g. "llk_unpack_A"
    callee: str  # qualified text as written, e.g. "::ckernel::sfpu::calculate_exp"
    template_args: tuple[str, ...]
    args: tuple[str, ...]
    line: int  # absolute line in the file
    offset: int  # offset into FunctionDef.masked_body (invocation offset for macro-derived calls)
    thread: str | None  # thread implied by UNPACK/MATH/PACK wrapper or TRISC_* region
    condition: (
        str | None
    )  # enclosing if/else and undecidable #if conditions, joined by " && "
    via_macro: str | None = None  # outermost macro whose expansion produced this call
    is_member: bool = False
    receiver: str | None = None  # receiver expression text for member calls
    receiver_type: str | None = None  # inferred class name for member calls / ctor type
    is_ctor: bool = False  # ``Type var(args)`` declaration
    is_macro: bool = (
        False  # the callee is a function-like macro (instruction macros included)
    )

    @property
    def qualifier(self) -> str:
        text = self.callee.replace(" ", "").lstrip(":")
        return text.rsplit("::", 1)[0] if "::" in text else ""

    def to_json(self) -> dict:
        return {
            "args": list(self.args),
            "callee": self.callee,
            "condition": self.condition,
            "is_ctor": self.is_ctor,
            "is_macro": self.is_macro,
            "is_member": self.is_member,
            "line": self.line,
            "name": self.name,
            "receiver": self.receiver,
            "receiver_type": self.receiver_type,
            "template_args": list(self.template_args),
            "thread": self.thread,
            "via_macro": self.via_macro,
        }


@dataclass(frozen=True)
class MacroExpansion:
    """Text produced by expanding a function-like macro invoked in a body."""

    name: str
    line: int
    offset: int
    text: str
    thread: str | None
    condition: str | None


@dataclass(frozen=True)
class MacroDef:
    name: str
    params: tuple[str, ...] | None  # None for object-like macros
    body: str
    path: str
    line: int
    condition: str | None  # undecidable #if text guarding the definition


@dataclass
class FunctionDef:
    """One function definition found in a scanned source file."""

    name: str
    qualified_name: str
    namespace: str
    owner: str  # enclosing/out-of-class owner class path ("" for free functions)
    key: str
    path: str
    layer: str
    line: int  # line of the function name token
    start_line: int  # line where the declaration (incl. template prefix) starts
    end_line: int
    body_line: int  # line holding the opening brace
    signature: str
    params: tuple[Param, ...]
    template_params: tuple[Param, ...]
    specialization_args: tuple[str, ...]
    attributes: tuple[str, ...]
    body: str  # raw body text between the braces
    masked_body: str  # body with comments/strings/inactive-#if/directives blanked
    pp_condition: str | None  # undecidable #if text guarding the definition
    pp_thread: str | None  # thread of an enclosing #ifdef TRISC_* region
    thread_hint: str | None  # thread derived from name, path or #ifdef region
    fingerprint: str
    is_ctor: bool = False
    calls: tuple[CallSite, ...] = ()
    expansions: tuple[MacroExpansion, ...] = ()
    locals: tuple[tuple[str, str], ...] = ()  # (variable, type) for class-typed locals
    using: tuple[
        str, ...
    ] = ()  # namespaces made visible by file-level "using namespace"
    _body_offset: int = 0  # absolute offset of body start in the file
    _line_table: tuple[int, ...] = ()  # absolute offsets of line starts in the file
    _line_info: tuple = ()  # per-line (thread, condition) from the preprocessor
    _blocks: list | None = None  # cached if/else control blocks of masked_body

    # ---- helpers -------------------------------------------------------- #
    @property
    def all_params(self) -> tuple[Param, ...]:
        return self.template_params + self.params

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.all_params if p.name)

    @property
    def is_template(self) -> bool:
        return bool(self.template_params) or bool(self.specialization_args)

    @property
    def loc(self) -> str:
        return f"{self.path}:{self.line}"

    def line_at(self, offset: int) -> int:
        """Absolute file line for an offset into ``masked_body``/``body``."""
        return self.body_line + self.masked_body.count("\n", 0, max(0, offset))

    def evidence_at(self, offset: int) -> str:
        return f"{self.path}:{self.line_at(offset)}"

    def pp_info_at_line(self, line: int) -> tuple[str | None, str | None]:
        if 1 <= line <= len(self._line_info):
            return self._line_info[line - 1]
        return (None, None)

    def condition_at(self, offset: int) -> str | None:
        """Enclosing ``if``/``else`` conditions plus undecidable ``#if`` text."""
        parts: list[str] = []
        _, pp = self.pp_info_at_line(self.line_at(offset))
        if pp and pp != self.pp_condition:
            parts.append(pp)
        control = condition_at(_control_blocks_cached(self), offset)
        if control:
            parts.append(control)
        return " && ".join(parts) if parts else None

    def to_json(self, *, include_calls: bool = True) -> dict:
        data = {
            "attributes": list(self.attributes),
            "end_line": self.end_line,
            "fingerprint": self.fingerprint,
            "is_ctor": self.is_ctor,
            "key": self.key,
            "layer": self.layer,
            "line": self.line,
            "name": self.name,
            "namespace": self.namespace,
            "owner": self.owner,
            "params": [p.to_json() for p in self.params],
            "path": self.path,
            "pp_condition": self.pp_condition,
            "qualified_name": self.qualified_name,
            "signature": self.signature,
            "specialization_args": list(self.specialization_args),
            "template_params": [p.to_json() for p in self.template_params],
            "thread_hint": self.thread_hint,
        }
        if include_calls:
            data["calls"] = [c.to_json() for c in self.calls]
        return data


def _control_blocks_cached(fn: FunctionDef) -> list[dict]:
    if fn._blocks is None:
        fn._blocks = control_blocks(fn.masked_body, fn.body)
    return fn._blocks


@dataclass
class SourceFile:
    path: str
    layer: str
    sha256: str
    functions: list[FunctionDef] = field(default_factory=list)
    macros: list[MacroDef] = field(default_factory=list)
    constants: dict[str, str] = field(default_factory=dict)
    globals: list[str] = field(default_factory=list)
    using: list[str] = field(default_factory=list)


@dataclass
class ScanIndex:
    """All functions, macros and constants found in the analysed Blackhole tree."""

    files: list[SourceFile]
    functions: list[FunctionDef]
    macros: dict[str, MacroDef]
    constants: dict[str, str]
    globals: tuple[str, ...]
    fingerprint: str
    _by_name: dict[str, list[FunctionDef]] = field(default_factory=dict)
    _by_key: dict[str, FunctionDef] = field(default_factory=dict)

    def by_name(self, name: str) -> list[FunctionDef]:
        return self._by_name.get(name, [])

    def get(self, key: str) -> FunctionDef | None:
        return self._by_key.get(key)

    def layer_counts(self) -> dict[str, int]:
        counts = {layer: 0 for layer in LAYERS}
        for fn in self.functions:
            counts[fn.layer] = counts.get(fn.layer, 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# Lexical masking
# --------------------------------------------------------------------------- #
def _blank(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] != "\n":
            chars[index] = " "


def _is_quote_start(text: str, position: int) -> bool:
    char = text[position]
    if char == '"':
        return True
    if char != "'":
        return False
    before = text[position - 1] if position > 0 else ""
    after = text[position + 1] if position + 1 < len(text) else ""
    return not (before.isalnum() and after.isalnum())  # digit separator 0x1'0000


_LEXICAL = re.compile(r"//|/\*|\"|'")


def mask_comments_and_strings(text: str) -> str:
    """Blank comments and string/char literals, preserving offsets and newlines."""
    chars = list(text)
    position = 0
    length = len(text)
    while True:
        match = _LEXICAL.search(text, position)
        if match is None:
            break
        start = match.start()
        token = match.group(0)
        if token == "//":
            end = text.find("\n", start + 2)
            end = length if end == -1 else end
            _blank(chars, start, end)
            position = end
        elif token == "/*":
            end = text.find("*/", start + 2)
            end = length if end == -1 else end + 2
            _blank(chars, start, end)
            position = end
        else:
            if not _is_quote_start(text, start):
                position = start + 1
                continue
            end = start + 1
            while end < length:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == token or text[end] == "\n":
                    end += 1
                    break
                end += 1
            _blank(chars, start, min(end, length))
            position = end
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Tri-state expression evaluation (preprocessor and condition folding)
# --------------------------------------------------------------------------- #
_EXPR_TOKEN = re.compile(
    r"\s*(?:(?P<num>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*"
    r"|(?P<id>(?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)"
    r"|(?P<op>\|\||&&|==|!=|<=|>=|<<|>>|[!()<>+\-*/%&|^~?:,]))"
)


class _Unknown:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "UNKNOWN"


UNKNOWN = _Unknown()


def _tokenize_expression(expression: str) -> list[tuple[str, str]] | None:
    tokens: list[tuple[str, str]] = []
    position = 0
    expression = expression.rstrip()
    while position < len(expression):
        match = _EXPR_TOKEN.match(expression, position)
        if match is None or match.end() == position:
            return None
        position = match.end()
        for kind in ("num", "id", "op"):
            value = match.group(kind)
            if value is not None:
                tokens.append((kind, compact(value).replace(" ", "")))
                break
    return tokens


class _Evaluator:
    """Recursive-descent tri-state evaluator.

    Values are Python ints/bools, ``UNKNOWN``, or ``("sym", text)`` for symbolic
    qualified names that can still be compared for (in)equality.
    """

    def __init__(self, tokens, resolve_identifier, resolve_defined, resolve_builtin):
        self.tokens = tokens
        self.index = 0
        self.resolve_identifier = resolve_identifier
        self.resolve_defined = resolve_defined
        self.resolve_builtin = resolve_builtin

    def peek(self):
        return (
            self.tokens[self.index] if self.index < len(self.tokens) else (None, None)
        )

    def take(self):
        token = self.peek()
        self.index += 1
        return token

    def parse(self):
        value = self.ternary()
        if self.index != len(self.tokens):
            raise ValueError("trailing tokens")
        return value

    def ternary(self):
        condition = self.logical_or()
        if self.peek() == ("op", "?"):
            self.take()
            left = self.ternary()
            if self.take() != ("op", ":"):
                raise ValueError("bad ternary")
            right = self.ternary()
            truth = _truth(condition)
            if truth is None:
                return left if left == right and left is not UNKNOWN else UNKNOWN
            return left if truth else right
        return condition

    def logical_or(self):
        value = self.logical_and()
        while self.peek() == ("op", "||"):
            self.take()
            right = self.logical_and()
            a, b = _truth(value), _truth(right)
            if a is True or b is True:
                value = True
            elif a is False and b is False:
                value = False
            else:
                value = UNKNOWN
        return value

    def logical_and(self):
        value = self.bit_or()
        while self.peek() == ("op", "&&"):
            self.take()
            right = self.bit_or()
            a, b = _truth(value), _truth(right)
            if a is False or b is False:
                value = False
            elif a is True and b is True:
                value = True
            else:
                value = UNKNOWN
        return value

    def bit_or(self):
        value = self.bit_xor()
        while self.peek() == ("op", "|"):
            self.take()
            value = _arith(value, self.bit_xor(), lambda a, b: a | b)
        return value

    def bit_xor(self):
        value = self.bit_and()
        while self.peek() == ("op", "^"):
            self.take()
            value = _arith(value, self.bit_and(), lambda a, b: a ^ b)
        return value

    def bit_and(self):
        value = self.equality()
        while self.peek() == ("op", "&"):
            self.take()
            value = _arith(value, self.equality(), lambda a, b: a & b)
        return value

    def equality(self):
        value = self.relational()
        while self.peek() in (("op", "=="), ("op", "!=")):
            operator = self.take()[1]
            right = self.relational()
            equal = _equal(value, right)
            if equal is None:
                value = UNKNOWN
            else:
                value = equal if operator == "==" else not equal
        return value

    def relational(self):
        value = self.shift()
        while self.peek() in (("op", "<"), ("op", ">"), ("op", "<="), ("op", ">=")):
            operator = self.take()[1]
            right = self.shift()
            ops = {
                "<": lambda a, b: a < b,
                ">": lambda a, b: a > b,
                "<=": lambda a, b: a <= b,
                ">=": lambda a, b: a >= b,
            }
            value = _arith(value, right, ops[operator], boolean=True)
        return value

    def shift(self):
        value = self.additive()
        while self.peek() in (("op", "<<"), ("op", ">>")):
            operator = self.take()[1]
            right = self.additive()
            value = _arith(
                value,
                right,
                (lambda a, b: a << b) if operator == "<<" else (lambda a, b: a >> b),
            )
        return value

    def additive(self):
        value = self.multiplicative()
        while self.peek() in (("op", "+"), ("op", "-")):
            operator = self.take()[1]
            right = self.multiplicative()
            value = _arith(
                value,
                right,
                (lambda a, b: a + b) if operator == "+" else (lambda a, b: a - b),
            )
        return value

    def multiplicative(self):
        value = self.unary()
        while self.peek() in (("op", "*"), ("op", "/"), ("op", "%")):
            operator = self.take()[1]
            right = self.unary()
            if operator == "*":
                value = _arith(value, right, lambda a, b: a * b)
            else:
                value = _arith(
                    value,
                    right,
                    (
                        (lambda a, b: a // b if b else None)
                        if operator == "/"
                        else (lambda a, b: a % b if b else None)
                    ),
                )
        return value

    def unary(self):
        token = self.peek()
        if token == ("op", "!"):
            self.take()
            truth = _truth(self.unary())
            return UNKNOWN if truth is None else (not truth)
        if token == ("op", "-"):
            self.take()
            return _arith(0, self.unary(), lambda a, b: a - b)
        if token == ("op", "+"):
            self.take()
            return self.unary()
        if token == ("op", "~"):
            self.take()
            return _arith(0, self.unary(), lambda a, b: ~b)
        return self.primary()

    def primary(self):
        kind, value = self.take()
        if kind == "num":
            return _parse_int(value)
        if kind == "op" and value == "(":
            inner = self.ternary()
            if self.take() != ("op", ")"):
                raise ValueError("unbalanced")
            return inner
        if kind == "id":
            if value == "defined":
                parenthesized = self.peek() == ("op", "(")
                if parenthesized:
                    self.take()
                name_kind, name = self.take()
                if name_kind != "id":
                    raise ValueError("bad defined")
                if parenthesized and self.take() != ("op", ")"):
                    raise ValueError("bad defined")
                return self.resolve_defined(name)
            if value == "__has_builtin" or value == "__has_include":
                if self.take() != ("op", "("):
                    raise ValueError("bad builtin")
                depth = 1
                parts = []
                while depth:
                    token = self.take()
                    if token == (None, None):
                        raise ValueError("bad builtin")
                    if token == ("op", "("):
                        depth += 1
                    elif token == ("op", ")"):
                        depth -= 1
                        if not depth:
                            break
                    parts.append(token[1])
                return self.resolve_builtin(value, "".join(parts))
            if value in ("true", "false"):
                return value == "true"
            if self.peek() == ("op", "("):
                # A function-like construct we cannot evaluate: skip its arguments.
                self.take()
                depth = 1
                while depth:
                    token = self.take()
                    if token == (None, None):
                        raise ValueError("unbalanced call")
                    if token == ("op", "("):
                        depth += 1
                    elif token == ("op", ")"):
                        depth -= 1
                return UNKNOWN
            return self.resolve_identifier(value)
        raise ValueError(f"unexpected token {value!r}")


def _parse_int(text: str):
    text = text.rstrip("uUlL")
    try:
        if text[:2] in ("0x", "0X"):
            return int(text, 16)
        if text[:2] in ("0b", "0B"):
            return int(text[2:], 2)
        if len(text) > 1 and text.startswith("0"):
            return int(text, 8)
        return int(text)
    except ValueError:
        return UNKNOWN


def _truth(value) -> bool | None:
    if value is UNKNOWN or isinstance(value, tuple):
        return None
    return bool(value)


def _equal(left, right) -> bool | None:
    if left is UNKNOWN or right is UNKNOWN:
        return None
    if isinstance(left, tuple) or isinstance(right, tuple):
        if isinstance(left, tuple) and isinstance(right, tuple):
            if left[1] == right[1]:
                return True
            left_scope = left[1].rsplit("::", 1)[0] if "::" in left[1] else None
            right_scope = right[1].rsplit("::", 1)[0] if "::" in right[1] else None
            if left_scope and left_scope == right_scope:
                return False  # two different enumerators of the same scope
        return None
    return int(left) == int(right)


def _arith(left, right, operation, *, boolean: bool = False):
    if (
        left is UNKNOWN
        or right is UNKNOWN
        or isinstance(left, tuple)
        or isinstance(right, tuple)
    ):
        return UNKNOWN
    try:
        result = operation(int(left), int(right))
    except (ArithmeticError, ValueError):
        return UNKNOWN
    if result is None:
        return UNKNOWN
    return bool(result) if boolean else result


def evaluate(
    expression: str,
    *,
    identifiers: dict[str, object] | None = None,
    defines: dict[str, bool | int] | None = None,
    has_rvtt_builtins: bool = True,
    symbolic_qualified: bool = True,
):
    """Evaluate a C/C++ (preprocessor or ``if constexpr``) expression tri-state.

    Returns ``True``/``False``/an int when decidable, otherwise ``UNKNOWN``.
    Qualified names (``a::B``) are symbolic: equal to themselves and unequal to
    another enumerator of the same scope.
    """
    tokens = _tokenize_expression(expression)
    if tokens is None:
        return UNKNOWN
    identifiers = identifiers or {}
    defines = BH_DEFINES if defines is None else defines

    def resolve_identifier(name: str):
        if name in identifiers:
            return identifiers[name]
        if name in defines:
            value = defines[name]
            if value is False:
                return 0
            return 1 if value is True else value
        if symbolic_qualified and "::" in name:
            return ("sym", name.lstrip(":"))
        return UNKNOWN

    def resolve_defined(name: str):
        if name in defines:
            return defines[name] is not False
        return UNKNOWN

    def resolve_builtin(kind: str, argument: str):
        if kind == "__has_builtin" and argument.startswith("__builtin_rvtt_"):
            return bool(has_rvtt_builtins)
        return UNKNOWN

    try:
        return _Evaluator(
            tokens, resolve_identifier, resolve_defined, resolve_builtin
        ).parse()
    except (ValueError, IndexError, TypeError):
        return UNKNOWN


def truth(expression: str | None, **kwargs) -> bool | None:
    """``True``/``False`` when the expression is decidable, ``None`` otherwise."""
    if expression is None or not expression.strip():
        return True
    return _truth(evaluate(expression, **kwargs))


# --------------------------------------------------------------------------- #
# Preprocessor pass
# --------------------------------------------------------------------------- #
_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(\w+)(.*)$")
_DEFINE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\(([^)]*)\))?(.*)$", re.S
)
_THREAD_IF = re.compile(r"^\s*(?:defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+))\s*$")
_HEADER_GUARD = re.compile(r"^\w+_(?:H|HPP|H_|HPP_)$")


@dataclass
class _PPResult:
    text: str  # masked text with inactive branches and directives blanked
    line_info: tuple  # per line: (thread, condition)
    macros: list[MacroDef]


def preprocess(
    masked: str,
    path: str,
    defines: dict[str, bool | int] | None = None,
    has_rvtt_builtins: bool = True,
) -> _PPResult:
    """Evaluate conditionals on comment/string-masked text (offset preserving)."""
    defines = BH_DEFINES if defines is None else defines
    lines = masked.split("\n")
    out_lines = list(lines)
    info: list[tuple[str | None, str | None]] = []
    macros: list[MacroDef] = []
    # frame: dict(parent_active, taken(True if some branch decidably true),
    #             all_false(bool), active, thread, cond)
    stack: list[dict] = []
    active = True
    index = 0
    count = len(lines)

    def current_thread() -> str | None:
        for frame in reversed(stack):
            if frame["thread"]:
                return frame["thread"]
        return None

    def current_condition() -> str | None:
        parts = [frame["cond"] for frame in stack if frame["cond"]]
        return " && ".join(parts) if parts else None

    def branch(
        expression: str, kind: str
    ) -> tuple[bool | None, str | None, str | None]:
        """Return (truth, thread, condition text) for an #if-like directive."""
        expression = expression.strip()
        if kind in ("ifdef", "ifndef"):
            name = expression.split()[0] if expression.split() else ""
            text = f"defined({name})"
            if kind == "ifndef":
                if name in THREAD_DEFINES:
                    return None, None, None
                if _HEADER_GUARD.match(name):
                    return True, None, None
                value = evaluate(
                    f"!{text}", defines=defines, has_rvtt_builtins=has_rvtt_builtins
                )
                return (
                    _truth(value),
                    None,
                    (f"!defined({name})" if _truth(value) is None else None),
                )
            if name in THREAD_DEFINES:
                return None, THREAD_DEFINES[name], None
            value = evaluate(text, defines=defines, has_rvtt_builtins=has_rvtt_builtins)
            return _truth(value), None, (text if _truth(value) is None else None)
        match = _THREAD_IF.match(expression)
        if match:
            name = match.group(1) or match.group(2)
            if name in THREAD_DEFINES:
                return None, THREAD_DEFINES[name], None
        value = evaluate(
            expression, defines=defines, has_rvtt_builtins=has_rvtt_builtins
        )
        result = _truth(value)
        return result, None, (compact(expression) if result is None else None)

    while index < count:
        line = lines[index]
        directive = _DIRECTIVE.match(line)
        if directive is None:
            if not active:
                out_lines[index] = " " * len(line)
            info.append(
                (current_thread(), current_condition()) if active else (None, None)
            )
            index += 1
            continue
        # Join continuation lines of this directive.
        span = [index]
        full = line
        while full.endswith("\\") and span[-1] + 1 < count:
            full = full[:-1] + " " + lines[span[-1] + 1]
            span.append(span[-1] + 1)
        kind = directive.group(1)
        tail = full[directive.end(1) :]
        if kind in ("if", "ifdef", "ifndef"):
            result, thread, cond = branch(tail, kind)
            frame = {
                "parent_active": active,
                "taken": result is True,
                "all_false": result is False,
                "thread": thread,
                "cond": cond,
                "prior": [cond] if cond else [],
                "prior_thread": thread,
            }
            stack.append(frame)
            active = active and result is not False
        elif kind == "elif" and stack:
            frame = stack[-1]
            if frame["taken"]:
                active = False
                frame["thread"], frame["cond"] = None, None
            else:
                result, thread, cond = branch(tail, "if")
                negated = [f"!({c})" for c in frame["prior"]]
                frame["thread"] = thread
                frame["cond"] = " && ".join(negated + ([cond] if cond else [])) or None
                frame["prior"] = frame["prior"] + ([cond] if cond else [])
                frame["taken"] = result is True and frame["all_false"]
                frame["all_false"] = frame["all_false"] and result is False
                active = frame["parent_active"] and result is not False
        elif kind == "else" and stack:
            frame = stack[-1]
            if frame["taken"]:
                active = False
            else:
                active = frame["parent_active"]
            frame["thread"] = None
            frame["cond"] = " && ".join(f"!({c})" for c in frame["prior"]) or None
            frame["taken"] = True
        elif kind == "endif" and stack:
            active = stack.pop()["parent_active"]
        elif kind == "define" and active:
            match = _DEFINE.match(full)
            if match:
                params = None
                if match.group(2) is not None and full[match.end(1)] == "(":
                    params = tuple(
                        compact(p) for p in match.group(3).split(",") if compact(p)
                    )
                    body = match.group(4)
                else:
                    body = (
                        (match.group(2) or "") + match.group(4)
                        if match.group(2) is not None
                        else match.group(4)
                    )
                macros.append(
                    MacroDef(
                        name=match.group(1),
                        params=params,
                        body=compact(body),
                        path=path,
                        line=index + 1,
                        condition=current_condition(),
                    )
                )
        for position in span:
            out_lines[position] = " " * len(lines[position])
            info.append((None, None))
        index = span[-1] + 1
    return _PPResult(text="\n".join(out_lines), line_info=tuple(info), macros=macros)


# --------------------------------------------------------------------------- #
# Delimiter helpers
# --------------------------------------------------------------------------- #
_PAREN = re.compile(r"[()]")
_BRACE = re.compile(r"[{}]")
_BRACKET = re.compile(r"[\[\]]")


def match_close(text: str, opening: int, pair: str = "()") -> int | None:
    """Offset of the delimiter closing ``text[opening]`` (``()``, ``{}`` or ``[]``)."""
    pattern = {"()": _PAREN, "{}": _BRACE, "[]": _BRACKET}[pair]
    depth = 0
    for match in pattern.finditer(text, opening):
        if match.group(0) == pair[0]:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return match.start()
    return None


def _is_template_open(text: str, position: int) -> bool:
    """Heuristic: ``<`` opens a template list when glued to an identifier."""
    if position + 1 < len(text) and text[position + 1] in "<=":
        return False
    if position > 0 and text[position - 1] == "<":
        return False
    before = position - 1
    while before >= 0 and text[before] == " ":
        before -= 1
    if before < 0:
        return False
    if text[before].isalnum() or text[before] == "_":
        # "template <" and "name<" both count; "a < b" (spaced) does not unless after template.
        if before == position - 1:
            return True
        word = re.search(r"[A-Za-z_]\w*$", text[: before + 1])
        return word is not None and word.group(0) in ("template", "operator")
    return False


def match_angle(text: str, opening: int) -> int | None:
    """Offset of the ``>`` closing the template list opened at ``text[opening]``."""
    depth = 0
    position = opening
    length = len(text)
    while position < length:
        char = text[position]
        if char == "<":
            if position == opening or _is_template_open(text, position):
                depth += 1
        elif char == ">":
            if position > 0 and text[position - 1] == "-":
                position += 1
                continue
            depth -= 1
            if depth == 0:
                return position
        elif char == "(":
            close = match_close(text, position, "()")
            if close is None:
                return None
            position = close
        elif char in ";{}":
            return None
        position += 1
    return None


def split_top_level(value: str) -> list[str]:
    """Split on commas outside (), [], {} and template <> lists."""
    parts: list[str] = []
    start = 0
    depth = 0
    angle = 0
    position = 0
    length = len(value)
    while position < length:
        char = value[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "<" and _is_template_open(value, position):
            angle += 1
        elif (
            char == ">" and angle and not (position > 0 and value[position - 1] == "-")
        ):
            angle -= 1
        elif char == "," and depth == 0 and angle == 0:
            parts.append(value[start:position])
            start = position + 1
        position += 1
    parts.append(value[start:])
    return parts


def split_args(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(compact(part) for part in split_top_level(value))


def statement_end(text: str, position: int) -> int:
    depth = 0
    length = len(text)
    while position < length:
        char = text[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return position
            depth -= 1
        elif char == ";" and depth == 0:
            return position
        position += 1
    return length


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def _split_default(declaration: str) -> tuple[str, str | None]:
    depth = 0
    angle = 0
    for position, char in enumerate(declaration):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "<" and _is_template_open(declaration, position):
            angle += 1
        elif char == ">" and angle:
            angle -= 1
        elif char == "=" and depth == 0 and angle == 0:
            following = declaration[position + 1 : position + 2]
            preceding = declaration[position - 1 : position] if position else ""
            if following != "=" and preceding not in ("=", "!", "<", ">"):
                return declaration[:position], compact(declaration[position + 1 :])
    return declaration, None


_TYPE_WORDS = frozenset(
    {
        "const",
        "volatile",
        "typename",
        "class",
        "struct",
        "enum",
        "unsigned",
        "signed",
        "long",
        "short",
        "register",
    }
)


def parse_params(value: str | None, kind: str) -> tuple[Param, ...]:
    """Parse a template or runtime parameter list into :class:`Param` records."""
    if value is None:
        return ()
    stripped = compact(value)
    if not stripped or stripped == "void":
        return ()
    result: list[Param] = []
    for position, part in enumerate(split_top_level(value)):
        declaration, default = _split_default(part)
        declaration = compact(declaration)
        variadic = "..." in declaration
        name = ""
        pointer = re.search(r"\(\s*[*&]+\s*([A-Za-z_]\w*)\s*\)\s*\(", declaration)
        if pointer:
            name = pointer.group(1)
        else:
            cleaned = (
                re.sub(r"\[[^\]]*\]\s*$", "", declaration).replace("...", " ").strip()
            )
            cleaned = re.sub(r"\[\[[^\]]*\]\]", " ", cleaned).strip()
            match = re.search(r"([A-Za-z_]\w*)\s*$", cleaned)
            if match:
                candidate = match.group(1)
                before = cleaned[: match.start()].rstrip()
                head_words = [
                    w
                    for w in re.findall(r"[A-Za-z_]\w*", before)
                    if w not in _TYPE_WORDS
                ]
                qualified = before.endswith("::")
                type_only = candidate in _PRIMITIVE_TYPES or candidate in _TYPE_WORDS
                if (
                    kind == "template"
                    and before.split()
                    and before.split()[-1] in ("typename", "class")
                    and not qualified
                ):
                    name = candidate
                elif (
                    before
                    and not qualified
                    and not type_only
                    and (head_words or before.endswith(("*", "&", ">")))
                ):
                    name = candidate
        type_text = declaration
        if name:
            type_text = compact(
                re.sub(
                    rf"\b{re.escape(name)}\b(?!.*\b{re.escape(name)}\b)",
                    "",
                    declaration,
                )
            )
        result.append(
            Param(
                name=name,
                type=type_text,
                default=default,
                kind=kind,
                position=position,
                variadic=variadic,
            )
        )
    return tuple(result)


# --------------------------------------------------------------------------- #
# Definition discovery
# --------------------------------------------------------------------------- #
_ATTRIBUTE = re.compile(r"__attribute__\s*\(\(")
_STD_ATTRIBUTE = re.compile(r"\[\[")
_ACCESS = re.compile(r"^\s*(?:public|private|protected)\s*:")
_NAMESPACE_HEADER = re.compile(
    r"^\s*(?:inline\s+)?namespace\b\s*((?:[A-Za-z_]\w*\s*(?:::\s*)?)*)\s*$"
)
_CLASS_HEADER = re.compile(
    r"\b(struct|class|union)\s+(?:alignas\s*\([^)]*\)\s*)?(?:[A-Za-z_]\w*\s+)*?([A-Za-z_]\w*)\s*(?:<.*>)?\s*(?:final\s*)?(?::[^{]*)?$",
    re.S,
)
_ENUM_HEADER = re.compile(r"\benum\b")
_EXTERN_HEADER = re.compile(r"^\s*extern\s*$")
_CONTROL_WORDS = re.compile(r"\b(return|if|else|for|while|switch|case|throw|do)\b")
_POST_PARAMS = re.compile(
    r"^\s*(?:(?:const|volatile|noexcept(?:\s*\([^)]*\))?|override|final|&&|&|"
    r"__attribute__\s*\(\([^)]*\)\)|->\s*[\w:<>,\s*&]+|requires\b.*)\s*)*$",
    re.S,
)


def _strip_attributes(text: str) -> tuple[str, list[str]]:
    """Blank ``__attribute__((..))`` and ``[[..]]`` (offset preserving)."""
    chars = list(text)
    attributes: list[str] = []
    for pattern, pair in ((_ATTRIBUTE, "()"), (_STD_ATTRIBUTE, "[]")):
        for match in pattern.finditer(text):
            opening = match.end() - 2
            close = match_close(text, opening, pair)
            if close is None:
                continue
            attributes.extend(
                word
                for word in re.findall(r"[A-Za-z_]\w*", text[match.end() : close])
                if word not in ("gnu", "maybe_unused", "nodiscard")
            )
            _blank(chars, match.start(), close + 1)
    return "".join(chars), attributes


def _template_prefixes(header: str) -> tuple[int, str | None]:
    """Return (offset after the last leading ``template <..>``, its parameter text)."""
    position = 0
    params = None
    while True:
        match = re.compile(r"\s*template\s*<").match(header, position)
        if match is None:
            return position, params
        opening = match.end() - 1
        close = match_angle(header, opening)
        if close is None:
            return position, params
        params = header[opening + 1 : close]
        position = close + 1


def _parse_function_header(header: str, scope_class: str | None) -> dict | None:
    """Recognize a definition header (text between the previous boundary and ``{``).

    Every top-level ``(`` is tried as the parameter list, so attribute-like macro
    prefixes (``RECONFIG_DF_DEPRECATED("..") ALWI void f(..)``) are skipped.
    """
    stripped, attributes = _strip_attributes(header)
    access = _ACCESS.match(stripped)
    base = access.end() if access else 0
    after_template, template_params = _template_prefixes(stripped[base:])
    after_template += base
    declarator = stripped[after_template:]
    position = 0
    length = len(declarator)
    while position < length:
        char = declarator[position]
        if char == "<" and _is_template_open(declarator, position):
            close = match_angle(declarator, position)
            if close is None:
                return None
            position = close + 1
            continue
        if char in "=;":
            return None
        if char == "(":
            close = match_close(declarator, position, "()")
            if close is None:
                return None
            parsed = _parse_declarator(declarator, position, close, scope_class)
            if parsed is not None:
                parsed["template"] = template_params
                parsed["name_offset"] += after_template
                parsed["attributes"] = attributes
                return parsed
            position = close + 1
            continue
        position += 1
    main = re.search(r"\bvoid\s+(MAIN)\s*$", declarator)
    if main:
        return {
            "name": "MAIN",
            "owner": "",
            "spec": None,
            "template": template_params,
            "params": "",
            "name_offset": after_template + main.start(1),
            "prefix": "void",
            "attributes": attributes,
            "is_ctor": False,
        }
    return None


def _parse_declarator(
    declarator: str, paren: int, close: int, scope_class: str | None
) -> dict | None:
    before = declarator[:paren].rstrip()
    spec = None
    if before.endswith(">"):
        depth = 0
        cursor = len(before) - 1
        while cursor >= 0:
            if before[cursor] == ">":
                depth += 1
            elif before[cursor] == "<":
                depth -= 1
                if depth == 0:
                    break
            cursor -= 1
        if cursor < 0:
            return None
        spec = before[cursor + 1 : -1]
        before = before[:cursor].rstrip()
    match = re.search(r"((?:[A-Za-z_]\w*\s*::\s*)*)(~?[A-Za-z_]\w*)$", before)
    if match is None:
        return None
    name = match.group(2)
    owner = compact(match.group(1)).replace(" ", "").rstrip(":")
    if name in _KEYWORDS or name.startswith("operator"):
        return None
    prefix = compact(before[: match.start()])
    if _CONTROL_WORDS.search(prefix) or "=" in prefix or prefix.startswith((".", "->")):
        return None
    if re.search(r"\boperator\b", prefix):
        return None
    tail = declarator[close + 1 :]
    owner_last = owner.rsplit("::", 1)[-1] if owner else (scope_class or "")
    is_ctor = bool(owner_last) and name.lstrip("~") == owner_last
    if not _POST_PARAMS.match(tail):
        if not (is_ctor and tail.lstrip().startswith(":")):
            return None
    if not prefix or not re.search(r"[A-Za-z_]", prefix):
        if not is_ctor:
            return None
    if prefix and not is_ctor:
        if not re.search(r"[\w>*&]\s*$", prefix):
            return None
        last_word = re.findall(r"[A-Za-z_]\w*", prefix)
        if last_word and last_word[-1] in _DECL_PREFIX_KEYWORDS:
            return None
    return {
        "name": name,
        "owner": owner,
        "spec": spec,
        "params": declarator[paren + 1 : close],
        "name_offset": match.start(2),
        "prefix": prefix,
        "is_ctor": is_ctor,
    }


def _line_starts(text: str) -> tuple[int, ...]:
    starts = [0]
    find = text.find
    position = find("\n")
    while position != -1:
        starts.append(position + 1)
        position = find("\n", position + 1)
    return tuple(starts)


def _line_of(starts: tuple[int, ...], offset: int) -> int:
    import bisect

    return bisect.bisect_right(starts, offset)


_CONSTEXPR_INT = re.compile(
    r"\b(?:static\s+)?constexpr\s+(?:static\s+)?(?:inline\s+)?(?:const\s+)?"
    r"(?:std::)?(?:u?int(?:8|16|32|64)_t|unsigned(?:\s+int)?|int|bool|size_t|uint|auto)\s+"
    r"([A-Za-z_]\w*)\s*(?:=\s*([^;{}]+)|\{([^;{}]*)\})\s*;"
)
_GLOBAL_DECL = re.compile(
    r"^\s*(?:extern\s+|static\s+|inline\s+|volatile\s+|thread_local\s+)*(?:std::)?"
    r"(?:u?int(?:8|16|32|64)_t|unsigned(?:\s+int)?|int|bool|uint|DataFormat|DstSync|[A-Za-z_]\w*_t)\s+"
    r"(?:volatile\s+)?([A-Za-z_]\w*)\s*(?:=[^;]*|\{[^}]*\})?\s*$"
)


def _scan_scopes(text: str, starts: tuple[int, ...]):
    """Yield definitions and namespace-scope declarations from masked text.

    Yields ("function", header_start, open_brace, close_brace, parsed, namespace, class_path)
    and ("decl", start, end, namespace, class_path) for namespace/class-scope
    statements (used for constants and globals).
    """
    stack: list[tuple[str, str]] = []  # (kind, name) for namespace/class/extern scopes
    boundary = 0
    position = 0
    length = len(text)
    pattern = re.compile(r"[{};]")
    while True:
        match = pattern.search(text, position)
        if match is None:
            return
        offset = match.start()
        char = match.group(0)
        if char == ";":
            namespace = "::".join(n for k, n in stack if k == "namespace" and n)
            cls = "::".join(n for k, n in stack if k == "class")
            yield ("decl", boundary, offset, namespace, cls)
            boundary = offset + 1
            position = offset + 1
            continue
        if char == "}":
            if stack:
                stack.pop()
            boundary = offset + 1
            position = offset + 1
            continue
        header = text[boundary:offset]
        namespace = "::".join(n for k, n in stack if k == "namespace" and n)
        class_path = "::".join(n for k, n in stack if k == "class")
        scope_class = class_path.rsplit("::", 1)[-1] if class_path else None
        stripped_header, _ = _strip_attributes(header)
        access = _ACCESS.match(stripped_header)
        if access:
            stripped_header = stripped_header[access.end() :]
        namespace_match = _NAMESPACE_HEADER.match(stripped_header)
        if namespace_match:
            names = compact(namespace_match.group(1)).replace(" ", "")
            if names:
                for part in names.split("::"):
                    stack.append(("namespace", part))
                # nested namespace a::b opens several scopes but closes with one brace
                extra = names.count("::")
                if extra:
                    joined = stack[-(extra + 1) :]
                    del stack[-(extra + 1) :]
                    stack.append(("namespace", "::".join(n for _, n in joined)))
            else:
                stack.append(("namespace", ""))
            boundary = offset + 1
            position = offset + 1
            continue
        if _EXTERN_HEADER.match(stripped_header):
            stack.append(("extern", ""))
            boundary = offset + 1
            position = offset + 1
            continue
        close = match_close(text, offset, "{}")
        if close is None:
            return
        parsed = None
        if "(" in stripped_header:
            parsed = _parse_function_header(header, scope_class)
        elif re.search(r"\bvoid\s+MAIN\s*$", stripped_header):
            parsed = _parse_function_header(header, scope_class)
        if parsed is not None:
            yield ("function", boundary, offset, close, parsed, namespace, class_path)
            boundary = close + 1
            position = close + 1
            continue
        after_template = stripped_header[_template_prefixes(stripped_header)[0] :]
        class_match = _CLASS_HEADER.search(after_template)
        if (
            class_match
            and "(" not in after_template.split(":")[0]
            and not _ENUM_HEADER.search(after_template)
            and "=" not in after_template
        ):
            stack.append(("class", class_match.group(2)))
            boundary = offset + 1
            position = offset + 1
            continue
        # enum bodies, aggregate initializers, lambdas at namespace scope, ...
        position = close + 1
        if "(" in stripped_header and "=" not in stripped_header:
            boundary = (
                close + 1
            )  # an unparsed function-like block: do not poison the next header
        # otherwise keep boundary so a trailing ';' closes the whole declaration


# --------------------------------------------------------------------------- #
# Body analysis: thread wrappers, control blocks, call sites
# --------------------------------------------------------------------------- #
_IF = re.compile(r"\bif\b")


def control_blocks(masked: str, body: str) -> list[dict]:
    """``if``/``else`` regions of a body with their condition text."""
    blocks: list[dict] = []
    for match in _IF.finditer(masked):
        position = match.end()
        while position < len(masked) and masked[position].isspace():
            position += 1
        constexpr = False
        if masked.startswith("constexpr", position):
            constexpr = True
            position += len("constexpr")
            while position < len(masked) and masked[position].isspace():
                position += 1
        if position >= len(masked) or masked[position] != "(":
            continue
        close = match_close(masked, position, "()")
        if close is None:
            continue
        condition = compact(masked[position + 1 : close])
        after = close + 1
        while after < len(masked) and masked[after].isspace():
            after += 1
        if after < len(masked) and masked[after] == "{":
            block_close = match_close(masked, after, "{}")
            if block_close is None:
                continue
            start, end = after + 1, block_close
        else:
            start, end = close + 1, statement_end(masked, close + 1)
        blocks.append(
            {"start": start, "end": end, "condition": condition, "constexpr": constexpr}
        )
        else_position = end + 1
        while else_position < len(masked) and masked[else_position].isspace():
            else_position += 1
        if not masked.startswith("else", else_position) or (
            else_position + 4 < len(masked)
            and (
                masked[else_position + 4].isalnum() or masked[else_position + 4] == "_"
            )
        ):
            continue
        else_body = else_position + 4
        while else_body < len(masked) and masked[else_body].isspace():
            else_body += 1
        if masked.startswith("if", else_body) and not (
            masked[else_body + 2 : else_body + 3].isalnum()
        ):
            else_end = _if_chain_end(masked, else_body)
            else_start = else_body
        elif else_body < len(masked) and masked[else_body] == "{":
            else_close = match_close(masked, else_body, "{}")
            if else_close is None:
                continue
            else_start, else_end = else_body + 1, else_close
        else:
            else_start, else_end = else_body, statement_end(masked, else_body)
        blocks.append(
            {
                "start": else_start,
                "end": else_end,
                "condition": f"!({condition})",
                "constexpr": constexpr,
            }
        )
    blocks.sort(key=lambda block: (block["start"], block["end"]))
    return blocks


def _if_chain_end(masked: str, if_position: int) -> int:
    position = if_position + 2
    while position < len(masked) and masked[position].isspace():
        position += 1
    if masked.startswith("constexpr", position):
        position += len("constexpr")
        while position < len(masked) and masked[position].isspace():
            position += 1
    if position >= len(masked) or masked[position] != "(":
        return if_position
    close = match_close(masked, position, "()")
    if close is None:
        return len(masked)
    body = close + 1
    while body < len(masked) and masked[body].isspace():
        body += 1
    if body < len(masked) and masked[body] == "{":
        end = match_close(masked, body, "{}")
        end = len(masked) if end is None else end
    else:
        end = statement_end(masked, body)
    after = end + 1
    while after < len(masked) and masked[after].isspace():
        after += 1
    if not masked.startswith("else", after):
        return end + 1
    else_body = after + 4
    while else_body < len(masked) and masked[else_body].isspace():
        else_body += 1
    if masked.startswith("if", else_body):
        return _if_chain_end(masked, else_body)
    if else_body < len(masked) and masked[else_body] == "{":
        close_else = match_close(masked, else_body, "{}")
        return len(masked) if close_else is None else close_else + 1
    return statement_end(masked, else_body) + 1


def condition_at(blocks: list[dict], offset: int) -> str | None:
    enclosing = [
        block["condition"]
        for block in blocks
        if block["start"] <= offset < block["end"]
    ]
    return " && ".join(enclosing) if enclosing else None


def thread_spans(masked: str) -> list[tuple[int, int, str]]:
    """(start, end, thread) spans of ``UNPACK(..)``/``MATH(..)``/``PACK(..)`` wrappers."""
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?<![\w.>])(UNPACK|MATH|PACK)\s*\(", masked):
        opening = match.end() - 1
        close = match_close(masked, opening, "()")
        if close is not None:
            spans.append((opening, close, THREAD_WRAPPERS[match.group(1)]))
    return spans


def _thread_at(spans: list[tuple[int, int, str]], offset: int) -> str | None:
    best = None
    for start, end, thread in spans:
        if start <= offset <= end and (best is None or start >= best[0]):
            best = (start, thread)
    return best[1] if best else None


def _receiver_before(masked: str, dot: int) -> str:
    """Postfix-expression text that ends right before ``.``/``->`` at ``dot``."""
    position = dot - 1
    while position >= 0 and masked[position].isspace():
        position -= 1
    end = position + 1
    while position >= 0:
        char = masked[position]
        if char in ")]}":
            pair = {")": "(", "]": "[", "}": "{"}[char]
            depth = 0
            while position >= 0:
                if masked[position] == char:
                    depth += 1
                elif masked[position] == pair:
                    depth -= 1
                    if depth == 0:
                        break
                position -= 1
            position -= 1
            while position >= 0 and masked[position] in " \t\n":
                position -= 1
            continue
        if char.isalnum() or char == "_":
            position -= 1
            continue
        if char == ":" and position > 0 and masked[position - 1] == ":":
            position -= 2
            continue
        if char == "." or (
            char == ">" and position > 0 and masked[position - 1] == "-"
        ):
            position -= 2 if char == ">" else 1
            continue
        break
    return compact(masked[position + 1 : end])


_LOCAL_DECL = re.compile(
    r"(?<![\w:.])((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)(?:\s*<[^;(){}]*>)?\s+(?:const\s+)?([A-Za-z_]\w*)\s*(?=[({=;])"
)


def _local_types(masked: str, class_names: set[str]) -> dict[str, str]:
    types: dict[str, str] = {}
    for match in _LOCAL_DECL.finditer(masked):
        type_name = match.group(1)
        if (
            type_name.rsplit("::", 1)[-1] in class_names
            and match.group(2) not in _KEYWORDS
        ):
            types.setdefault(match.group(2), type_name)
    return types


def _expand_macro(macro: MacroDef, args: tuple[str, ...]) -> str:
    body = macro.body
    params = list(macro.params or ())
    variadic = None
    if params and params[-1] == "...":
        variadic = ", ".join(args[len(params) - 1 :])
        params = params[:-1]
    elif params and params[-1].endswith("..."):
        variadic = ", ".join(args[len(params) - 1 :])
        params[-1] = params[-1][:-3].strip()
    mapping = {
        name: (args[index] if index < len(args) else "")
        for index, name in enumerate(params)
    }
    if variadic is not None:
        # ", ##__VA_ARGS__" drops the comma when empty
        body = re.sub(
            r",\s*##\s*__VA_ARGS__", (", __VA_ARGS__" if variadic else ""), body
        )
        mapping["__VA_ARGS__"] = variadic
    body = (
        re.sub(r"#\s*([A-Za-z_]\w*)", r"\1", body)
        if "#" in body and "##" not in body
        else body
    )
    body = body.replace("##", "")

    def substitute(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return re.sub(r"[A-Za-z_]\w*", substitute, body)


def _expandable(macro: MacroDef | None) -> bool:
    return (
        macro is not None
        and macro.params is not None
        and macro.name not in NO_EXPAND_MACROS
        and not macro.path.endswith(INSTRUCTION_MACRO_FILES)
        and not re.match(r"^(?:TTI_|TT_OP_|TT_)[A-Z0-9_]+$", macro.name)
    )


_MACRO_INVOCATION = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")


def expand_all(text: str, macros: dict[str, MacroDef], depth: int = 0) -> str:
    """Expand every expandable function-like macro in ``text`` (bounded depth)."""
    if depth > 8:
        return text
    pieces: list[str] = []
    position = 0
    changed = False
    for match in _MACRO_INVOCATION.finditer(text):
        if match.start() < position:
            continue
        macro = macros.get(match.group(1))
        if not _expandable(macro):
            continue
        opening = match.end() - 1
        close = match_close(text, opening, "()")
        if close is None:
            continue
        pieces.append(text[position : match.start()])
        pieces.append(
            " " + _expand_macro(macro, split_args(text[opening + 1 : close])) + " "
        )
        position = close + 1
        changed = True
    if not changed:
        return text
    pieces.append(text[position:])
    return expand_all("".join(pieces), macros, depth + 1)


def find_calls(
    masked: str,
    *,
    base_line: int,
    line_info: tuple = (),
    macros: dict[str, MacroDef] | None = None,
    class_names: set[str] | None = None,
    blocks: list[dict] | None = None,
    pp_condition: str | None = None,
    pp_thread: str | None = None,
    _depth: int = 0,
    _via: str | None = None,
    _fixed: tuple[int, int, str | None, str | None] | None = None,
) -> tuple[list[CallSite], list[MacroExpansion]]:
    """Extract call sites (and macro expansions) from a masked code segment.

    ``_fixed`` is used for macro expansion text: (offset, line, thread, condition)
    of the invocation, applied to every call found in the expansion.
    """
    macros = macros or {}
    class_names = class_names or set()
    blocks = blocks if blocks is not None else control_blocks(masked, masked)
    spans = thread_spans(masked)
    locals_ = _local_types(masked, class_names)
    calls: list[CallSite] = []
    expansions: list[MacroExpansion] = []
    for match in _QUALIFIED_IDENT.finditer(masked):
        text = match.group(0)
        start = match.start()
        name = text.replace(" ", "").rsplit("::", 1)[-1]
        if name in _KEYWORDS or name in THREAD_WRAPPERS:
            continue
        if name.startswith("~"):
            continue
        position = match.end()
        length = len(masked)
        while position < length and masked[position] in " \t\n":
            position += 1
        template_args: tuple[str, ...] = ()
        if (
            position < length
            and masked[position] == "<"
            and _is_template_open(masked, position)
            and position == match.end()
        ):
            close = match_angle(masked, position)
            if close is None:
                continue
            template_args = split_args(masked[position + 1 : close])
            position = close + 1
            while position < length and masked[position] in " \t\n":
                position += 1
        if position >= length or masked[position] != "(":
            continue
        close = match_close(masked, position, "()")
        if close is None:
            continue
        args = split_args(masked[position + 1 : close])
        # classify: member call, constructor declaration, plain call
        before = start - 1
        while before >= 0 and masked[before] in " \t\n":
            before -= 1
        is_member = False
        receiver = None
        receiver_type = None
        is_ctor = False
        callee = compact(text).replace(" ", "")
        if before >= 0 and (
            masked[before] == "."
            or (masked[before] == ">" and before > 0 and masked[before - 1] == "-")
        ):
            is_member = True
            receiver = _receiver_before(
                masked, before - (1 if masked[before] == ">" else 0)
            )
            head = re.match(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)", receiver or "")
            if head:
                if head.group(1) in class_names:
                    receiver_type = head.group(1)
                elif head.group(0) in locals_:
                    receiver_type = locals_[head.group(0)].rsplit("::", 1)[-1]
        elif before >= 0 and (masked[before].isalnum() or masked[before] in "_>"):
            word = re.search(r"([A-Za-z_]\w*)\s*$", masked[: before + 1])
            word_text = word.group(1) if word else ""
            if word_text in _DECL_PREFIX_KEYWORDS:
                pass
            elif word_text and word_text not in _KEYWORDS:
                type_match = re.search(
                    r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)(?:\s*<[^;(){}]*>)?\s*$",
                    masked[: before + 1],
                )
                type_name = type_match.group(1) if type_match else word_text
                if type_name.rsplit("::", 1)[-1] in class_names:
                    is_ctor = True
                    receiver = callee
                    receiver_type = type_name.rsplit("::", 1)[-1]
                    callee = type_name
                    name = receiver_type
                else:
                    continue  # a declaration like "uint32_t x(5)" -- not a call
            elif masked[before] == ">":
                pass
        if _fixed is not None:
            offset, line, thread, condition = _fixed
        else:
            offset = start
            line = base_line + masked.count("\n", 0, start)
            thread = _thread_at(spans, start)
            if thread is None and line_info and 1 <= line <= len(line_info):
                thread = line_info[line - 1][0]
            if thread is None:
                thread = pp_thread
            parts = []
            if line_info and 1 <= line <= len(line_info):
                pp = line_info[line - 1][1]
                if pp and pp != pp_condition:
                    parts.append(pp)
            control = condition_at(blocks, start)
            if control:
                parts.append(control)
            condition = " && ".join(parts) if parts else None
        macro = (
            macros.get(name)
            if not is_member and not is_ctor and "::" not in callee
            else None
        )
        is_macro = macro is not None and macro.params is not None
        if not is_macro and re.match(r"^(?:TTI_|TT_OP_|TT_)[A-Z0-9_]+$", name):
            is_macro = True
        calls.append(
            CallSite(
                name=name,
                callee=callee,
                template_args=template_args,
                args=args,
                line=line,
                offset=offset,
                thread=thread,
                condition=condition,
                via_macro=_via,
                is_member=is_member,
                receiver=receiver,
                receiver_type=receiver_type,
                is_ctor=is_ctor,
                is_macro=is_macro,
            )
        )
        if is_macro and _depth < 6 and _expandable(macro):
            expanded = compact(expand_all(_expand_macro(macro, args), macros))
            expansions.append(
                MacroExpansion(
                    name=name,
                    line=line,
                    offset=offset,
                    text=expanded,
                    thread=thread,
                    condition=condition,
                )
            )
            inner_calls, inner_expansions = find_calls(
                expanded,
                base_line=line,
                macros=macros,
                class_names=class_names,
                blocks=[],
                _depth=_depth + 1,
                _via=_via or name,
                _fixed=(offset, line, thread, condition),
            )
            # calls in the expansion inherit a thread wrapper found inside the expansion
            inner_spans = thread_spans(expanded)
            for call in inner_calls:
                if call.thread is None and inner_spans:
                    call = _replace_call(call, thread=_thread_at(inner_spans, 0))
                calls.append(call)
            expansions.extend(inner_expansions)
    return calls, expansions


def _replace_call(call: CallSite, **changes) -> CallSite:
    values = {name: getattr(call, name) for name in call.__dataclass_fields__}
    values.update(changes)
    return CallSite(**values)


# --------------------------------------------------------------------------- #
# File and tree scanning
# --------------------------------------------------------------------------- #
def layer_of(path: str) -> str | None:
    """Layer of a repo-relative path, or ``None`` when outside the analysed roots."""
    for root, layer in SOURCE_ROOTS:
        if path == root or path.startswith(root + "/"):
            basename = path.rsplit("/", 1)[-1]
            if "/common/inc/sfpu/" in path or basename.startswith("ckernel_sfpu"):
                return "sfpu"
            return layer
    return None


def iter_sources(repo_root: Path | str) -> list[tuple[str, str]]:
    """Sorted (repo-relative path, layer) pairs of every analysed header."""
    root = Path(repo_root)
    found: dict[str, str] = {}
    for directory, _ in SOURCE_ROOTS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in HEADER_SUFFIXES:
                relative = path.relative_to(root).as_posix()
                layer = layer_of(relative)
                if layer is not None:
                    found[relative] = layer
    return sorted(found.items())


def _thread_from_name(name: str) -> str | None:
    stripped = name.lstrip("_")
    for prefix, thread in (
        ("llk_unpack", "unpack"),
        ("llk_math", "math"),
        ("llk_pack", "pack"),
    ):
        if stripped.startswith(prefix):
            return thread
    return None


def _thread_from_path(path: str, layer: str) -> str | None:
    if layer == "sfpu" or layer == "compute_api":
        return None
    basename = path.rsplit("/", 1)[-1]
    for prefixes, thread in (
        (("llk_unpack", "cunpack", "llk_io_unpack"), "unpack"),
        (("llk_math", "cmath"), "math"),
        (("llk_pack", "cpack", "llk_io_pack"), "pack"),
    ):
        if basename.startswith(prefixes):
            return thread
    return None


def thread_hint(name: str, path: str, layer: str, pp_thread: str | None) -> str | None:
    return _thread_from_name(name) or pp_thread or _thread_from_path(path, layer)


def _signature_key(
    qualified: str,
    template_params: tuple[Param, ...],
    params: tuple[Param, ...],
    spec: str | None,
) -> str:
    def type_of(param: Param) -> str:
        text = re.sub(r"\s*([^\w\s])\s*", r"\1", compact(param.type))
        return text + ("..." if param.variadic and "..." not in text else "")

    template = ""
    if template_params:
        template = "<" + ",".join(type_of(p) for p in template_params) + ">"
    elif spec is not None:
        template = "<>"
    specialization = f"<{compact(spec).replace(' ', '')}>" if spec else ""
    return (
        f"{qualified}{template}{specialization}("
        + ",".join(type_of(p) for p in params)
        + ")"
    )


def scan_text(
    text: str,
    path: str,
    layer: str,
    *,
    defines: dict[str, bool | int] | None = None,
    has_rvtt_builtins: bool = True,
) -> SourceFile:
    """Scan one file: definitions, macros, constants and globals (no calls yet)."""
    masked0 = mask_comments_and_strings(text)
    pp = preprocess(masked0, path, defines=defines, has_rvtt_builtins=has_rvtt_builtins)
    analysis = pp.text
    starts = _line_starts(text)
    source = SourceFile(
        path=path,
        layer=layer,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        macros=pp.macros,
    )
    for item in _scan_scopes(analysis, starts):
        if item[0] == "decl":
            _, start, end, namespace, cls = item
            statement = analysis[start : end + 1]
            using = re.match(
                r"\s*using\s+namespace\s+((?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)\s*;",
                statement,
            )
            if using:
                name = compact(using.group(1)).replace(" ", "").lstrip(":")
                qualified_using = (
                    name if not namespace or name.startswith(namespace) else name
                )
                if qualified_using not in source.using:
                    source.using.append(qualified_using)
                continue
            for constant in _CONSTEXPR_INT.finditer(statement):
                value = compact(constant.group(2) or constant.group(3) or "")
                source.constants.setdefault(constant.group(1), value)
                if cls:
                    source.constants.setdefault(
                        f"{cls.rsplit('::', 1)[-1]}::{constant.group(1)}", value
                    )
            if (
                not cls
                and "constexpr" not in statement
                and "(" not in statement
                and "using" not in statement
            ):
                declaration = _GLOBAL_DECL.match(compact(statement.rstrip(";")))
                if declaration and not re.search(
                    r"\b(const|typedef|template|return)\b", statement
                ):
                    source.globals.append(declaration.group(1))
            continue
        _, header_start, open_brace, close_brace, parsed, namespace, class_path = item
        name_offset = header_start + parsed["name_offset"]
        owner = parsed["owner"]
        owner_path = owner or class_path
        qualified_parts = [
            part for part in (namespace, owner_path, parsed["name"]) if part
        ]
        qualified = "::".join(qualified_parts)
        template_params = parse_params(parsed["template"], "template")
        params = parse_params(parsed["params"], "runtime")
        body_offset = open_brace + 1
        line = _line_of(starts, name_offset)
        declaration_start = header_start
        while declaration_start < name_offset and text[declaration_start].isspace():
            declaration_start += 1
        pp_thread, pp_condition = (
            pp.line_info[line - 1] if line - 1 < len(pp.line_info) else (None, None)
        )
        signature = compact(analysis[declaration_start:open_brace])
        source.functions.append(
            FunctionDef(
                name=parsed["name"],
                qualified_name=qualified,
                namespace=namespace,
                owner=owner_path,
                key=_signature_key(qualified, template_params, params, parsed["spec"]),
                path=path,
                layer=layer,
                line=line,
                start_line=_line_of(starts, declaration_start),
                end_line=_line_of(starts, close_brace),
                body_line=_line_of(starts, open_brace),
                signature=signature,
                params=params,
                template_params=template_params,
                specialization_args=(
                    split_args(parsed["spec"]) if parsed["spec"] else ()
                ),
                attributes=tuple(
                    sorted(
                        set(parsed["attributes"])
                        | set(_prefix_attributes(parsed["prefix"]))
                    )
                ),
                body=text[body_offset:close_brace],
                masked_body=analysis[body_offset:close_brace],
                pp_condition=pp_condition,
                pp_thread=pp_thread,
                thread_hint=thread_hint(parsed["name"], path, layer, pp_thread),
                fingerprint=hashlib.sha256(
                    compact(analysis[body_offset:close_brace]).encode("utf-8")
                ).hexdigest(),
                is_ctor=parsed["is_ctor"],
                _body_offset=body_offset,
                _line_table=(),
                _line_info=pp.line_info,
            )
        )
    return source


def _prefix_attributes(prefix: str) -> list[str]:
    words = set(re.findall(r"[A-Za-z_]\w*", prefix))
    return sorted(
        words
        & {
            "ALWI",
            "inline",
            "sfpi_inline",
            "static",
            "constexpr",
            "always_inline",
            "noinline",
            "FORCE_INLINE",
        }
    )


def attach_calls(
    fn: FunctionDef, macros: dict[str, MacroDef], class_names: set[str]
) -> None:
    """Populate ``fn.calls``/``fn.expansions``/``fn.locals`` (needs the global macro table)."""
    blocks = _control_blocks_cached(fn)
    calls, expansions = find_calls(
        fn.masked_body,
        base_line=fn.body_line,
        line_info=fn._line_info,
        macros=macros,
        class_names=class_names,
        blocks=blocks,
        pp_condition=fn.pp_condition,
        pp_thread=fn.pp_thread,
    )
    calls.sort(
        key=lambda call: (
            call.offset,
            call.line,
            call.name,
            call.via_macro or "",
            call.args,
        )
    )
    fn.calls = tuple(calls)
    fn.expansions = tuple(expansions)
    fn.locals = tuple(sorted(_local_types(fn.masked_body, class_names).items()))


def scan_tree(
    repo_root: Path | str,
    *,
    defines: dict[str, bool | int] | None = None,
    has_rvtt_builtins: bool = True,
    sources: Iterable[tuple[str, str]] | None = None,
) -> ScanIndex:
    """Scan every analysed Blackhole header below ``repo_root``."""
    root = Path(repo_root)
    files: list[SourceFile] = []
    digest = hashlib.sha256()
    for relative, layer in (sources if sources is not None else iter_sources(root)):
        raw = (root / relative).read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + raw + b"\0")
        text = raw.decode("utf-8", errors="replace")
        files.append(
            scan_text(
                text,
                relative,
                layer,
                defines=defines,
                has_rvtt_builtins=has_rvtt_builtins,
            )
        )
    return build_index(files, digest.hexdigest())


def build_index(files: list[SourceFile], fingerprint: str) -> ScanIndex:
    macros: dict[str, MacroDef] = {}
    constants: dict[str, str] = {}
    global_names: set[str] = set()
    for source in files:
        for macro in source.macros:
            existing = macros.get(macro.name)
            # Prefer unconditional definitions; otherwise the first in path order.
            if existing is None or (existing.condition and not macro.condition):
                macros[macro.name] = macro
        for name, value in source.constants.items():
            constants.setdefault(name, value)
        global_names.update(source.globals)
    functions = [fn for source in files for fn in source.functions]
    class_names = {fn.owner.rsplit("::", 1)[-1] for fn in functions if fn.owner}
    for source in files:
        for fn in source.functions:
            fn.using = tuple(sorted(set(source.using)))
    for fn in functions:
        attach_calls(fn, macros, class_names)
    # Unique keys: suffix duplicates (e.g. #if/#else twins) in path/line order.
    seen: dict[str, int] = {}
    for fn in functions:
        count = seen.get(fn.key, 0)
        seen[fn.key] = count + 1
        if count:
            fn.key = f"{fn.key}#{count + 1}"
    functions.sort(key=lambda fn: (fn.path, fn.line, fn.key))
    index = ScanIndex(
        files=files,
        functions=functions,
        macros=macros,
        constants=constants,
        globals=tuple(sorted(global_names)),
        fingerprint=fingerprint,
    )
    for fn in functions:
        index._by_name.setdefault(fn.name, []).append(fn)
        index._by_key[fn.key] = fn
    return index


def scan_source(
    text: str,
    path: str,
    layer: str,
    *,
    index: ScanIndex | None = None,
    defines: dict[str, bool | int] | None = None,
) -> SourceFile:
    """Scan an arbitrary file (e.g. a compute kernel) and attach call sites.

    Macros and class names of ``index`` (the BH tree) are used for expansion and
    member-call typing, so kernel calls to ``SFPU_*`` macros resolve too.
    """
    source = scan_text(text, path, layer, defines=defines)
    macros = dict(index.macros) if index is not None else {}
    for macro in source.macros:
        macros.setdefault(macro.name, macro)
    class_names = {
        fn.owner.rsplit("::", 1)[-1]
        for fn in (index.functions if index else [])
        if fn.owner
    }
    class_names |= {fn.owner.rsplit("::", 1)[-1] for fn in source.functions if fn.owner}
    for fn in source.functions:
        fn.using = tuple(sorted(set(source.using)))
        attach_calls(fn, macros, class_names)
    return source
