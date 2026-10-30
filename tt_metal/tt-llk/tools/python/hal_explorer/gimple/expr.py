# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GIMPLE expressions: parsed definitions, evaluation and source text.

Every SSA definition is parsed once into a small node tree. A Scope binds SSA
names to their definitions and to the values known for one walk of the function:
concrete integers, symbolic pointers (Pointer) or nothing (unknown at runtime).
"""

import re
from dataclasses import dataclass

from hal_explorer.gimple.reader import SSA, base_name
from hal_explorer.gimple.types import UNSIGNED32, Pointer, TypeInfo, type_info

INTEGER = re.compile(r"^-?\d+[uUlLB]*$")
BINARY_OPS = (
    "r<<", "r>>", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "+", "-", "*", "/", "%", "&", "|", "^", "<", ">",
)  # fmt: skip
INVERT = {"!=": "==", "==": "!=", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


# ---------------------------------------------------------------- nodes


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Symbol:
    """A global read by name, e.g. cfg_state_id, or its address (&sym)."""

    name: str
    address: bool = False


@dataclass(frozen=True)
class Cast:
    type: TypeInfo
    operand: object


@dataclass(frozen=True)
class Unary:
    op: str
    operand: object


@dataclass(frozen=True)
class Binary:
    op: str
    lhs: object
    rhs: object


@dataclass(frozen=True)
class Select:
    condition: object
    then: object
    otherwise: object


@dataclass(frozen=True)
class Opaque:
    text: str


def parse_atom(token):
    token = token.strip()
    if INTEGER.match(token):
        return Const(int(re.sub(r"[uUlLB]+$", "", token)))
    if token.startswith("&") and re.fullmatch(r"&[A-Za-z_]\w*", token):
        return Symbol(token[1:], address=True)
    if re.fullmatch(SSA, token):
        if re.search(r"(_\d+(\(D\))?|\.\d+_\d+)$", token) or "$" in token:
            return Ref(token)
        return Symbol(token)
    return Opaque(token)


def parse(rhs):
    rhs = rhs.strip()
    if rhs == "OPAQUE":
        return Opaque("runtime value")
    m = re.fullmatch(r"\(([^()]+)\) (\S+)", rhs)
    if m:
        return Cast(type_info(m.group(1)), parse_atom(m.group(2)))
    m = re.fullmatch(r"(MIN|MAX)_EXPR <(\S+), (\S+)>", rhs)
    if m:
        return Binary(
            m.group(1).lower(), parse_atom(m.group(2)), parse_atom(m.group(3))
        )
    m = re.fullmatch(r"ABS_EXPR <(\S+)>", rhs)
    if m:
        return Unary("abs", parse_atom(m.group(1)))
    m = re.fullmatch(r"(\S+) \? (\S+) : (\S+)", rhs)
    if m:
        return Select(
            parse_atom(m.group(1)), parse_atom(m.group(2)), parse_atom(m.group(3))
        )
    m = re.fullmatch(r"(\S+) (\S+) (\S+)", rhs)
    if m and m.group(2) in BINARY_OPS:
        return Binary(m.group(2), parse_atom(m.group(1)), parse_atom(m.group(3)))
    m = re.fullmatch(r"([~!-])(\S+)", rhs)
    if m and not INTEGER.match(rhs):
        return Unary(m.group(1), parse_atom(m.group(2)))
    atom = parse_atom(rhs)
    return atom if not isinstance(atom, Opaque) else Opaque(rhs)


# ---------------------------------------------------------------- scope


class Scope:
    """Definitions, declared types and the values known for one walk."""

    def __init__(self, function, definitions=None, values=None, unknown=()):
        self.function = function
        self.definitions = (
            definitions if definitions is not None else definitions_of(function)
        )
        # PHI results bound to the incoming value of the edge this walk took.
        self.aliases = {}
        # Concrete values by SSA name, parameter name or global symbol name.
        self.values = dict(values or {})
        self.unknown = set(unknown)
        self.cache = {}
        # Optional value for memory loads the dump leaves opaque, by their text.
        self.load_value = None

    def definition(self, name):
        if name in self.aliases:
            return self.aliases[name]
        return self.definitions.get(name)

    def bound(self, name):
        """The concrete value bound to an SSA name, a parameter or a global, if any."""
        if name in self.values:
            return True, self.values[name]
        if name.endswith("(D)") and base_name(name) in self.values:
            return True, self.values[base_name(name)]
        return False, None

    def type_of(self, name):
        declarations = self.function.declarations
        if name in declarations:
            return type_info(declarations[name])
        base = base_name(name)
        if base in declarations:
            return type_info(declarations[base])
        for parameter_type, parameter in self.function.parameters:
            if parameter == base:
                return type_info(parameter_type)
        return UNSIGNED32

    def node(self, name):
        return self.definitions.get(name)

    # -- concrete values ------------------------------------------------

    def value(self, node, depth=0):
        """Integer or Pointer value of node, or None when it depends on unknowns."""
        if isinstance(node, str):
            node = parse_atom(node)
        if depth > 200:
            return None
        if isinstance(node, Const):
            return node.value
        if isinstance(node, Opaque):
            return self.load_value(node.text) if self.load_value else None
        if isinstance(node, Symbol):
            if node.address:
                return Pointer(node.name)
            return self.values.get(node.name)
        if isinstance(node, Ref):
            known, value = self.bound(node.name)
            if known:
                return (
                    self.type_of(node.name).wrap(value) if value is not None else None
                )
            if node.name in self.unknown:
                return None
            if node.name not in self.cache:
                definition = self.definition(node.name)
                result = (
                    None if definition is None else self.value(definition, depth + 1)
                )
                self.cache[node.name] = (
                    self.type_of(node.name).wrap(result) if result is not None else None
                )
            return self.cache[node.name]
        if isinstance(node, Cast):
            operand = self.value(node.operand, depth + 1)
            if operand is None:
                return None
            return node.type.wrap(operand)
        if isinstance(node, Unary):
            operand = self.value(node.operand, depth + 1)
            if operand is None or isinstance(operand, Pointer):
                return None
            return {
                "~": lambda v: ~v,
                "-": lambda v: -v,
                "!": lambda v: int(not v),
                "abs": abs,
            }[node.op](operand)
        if isinstance(node, Select):
            condition = self.value(node.condition, depth + 1)
            if condition is None:
                return None
            return self.value(node.then if condition else node.otherwise, depth + 1)
        if isinstance(node, Binary):
            return binary(
                node.op,
                self.value(node.lhs, depth + 1),
                self.value(node.rhs, depth + 1),
            )
        return None

    # -- source text ------------------------------------------------------

    def text(self, node, depth=0):
        if isinstance(node, str):
            node = parse_atom(node)
        if isinstance(node, Const):
            return hex(node.value) if abs(node.value) > 255 else str(node.value)
        if isinstance(node, Symbol):
            return ("&" if node.address else "") + node.name
        if isinstance(node, Ref):
            _, value = self.bound(node.name)
            if isinstance(value, int):
                return self.text(Const(value))
            definition = self.definition(node.name)
            if definition is None or depth > 24 or node.name in self.unknown:
                return (
                    base_name(node.name)
                    if node.name.endswith("(D)") or "." in node.name
                    else source_name(node.name)
                )
            return self.text(definition, depth + 1)
        if isinstance(node, Cast):
            return self.text(node.operand, depth + 1)
        if isinstance(node, Unary):
            inner = self.text(node.operand, depth + 1)
            return f"abs({inner})" if node.op == "abs" else f"{node.op}{inner}"
        if isinstance(node, Select):
            return f"({self.text(node.condition, depth + 1)} ? {self.text(node.then, depth + 1)} : {self.text(node.otherwise, depth + 1)})"
        if isinstance(node, Binary):
            value = self.value(node)
            if isinstance(value, int):
                return self.text(Const(value))
            lhs, rhs = self.text(node.lhs, depth + 1), self.text(node.rhs, depth + 1)
            if node.op in ("min", "max"):
                return f"{node.op}({lhs}, {rhs})"
            return f"({lhs} {node.op} {rhs})"
        return node.text

    # -- runtime words ------------------------------------------------------

    def runtime_mask(self, node, depth=0):
        """Bits of node's value that can depend on runtime inputs."""
        if isinstance(node, str):
            node = parse_atom(node)
        if depth > 200 or isinstance(node, Opaque):
            return 0xFFFFFFFF
        if self.value(node) is not None:
            return 0
        if isinstance(node, (Symbol,)):
            return 0xFFFFFFFF
        if isinstance(node, Ref):
            definition = self.definition(node.name)
            if definition is None or node.name in self.unknown:
                bits = self.type_of(node.name).bits
                return (1 << bits) - 1
            return self.runtime_mask(definition, depth + 1) & (
                (1 << self.type_of(node.name).bits) - 1
            )
        if isinstance(node, Cast):
            mask = self.runtime_mask(node.operand, depth + 1)
            return mask & ((1 << node.type.bits) - 1) if not node.type.signed else mask
        if isinstance(node, Select):
            return 0xFFFFFFFF
        if isinstance(node, Unary):
            return (
                0xFFFFFFFF
                if node.op != "~"
                else self.runtime_mask(node.operand, depth + 1)
            )
        if isinstance(node, Binary):
            a, b = self.runtime_mask(node.lhs, depth + 1), self.runtime_mask(
                node.rhs, depth + 1
            )
            shift = self.value(node.rhs)
            if node.op == "<<" and isinstance(shift, int):
                return (a << shift) & 0xFFFFFFFF
            if node.op == ">>" and isinstance(shift, int):
                return a >> shift
            if node.op == "&":
                constant = (
                    self.value(node.rhs)
                    if b == 0
                    else self.value(node.lhs) if a == 0 else None
                )
                return (a | b) & constant if isinstance(constant, int) else a | b
            if node.op in ("|", "^", "+", "-"):
                # HAL words add disjoint fields, so a runtime term only reaches its own bits.
                return a | b
            if node.op == "*":
                factor = (
                    self.value(node.rhs)
                    if b == 0
                    else self.value(node.lhs) if a == 0 else None
                )
                if (
                    isinstance(factor, int)
                    and factor > 0
                    and factor & (factor - 1) == 0
                ):
                    return ((a | b) << (factor.bit_length() - 1)) & 0xFFFFFFFF
            low = (a | b) & -(a | b)
            return (0xFFFFFFFF ^ (low - 1)) & 0xFFFFFFFF if a | b else 0
        return 0xFFFFFFFF

    def constant_part(self, node):
        """node's value with every runtime leaf taken as zero."""
        zeroed = Scope(self.function, self.definitions, self.values)
        zeroed.aliases = self.aliases
        for name in self.leaves(node):
            zeroed.values[name] = 0
        value = zeroed.value(node)
        return value & 0xFFFFFFFF if isinstance(value, int) else 0

    def leaves(self, node, depth=0):
        """Names of the runtime inputs node depends on."""
        if isinstance(node, str):
            node = parse_atom(node)
        if depth > 200:
            return set()
        if isinstance(node, Symbol) and not node.address:
            return set() if node.name in self.values else {node.name}
        if isinstance(node, Ref):
            if self.bound(node.name)[0]:
                return set()
            definition = self.definition(node.name)
            if (
                definition is None
                or node.name in self.unknown
                or isinstance(definition, Opaque)
            ):
                return {node.name}
            return self.leaves(definition, depth + 1)
        children = {
            Cast: lambda n: [n.operand],
            Unary: lambda n: [n.operand],
            Binary: lambda n: [n.lhs, n.rhs],
            Select: lambda n: [n.condition, n.then, n.otherwise],
        }.get(type(node), lambda n: [])(node)
        return (
            set().union(*(self.leaves(child, depth + 1) for child in children))
            if children
            else set()
        )

    def condition_text(self, condition, taken):
        m = re.fullmatch(r"(\S+) (!=|==|<=|>=|<|>) (\S+)", condition)
        if not m:
            text = self.text(parse(condition))
            return text if taken else f"!({text})"
        a, op, b = m.groups()
        op = op if taken else INVERT[op]
        lhs, rhs = self.text(a), self.text(b)
        if rhs == "0" and op in ("!=", "=="):
            return lhs if op == "!=" else f"!{lhs}"
        return f"{lhs} {op} {rhs}"


def definitions_of(function):
    return {
        stmt.data["name"]: parse(stmt.data["rhs"])
        for block in function.blocks.values()
        for stmt in block.statements
        if stmt.kind == "def"
    }


def source_name(token):
    """tile_size_2(D) -> tile_size, cfg_state_id.0_9 -> cfg_state_id, _5 -> _5."""
    m = re.match(r"^([A-Za-z_][\w$]*?)(?:_\d+\(D\)|\.\d+_\d+|_\d+)$", token)
    return m.group(1) if m else token


def binary(op, a, b):
    """C semantics on already-wrapped operands; the caller wraps the result."""
    if a is None or b is None:
        return None
    if isinstance(a, Pointer) or isinstance(b, Pointer):
        if op in ("+", "-") and isinstance(a, Pointer) and isinstance(b, int):
            return Pointer(a.symbol, a.offset + (b if op == "+" else -b))
        if op == "+" and isinstance(b, Pointer) and isinstance(a, int):
            return Pointer(b.symbol, b.offset + a)
        if op in ("==", "!=") and isinstance(a, Pointer) and isinstance(b, Pointer):
            return int((a == b) == (op == "=="))
        return None
    if op in ("/", "%") and b == 0:
        # C leaves this undefined; GCC emits div/rem, and RISC-V defines the result.
        return -1 if op == "/" else a
    operations = {
        "+": lambda: a + b,
        "-": lambda: a - b,
        "*": lambda: a * b,
        "/": lambda: abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1),
        "%": lambda: a - b * (abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)),
        "&": lambda: a & b,
        "|": lambda: a | b,
        "^": lambda: a ^ b,
        "<<": lambda: a << (b & 63),
        ">>": lambda: a >> (b & 63),
        "r<<": lambda: (
            (a & 0xFFFFFFFF) << (b % 32) | (a & 0xFFFFFFFF) >> (32 - b % 32)
        ),
        "r>>": lambda: (
            (a & 0xFFFFFFFF) >> (b % 32) | (a & 0xFFFFFFFF) << (32 - b % 32)
        ),
        "==": lambda: int(a == b),
        "!=": lambda: int(a != b),
        "<": lambda: int(a < b),
        "<=": lambda: int(a <= b),
        ">": lambda: int(a > b),
        ">=": lambda: int(a >= b),
        "&&": lambda: int(bool(a) and bool(b)),
        "||": lambda: int(bool(a) or bool(b)),
        "min": lambda: min(a, b),
        "max": lambda: max(a, b),
    }
    return operations[op]() if op in operations else None
