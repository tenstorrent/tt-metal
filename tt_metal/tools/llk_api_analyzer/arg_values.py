# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Recover compile-time-constant values of runtime arguments from DWARF.

At ``-O3`` the compiler rarely emits ``DW_AT_const_value`` on an inlined
function's parameters. Instead, a parameter whose value the optimizer knows is
described by a *location* that evaluates to a constant: a DWARF expression of
the form ``DW_OP_litN`` / ``DW_OP_constNu <v>`` terminated by
``DW_OP_stack_value`` (an "implicit value"). Parameters that live in a register
or on the stack at runtime use ``DW_OP_regN`` / ``DW_OP_bregN`` instead and are
genuinely dynamic.

:class:`ConstantArgEvaluator` reads a parameter's ``DW_AT_const_value`` or
``DW_AT_location`` (single expression or location list) and returns its constant
value when every location that describes it is the *same* implicit constant.
"""

from __future__ import annotations

from elftools.dwarf.die import DIE
from elftools.dwarf.dwarf_expr import DWARFExprParser
from elftools.dwarf.locationlists import LocationExpr, LocationParser

# Expression operators that push an explicit constant taken from their operand.
_CONST_OPERAND_OPS = frozenset(
    {
        "DW_OP_const1u",
        "DW_OP_const1s",
        "DW_OP_const2u",
        "DW_OP_const2s",
        "DW_OP_const4u",
        "DW_OP_const4s",
        "DW_OP_const8u",
        "DW_OP_const8s",
        "DW_OP_constu",
        "DW_OP_consts",
        "DW_OP_addr",
    }
)
_LIT_PREFIX = "DW_OP_lit"

# Stack operators whose result depends only on already-pushed constants. As long
# as an expression uses *only* these plus constant pushes, the optimizer fully
# folded it and the value is compile-time constant. Any other operator
# (registers, frame base, dereferences, ...) means the value is computed at
# runtime, so the whole expression is treated as dynamic.
_BINARY_OPS = {
    "DW_OP_plus": lambda a, b: a + b,
    "DW_OP_minus": lambda a, b: a - b,
    "DW_OP_mul": lambda a, b: a * b,
    "DW_OP_div": lambda a, b: a // b if b != 0 else None,
    "DW_OP_mod": lambda a, b: a % b if b != 0 else None,
    "DW_OP_and": lambda a, b: a & b,
    "DW_OP_or": lambda a, b: a | b,
    "DW_OP_xor": lambda a, b: a ^ b,
    "DW_OP_shl": lambda a, b: a << b if 0 <= b < 256 else None,
    "DW_OP_shr": lambda a, b: a >> b if 0 <= b < 256 else None,
    "DW_OP_shra": lambda a, b: a >> b if 0 <= b < 256 else None,
}
_UNARY_OPS = {
    "DW_OP_neg": lambda a: -a,
    "DW_OP_not": lambda a: ~a,
    "DW_OP_abs": abs,
}


class ConstantArgEvaluator:
    """Evaluates whether a formal parameter has a known compile-time constant."""

    def __init__(self, dwarf):
        self._expr_parser = DWARFExprParser(dwarf.structs)
        loclists = dwarf.location_lists()
        self._loc_parser = LocationParser(loclists) if loclists is not None else None

    def evaluate(self, param_die: DIE) -> tuple[int, ...] | None:
        """Return a parameter's compile-time constant value(s), or ``None``.

        The result is a tuple of the distinct constant values the parameter
        takes (sorted):

        * ``(v,)`` for an ordinary constant argument;
        * ``(0, 1, ..., 7)`` when an unrolled loop reuses one inlined call over
          several constant indices (the optimizer keeps a single inlined DIE
          whose parameter location list spans several PC ranges, one per
          unrolled iteration);
        * ``None`` if any location is a register / memory slot, i.e. the value
          is only known at runtime.
        """
        const = param_die.attributes.get("DW_AT_const_value")
        if const is not None and isinstance(const.value, int):
            return (const.value,)

        location_attr = param_die.attributes.get("DW_AT_location")
        if location_attr is None or self._loc_parser is None:
            return None

        try:
            parsed = self._loc_parser.parse_from_attribute(location_attr, param_die.cu["version"], die=param_die)
        except Exception:  # noqa: BLE001 - malformed locations are simply "dynamic"
            return None

        values: set[int] = set()
        for expr in self._location_expressions(parsed):
            value = self._implicit_constant(expr)
            if value is None:
                return None  # at least one location is a register/memory => dynamic
            values.add(value)

        return tuple(sorted(values)) if values else None

    @staticmethod
    def _location_expressions(parsed) -> list:
        """Yield the raw expression bytes for a single expr or a location list."""
        if isinstance(parsed, LocationExpr):
            return [parsed.loc_expr]
        expressions = []
        for entry in parsed:
            expr = getattr(entry, "loc_expr", None)
            if expr is not None:  # skip base-address / default entries
                expressions.append(expr)
        return expressions

    def _implicit_constant(self, expr_bytes) -> int | None:
        """Return the constant a location expression yields, or ``None``.

        Evaluates the expression on a small operand stack. It succeeds only when
        the expression is built purely from constant pushes and constant-folding
        arithmetic (so the optimizer baked the value into the debug info). Any
        register, frame-base, dereference or otherwise unrecognised operator
        makes the value runtime-dependent and yields ``None``.
        """
        try:
            ops = self._expr_parser.parse_expr(expr_bytes)
        except Exception:  # noqa: BLE001
            return None

        stack: list[int] = []
        for op in ops:
            name = op.op_name
            # DW_OP_stack_value marks the computed value as the object's value
            # rather than its address; it does not change the stack contents.
            if name == "DW_OP_stack_value":
                continue
            if name.startswith(_LIT_PREFIX):
                suffix = name[len(_LIT_PREFIX) :]
                if not suffix.isdigit():
                    return None
                stack.append(int(suffix))
            elif name in _CONST_OPERAND_OPS and op.args:
                stack.append(op.args[0])
            elif name == "DW_OP_plus_uconst" and op.args:
                if not stack:
                    return None
                stack.append(stack.pop() + op.args[0])
            elif name in _BINARY_OPS:
                if len(stack) < 2:
                    return None
                rhs = stack.pop()
                lhs = stack.pop()
                result = _BINARY_OPS[name](lhs, rhs)
                if result is None:
                    return None
                stack.append(result)
            elif name in _UNARY_OPS:
                if not stack:
                    return None
                stack.append(_UNARY_OPS[name](stack.pop()))
            else:
                return None  # register / memory / unsupported => dynamic

        return stack[0] if len(stack) == 1 else None
