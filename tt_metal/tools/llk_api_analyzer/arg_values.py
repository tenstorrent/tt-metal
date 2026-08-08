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
# runtime, so the whole expression is treated as dynamic. The operators are
# applied on the DWARF "generic type" (see :meth:`ConstantArgEvaluator._apply_binary`).
_BINARY_OPS = frozenset(
    {
        "DW_OP_plus",
        "DW_OP_minus",
        "DW_OP_mul",
        "DW_OP_div",
        "DW_OP_mod",
        "DW_OP_and",
        "DW_OP_or",
        "DW_OP_xor",
        "DW_OP_shl",
        "DW_OP_shr",
        "DW_OP_shra",
    }
)
_UNARY_OPS = frozenset({"DW_OP_neg", "DW_OP_not", "DW_OP_abs"})


class ConstantArgEvaluator:
    """Evaluates whether a formal parameter has a known compile-time constant."""

    def __init__(self, dwarf):
        self._expr_parser = DWARFExprParser(dwarf.structs)
        # Build the location parser unconditionally. Inline exprloc constants
        # (``DW_OP_constu N; DW_OP_stack_value``) — this tool's primary target —
        # parse without any loclist section, so we must not gate on
        # ``location_lists()`` being present (it is ``None`` when the ELF has no
        # ``.debug_loc`` / ``.debug_loclists``). ``LocationParser`` only touches
        # the loclists for loclist-pointer attributes, which cannot occur when
        # there is no such section.
        self._loc_parser = LocationParser(dwarf.location_lists())
        # DWARF stack values are of the "generic type": an integral type the
        # size of a target address, with two's-complement wraparound. All
        # folded results are reduced modulo ``2**bits``.
        self._bits = 8 * dwarf.structs.address_size
        self._mask = (1 << self._bits) - 1

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
        if location_attr is None:
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
        saw_stack_value = False
        for op in ops:
            name = op.op_name
            # DW_OP_stack_value marks the computed value as the object's value
            # rather than its address; it does not change the stack contents.
            # It must appear (and terminate the value computation) for the
            # result to be a real constant rather than a memory address.
            if name == "DW_OP_stack_value":
                saw_stack_value = True
                continue
            if saw_stack_value:
                return None  # trailing ops (e.g. composite pieces) => not a simple constant
            if name.startswith(_LIT_PREFIX):
                suffix = name[len(_LIT_PREFIX) :]
                if not suffix.isdigit():
                    return None
                stack.append(int(suffix) & self._mask)
            elif name in _CONST_OPERAND_OPS and op.args:
                stack.append(op.args[0] & self._mask)
            elif name == "DW_OP_plus_uconst" and op.args:
                if not stack:
                    return None
                stack.append((stack.pop() + op.args[0]) & self._mask)
            elif name in _BINARY_OPS:
                if len(stack) < 2:
                    return None
                rhs = stack.pop()
                lhs = stack.pop()
                result = self._apply_binary(name, lhs, rhs)
                if result is None:
                    return None
                stack.append(result)
            elif name in _UNARY_OPS:
                if not stack:
                    return None
                result = self._apply_unary(name, stack.pop())
                if result is None:
                    return None
                stack.append(result)
            else:
                return None  # register / memory / unsupported => dynamic

        if not saw_stack_value:
            return None  # a memory/register location, not an implicit constant value
        return stack[0] if len(stack) == 1 else None

    def _to_signed(self, value: int) -> int:
        """Reinterpret a generic-type (unsigned) stack value as two's-complement."""
        value &= self._mask
        if value & (1 << (self._bits - 1)):
            value -= 1 << self._bits
        return value

    def _apply_binary(self, name: str, lhs: int, rhs: int) -> int | None:
        """Apply a binary operator on two generic-type (address-sized) values.

        Per DWARF5 §2.5.1.4 the generic type is an address-sized integral type
        with two's-complement wraparound (overflow never traps). Arithmetic is
        unsigned modulo ``2**bits`` *except* ``DW_OP_div`` (signed); the result
        is reduced back into the generic type.
        """
        mask = self._mask
        if name == "DW_OP_plus":
            return (lhs + rhs) & mask
        if name == "DW_OP_minus":
            return (lhs - rhs) & mask
        if name == "DW_OP_mul":
            return (lhs * rhs) & mask
        if name == "DW_OP_and":
            return lhs & rhs & mask
        if name == "DW_OP_or":
            return (lhs | rhs) & mask
        if name == "DW_OP_xor":
            return (lhs ^ rhs) & mask
        if name == "DW_OP_div":
            # Signed division truncating toward zero: DW_OP_div is signed, and
            # GCC's evaluator (like C++) rounds toward zero, unlike Python //.
            a, b = self._to_signed(lhs), self._to_signed(rhs)
            if b == 0:
                return None
            q = abs(a) // abs(b)
            return (-q if (a < 0) != (b < 0) else q) & mask
        if name == "DW_OP_mod":
            # For the generic type DW_OP_mod is unsigned (modulo == remainder).
            divisor = rhs & mask
            if divisor == 0:
                return None
            return (lhs & mask) % divisor
        if name in ("DW_OP_shl", "DW_OP_shr", "DW_OP_shra"):
            count = rhs & mask
            if name == "DW_OP_shl":
                return (lhs << count) & mask if count < self._bits else 0
            if name == "DW_OP_shr":  # logical right shift, zero-fill
                return (lhs & mask) >> count if count < self._bits else 0
            # DW_OP_shra: arithmetic right shift, sign-preserving.
            if count >= self._bits:
                return mask if (lhs >> (self._bits - 1)) & 1 else 0
            return (self._to_signed(lhs) >> count) & mask
        return None

    def _apply_unary(self, name: str, value: int) -> int | None:
        """Apply a unary operator on a generic-type (address-sized) value."""
        mask = self._mask
        if name == "DW_OP_neg":
            return (-value) & mask
        if name == "DW_OP_not":
            return (~value) & mask
        if name == "DW_OP_abs":
            return abs(self._to_signed(value)) & mask
        return None
