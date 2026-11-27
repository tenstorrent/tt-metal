# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Evaluating the contract's values for concrete operands (used by check-sequence).

A value in the contract is a small expression: names and attributes (srcA.format.l1, op.transpose,
kernel.fp32_dest_acc), numbers, + - * //, comparisons, `and`/`or`/`not`, `X if C else Y`, `X if C`
(X when C holds, otherwise the field is not changed), tuples and function calls.

Evaluation, given the operands of one sequence:
  - attributes of the operands, op parameters and kernel settings are looked up;
  - the format predicates and sizes below are computed, as the LLK code computes them;
  - any other function (canonical_z_stride, matmul, datacopy ...) is left as a term: the same function
    of the same arguments is the same value, which is all a contract needs to compare;
  - a name that is not bound (Float16_b, pass_through) is a constant.
When something needed is not given (an attribute missing from the sequence), the value is UNKNOWN, and
check-sequence says it could not decide instead of guessing.
"""

import ast
import re

# DataFormat, from tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix_types.h
FORMATS = {
    "Float32": 0,
    "Float16": 1,
    "Bfp8": 2,
    "Bfp4": 3,
    "Bfp2": 11,
    "Float16_b": 5,
    "Bfp8_b": 6,
    "Bfp4_b": 7,
    "Bfp2_b": 15,
    "Lf8": 10,
    "Int8": 14,
    "UInt8": 30,
    "UInt16": 9,
    "Int32": 8,
    "UInt32": 24,
    "Tf32": 4,
    "Fp8_e4m3": 26,
}

# bytes per datum in a source register or DEST row (approximate where formats are block formats)
DATUM_BYTES = {
    "Float32": 4,
    "Tf32": 4,
    "Int32": 4,
    "UInt32": 4,
    "Float16": 2,
    "Float16_b": 2,
    "UInt16": 2,
    "Bfp8": 1,
    "Bfp8_b": 1,
    "Bfp4": 1,
    "Bfp4_b": 1,
    "Bfp2": 1,
    "Bfp2_b": 1,
    "Lf8": 1,
    "Fp8_e4m3": 1,
    "Int8": 1,
    "UInt8": 1,
}


def _fmt(x):
    if isinstance(x, str) and x in FORMATS:
        return x
    raise Unknown(f"not a format: {x!r}")


KNOWN = {
    # ckernel_defs.h:236 IS_8BIT_FORMAT
    "is_8bit": lambda f: int(_fmt(f) in ("Int8", "UInt8", "Fp8_e4m3", "Lf8")),
    # ckernel_defs.h:280: masked format Int8 (Int8, UInt8), or exactly Int32
    "is_int8_or_int32": lambda f: int(_fmt(f) in ("Int8", "UInt8", "Int32")),
    "is_fp8_e4m3": lambda f: int(_fmt(f) == "Fp8_e4m3"),
    "tile_datums": lambda t: t["face_r_dim"] * t["num_faces"] * 16,
    "datum_size": lambda f: DATUM_BYTES[_fmt(f)],
}


class Unknown(Exception):
    pass


UNCHANGED = object()


class Params(dict):
    """An operation's parameters. One not given stays a symbol (op.ct_dim), equal only to itself."""


class Ref(tuple):
    """A whole operand or parameter set passed to a function: shown by name, compared by contents."""

    def __repr__(self):
        return self[0]


class Term(tuple):
    """An uninterpreted function applied to arguments; equal when name and arguments are equal."""

    def __repr__(self):
        return f"{self[0]}({', '.join(map(fmt_value, self[1:]))})"


def fmt_value(v):
    if isinstance(v, dict):
        return "{...}"
    return repr(v) if not isinstance(v, str) else v


def _split_if(text):
    """'X if C' with no else, at the top level: (X, C). Otherwise None."""
    depth = 0
    toks = re.split(r"(\(|\)|\[|\]|\bif\b|\belse\b)", text)
    pos_if, has_else = None, False
    for i, t in enumerate(toks):
        if t in ("(", "["):
            depth += 1
        elif t in (")", "]"):
            depth -= 1
        elif t == "if" and depth == 0 and pos_if is None:
            pos_if = i
        elif t == "else" and depth == 0:
            has_else = True
    if pos_if is None or has_else:
        return None
    return "".join(toks[:pos_if]).strip(), "".join(toks[pos_if + 1 :]).strip()


def evaluate(text, env):
    """The value of a contract expression, UNCHANGED for 'X if C' when C is false; raises Unknown."""
    s = str(text).strip()
    cond = _split_if(s)
    if cond:
        x, c = cond
        return (
            _eval(ast.parse(x, mode="eval").body, env)
            if _truth(_eval(ast.parse(c, mode="eval").body, env))
            else UNCHANGED
        )
    return _eval(ast.parse(s, mode="eval").body, env)


def _truth(v):
    if isinstance(v, (int, bool)):
        return bool(v)
    raise Unknown(f"cannot decide the condition: {fmt_value(v)}")


def _num(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    raise Unknown(f"not a number: {fmt_value(v)}")


def _eval(n, env):
    if isinstance(n, ast.Constant):
        return int(n.value) if isinstance(n.value, bool) else n.value
    if isinstance(n, ast.Name):
        if n.id in env:
            return env[n.id]
        if n.id in ("True", "False"):
            return int(n.id == "True")
        return n.id  # a constant: Float16_b, pass_through, default
    if isinstance(n, ast.Attribute):
        base = _eval(n.value, env)
        if isinstance(base, dict):
            if n.attr not in base:
                if isinstance(base, Params):
                    return ast.unparse(
                        n
                    )  # a parameter the sequence does not fix: a symbol
                raise Unknown(f"{ast.unparse(n)} is not given")
            v = base[n.attr]
            return int(v) if isinstance(v, bool) else v
        raise Unknown(f"{ast.unparse(n)}: no such attribute")
    if isinstance(n, ast.Tuple):
        return tuple(_eval(e, env) for e in n.elts)
    if isinstance(n, ast.Call):
        name = ast.unparse(n.func)
        args = [_eval(a, env) for a in n.args]
        if name in KNOWN:
            return KNOWN[name](*args)
        return Term(
            (
                name,
                *[
                    (
                        Ref((ast.unparse(e), _freeze(a)))
                        if isinstance(a, dict)
                        else _freeze(a)
                    )
                    for e, a in zip(n.args, args)
                ],
            )
        )
    if isinstance(n, ast.BinOp):
        a, b = _num(_eval(n.left, env)), _num(_eval(n.right, env))
        ops = {
            ast.Add: lambda: a + b,
            ast.Sub: lambda: a - b,
            ast.Mult: lambda: a * b,
            ast.FloorDiv: lambda: a // b,
        }
        if type(n.op) in ops:
            return ops[type(n.op)]()
    if isinstance(n, ast.BoolOp):
        vals = [_eval(v, env) for v in n.values]
        if isinstance(n.op, ast.And):
            return int(all(_truth(v) for v in vals))
        return int(any(_truth(v) for v in vals))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
        return int(not _truth(_eval(n.operand, env)))
    if isinstance(n, ast.IfExp):
        return (
            _eval(n.body, env) if _truth(_eval(n.test, env)) else _eval(n.orelse, env)
        )
    if isinstance(n, ast.Compare) and len(n.ops) == 1:
        a, b = _eval(n.left, env), _eval(n.comparators[0], env)
        op = n.ops[0]
        if isinstance(op, (ast.Eq, ast.NotEq)):
            if isinstance(a, Term) or isinstance(b, Term):
                if a != b:
                    raise Unknown(f"cannot compare {fmt_value(a)} and {fmt_value(b)}")
            r = _freeze(a) == _freeze(b)
            return int(r if isinstance(op, ast.Eq) else not r)
        if isinstance(op, (ast.In, ast.NotIn)):  # codespell:ignore notin
            r = a in b
            return int(r if isinstance(op, ast.In) else not r)
        a, b = _num(a), _num(b)
        return int(
            {ast.Gt: a > b, ast.GtE: a >= b, ast.Lt: a < b, ast.LtE: a <= b}[type(op)]
        )
    raise Unknown(f"cannot evaluate {ast.unparse(n)}")


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v
