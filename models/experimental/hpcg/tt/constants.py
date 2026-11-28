# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The constants of shared_constants.h, read from the header: one definition for the Python ops, xhpcg's C++ ops and
the kernels. Each `constexpr` becomes a module attribute (an int, or a tuple for an array); each
`HPCG_<OP>_CBS(X)` table becomes CBS["<OP>"], a dict of circular-buffer name -> index."""

import ast
import operator
import pathlib
import re

HEADER = pathlib.Path(__file__).with_name("shared_constants.h")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.floordiv}


def _value(expr, known):
    """An integer expression of the header (integers, + - * /, parentheses, names defined above it)."""

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name) and n.id in known:
            return known[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError(f"{HEADER.name}: unsupported expression {expr!r}")

    return ev(ast.parse(expr.strip(), mode="eval").body)


def _read():
    text = re.sub(r"//[^\n]*", "", HEADER.read_text())
    known, cbs = {}, {}
    for name, array, expr in re.findall(r"constexpr\s+[\w:]+\s+(\w+)\s*(\[\s*\])?\s*=\s*([^;]+);", text):
        if array:
            known[name] = tuple(_value(v, known) for v in expr.strip().strip("{}").split(","))
        else:
            known[name] = _value(expr, known)
    for op, body in re.findall(r"#define\s+HPCG_(\w+)_CBS\(X\)((?:[^\n]*\\\n)*[^\n]*)", text):
        cbs[op] = {k: int(v) for k, v in re.findall(r"X\((\w+),\s*(\d+)\)", body)}
    return known, cbs


_known, CBS = _read()
globals().update(_known)
