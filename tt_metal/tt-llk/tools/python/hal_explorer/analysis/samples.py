# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Concrete inputs to run a function on, and exact linear fits of its instruction count."""

import itertools
import random
import re
from fractions import Fraction

from hal_explorer.gimple.expr import Symbol
from hal_explorer.gimple.types import SCALAR_TYPES, Pointer, type_info

MAX_VECTORS = 256
INTEGER_VALUES = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    255,
    256,
    1023,
    1024,
    65535,
    65536,
)
ENUM_VALUES = (0, 1, 2, 3, 4, 7)
GLOBAL_VALUES = (0, 1)


def known_type(text):
    text = " ".join(re.sub(r"\b(const|volatile)\b", " ", text).split()).removeprefix(
        "std::"
    )
    return (
        text in SCALAR_TYPES
        or bool(re.fullmatch(r"u?int(8|16|32|64)_t", text))
        or "*" in text
        or "&" in text
    )


def candidates(type_text):
    """Values to try for one parameter; None means unknown (pointers and references)."""
    info = type_info(type_text)
    if info.pointer:
        return [None]
    if info.bits == 1:
        return [0, 1]
    if not known_type(type_text):
        return list(ENUM_VALUES)
    top = (1 << (info.bits - 1)) - 1 if info.signed else (1 << info.bits) - 1
    values = [v for v in INTEGER_VALUES if v <= top] + [top]
    if info.signed:
        values += [-1, -2, -(1 << (info.bits - 1))]
    return list(dict.fromkeys(values))


def globals_read(definitions, scope):
    """{name: 'pointer' | 'integer'} for globals the function loads by name."""
    found = {}
    for name, node in definitions.items():
        if isinstance(node, Symbol) and not node.address:
            found[node.name] = "pointer" if scope.type_of(name).pointer else "integer"
    return found


def input_space(parameters, globals_found):
    """[(input name, candidate values)] for parameters, then globals."""
    space = [(name, candidates(type_text)) for type_text, name in parameters]
    for name, kind in sorted(globals_found.items()):
        space.append(
            (name, [Pointer(name)] if kind == "pointer" else list(GLOBAL_VALUES))
        )
    return space


def vectors(space, limit=MAX_VECTORS):
    """Input vectors covering every candidate of every input, all combinations when they fit."""
    names = [name for name, _ in space]
    choices = [values for _, values in space]
    total = 1
    for values in choices:
        total *= len(values)
    if total <= limit:
        return [dict(zip(names, combo)) for combo in itertools.product(*choices)]
    defaults = [values[0] for values in choices]
    chosen, seen = [], set()

    def add(combo):
        key = tuple(str(v) for v in combo)
        if key not in seen:
            seen.add(key)
            chosen.append(dict(zip(names, combo)))

    add(defaults)
    for position, values in enumerate(choices):
        for value in values:
            combo = list(defaults)
            combo[position] = value
            add(combo)
    rng = random.Random(0)
    attempts = 0
    while len(chosen) < limit and attempts < limit * 20:
        attempts += 1
        add([rng.choice(values) for values in choices])
    return chosen


def fit_linear(points, names):
    """'a + b × x + …' reproducing every (inputs, count) point exactly, else None."""
    variables = [n for n in names if all(isinstance(p[0].get(n), int) for p in points)]
    rows = [
        [Fraction(1)] + [Fraction(p[0][n]) for n in variables] + [Fraction(p[1])]
        for p in points
    ]
    solution = solve(rows, len(variables) + 1)
    if solution is None:
        return None
    for inputs, count in points:
        predicted = solution[0] + sum(
            c * inputs[n] for c, n in zip(solution[1:], variables)
        )
        if predicted != count:
            return None
    terms = (
        [] if solution[0] == 0 and any(solution[1:]) else [format_number(solution[0])]
    )
    for coefficient, name in zip(solution[1:], variables):
        if coefficient:
            terms.append(
                name if coefficient == 1 else f"{format_number(coefficient)} × {name}"
            )
    return " + ".join(terms).replace("+ -", "- ")


def format_number(value):
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def solve(rows, unknowns):
    """Exact Gauss-Jordan solve of the system; unknowns without a pivot are zero."""
    matrix = [list(r) for r in rows]
    pivots, row = [], 0
    for column in range(unknowns):
        pivot = next(
            (r for r in range(row, len(matrix)) if matrix[r][column] != 0), None
        )
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        matrix[row] = [v / matrix[row][column] for v in matrix[row]]
        for r in range(len(matrix)):
            if r != row and matrix[r][column] != 0:
                factor = matrix[r][column]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[row])]
        pivots.append(column)
        row += 1
    for r in range(row, len(matrix)):
        if matrix[r][-1] != 0:
            return None
    solution = [Fraction(0)] * unknowns
    for r, column in enumerate(pivots):
        solution[column] = matrix[r][-1]
    return solution
