# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Everything shown for one function: its distinct sequences, concrete samples and warnings."""

import hashlib

from hal_explorer.analysis import paths as path_walk
from hal_explorer.analysis import samples as sampling
from hal_explorer.analysis.interp import Interpreter
from hal_explorer.gimple import loops as loop_analysis
from hal_explorer.gimple import reader
from hal_explorer.gimple.expr import Scope, definitions_of
from hal_explorer.gimple.types import Pointer
from hal_explorer.model import Access, Function, Group, Instruction, Loop, Sample

MAX_SHOWN_SAMPLES = 64
SAMPLE_STEP_BUDGET = 200_000


def fingerprint(items):
    return hashlib.sha256(repr([item.key() for item in items]).encode()).hexdigest()[
        :12
    ]


def count_items(items):
    """(fixed count, per-iteration counts of each runtime loop)."""
    fixed, loops = 0, []
    for item in items:
        if isinstance(item, Instruction):
            fixed += 1
        elif isinstance(item, Loop):
            bodies = (
                [item.body] if not item.variants else [v["body"] for v in item.variants]
            )
            per = sorted({sum(isinstance(i, Instruction) for i in b) for b in bodies})
            loops.append(per)
    return fixed, loops


def count_text(fixed, loops):
    if not loops:
        return str(fixed)
    parts = [str(fixed)] if fixed else []
    for per in loops:
        parts.append(
            f"{per[0]} per iteration"
            if len(per) == 1
            else f"{per[0]}–{per[-1]} per iteration"
        )
    return " + ".join(parts)


class Context:
    """Parsed function plus the shared analyses both walkers need."""

    def __init__(self, function, tables, sources):
        self.function = function
        self.tables = tables
        self.sources = sources
        self.loops = loop_analysis.find_loops(function)
        self.definitions = definitions_of(function)
        self.interpreter = Interpreter(
            function, self.loops, tables, sources, self.definitions
        )
        scope = Scope(function, self.definitions)
        self.globals = sampling.globals_read(self.definitions, scope)

    def input_space(self):
        return sampling.input_space(self.function.parameters, self.globals)


def analyse(context, gimple_text=""):
    function, tables, sources = context.function, context.tables, context.sources
    pointers = {
        name: Pointer(name)
        for name, kind in context.globals.items()
        if kind == "pointer"
    }
    walker = path_walk.Walker(
        function, context.loops, tables, sources, context.definitions, pointers
    )
    paths = walker.paths()

    warnings = {}
    if loop_analysis.has_irreducible_cycle(function, context.loops):
        warnings["irreducible"] = {
            "message": "the function has a cycle that is not a natural loop; paths through it are not listed",
            "origin": None,
        }
    if walker.truncated:
        warnings["truncated"] = {
            "message": f"more than {path_walk.MAX_PATHS} paths; only the first {path_walk.MAX_PATHS} are listed",
            "origin": None,
        }
    for path in paths:
        for warning in path.warnings:
            warnings.setdefault(warning["message"], warning)

    groups, by_decisions = {}, {}
    for path in paths:
        key = fingerprint(path.items)
        if key not in groups:
            fixed, loops = count_items(path.items)
            groups[key] = Group(
                key, path.items, None if loops else fixed, count_text(fixed, loops), []
            )
        groups[key].when.append(path.conditions)
        by_decisions[path.decisions] = groups[key]

    shown, points, concrete = [], {}, {}
    for inputs in sampling.vectors(context.input_space()):
        if context.interpreter.steps > SAMPLE_STEP_BUDGET:
            break
        for trace in context.interpreter.run(inputs):
            group = by_decisions.get(trace.decisions)
            count = sum(isinstance(item, Instruction) for item in trace.items)
            check_against_group(trace, group, warnings, walker.truncated)
            if trace.complete and group is not None and group.count is None:
                points.setdefault(group.fingerprint, []).append((plain(inputs), count))
                concrete.setdefault(group.fingerprint, {})[
                    tuple(i.key() for i in trace.items)
                ] = trace.items
            if len(shown) < MAX_SHOWN_SAMPLES:
                fingerprint_ = group.fingerprint if group else None
                shown.append(
                    Sample(
                        plain(inputs), count, fingerprint_, trace.complete, trace.note
                    )
                )
            for warning in trace.warnings:
                warnings.setdefault(warning["message"], warning)

    for key, runs in concrete.items():
        # A loop that issues the same words on every run is shown with those exact words.
        if len(runs) == 1 and groups[key].count is None:
            (items,) = runs.values()
            groups[key].sequence = list(items)
            groups[key].count = sum(isinstance(i, Instruction) for i in items)
            groups[key].count_text = str(groups[key].count)
            points.pop(key, None)
    names = [name for name, _ in context.input_space()]
    for key, group_points in points.items():
        formula = sampling.fit_linear(group_points, names)
        if formula:
            groups[key].count_text = (
                f"{formula}  (fitted on {len(group_points)} samples; {groups[key].count_text})"
            )

    ordered = sorted(groups.values(), key=lambda g: (g.count is None, -(g.count or 0)))
    return Function(
        name=function.name,
        signature=function.signature,
        mangled=function.mangled,
        path_count=len(paths),
        truncated=walker.truncated,
        groups=ordered,
        samples=shown,
        count_formula=None,
        warnings=list(warnings.values()),
        gimple=[
            {"text": line, "mark": reader.classify_line(line)}
            for line in gimple_text.splitlines()
        ],
    )


def check_against_group(trace, group, warnings, truncated):
    """A fully constant path must issue exactly what the concrete run issued."""
    if group is None:
        if trace.complete and not truncated:
            warnings.setdefault(
                "unmatched",
                {
                    "message": "a concrete run took a path the symbolic walk did not list",
                    "origin": None,
                },
            )
        return
    if not trace.complete or not constant_sequence(group.sequence):
        return
    if [item.key() for item in trace.items] != [item.key() for item in group.sequence]:
        warnings.setdefault(
            "disagree",
            {
                "message": "internal check failed: a concrete run disagrees with the listed sequence; please report this input",
                "origin": None,
            },
        )


def constant_sequence(items):
    """True when no item depends on a runtime value, so every run issues it identically."""
    for item in items:
        if isinstance(item, Loop):
            return False
        if isinstance(item, Instruction) and item.runtime_mask:
            return False
        if isinstance(item, Access):
            if item.address is None or (
                item.operation == "write" and item.value is None
            ):
                return False
    return True


def plain(inputs):
    return {
        name: (f"&{value.symbol}" if isinstance(value, Pointer) else value)
        for name, value in inputs.items()
    }
