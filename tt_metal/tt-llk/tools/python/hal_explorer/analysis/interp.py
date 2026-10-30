# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run a function's GIMPLE on concrete inputs and record exactly what it issues.

Parameters and globals take the given values. MMIO reads and other values only
known on hardware stay unknown; a branch on them forks the run, up to a budget.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List

from hal_explorer.analysis.paths import case_target, outermost_headers
from hal_explorer.gimple.expr import Scope, parse, parse_atom
from hal_explorer.isa import events

STEP_LIMIT = 4000
FORK_LIMIT = 8
# After this many forks at one block, an MMIO-dependent branch leaves its loop: a poll ends.
FORKS_PER_BLOCK = 2


@dataclass
class Trace:
    items: List = field(default_factory=list)
    decisions: tuple = ()
    warnings: List[dict] = field(default_factory=list)
    complete: bool = True
    note: str = ""


@dataclass
class Run:
    block: int
    pred: object
    values: dict
    trace: Trace
    steps: int = 0
    forks: dict = field(default_factory=dict)
    reads: int = 0


def derived(key):
    """A fixed 32-bit value for key, the same in every function that asks for it."""
    return int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)


def load_key(text, positions):
    """(parameter position, byte offset) of a load through a pointer parameter, else None."""
    m = re.search(r"([A-Za-z_]\w*?)_\d+\(D\)", text)
    if not m or m.group(1) not in positions:
        return None
    offset = sum(int(x) for x in re.findall(r"\+ (\d+)B?\]", text))
    offset += sum(4 * int(x) for x in re.findall(r"\[(\d+)\]", text))
    return ("load", positions[m.group(1)], offset)


class Interpreter:
    """read_value: None forks on MMIO reads; an int is returned by every read; "derived"
    gives the k-th read of a run a fixed value. derive_loads gives loads through pointer
    parameters a value fixed by parameter position and offset, so two functions that
    read the same memory see the same values."""

    def __init__(
        self,
        function,
        loops,
        tables,
        sources,
        definitions,
        read_value=None,
        derive_loads=False,
    ):
        self.function = function
        self.read_value = read_value
        positions = {name: index for index, (_, name) in enumerate(function.parameters)}
        self.load_value = (
            (
                lambda text: (lambda key: derived(key) if key else None)(
                    load_key(text, positions)
                )
            )
            if derive_loads
            else None
        )
        self.loops = loops
        self.tables = tables
        self.sources = sources
        self.definitions = definitions
        self.in_loop = set().union(*(loop.body for loop in loops.values()))
        self.outermost = {h: loops[h].body for h in outermost_headers(loops)}
        self.steps = 0  # blocks executed by every run so far

    def run(self, inputs):
        """Every trace for these inputs; more than one when a branch depends on MMIO."""
        pending = [Run(self.function.entry, None, dict(inputs), Trace())]
        traces = []
        while pending:
            run = pending.pop()
            forks = self.step_until_end(run)
            self.steps += run.steps
            for fork in forks:
                if len(traces) + len(pending) < FORK_LIMIT:
                    pending.append(fork)
                else:
                    run.trace.complete = False
                    run.trace.note = "too many MMIO-dependent branches"
            traces.append(run.trace)
        return traces

    def scope(self, values):
        scope = Scope(self.function, self.definitions, values=values)
        scope.load_value = self.load_value
        return scope

    def step_until_end(self, run):
        """Advance run to a function exit; return runs forked off on unknown branches."""
        forks = []
        while True:
            if run.steps >= STEP_LIMIT:
                run.trace.complete = False
                run.trace.note = f"stopped after {STEP_LIMIT} blocks"
                return forks
            run.steps += 1
            block = self.function.blocks[run.block]
            self.assign_phis(block, run)
            scope = self.scope(run.values)
            for stmt in block.statements:
                if stmt.kind in events.HIDING:
                    if not any(
                        w["message"] == events.warning(stmt, self.sources)["message"]
                        for w in run.trace.warnings
                    ):
                        run.trace.warnings.append(events.warning(stmt, self.sources))
                    continue
                item = events.describe(stmt, scope, self.tables, self.sources)
                if item is not None:
                    run.trace.items.append(item)
                if stmt.kind == "mmio_read" and self.read_value is not None:
                    run.reads += 1
                    value = (
                        derived(("read", run.reads))
                        if self.read_value == "derived"
                        else self.read_value
                    )
                    run.values[stmt.data["target"]] = value
                    scope = self.scope(run.values)
            term = block.terminator
            if term[0] in ("return", "noreturn"):
                self.end_decision(run, term[0])
                return forks
            targets = self.targets(term, scope)
            if len(targets) > 1 and run.forks.get(run.block, 0) >= FORKS_PER_BLOCK:
                targets = [self.leaving_loop(run.block, targets)]
                run.trace.note = "assumed MMIO-dependent loops end"
            elif len(targets) > 1:
                run.forks[run.block] = run.forks.get(run.block, 0) + 1
            for target in targets[1:]:
                forks.append(self.fork(run, target))
            self.take(run, targets[0])

    def targets(self, term, scope):
        if term[0] == "goto":
            return [term[1]]
        if term[0] == "if":
            value = scope.value(parse(term[1]))
            if isinstance(value, int):
                return [term[2] if value else term[3]]
            return [term[2], term[3]]
        value = scope.value(parse_atom(term[1]))
        if isinstance(value, int):
            return [case_target(term[2], value)]
        return list(dict.fromkeys(target for _, _, target in term[2]))

    def leaving_loop(self, block, targets):
        """The target that exits the innermost loop around block, else the first target."""
        around = [loop for loop in self.loops.values() if block in loop.body]
        if around:
            innermost = min(around, key=lambda loop: len(loop.body))
            for target in targets:
                if target not in innermost.body:
                    return target
        return targets[0]

    def take(self, run, target):
        source = run.block
        branching = self.function.blocks[source].terminator[0] in ("if", "switch")
        if branching and source not in self.in_loop:
            run.trace.decisions += ((source, target),)
        for header, body in self.outermost.items():
            if source in body and target not in body:
                run.trace.decisions += ((header, target),)
        run.pred, run.block = source, target

    def fork(self, run, target):
        trace = Trace(
            list(run.trace.items), run.trace.decisions, list(run.trace.warnings)
        )
        copy = Run(
            run.block, run.pred, dict(run.values), trace, run.steps, dict(run.forks)
        )
        self.take(copy, target)
        return copy

    def assign_phis(self, block, run):
        if not block.phis:
            return
        scope = self.scope(run.values)
        incoming = {}
        for name, sources in block.phis:
            if run.pred in sources:
                incoming[name] = scope.value(parse_atom(sources[run.pred]))
        run.values.update(incoming)

    def end_decision(self, run, kind):
        for header, body in self.outermost.items():
            if run.block in body:
                run.trace.decisions += ((header, kind),)
