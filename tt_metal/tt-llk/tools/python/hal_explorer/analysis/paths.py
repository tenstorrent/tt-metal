# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Symbolic walk of every entry-to-exit path, with each loop folded into one Loop item.

Values that are only known at runtime (parameters, globals, MMIO reads and
loop-carried PHIs) stay symbolic. Each path records its branch decisions outside
loops and the exits of outermost loops, the same decisions the interpreter
records, so a concrete run can be matched to the path it took.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

from hal_explorer.gimple.expr import Scope, parse, parse_atom
from hal_explorer.isa import events
from hal_explorer.model import Loop

MAX_PATHS = 1024


@dataclass
class State:
    conditions: Tuple[
        Tuple[str, bool], ...
    ] = ()  # (text, depends on a loop-carried value)
    items: Tuple = ()
    warnings: Tuple = ()
    decisions: Tuple = ()
    aliases: Dict[str, object] = field(default_factory=dict)
    unknown: frozenset = frozenset()


@dataclass
class Path:
    conditions: List[str]
    items: List
    warnings: List[dict]
    decisions: tuple
    ended: str  # "return" or "noreturn"


def outermost_headers(loops):
    return {
        h
        for h, loop in loops.items()
        if not any(h in other.body for o, other in loops.items() if o != h)
    }


class Walker:
    def __init__(self, function, loops, tables, sources, definitions, values=None):
        self.function = function
        self.values = values or {}  # inputs known to every walk, e.g. global pointers
        self.loops = loops
        self.tables = tables
        self.sources = sources
        self.definitions = definitions
        self.truncated = False
        self.finished = 0
        self.in_loop = set().union(*(loop.body for loop in loops.values()))
        self.outermost = outermost_headers(loops)
        # Values that change from one iteration to the next: loop-carried PHIs and MMIO reads.
        self.varying = {
            header: loop.carried
            | {
                stmt.data["target"]
                for index in loop.body
                for stmt in function.blocks[index].statements
                if stmt.kind == "mmio_read"
            }
            for header, loop in loops.items()
        }

    def scope(self, state):
        scope = Scope(
            self.function, self.definitions, values=self.values, unknown=state.unknown
        )
        scope.aliases = state.aliases
        return scope

    def paths(self):
        result = []
        for kind, state in self.walk(self.function.entry, None, State(), None):
            conditions = [text for text, _ in state.conditions]
            result.append(
                Path(
                    conditions,
                    list(state.items),
                    list(state.warnings),
                    state.decisions,
                    kind,
                )
            )
        return result

    def walk(self, start, pred, state, region, skip_phis=False):
        """Yield ('back', state), ('exit', target, source, state), or (end, state) at a function exit."""
        stack = [(start, pred, state, skip_phis)]
        while stack:
            if self.finished >= MAX_PATHS:
                self.truncated = True
                return
            index, pred, state, skip = stack.pop()
            if index is None:  # a function exit reached inside a loop
                yield from self.finish(pred, state, region)
                continue
            if region is not None and not skip:
                if index == region.header:
                    yield ("back", state)
                    continue
                if index not in region.body:
                    yield ("exit", index, pred, state)
                    continue
            if (
                index in self.loops
                and (region is None or index != region.header)
                and not skip
            ):
                stack.extend(self.enter_loop(self.loops[index], pred, state))
                continue
            block = self.function.blocks[index]
            if not skip:
                state = self.apply_phis(block, pred, state)
            state = self.run_statements(block, state)
            if block.terminator[0] in ("return", "noreturn"):
                yield from self.finish(block.terminator[0], state, region)
                continue
            branching = (
                block.terminator[0] in ("if", "switch") and index not in self.in_loop
            )
            for target, condition in self.edges(block, state):
                next_state = state
                if condition is not None:
                    next_state = replace(
                        state, conditions=state.conditions + (condition,)
                    )
                if branching:
                    next_state = replace(
                        next_state, decisions=next_state.decisions + ((index, target),)
                    )
                stack.append((target, index, next_state, False))

    def finish(self, kind, state, region):
        if region is None:
            self.finished += 1
        yield (kind, state)

    def apply_phis(self, block, pred, state):
        if not block.phis:
            return state
        aliases = dict(state.aliases)
        scope = self.scope(state)
        for name, incoming in block.phis:
            if pred in incoming:
                atom = parse_atom(incoming[pred])
                value = scope.value(atom)
                if isinstance(value, int):
                    aliases[name] = parse_atom(str(value))
                else:
                    aliases[name] = state.aliases.get(getattr(atom, "name", None), atom)
        return replace(state, aliases=aliases)

    def run_statements(self, block, state):
        if not block.statements:
            return state
        scope = self.scope(state)
        items, warnings = list(state.items), list(state.warnings)
        for stmt in block.statements:
            if stmt.kind in events.HIDING:
                warnings.append(events.warning(stmt, self.sources))
                continue
            item = events.describe(stmt, scope, self.tables, self.sources)
            if item is not None:
                items.append(item)
        return replace(state, items=tuple(items), warnings=tuple(warnings))

    def edges(self, block, state):
        """(target, condition or None) per feasible edge; constant branches have one."""
        term = block.terminator
        if term[0] == "goto":
            return [(term[1], None)]
        scope = self.scope(state)
        if term[0] == "if":
            node = parse(term[1])
            value = scope.value(node)
            if isinstance(value, int):
                return [(term[2] if value else term[3], None)]
            carried = bool(scope.leaves(node) & state.unknown)
            return [
                (term[2], (scope.condition_text(term[1], True), carried)),
                (term[3], (scope.condition_text(term[1], False), carried)),
            ]
        index = parse_atom(term[1])
        value = scope.value(index)
        cases = term[2]
        if isinstance(value, int):
            return [(case_target(cases, value), None)]
        name = scope.text(index)
        carried = bool(scope.leaves(index) & state.unknown)
        texts = {}
        for low, high, target in cases:
            if low is None:
                listed = ", ".join(
                    str(l) if l == h else f"{l}..{h}"
                    for l, h, _ in cases
                    if l is not None
                )
                text = f"{name} not in {{{listed}}}"
            else:
                text = (
                    f"{name} == {low}" if low == high else f"{low} <= {name} <= {high}"
                )
            texts.setdefault(target, []).append(text)
        return [(target, (" || ".join(t), carried)) for target, t in texts.items()]

    def enter_loop(self, loop, pred, state):
        """Stack entries continuing after the loop, one per way out of it."""
        header = self.function.blocks[loop.header]
        entry = self.apply_phis(header, pred, state)
        aliases = {k: v for k, v in entry.aliases.items() if k not in loop.carried}
        inner = State(
            aliases=aliases, unknown=state.unknown | self.varying[loop.header]
        )
        backs, exits = [], []
        for outcome in self.walk(loop.header, pred, inner, loop, skip_phis=True):
            (backs if outcome[0] == "back" else exits).append(outcome)
        item = loop_item(backs)
        if item is not None and not exits:
            never = State(
                state.conditions,
                state.items + (item,),
                state.warnings,
                state.decisions,
                state.aliases,
                state.unknown,
            )
            return [(None, "noreturn", never, False)]
        entries = []
        for outcome in exits:
            kind, out = outcome[0], outcome[-1]
            items, tail = list(state.items), list(out.items)
            if item is not None:
                items.append(item)
                if any(keys(b[-1].items) == keys(tail) for b in backs):
                    tail = []  # the way out runs the same body as a full iteration
            target = outcome[1] if kind == "exit" else kind
            decisions = state.decisions
            if loop.header in self.outermost:
                decisions += ((loop.header, target),)
            merged = State(
                state.conditions + tuple(c for c in out.conditions if not c[1]),
                tuple(items + tail),
                state.warnings + out.warnings,
                decisions,
                {
                    **state.aliases,
                    **{k: v for k, v in out.aliases.items() if k not in loop.carried},
                },
                state.unknown,
            )
            if kind == "exit":
                entries.append((outcome[1], outcome[2], merged, False))
            else:
                entries.append((None, kind, merged, False))
        return entries


def case_target(cases, value):
    for low, high, target in cases:
        if low is not None and low <= value <= high:
            return target
    return next(target for low, _, target in cases if low is None)


def keys(items):
    return [item.key() for item in items]


def loop_item(backs):
    """One Loop for all ways around the loop; differing bodies become variants.

    The repeat condition is the last iteration-dependent condition on the way back to
    the header; the other conditions tell the variants apart.
    """
    if not backs:
        return None
    variants, by_body, repeats = [], {}, []
    for _, state in backs:
        varying = [i for i, (_, carried) in enumerate(state.conditions) if carried]
        repeat = state.conditions[varying[-1]][0] if varying else "always"
        repeats.append(repeat)
        conditions = [
            t
            for i, (t, _) in enumerate(state.conditions)
            if not varying or i != varying[-1]
        ]
        body = list(state.items)
        signature = tuple(keys(body))
        if signature not in by_body:
            by_body[signature] = {"when": [], "body": body}
            variants.append(by_body[signature])
        by_body[signature]["when"].append(conditions)
    repeat_while = " || ".join(dict.fromkeys(repeats))
    if len(variants) == 1:
        return Loop(repeat_while, variants[0]["body"])
    return Loop(repeat_while, [], variants)
