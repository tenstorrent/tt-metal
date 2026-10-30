# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Natural loops of a function's control-flow graph."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List


@dataclass(frozen=True)
class Loop:
    header: int
    body: FrozenSet[int]  # blocks of the loop, header included
    latches: FrozenSet[int]  # blocks with an edge back to the header
    carried: FrozenSet[str]  # header PHIs fed from inside the loop


def successors(block):
    term = block.terminator
    if term[0] == "goto":
        return [term[1]]
    if term[0] == "if":
        return [term[2], term[3]]
    if term[0] == "switch":
        return list(dict.fromkeys(target for _, _, target in term[2]))
    return []


def dominators(function):
    blocks = function.blocks
    order = reachable(function)
    preds = {b: [] for b in order}
    for b in order:
        for s in successors(blocks[b]):
            if s in preds:
                preds[s].append(b)
    dom = {b: set(order) for b in order}
    dom[function.entry] = {function.entry}
    changed = True
    while changed:
        changed = False
        for b in order:
            if b == function.entry:
                continue
            new = (
                {b} | set.intersection(*(dom[p] for p in preds[b])) if preds[b] else {b}
            )
            if new != dom[b]:
                dom[b], changed = new, True
    return dom, preds


def reachable(function):
    seen, stack, order = set(), [function.entry], []
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        order.append(b)
        stack.extend(reversed(successors(function.blocks[b])))
    return order


def find_loops(function) -> Dict[int, Loop]:
    """Loops by header. Irreducible cycles (no dominating header) are not loops here."""
    if function.entry is None:
        return {}
    dom, preds = dominators(function)
    latches: Dict[int, List[int]] = {}
    for b in dom:
        for s in successors(function.blocks[b]):
            if s in dom.get(b, ()):
                latches.setdefault(s, []).append(b)
    loops = {}
    for header, tails in latches.items():
        body, stack = {header}, list(tails)
        while stack:
            b = stack.pop()
            if b not in body:
                body.add(b)
                stack.extend(preds[b])
        carried = frozenset(
            name
            for name, incoming in function.blocks[header].phis
            if any(pred in body for pred in incoming)
        )
        loops[header] = Loop(header, frozenset(body), frozenset(tails), carried)
    return loops


def has_irreducible_cycle(function, loops):
    """True when a cycle exists that no natural loop accounts for."""
    order = reachable(function)
    position = {b: i for i, b in enumerate(order)}
    state, stack = {}, [
        (function.entry, iter(successors(function.blocks[function.entry])))
    ]
    state[function.entry] = 1
    while stack:
        block, children = stack[-1]
        child = next(children, None)
        if child is None:
            state[block] = 2
            stack.pop()
            continue
        if state.get(child) == 1 and child not in loops:
            return True
        if child not in state and child in position:
            state[child] = 1
            stack.append((child, iter(successors(function.blocks[child]))))
    return False
