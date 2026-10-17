# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stacks found by watching WHO ran, not by classifying what the object graph looks like.

THE OP STREAM CANNOT PRODUCE A BOUNDARY. Two towers built from the same layer architecture emit
identical op signatures, so 32 encoder layers followed by another 32 read as one uninterrupted run of
64 -- nothing in the sequence says where the first tower ended. Periodicity is worse than useless
there: a hybrid stack alternating attention and Mamba layers has a period of TWO layers, so a
30-layer stack scans as 15 repeats of a 2-layer motif and reports the wrong depth. Boundaries are a
structural fact, and the ops are downstream of structure.

BUT OBJECT IDENTITY HAS THE BOUNDARY EXACTLY. Two identical towers are indistinguishable in ops and
never in identity: they are different containers holding different objects. So instead of asking the
walk to CLASSIFY a list as a stack -- the step that fails when the wrappers are differently typed and
share no base -- this records which object emitted each op and reads the stacks off the execution.

Traversal already reaches these objects. What the walk declines to do is classify them, and
classification is exactly what this does not need: a container whose elements each ran and each
emitted the same op subsequence IS a stack, whatever its elements are called, whether they share a
base, and however many of them there are.

REFERENCE-FREE AND FRAMEWORK-FREE. No torch, no HF config, no checkpoint, no naming convention -- it
observes the model the tool was going to run anyway. That makes it the one witness available to a
model with no HF lineage at all, which is the case the reference census cannot serve.

WHAT IT COSTS. Every element of every plausible container gets its __call__ wrapped, which is far
more instrumentation than tagging a single discovered stack. It is safe in the op-signature probe,
which runs without tracy; it must NOT leak into a profiling run, where broad instrumentation is what
exceeded tracy's 32K source-location limit and left a pytest process that never exited.
"""

from __future__ import annotations

MARK = "PERF_CALLER:"
_MAX_SEQ_SCAN = 512
_MAX_DEPTH = 8
_MAX_NODES = 4000


def _children(node):
    """(label, value) for every child, through the same container set the walk uses."""
    d = getattr(node, "__dict__", None)
    if isinstance(d, dict):
        for k, v in list(d.items()):
            yield str(k), v
    for slot in getattr(type(node), "__slots__", ()) or ():
        try:
            yield str(slot), getattr(node, slot)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(node, dict):
        for k, v in list(node.items())[:_MAX_SEQ_SCAN]:
            yield str(k), v
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(list(node)[:_MAX_SEQ_SCAN]):
            yield str(i), v


def _sequence(node):
    """`node` as an ordered sequence when it is one. dict included so torch's ModuleList counts --
    it keeps its children in the _modules OrderedDict, not in a list."""
    if isinstance(node, (list, tuple)) and 2 <= len(node) <= _MAX_SEQ_SCAN:
        return list(node)
    if isinstance(node, dict) and 2 <= len(node) <= _MAX_SEQ_SCAN:
        return list(node.values())
    mods = getattr(node, "_modules", None)
    if isinstance(mods, dict) and 2 <= len(mods) <= _MAX_SEQ_SCAN:
        return list(mods.values())
    return None


def _plausible(members) -> bool:
    """Worth instrumenting: every element callable, and at least one owning callable children.

    Deliberately weak. This is not a classification -- getting it wrong costs a wrapper on something
    that never runs, and anything that does not run produces no marker and no stack. The strictness
    that matters lives in `observed_stacks`, which requires the elements to have actually executed.
    """
    if not members or len(members) < 2:
        return False
    if not all(callable(m) and hasattr(m, "__dict__") for m in members):
        return False
    return any(_composite(m) for m in members)


def _composite(node) -> bool:
    for _, child in _children(node):
        if child is not None and callable(child) and hasattr(child, "__dict__"):
            return True
        if isinstance(child, (list, tuple, dict)):
            seq = list(child.values()) if isinstance(child, dict) else list(child)
            if any(c is not None and callable(c) and hasattr(c, "__dict__") for c in seq[:_MAX_SEQ_SCAN]):
                return True
    return False


def candidate_containers(root) -> list:
    """[(path, members)] for every sequence worth instrumenting, reachable from `root`.

    Every list of callables is a candidate, INCLUDING the ones the walk rejects: differently-typed
    wrappers with no shared base, and lists too short for the hybrid rule. Whether they form a stack
    is decided later by what ran.
    """
    out, seen, budget = [], set(), [_MAX_NODES]

    def visit(node, path, depth):
        if depth > _MAX_DEPTH or budget[0] <= 0 or node is None:
            return
        budget[0] -= 1
        if isinstance(node, (str, bytes, int, float, bool)):
            return
        key = id(node)
        if key in seen:
            return
        seen.add(key)
        members = _sequence(node)
        if members is not None and _plausible(members):
            out.append((path or "root", members))
        for label, child in _children(node):
            visit(child, "%s.%s" % (path, label) if path else label, depth + 1)

    visit(root, "", 0)
    return out


def marker(path, idx, edge) -> str:
    return "%s%s:%d:%s" % (MARK, path, idx, edge)


PATH_TAG = "_perf_caller_path"
IDX_TAG = "_perf_caller_idx"
_WRAPPED = "_perf_caller_wrapped"


def instrument(root, emit) -> int:
    """Bracket every candidate element's call with an identity marker. Returns elements tagged.

    TAG THE INSTANCE, WRAP THE CLASS. `obj()` resolves __call__ on the TYPE, so an instance
    attribute named __call__ is never consulted -- a wrapper installed that way is silently dead and
    every stack reads as never having run. The identity therefore lives on the instance (which
    container, which index) and the hook lives on its class, which is also what makes an element
    reached through two containers unambiguous: the tag says which one it was found by.

    Classes are patched, not their bases: setting __call__ on a subclass shadows torch's without
    touching torch.nn.Module, so no object outside this model is affected. The probe process is
    throwaway, and nothing here is restored.
    """
    n, classes = 0, {}
    for path, members in candidate_containers(root):
        for i, m in enumerate(members):
            try:
                if getattr(m, PATH_TAG, None) is not None:
                    continue  # already reached by another container; first path wins
                setattr(m, PATH_TAG, path)
                setattr(m, IDX_TAG, i)
            except Exception:  # noqa: BLE001 -- __slots__, frozen dataclass: not observable
                continue
            classes.setdefault(type(m), None)
            n += 1
    for cls in classes:
        _wrap_class(cls, emit)
    return n


def _wrap_class(cls, emit) -> bool:
    """Install the bracket on a class, once, driven entirely by the per-instance tags."""
    try:
        if cls.__dict__.get(_WRAPPED):
            return False
        inner = cls.__call__

        def wrapped(self, *a, **k):
            path = getattr(self, PATH_TAG, None)
            if path is None:
                return inner(self, *a, **k)
            idx = getattr(self, IDX_TAG, 0)
            emit(marker(path, idx, "in"))
            try:
                return inner(self, *a, **k)
            finally:
                emit(marker(path, idx, "out"))

        cls.__call__ = wrapped
        setattr(cls, _WRAPPED, True)
        return True
    except Exception:  # noqa: BLE001 -- a class that cannot be patched is simply not observed
        return False


def parse(tok):
    """(path, idx, edge) from a marker, or None."""
    if not isinstance(tok, str) or not tok.startswith(MARK):
        return None
    body = tok[len(MARK) :]
    try:
        path, idx, edge = body.rsplit(":", 2)
        return path, int(idx), edge
    except ValueError:
        return None


def observed_stacks(seq) -> list:
    """The stacks that actually RAN, from a bracketed sequence.

    [{"path", "blocks", "ran", "depth", "uniform"}], one entry per container whose elements executed.
    `blocks` is how many elements exist, `ran` how many were invoked, `depth` the number of distinct
    elements that produced ops, and `uniform` whether at least two of them emitted the identical op
    subsequence -- the evidence that they are repeats of one block rather than a list of different
    things that happen to sit together.

    Boundaries fall out of the paths: two towers of identical layers are two containers, so they are
    two stacks here even though their op streams are indistinguishable.
    """
    per, stack = {}, []
    for tok in seq or []:
        p = parse(tok)
        if p is None:
            if stack:
                per[stack[-1]].append(tok)
            continue
        path, idx, edge = p
        if edge == "in":
            per.setdefault((path, idx), [])
            stack.append((path, idx))
        elif stack:
            stack.pop()

    by_path = {}
    for (path, idx), ops in per.items():
        by_path.setdefault(path, {})[idx] = tuple(ops)

    out = []
    for path, elems in sorted(by_path.items()):
        bodies = [b for b in elems.values() if b]
        counts = {}
        for b in bodies:
            counts[b] = counts.get(b, 0) + 1
        out.append(
            {
                "path": path,
                "blocks": max(elems) + 1 if elems else 0,
                "ran": len(elems),
                "depth": len(bodies),
                "uniform": bool(counts) and max(counts.values()) >= 2,
            }
        )
    return out


def stacks_that_ran(seq) -> list:
    """Only the observed containers that look like real repeated stacks."""
    return [s for s in observed_stacks(seq) if s["uniform"] and s["depth"] >= 2]
