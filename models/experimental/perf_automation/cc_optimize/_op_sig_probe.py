# SPDX-License-Identifier: Apache-2.0
"""Generic op-type-coverage probe (MODEL-AGNOSTIC).

Runs a perf test's forward at the current TT_PERF_LAYERS depth, intercepts EVERY dispatched ttnn op by
type (the same FastOperation-by-type technique the perf test uses to drain the profiler), and prints the
SET of distinct op signatures (op name + input-tensor shapes) as `PERF_OP_SIGS=<json>`. It wraps ttnn
itself and runs the given pytest node, so it needs no per-model knowledge — it works for any pipeline.

The coverage-window sizing (run.py:_coverage_layers) grows the profiled depth and compares these sets:
when a deeper window adds no new signature, every block type is covered and the profiled slice is a valid
representative sample. Homogeneous models saturate at 1-2 layers; heterogeneous ones (e.g. mamba + attention
+ MoE interleaved) grow until every type has appeared — with no model-specific layer maps.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))
sys.path.insert(0, str(_PKG))

_SIGS = set()
_SEQ = []


def _shape_sig(args):
    out = []
    for x in args:
        s = getattr(x, "shape", None)
        if s is None:
            continue
        try:
            dims = tuple(int(d) for d in s)
        except Exception:  # noqa: BLE001
            dims = str(s)
        dt = getattr(getattr(x, "dtype", None), "name", None) or str(getattr(x, "dtype", "") or "")
        out.append((dims, dt) if dt else dims)
    return tuple(out)


def _wrap(fn, name):
    def inner(*a, **k):
        try:
            sig = "%s%s" % (name, _shape_sig(a))
            _SIGS.add(sig)
            _SEQ.append(sig)
        except Exception:  # noqa: BLE001
            pass
        return fn(*a, **k)

    return inner


def _install():
    import ttnn

    mods = [ttnn] + [getattr(ttnn, m, None) for m in ("transformer", "experimental")]
    for mod in [m for m in mods if m is not None]:
        for n in dir(mod):
            op = getattr(mod, n, None)
            if type(op).__name__ == "FastOperation":
                setattr(mod, n, _wrap(op, "%s.%s" % (getattr(mod, "__name__", "ttnn"), n)))


_BLOCK_TAG = "_perf_block_idx"
_SIGNPOST_PREFIX = "PERF_BLOCK_SIGNPOST:"


_ATOMIC = (str, bytes, bytearray, int, float, bool, complex, type(None))
_MAX_STACK_DEPTH = 8
_MAX_STACK_NODES = 20000
_MAX_SEQ_SCAN = 4096


def _is_atomic(node) -> bool:
    if isinstance(node, _ATOMIC):
        return True
    return "Tensor" in type(node).__name__


def _stack_members(seq):
    """The elements of `seq` that could be blocks: objects carrying attributes."""
    return [v for v in seq if v is not None and hasattr(v, "__dict__") and not _is_atomic(v)]


def _stack_tier(members) -> int:
    """How block-like this stack is: 2 invocable and composite, 1 either, 0 neither.

    A real block is CALLABLE (it exists to be invoked, and the signpost wrapper hooks exactly that
    call) and COMPOSITE (it owns sub-modules). Ranking rather than filtering matters: a bare list of
    ``Linear`` projections outranks nothing, so it never displaces a real stack, yet a stack of inert
    same-typed objects is still found when a model has nothing better -- which is the long-standing
    contract test_finds_plain_python_list_of_same_typed_blocks pins down.
    """
    head = members[0]
    return int(callable(head)) + int(_is_composite(head))


def _shared_base(kinds):
    """The most specific class every kind in `kinds` derives from, or None when that is only object."""
    mros = [[c for c in t.__mro__ if c is not object] for t in kinds]
    if not mros or any(not m for m in mros):
        return None
    common = set(mros[0]).intersection(*(set(m) for m in mros[1:]))
    if not common:
        return None
    return min(common, key=lambda c: len(c.__subclasses__()))


def _is_composite(node) -> bool:
    """Does `node` own callable children, i.e. is it built out of sub-modules?

    A decoder block contains attention/MLP/norm sub-modules; a bare ``Linear`` or ``Conv1d`` owns
    only weights. Without this, the longest list of leaf projections wins over the real stack --
    lw-detr picks a 13-element ``Linear`` list over its 10-layer encoder.
    """
    for child in _child_nodes(node):
        if child is not None and callable(child) and hasattr(child, "__dict__") and not _is_atomic(child):
            return True
        if isinstance(child, (list, tuple, dict)):
            seq = list(child.values()) if isinstance(child, dict) else list(child)
            if any(c is not None and callable(c) and hasattr(c, "__dict__") for c in seq[:_MAX_SEQ_SCAN]):
                return True
    return False


def _is_block_stack(members) -> bool:
    """Does this sequence of callables look like a repeated block stack?

    Two accepted shapes. HOMOGENEOUS -- N instances of one class -- is the original signal and still
    the common case. HYBRID -- a few interleaved block classes sharing a base, e.g. NemotronH's
    alternating Mamba and attention blocks -- was rejected outright by the old ``len(kinds) == 1``
    test, so hybrid models silently produced no signposts. Hybrids are held to stricter bounds (at
    least 4 blocks, at most 3 classes, a shared base, and one class covering at least a third) so an
    ordinary list of a few unrelated submodules is not mistaken for a stack.
    """
    if len(members) < 2:
        return False
    kinds = {type(v) for v in members}
    if len(kinds) == 1:
        return True
    if len(members) < 4 or len(kinds) > 3:
        return False
    if _shared_base(kinds) is None:
        return False
    dominant = max(sum(1 for v in members if type(v) is k) for k in kinds)
    return dominant / len(members) >= 1 / 3


def _child_nodes(node):
    """Every child reachable from `node`, whatever container holds it.

    The old walk followed only ``__dict__`` values and treated a list as a leaf, so a stack nested
    inside ANY list or dict was unreachable. gemma3 hits this: ``prepare_generator_args`` builds one
    model per submesh, so with data_parallel=1 the generator holds a ONE-element list whose single
    element owns the 48 blocks -- too short to be a stack itself, and never opened.

    Containers are a closed set (object attributes, __slots__, list/tuple, dict), unlike the open set
    of shapes models arrange them in, so covering them here is what keeps this model-agnostic.
    """
    d = getattr(node, "__dict__", None)
    if isinstance(d, dict):
        yield from list(d.values())
    for slot in getattr(type(node), "__slots__", ()) or ():
        try:
            yield getattr(node, slot)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(node, dict):
        yield from list(node.values())[:_MAX_SEQ_SCAN]
    elif isinstance(node, (list, tuple)):
        yield from list(node)[:_MAX_SEQ_SCAN]


def _node_sequence(node):
    """`node` viewed as an ordered sequence, when it is one.

    dict is included so torch's ``nn.ModuleList`` is covered: it keeps its children in the
    ``_modules`` OrderedDict, not in a list.
    """
    if isinstance(node, (list, tuple)) and 2 <= len(node) <= _MAX_SEQ_SCAN:
        return list(node)
    if isinstance(node, dict) and 2 <= len(node) <= _MAX_SEQ_SCAN:
        return list(node.values())
    return None


def _largest_repeated_stack(root, _depth: int = 0, _seen=None, _budget=None):
    """The largest repeated block stack reachable from `root`, through any container nesting.

    A TTNN model is typically NOT a torch.nn.Module: models/common/lightweightmodule.py exists
    precisely to avoid torch's per-call host overhead, and such models hold their decoder blocks in a
    PLAIN PYTHON LIST (``self.layers = [TransformerBlock(...) for _ in range(n_layers)]``). Looking
    only for nn.ModuleList therefore finds nothing on most tt-metal models, the probe emits no
    signposts, and run.py has to fall back to probing depth 2/4/8/16 to discover what the signposts
    would have said for free.

    Structure is the signal, not the attribute name: a stack is N callable blocks of one class (or of
    a few interleaved classes sharing a base), so no per-model knowledge and no 'layers'/'blocks'
    name list is needed.
    """
    if _seen is None:
        _seen = set()
    if _budget is None:
        _budget = [_MAX_STACK_NODES]
    if root is None or _depth > _MAX_STACK_DEPTH or _budget[0] <= 0 or _is_atomic(root):
        return None
    if id(root) in _seen:
        return None
    _seen.add(id(root))
    _budget[0] -= 1
    best: dict = {}
    seq = _node_sequence(root)
    if seq is not None:
        members = _stack_members(seq)
        if _is_block_stack(members):
            best[_stack_tier(members)] = members
    for child in _child_nodes(root):
        deeper = _largest_repeated_stack(child, _depth + 1, _seen, _budget)
        if deeper is None:
            continue
        tier = _stack_tier(deeper)
        if tier not in best or len(deeper) > len(best[tier]):
            best[tier] = deeper
    if not best:
        return None
    return best[max(best)]


def _enclosing_stack(block):
    """The stack that CONTAINS `block`, for when the walk is rooted at a leaf block.

    Walking DOWN only works when the wrapper first fires on the top module. It often does not: a
    model whose top module is invoked by METHOD rather than ``__call__`` never triggers the hook
    itself. tt_transformers is exactly this shape -- generator.py calls
    ``self.model[i].ttnn_prefill_forward(...)``, so the first ``LightweightModule.__call__`` is
    ``x = layer(...)`` inside the Transformer, i.e. ``layers[0]``. A single block holds attention,
    MLP and norms, never a 48-element stack, so the downward walk finds nothing and no signpost is
    ever emitted.

    The list holding the block is one referrer away, so recover the stack upward instead.
    """
    import gc

    best = None
    for ref in gc.get_referrers(block):
        if isinstance(ref, (list, tuple)):
            seq = list(ref)
        elif isinstance(ref, dict):
            seq = list(ref.values())
        else:
            continue
        if len(seq) < 2 or len(seq) > _MAX_SEQ_SCAN or not any(v is block for v in seq):
            continue
        members = _stack_members(seq)
        if _is_block_stack(members) and (best is None or len(members) > len(best)):
            best = members
    return best


class StackInfo:
    """Describes one repeated block stack found in a model.

    Attributes:
        path         Dot-separated attribute path from the root to the container,
                     e.g. ``"audio_tower.layers"``.  Empty string when the stack
                     is discovered upward (via GC referrers) rather than downward.
        stack        The actual list of module objects in traversal order.
        element_type The shared class of every element (the most-specific common
                     base, for hybrid stacks; the single class for homogeneous ones).
        count        ``len(stack)``.
        stack_idx    0-based integer assigned in depth-first discovery order so
                     callers can refer to stacks without holding the list.
    """

    __slots__ = ("path", "stack", "element_type", "count", "stack_idx")

    def __init__(self, path, stack, element_type, count, stack_idx):
        self.path = path
        self.stack = stack
        self.element_type = element_type
        self.count = count
        self.stack_idx = stack_idx

    def __repr__(self):
        return "StackInfo(path=%r, element_type=%s, count=%d, stack_idx=%d)" % (
            self.path,
            getattr(self.element_type, "__name__", repr(self.element_type)),
            self.count,
            self.stack_idx,
        )


def _dominant_type(members):
    """Return the most-specific shared base class for `members`, or the most common type."""
    kinds = {type(v) for v in members}
    if len(kinds) == 1:
        return next(iter(kinds))
    base = _shared_base(kinds)
    if base is not None:
        return base
    # fall back to the most common concrete type
    return max(kinds, key=lambda k: sum(1 for v in members if type(v) is k))


def _walk_for_stacks(root):
    """Depth-first walk that yields (path, members) for every candidate block stack.

    Visits nodes in DFS pre-order so the resulting list respects execution order.
    Handles both ``torch.nn.Module`` trees (via ``named_modules()``) and arbitrary
    Python objects (via ``__dict__``).  Each node is visited at most once.
    """
    seen_ids = set()

    def _visit(node, prefix: str):
        nid = id(node)
        if nid in seen_ids or node is None or _is_atomic(node):
            return
        seen_ids.add(nid)

        # --- Try torch.nn.Module.named_children() for proper torch trees ----------
        named_children_fn = getattr(node, "named_children", None)
        if named_children_fn is not None:
            try:
                for child_name, child in list(named_children_fn()):
                    child_path = "%s.%s" % (prefix, child_name) if prefix else child_name
                    # Check if the child itself is a ModuleList-like sequence
                    seq = _node_sequence(child)
                    if seq is None:
                        # Also check via _modules dict directly
                        modules_dict = getattr(child, "_modules", None)
                        if isinstance(modules_dict, dict) and modules_dict:
                            seq = list(modules_dict.values())
                    if seq is not None:
                        members = _stack_members(seq)
                        if _is_block_stack(members) and len(members) >= 3:
                            yield child_path, members
                    # Recurse into the child
                    yield from _visit(child, child_path)
            except Exception:  # noqa: BLE001
                pass
            return  # torch.nn.Module children handled above; skip __dict__ walk

        # --- Generic __dict__ walk for LightweightModule and plain objects --------
        d = getattr(node, "__dict__", None)
        if not isinstance(d, dict):
            return
        for attr, val in list(d.items()):
            if val is None or _is_atomic(val):
                continue
            attr_path = "%s.%s" % (prefix, attr) if prefix else attr
            seq = _node_sequence(val)
            if seq is not None:
                members = _stack_members(seq)
                if _is_block_stack(members) and len(members) >= 3:
                    yield attr_path, members
            # Recurse: unwrap single-element containers that might wrap a real stack
            if not _is_atomic(val) and id(val) not in seen_ids:
                if isinstance(val, (list, tuple)):
                    for item in list(val)[:_MAX_SEQ_SCAN]:
                        yield from _visit(item, attr_path)
                elif isinstance(val, dict):
                    for item in list(val.values())[:_MAX_SEQ_SCAN]:
                        yield from _visit(item, attr_path)
                else:
                    yield from _visit(val, attr_path)

    yield from _visit(root, "")


def find_all_stacks(root) -> list:
    """Discover ALL repeating block stacks in `root`, not just the largest one.

    Returns a list of :class:`StackInfo` sorted in depth-first (execution) order,
    with ``stack_idx`` values 0, 1, 2, ...

    Algorithm
    ---------
    1.  Walk the model depth-first, yielding every list/ModuleList with ≥3
        same-typed (or same-base) elements.
    2.  Deduplicate: when stack A's path is a strict prefix of stack B's path
        *and* they share the same element type, keep only the deeper one (B).
        This prevents double-counting a ``ModuleList`` that appears both as
        ``model.layers`` and as its own ``_modules`` dict.
    3.  Sort the survivors in discovery order and assign ``stack_idx``.

    Both ``torch.nn.ModuleList`` and plain Python lists of ``LightweightModule``
    (or any same-typed object with ``__dict__``) are found; no per-model code
    is required.
    """
    # Step 1: collect all candidates in DFS order (path, members)
    candidates = list(_walk_for_stacks(root))

    if not candidates:
        return []

    # Step 2: deduplicate
    # Build (path, members, element_type) triples
    triples = []
    for path, members in candidates:
        etype = _dominant_type(members)
        triples.append((path, members, etype))

    # Remove any triple whose path is a strict prefix of another triple with the
    # same element type — keep the deeper (more specific) one.
    def _is_prefix_of(a_path: str, b_path: str) -> bool:
        """True when a_path is a strict prefix component of b_path."""
        if not a_path:
            return bool(b_path)  # "" is prefix of everything non-empty
        return b_path.startswith(a_path + ".")

    n = len(triples)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path_i, _, etype_i = triples[i]
            path_j, _, etype_j = triples[j]
            if etype_i == etype_j and _is_prefix_of(path_i, path_j):
                dominated[i] = True
                break

    survivors = [t for t, dom in zip(triples, dominated) if not dom]

    # Step 3: assign stack_idx and build StackInfo objects
    result = []
    for idx, (path, members, etype) in enumerate(survivors):
        result.append(
            StackInfo(
                path=path,
                stack=members,
                element_type=etype,
                count=len(members),
                stack_idx=idx,
            )
        )
    return result


def _find_stack(root):
    """The block stack for `root`, considering BOTH directions and keeping the better one.

    Down-first-then-up is wrong, and the live gemma3 generator proves it: rooted at layers[0] -- the
    root the wrapper actually gets -- the downward walk finds a 9-element stack of sub-modules INSIDE
    the block and returns it, so the enclosing 48-block stack is never even looked for. Tagging 9
    sub-modules as if they were the decoder stack would attribute every op to the wrong depth.

    Rank the two candidates the same way sibling candidates are ranked: block-likeness first, then
    length.
    """
    down = _largest_repeated_stack(root)
    up = _enclosing_stack(root)
    if down is None:
        return up
    if up is None:
        return down
    return up if (_stack_tier(up), len(up)) > (_stack_tier(down), len(down)) else down


def _tag_stack(stack) -> bool:
    """Index every block in `stack` so entering one can be attributed to an exact depth."""
    if not stack:
        return False
    tagged = False
    for i, blk in enumerate(stack):
        try:
            setattr(blk, _BLOCK_TAG, i)
            tagged = True
        except Exception:  # noqa: BLE001
            pass
    return tagged


def _install_block_signposts():
    """Emit a real per-block signpost into the op stream at every repeated-block invocation, so a
    consumer can attribute each op to an exact block (not an inferred boundary).

    MODEL-AGNOSTIC, and it must cover BOTH shapes tt-metal models come in:

      torch-shaped      the largest nn.ModuleList is the stack; torch.nn.Module.__call__ is wrapped.
      TTNN-shaped       blocks subclass LightweightModule (NOT torch.nn.Module) and live in a plain
                        Python list, so the torch hook never fires for them. LightweightModule.__call__
                        is wrapped too, and the stack is found by looking for the largest list of
                        same-typed objects.

    Covering only the torch shape is why llama3_1_8b_p150 reported full_blocks=0 and run.py had to
    climb a 2/4/8/16 ladder — four extra device probes — to recover depths this would have supplied
    from the single all-layers probe.

    No per-model code, no markers baked into model source; probe-local only.
    """
    state = {"tagged": False}

    def _emit(self):
        idx = getattr(self, _BLOCK_TAG, None)
        if idx is not None:
            try:
                _SEQ.append("%s%d" % (_SIGNPOST_PREFIX, idx))
            except Exception:  # noqa: BLE001
                pass

    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    if torch is not None:
        _torch_orig = torch.nn.Module.__call__

        def _torch_tag(root):
            best = None
            for m in root.modules():
                for _, child in m.named_children():
                    if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                        if best is None or len(child) > len(best):
                            best = child
            if best is None:
                return False
            return _tag_stack(list(best))

        def _torch_wrapped(self, *a, **k):
            if not state["tagged"]:
                try:
                    if sum(1 for _ in self.modules()) > 8:
                        state["tagged"] = _torch_tag(self)
                except Exception:  # noqa: BLE001
                    pass
            _emit(self)
            return _torch_orig(self, *a, **k)

        torch.nn.Module.__call__ = _torch_wrapped

    try:
        from models.common.lightweightmodule import LightweightModule
    except Exception:  # noqa: BLE001
        return

    _lw_orig = LightweightModule.__call__

    def _lw_wrapped(self, *a, **k):
        if not state["tagged"]:
            try:
                state["tagged"] = _tag_stack(_find_stack(self))
            except Exception:  # noqa: BLE001
                pass
        _emit(self)
        return _lw_orig(self, *a, **k)

    LightweightModule.__call__ = _lw_wrapped


def main(node: str, case: str | None = None) -> None:
    _install()
    _install_block_signposts()
    import pytest

    # The probe asks for ALL layers (the caller removed the cap); load the depth guard so a
    # setdefault in the test module cannot quietly reinstate one before the model is built.
    argv = ["-s", "-p", "models.experimental.perf_automation.agent.depth_guard_plugin", "-o", "timeout=0", node]
    if case:
        argv += ["-k", case]
    try:
        pytest.main(argv)
    except SystemExit:
        pass
    print("PERF_OP_SIGS=" + json.dumps(sorted(_SIGS)), flush=True)
    print("PERF_OP_SIG_COUNTS=" + json.dumps(Counter(_SEQ)), flush=True)
    print("PERF_OP_SIG_SEQUENCE=" + json.dumps(_SEQ[:50000]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
