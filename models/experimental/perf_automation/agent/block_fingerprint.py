# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a block IS, measured from the tensors it owns rather than from what its class is called.

THE WALK CLASSIFIES BY CLASS IDENTITY, which is a proxy for "same kind of block" and a poor one. A
pipeline that wraps each layer in a different class -- a counting proxy here, a parts-assembled layer
there -- holds a real stack that the walk reads as unrelated objects. Class identity says no; every
one of those wrappers owns the same weight shapes, because they are the same layer.

TWO EARLIER ATTEMPTS COMPARED THE WRONG THING and both regressed the walk (5 stacks to 3, an encoder
lost). Comparing ATTRIBUTE NAMES scored every pair of torch modules as identical -- they all carry
_parameters, _buffers, _modules, training -- so three unrelated top-level submodules grouped and
shadowed the real stacks. Comparing CHILD MODULE NAMES could not separate "three wrappers around one
layer kind" from "three submodules of a model".

Tensor shapes do not have that failure. Names are shared by everything a framework builds; shapes are
what a layer actually is. The case that broke both attempts -- an audio tower beside a language model
-- separates immediately here, because a 1280-wide encoder layer and a 3072-wide decoder layer own
different weights. The signal is also reference-free and framework-free: it reads `.shape` off
whatever the object holds, so a ttnn tensor, a torch parameter and a numpy array all work, and a
model with no HF lineage is served exactly as well as one built from HF weights.

WHAT THIS DELIBERATELY DOES NOT DO is decide anything on its own. It answers "are these two objects
the same kind of block", and the walk uses that only to ACCEPT a list it would otherwise have
skipped -- never to displace a stack it already found. That containment is the lesson of the two
regressions: a widened rule that can shadow is worse than a narrow one.
"""

from __future__ import annotations

_MAX_DEPTH = 4
_MAX_NODES = 400


def _dims(x):
    """(shape, dtype) for a tensor-like, else None. Duck-typed: torch, ttnn and numpy all qualify."""
    s = getattr(x, "shape", None)
    if s is None:
        return None
    try:
        dims = tuple(int(d) for d in s)
    except Exception:  # noqa: BLE001
        return None
    if not dims:
        return None
    dt = getattr(getattr(x, "dtype", None), "name", None) or str(getattr(x, "dtype", "") or "")
    return (dims, dt)


def _children(node):
    """Same container set the walk uses, so a block is fingerprinted through the shapes it holds
    however it holds them -- attribute, slot, list, tuple or dict."""
    d = getattr(node, "__dict__", None)
    if isinstance(d, dict):
        yield from list(d.values())
    for slot in getattr(type(node), "__slots__", ()) or ():
        try:
            yield getattr(node, slot)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(node, dict):
        yield from list(node.values())
    elif isinstance(node, (list, tuple)):
        yield from list(node)


def fingerprint(block) -> tuple:
    """The multiset of tensor shapes `block` owns, as a sorted tuple.

    Bounded in depth and node count: a block that reaches the whole model through a back-reference
    would otherwise fingerprint as the model. Depth 4 covers block -> submodule -> weight, which is
    every layer shape seen so far, plus one.
    """
    found, seen, budget = [], set(), [_MAX_NODES]

    def walk(node, depth):
        if depth > _MAX_DEPTH or budget[0] <= 0 or node is None:
            return
        budget[0] -= 1
        d = _dims(node)
        if d is not None:
            found.append(d)
            return  # a tensor has no interesting children
        if isinstance(node, (str, bytes, int, float, bool)):
            return
        key = id(node)
        if key in seen:
            return
        seen.add(key)
        for child in _children(node):
            if _owns(child, block):
                continue  # a parent pointer: see _owns
            walk(child, depth + 1)

    walk(block, 0)
    return tuple(sorted(found))


def _owns(node, target) -> bool:
    """Does `node` hold `target`, directly or one container deep?

    A BLOCK THAT CAN REACH ITS PARENT FINGERPRINTS AS THE MODEL. Layers commonly keep a back
    reference -- to the parent module, the config object, a shared cache -- and following it pulls in
    the embedding table and every other layer's weights. Every block would then produce the same
    enormous fingerprint and match every other block, including blocks from a different section:
    the exact shadowing failure that made the two similarity attempts worse than no rule at all.

    Cycle detection does not catch this, because the parent is a different object each hop and is
    only revisited after the damage is done. What identifies it is that the parent contains the
    block, so anything holding the starting block is not part of what that block owns.
    """
    try:
        for child in _children(node):
            if child is target:
                return True
            if isinstance(child, (list, tuple)):
                if any(c is target for c in child):
                    return True
            elif isinstance(child, dict):
                if any(c is target for c in child.values()):
                    return True
    except Exception:  # noqa: BLE001
        return False
    return False


def same_kind(a, b) -> bool:
    """Do these two objects own the same weights? Exact equality, and deliberately no threshold.

    A similarity score needs a cutoff, and a cutoff is where both earlier attempts went wrong: the
    mean included the reference compared with itself, so any two-element list passed. Equal or not
    equal has no such dial. Two objects owning no tensors at all are NOT the same kind -- an empty
    fingerprint is absence of evidence, and treating it as a match is how a list of stubs would
    group.
    """
    fa = fingerprint(a)
    return bool(fa) and fa == fingerprint(b)


def uniform_kind(members) -> bool:
    """Is every element of this list the same kind of block?

    This is the question the walk asks. It says nothing about how many elements there are or what
    they are called -- the walk keeps its own bounds and its own ranking.
    """
    members = list(members or [])
    if len(members) < 2:
        return False
    first = fingerprint(members[0])
    if not first:
        return False
    return all(fingerprint(m) == first for m in members[1:])
