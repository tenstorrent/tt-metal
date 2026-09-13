"""Decompose a stuck large component into smaller sub-components.

When a component (e.g. `vision_encoder`) repeatedly fails to graduate
through agent iteration, the auto-iterate loop used to push it to CPU
fallback. That's wasteful: a too-large component often has children
that *would* graduate independently. This module enumerates those
children, giving the bringup loop a chance to retry the smaller pieces
before accepting CPU placement.

Decomposition rules
-------------------

For a stuck component anchored at ``parent_submodule_path`` in the HF
model, the decomposer walks ``named_children()`` of the torch submodule
and emits a `DecomposedChild` for each non-trivial child:

  * skip container classes (`Sequential`, `ModuleList`, `ModuleDict`,
    `ParameterList`, `ParameterDict`) — they have no forward and the
    Phase 2 framework already handles them via `no_emit_tests.json`.
    Containers' grandchildren are NOT recursed into here; the user can
    re-decompose the child after it graduates if needed.
  * skip leaf classes (`Linear`, `LayerNorm`, `Dropout`, ...) that the
    bringup pipeline can already adapt directly; spawning a new
    component for them is wasted budget.
  * skip trivial-leaf-count children (default <=1 leaf descendants).

The decomposer is intentionally LOCAL: it returns the immediate
children of the stuck component, not the full transitive descendant
tree. The caller can drive multiple rounds of decomposition if needed
(stuck child → its children → its grandchildren → ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


# Re-use the canonical class-name sets that `module_tree` already curates.
# Anything that lives in those sets is NOT decomposable into a new
# sub-component (either no forward, or already a leaf-class the harness
# adapts directly).
def _container_class_names():
    from .module_tree import _CONTAINER_CLASS_NAMES

    return _CONTAINER_CLASS_NAMES


def _leaf_class_names():
    from .module_tree import _LEAF_CLASS_NAMES

    return _LEAF_CLASS_NAMES


def _count_leaves(mod: Any) -> int:
    from .module_tree import _count_leaves as _impl

    return _impl(mod)


@dataclass
class DecomposedChild:
    """A non-trivial child of a stuck component, ready to be re-onboarded
    as its own bring-up component."""

    name: str  # short identifier, e.g. "neck", "layers_0"
    parent_path: str  # original parent submodule path
    submodule_path: str  # full HF submodule path (parent + "." + name)
    class_name: str  # torch class of the child
    leaf_count: int  # transitive leaf-count under this child


def decompose_component(
    *,
    parent_path: str,
    parent_module: Any,
    min_leaf_count: int = 2,
) -> List[DecomposedChild]:
    """Return the decomposable children of ``parent_module``.

    Args:
      parent_path:   the HF dotted submodule path of the parent (used
                     verbatim as the prefix of each child's path).
      parent_module: the torch ``nn.Module`` instance at that path.
      min_leaf_count: children with fewer leaves than this are filtered
                     out as trivial. Default 2 (a child with one leaf
                     is usually a wrapper around a single linear/norm
                     that the harness already adapts).

    Returns:
      A list of `DecomposedChild` records — empty if no non-trivial
      child exists (caller should treat that as "decomposition exhausted;
      proceed with KERNEL_VERIFIED_MISSING verdict").
    """
    if parent_module is None or not hasattr(parent_module, "named_children"):
        return []
    containers = _container_class_names()
    leaves = _leaf_class_names()
    out: List[DecomposedChild] = []
    seen_names = set()
    for child_name, child_mod in parent_module.named_children():
        cls = type(child_mod).__name__
        if cls in containers:
            continue
        if cls in leaves:
            continue
        leaf_count = _count_leaves(child_mod)
        if leaf_count < min_leaf_count:
            continue
        # `named_children()` yields keys for `nn.ModuleDict` and string
        # indices for `nn.ModuleList`. Normalize either form.
        if child_name.isdigit():
            safe_name = f"{_basename_of(parent_path)}_{child_name}"
            sub_path = f"{parent_path}[{child_name}]" if parent_path else f"[{child_name}]"
        else:
            safe_name = child_name
            sub_path = f"{parent_path}.{child_name}" if parent_path else child_name
        if safe_name in seen_names:
            continue
        seen_names.add(safe_name)
        out.append(
            DecomposedChild(
                name=safe_name,
                parent_path=parent_path,
                submodule_path=sub_path,
                class_name=cls,
                leaf_count=leaf_count,
            )
        )
    out.sort(key=lambda c: (-c.leaf_count, c.name))
    return out


def _basename_of(dotted: str) -> str:
    """Return the last token of a dotted/indexed submodule path,
    stripping any [N] index suffix. Used to derive readable child
    names when the child key itself is a numeric index from ModuleList.

    Examples::

      "vision_encoder"           -> "vision_encoder"
      "vision_encoder.neck"      -> "neck"
      "encoder.layers[0]"        -> "layers"
      ""                         -> ""
    """
    if not dotted:
        return ""
    last = dotted.rsplit(".", 1)[-1]
    if "[" in last:
        last = last.split("[", 1)[0]
    return last


_DECOMPOSITION_ELIGIBLE_CLASSES = {
    "AGENT_STUCK",
    "KERNEL_VERIFIED_MISSING",
    "CONSTRAINT_MISMATCH",
    "ITERATION_BUDGET",
}


def failure_class_warrants_decomposition(failure_class: str) -> bool:
    """Class-only gate: does this failure class warrant a decomposition
    suggestion, INDEPENDENT of whether we currently have the torch
    module in hand?

    Used by auto-iterate to emit a CTA without paying the cost of
    loading the HF reference inline. The actual `decompose_component`
    call (which DOES need the torch module) happens out-of-band via the
    `decompose` CLI command.
    """
    return failure_class in _DECOMPOSITION_ELIGIBLE_CLASSES


def should_attempt_decomposition(
    *,
    parent_module: Optional[Any],
    failure_class: str,
) -> bool:
    """Full pre-check: is decomposition worth attempting RIGHT NOW?

    Returns True iff:
      * `failure_class_warrants_decomposition(failure_class)` is True.
      * `parent_module` is non-None and exposes `named_children` (else
        the decomposer can't look at it).

    The actual decomposition is gated separately on whether non-trivial
    children exist — caller still needs to inspect `decompose_component`
    output for emptiness.
    """
    if not failure_class_warrants_decomposition(failure_class):
        return False
    if parent_module is None:
        return False
    return hasattr(parent_module, "named_children")
