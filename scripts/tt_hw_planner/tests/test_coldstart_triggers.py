# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cold-start is for "nothing exists to copy from", not "you matched the catch-all".

Matching a GENERIC backend used to raise ColdStartScaffoldError, on the reasoning
that generic backends have no per-model `tt/` folder to copy. But "generic"
describes the registry ENTRY; the model can still be loaded and walked, and
walking it yields real bound components. Every generic backend carries
use_module_tree=False -- a flag predating module-tree discovery -- so that branch
fired for every model that fell through to the catch-all and ended the run
instead of bringing it up per component.

Cold-start remains for the two cases it was written for: no ported sibling for the
architecture family, and no backend registered for the category.
"""

from __future__ import annotations

from scripts.tt_hw_planner.family_backends import (
    FamilyBackend,
    ROUTING_GENERIC,
    all_backends,
    is_generic,
    prefers_module_tree,
)


def _backend(routing_mode="template", use_module_tree=False):
    return FamilyBackend(
        category="LLM",
        name="test backend",
        demo_path="models/demos/whatever",
        routing_mode=routing_mode,
        canonical_hf_id=None,
        use_module_tree=use_module_tree,
    )


def test_generic_backend_prefers_the_module_tree() -> None:
    """The case that ended runs: generic + use_module_tree=False."""
    b = _backend(routing_mode=ROUTING_GENERIC, use_module_tree=False)
    assert is_generic(b)
    assert prefers_module_tree(b), "a generic backend must walk the model, not cold-start"


def test_declared_module_tree_still_prefers_it() -> None:
    assert prefers_module_tree(_backend(use_module_tree=True))


def test_plain_template_backend_does_not_prefer_the_module_tree() -> None:
    """A template backend with a real sibling folder still copies it."""
    b = _backend(routing_mode="template", use_module_tree=False)
    assert not is_generic(b)
    assert not prefers_module_tree(b)


def test_predicate_tolerates_missing_attributes() -> None:
    class _Bare:
        pass

    assert is_generic(_Bare()) is False
    assert prefers_module_tree(_Bare()) is False


def test_every_registered_generic_backend_walks_the_model() -> None:
    """Pins the whole class: no registered generic backend may route to cold-start
    merely because it has no template folder."""
    generic = [b for b in all_backends() if is_generic(b)]
    assert generic, "expected the registry to have generic catch-all backends"
    unwalkable = [b.name for b in generic if not prefers_module_tree(b)]
    assert not unwalkable, f"generic backends that would still cold-start: {unwalkable}"


def test_routing_mode_literal_is_defined_once() -> None:
    """The mode string lives in one constant, not scattered comparisons."""
    assert ROUTING_GENERIC == "generic"
    assert is_generic(_backend(routing_mode=ROUTING_GENERIC))
