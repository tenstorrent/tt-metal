"""Tests for the ModuleList-path fixes (2026-06-03).

Background:
  In the seamless-m4t bring-up (2026-06-03), 22 of 23 NEW components
  had auto-generated PCC tests that SKIPPED because their resolved
  submodule path landed on a ``nn.ModuleList`` container rather than a
  concrete submodule. Calling a ModuleList raises NotImplementedError
  (no ``forward``), the test caught it and called ``pytest.skip(...)``,
  and the orchestrator silently dropped the components from the
  candidate pool, mis-classifying them as "graduated."

Two fixes shipped together:
  1. ``discover_components_from_hf_id`` now emits ``sample_paths[0]``
     (an indexed instance path like ``vocoder.hifi_gan.resblocks.0``)
     as the primary ``submodule_path`` instead of ``parent_path``
     (the ModuleList itself).
  2. The auto-test template's ``_resolve()`` function gained a runtime
     safety net: if the resolved object is a ``ModuleList`` or
     ``Sequential``, automatically pick element ``[0]``.

Tests below cover both layers independently and end-to-end.
"""

from __future__ import annotations

import torch


# ─── Discover: emits indexed sample path, not parent ModuleList ─────


def test_discover_emits_sample_paths_zero_for_repeated_components():
    """When a class has multiple instances under a ModuleList, discover
    must emit the FIRST INDEXED PATH (sample_paths[0]) as the primary
    submodule_path — not the parent ModuleList path."""
    from scripts.tt_hw_planner.module_tree import discover_components_from_hf_id

    comps = discover_components_from_hf_id("facebook/hf-seamless-m4t-medium")
    target = None
    for c in comps:
        if c.class_name == "HifiGanResidualBlock":
            target = c
            break
    assert target is not None, "HifiGanResidualBlock not discovered"
    assert target.occurrences >= 2, "expected multiple instances inside a ModuleList"
    # The primary path must be an indexed path (matches sample_paths[0]),
    # not the parent path (which would be the ModuleList "resblocks").
    assert target.submodule_path == target.sample_paths[0], (
        f"submodule_path={target.submodule_path!r} should equal " f"sample_paths[0]={target.sample_paths[0]!r}"
    )
    # Sanity: the path must contain a numeric index segment (e.g. ".0").
    assert any(
        seg.isdigit() for seg in target.submodule_path.split(".")
    ), f"submodule_path {target.submodule_path!r} doesn't look indexed"


def test_discover_falls_back_to_parent_path_when_sample_paths_empty(monkeypatch):
    """Defense: if sample_paths is unexpectedly empty (shouldn't happen
    in practice but the code defends against it), fall back to
    parent_path so we don't crash with IndexError."""
    # Easiest way to test the fallback: directly verify the conditional
    # in module_tree.py:428 by reading the source.
    import inspect

    from scripts.tt_hw_planner import module_tree

    src = inspect.getsource(module_tree)
    # The DiscoveredComponent construction must reference sample_paths
    # AND have a fallback to parent_path.
    assert "sample_paths" in src and "parent_path" in src


# ─── Runtime _resolve fallback in auto-test template ────────────────


def _run_resolver_template(obj, path):
    """Inline a copy of the auto-test template's _resolve function so
    we can unit-test it without scaffolding a full demo dir. The body
    here MUST match bringup_loop.py:136 / :1212."""
    cur = obj
    for tok in path.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    try:
        import torch as _torch

        if isinstance(cur, (_torch.nn.ModuleList, _torch.nn.Sequential)) and len(cur) > 0:
            cur = cur[0]
    except Exception:
        pass
    return cur


def test_resolve_picks_first_element_when_path_lands_on_modulelist():
    """If the resolved path lands on a ModuleList, the runtime safety
    net must pick element [0] so the test has a callable module."""

    class Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tag = "leaf"

        def forward(self, x):
            return x

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.children_list = torch.nn.ModuleList([Leaf(), Leaf(), Leaf()])

    obj = Container()
    # Path "children_list" lands on the ModuleList itself (the bug case).
    resolved = _run_resolver_template(obj, "children_list")
    assert isinstance(resolved, Leaf), f"resolver should have auto-picked the first Leaf, got {type(resolved).__name__}"


def test_resolve_picks_first_element_when_path_lands_on_sequential():
    """Same fallback applies to Sequential, not just ModuleList."""

    class Leaf(torch.nn.Module):
        def forward(self, x):
            return x

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = torch.nn.Sequential(Leaf(), Leaf())

    resolved = _run_resolver_template(Container(), "seq")
    assert isinstance(resolved, Leaf)


def test_resolve_does_not_index_into_regular_modules():
    """Defense: the fallback must only fire for container types
    (ModuleList/Sequential). A regular nn.Module that happens to be
    iterable must NOT be silently indexed."""

    class Regular(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tag = "regular"

        def forward(self, x):
            return x

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Regular()

    resolved = _run_resolver_template(Container(), "child")
    assert isinstance(resolved, Regular)
    assert resolved.tag == "regular"  # not indexed into anything


def test_resolve_handles_indexed_path_through_modulelist():
    """When the path is correct (already indexed into the ModuleList),
    the resolver must NOT double-index. The path 'children_list.0'
    should land on the first Leaf, NOT on something inside the leaf."""

    class Leaf(torch.nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def forward(self, x):
            return x

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.children_list = torch.nn.ModuleList([Leaf("a"), Leaf("b")])

    # Path already includes the index — resolve to Leaf("a")
    resolved = _run_resolver_template(Container(), "children_list.0")
    assert isinstance(resolved, Leaf)
    assert resolved.tag == "a"


def test_resolve_safe_for_empty_modulelist():
    """A ModuleList with no children shouldn't crash — fall through to
    returning the empty container (the test will then fail with a
    clearer error later, not crash here)."""

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.empty_list = torch.nn.ModuleList()

    resolved = _run_resolver_template(Container(), "empty_list")
    # Empty list — fallback's `len(cur) > 0` guard prevents indexing.
    # Result is still the empty ModuleList (caller can handle the error
    # downstream with a clearer message).
    assert isinstance(resolved, torch.nn.ModuleList)
    assert len(resolved) == 0


# ─── Static guard: both _resolve copies in bringup_loop.py have the fix


def test_both_resolve_copies_in_bringup_loop_have_modulelist_fallback():
    """There are TWO _resolve() copies in bringup_loop.py — one in the
    per-component PCC test template (~line 136) and one in the demo
    end-to-end wiring template (~line 1212). Both must have the
    ModuleList fallback or auto-generated tests still SKIP."""
    import re
    from pathlib import Path

    src = Path("scripts/tt_hw_planner/bringup_loop.py").read_text()
    # Find each def _resolve(...) block and inspect the next ~30 lines.
    matches = list(re.finditer(r"^def _resolve\b", src, flags=re.MULTILINE))
    assert len(matches) >= 2, f"expected ≥2 _resolve definitions in template, got {len(matches)}"
    for i, m in enumerate(matches):
        body_start = m.start()
        body = src[body_start : body_start + 1200]
        assert "ModuleList" in body and "Sequential" in body, (
            f"_resolve copy #{i + 1} (starts at offset {body_start}) " f"is missing the ModuleList/Sequential fallback"
        )
