"""A recomposed parent that has graduated its own whole-module test must ABSORB its
decomposition-children: the child stub files stay on disk (the parent imports them, and so may other
parents), but the children stop counting as separate top-level components — so the tally collapses
back to the pre-decomposition surface (e.g. 36 -> 32) instead of double-counting parent + its pieces.

Self-contained: builds a synthetic bringup_status.json, no device / no registry needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tt_hw_planner.final_categorization import (
    absorbed_decomposition_children,
    build_final_categorization,
    effective_component_tally,
)

_PARENT = "g_p_t"
_CHILDREN = [f"gpt_child{j}" for j in range(4)]
_LEAVES = [f"leaf{i}" for i in range(31)]


def _write_status(demo: Path) -> list:
    """Write a synthetic 36-component status: 31 leaves + parent + 4 decomposition children."""
    comps = [{"name": n, "status": "NEW", "submodule_path": f"m.{n}"} for n in _LEAVES]
    comps.append({"name": _PARENT, "status": "NEW", "submodule_path": "gpt"})
    for j, c in enumerate(_CHILDREN):
        comps.append({"name": c, "status": "NEW", "submodule_path": f"gpt.c{j}", "_added_by_decomposition_of": _PARENT})
    (demo / "bringup_status.json").write_text(json.dumps({"components": comps}))
    return [c["name"] for c in comps]


def test_no_absorption_while_parent_pending(tmp_path):
    _write_status(tmp_path)
    assert absorbed_decomposition_children(tmp_path, {"leaf0", "leaf1"}) == set()


def test_children_absorbed_once_parent_graduates(tmp_path):
    _write_status(tmp_path)
    assert absorbed_decomposition_children(tmp_path, {_PARENT}) == set(_CHILDREN)


def test_missing_status_is_safe(tmp_path):
    assert absorbed_decomposition_children(tmp_path, {_PARENT}) == set()


def test_count_collapses_36_to_32_on_recompose(tmp_path):
    all_names = _write_status(tmp_path)
    assert len(all_names) == 36
    rep = build_final_categorization(model_id="fake/xtts", demo_dir=tmp_path, graduated_set=set(all_names))
    total = len(rep.on_device) + len(rep.pending) + len(rep.kernel_missing) + len(rep.cpu_reuse)
    assert total == 32, f"expected 32 top-level, got {total}"
    assert _PARENT in rep.on_device
    for c in _CHILDREN:
        assert c not in rep.on_device, f"absorbed child {c} should not count separately"


def test_children_still_counted_before_parent_graduates(tmp_path):
    _write_status(tmp_path)
    rep = build_final_categorization(
        model_id="fake/xtts", demo_dir=tmp_path, graduated_set=set(_LEAVES) | set(_CHILDREN)
    )
    total = len(rep.on_device) + len(rep.pending) + len(rep.kernel_missing) + len(rep.cpu_reuse)
    assert total == 36, f"expected 36 while parent pending, got {total}"
    for c in _CHILDREN:
        assert c in rep.on_device
    assert _PARENT not in rep.on_device


def test_tally_is_single_source_and_collapses(tmp_path):
    all_names = _write_status(tmp_path)
    grad, total = effective_component_tally("fake/xtts", tmp_path, set(all_names))
    assert (grad, total) == (32, 32), f"expected (32, 32), got ({grad}, {total})"


def test_tally_matches_the_report_it_derives_from(tmp_path):
    all_names = _write_status(tmp_path)
    gset = set(_LEAVES) | set(_CHILDREN)
    grad, total = effective_component_tally("fake/xtts", tmp_path, gset)
    rep = build_final_categorization(model_id="fake/xtts", demo_dir=tmp_path, graduated_set=gset)
    assert grad == len(rep.on_device)
    assert total == len(rep.on_device) + len(rep.pending) + len(rep.kernel_missing) + len(rep.cpu_reuse)


def test_tally_empty_when_no_status(tmp_path):
    assert effective_component_tally("fake/xtts", tmp_path, {_PARENT}) == (0, 0)
