# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dropping a composite's overlays must reach its components' overlays too.

Overlay scopes are keyed by model id. A composite's parts are brought up as models
in their own right, so each holds its OWN scope keyed by that part's directory --
`overlay-drop <parent>` reported "no overlays registered" while six component
scopes sat on disk and were replayed on the next run. "Clean slate" was not clean,
and the stale scopes had to be removed by hand.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture()
def om(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_OVERLAYS_HOME", str(tmp_path / "overlays"))
    import scripts.tt_hw_planner.overlay_manager as m

    importlib.reload(m)
    return m, tmp_path / "overlays"


def _scope(root, slug, rel="some/file.py"):
    """A scope on disk with one registered overlay."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_text(json.dumps({rel: {"patch_file": "p.patch", "line_count": 1}}))
    (d / "p.patch").write_text("--- a\n+++ b\n")
    return d


PARENT = "some-org/Some-Model-9B"


def test_component_scopes_finds_parent_qualified_scopes(om) -> None:
    m, root = om
    _scope(root, m._slug(PARENT))
    _scope(root, "_tmp_components_some_model_9b__part_one")
    _scope(root, "_tmp_components_some_model_9b__part_two")
    found = m.component_scopes(PARENT)
    assert sorted(found) == [
        "_tmp_components_some_model_9b__part_one",
        "_tmp_components_some_model_9b__part_two",
    ]


def test_component_scopes_excludes_the_parent_itself(om) -> None:
    m, root = om
    _scope(root, m._slug(PARENT))
    assert m._slug(PARENT) not in m.component_scopes(PARENT)


def test_component_scopes_ignores_unrelated_models(om) -> None:
    m, root = om
    _scope(root, "_tmp_components_other_model__part_one")
    _scope(root, "some-org_Another-Model")
    assert m.component_scopes(PARENT) == []


def test_component_scopes_is_empty_when_nothing_is_registered(om) -> None:
    m, _ = om
    assert m.component_scopes(PARENT) == []


def test_drop_scope_removes_a_component_scope(om) -> None:
    """The scopes component_scopes returns must be droppable as-is."""
    m, root = om
    slug = "_tmp_components_some_model_9b__part_one"
    _scope(root, slug)
    count, dropped = m.drop_scope(slug)
    assert count == 1 and dropped == ["some/file.py"]
    assert not (root / slug).exists()


def test_overlay_drop_command_clears_parent_and_components(om, capsys) -> None:
    import argparse

    from scripts.tt_hw_planner.commands.overlay_drop import cmd_overlay_drop

    m, root = om
    _scope(root, m._slug(PARENT))
    _scope(root, "_tmp_components_some_model_9b__part_one")
    rc = cmd_overlay_drop(argparse.Namespace(model_id=PARENT, rel_path=None))
    assert rc == 0
    assert not (root / m._slug(PARENT)).exists()
    assert not (root / "_tmp_components_some_model_9b__part_one").exists(), "component scope survived the drop"


def test_nothing_to_drop_is_still_reported(om, capsys) -> None:
    import argparse

    from scripts.tt_hw_planner.commands.overlay_drop import cmd_overlay_drop

    rc = cmd_overlay_drop(argparse.Namespace(model_id=PARENT, rel_path=None))
    assert rc == 0
    assert "nothing to drop" in capsys.readouterr().out
