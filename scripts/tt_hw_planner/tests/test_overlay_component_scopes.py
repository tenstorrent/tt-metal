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
    _scope(root, "_tmp_components_some_model_9b_part_one")
    _scope(root, "_tmp_components_some_model_9b_part_two")
    found = m.component_scopes(PARENT)
    assert sorted(found) == [
        "_tmp_components_some_model_9b_part_one",
        "_tmp_components_some_model_9b_part_two",
    ]


def test_component_scopes_excludes_the_parent_itself(om) -> None:
    m, root = om
    _scope(root, m._slug(PARENT))
    assert m._slug(PARENT) not in m.component_scopes(PARENT)


def test_component_scopes_ignores_unrelated_models(om) -> None:
    m, root = om
    _scope(root, "_tmp_components_other_model_part_one")
    _scope(root, "some-org_Another-Model")
    assert m.component_scopes(PARENT) == []


def test_component_scopes_is_empty_when_nothing_is_registered(om) -> None:
    m, _ = om
    assert m.component_scopes(PARENT) == []


def test_drop_scope_removes_a_component_scope(om) -> None:
    """The scopes component_scopes returns must be droppable as-is."""
    m, root = om
    slug = "_tmp_components_some_model_9b_part_one"
    _scope(root, slug)
    count, dropped = m.drop_scope(slug)
    assert count == 1 and dropped == ["some/file.py"]
    assert not (root / slug).exists()


def test_overlay_drop_command_clears_parent_and_components(om, capsys) -> None:
    import argparse

    from scripts.tt_hw_planner.commands.overlay_drop import cmd_overlay_drop

    m, root = om
    _scope(root, m._slug(PARENT))
    _scope(root, "_tmp_components_some_model_9b_part_one")
    rc = cmd_overlay_drop(argparse.Namespace(model_id=PARENT, rel_path=None))
    assert rc == 0
    assert not (root / m._slug(PARENT)).exists()
    assert not (root / "_tmp_components_some_model_9b_part_one").exists(), "component scope survived the drop"


def test_nothing_to_drop_is_still_reported(om, capsys) -> None:
    import argparse

    from scripts.tt_hw_planner.commands.overlay_drop import cmd_overlay_drop

    rc = cmd_overlay_drop(argparse.Namespace(model_id=PARENT, rel_path=None))
    assert rc == 0
    assert "nothing to drop" in capsys.readouterr().out


def test_component_name_survives_slugging_unchanged() -> None:
    """One separator, deliberately.

    Every downstream name -- demo folder, overlay scope, worktree -- is derived from
    the alias basename through _slug, which collapses any run of non-alphanumerics
    to a single underscore. A doubled separator could never survive, so the
    component ended up with two spellings for the same thing. Pin that the alias is
    its own slug."""
    from scripts.tt_hw_planner.probe import _component_alias
    from scripts.tt_hw_planner.scaffold_demo_folder import _slug
    import os
    import tempfile

    base = tempfile.mkdtemp()
    target = os.path.join(base, "real_dir")
    os.makedirs(target, exist_ok=True)
    os.environ["TT_HW_PLANNER_COMPONENT_BASE"] = os.path.join(base, "aliases")
    alias = _component_alias("some-org/Some-Model-9B", "some_part", target)
    name = os.path.basename(alias)
    assert name == _slug(name), f"alias {name!r} changes under slugging -> two names for one component"
    assert "some_model_9b" in name and "some_part" in name
