# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An unattended auto-onboard must not write into tracked source.

Auto-onboard runs mid-bring-up and drafts a FamilyBackend with an LLM. It used to
splice that into family_backends.py, so an unreviewed entry landed in git -- which
is how a machine-local path (canonical_hf_id='/home/.../vae') and a blank
model_type_key reached tracked source. The registry sync's own additions go to a
cache file for exactly this reason.

Three properties are pinned here: the entry lands in the cache store, the running
process can see it immediately, and an entry keyed on a whole architecture family
while pointing at a target-specific folder is flagged.
"""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_CACHE", str(tmp_path / "cache"))
    import scripts.tt_hw_planner.family_backends as fb
    import scripts.tt_hw_planner.registry_sync as rs

    importlib.reload(rs)
    importlib.reload(fb)
    fb.invalidate_overlay_cache()
    yield rs, fb
    fb.invalidate_overlay_cache()


def _entry(name="probe backend", **over):
    e = {
        "name": name,
        "category": "LLM",
        "demo_path": "models/tt_transformers",
        "routing_mode": "template",
        "model_type_keys": ["zzz_probe_arch"],
        "use_module_tree": True,
    }
    e.update(over)
    return e


# ─── the store ───────────────────────────────────────────────────


def test_added_entry_is_readable(stores) -> None:
    rs, _ = stores
    assert rs.add_onboarded(_entry())
    names = [f.get("name") for f in rs.load_onboarded().get("families", [])]
    assert "probe backend" in names


def test_re_onboarding_replaces_rather_than_duplicates(stores) -> None:
    rs, _ = stores
    rs.add_onboarded(_entry(category="LLM"))
    rs.add_onboarded(_entry(category="Image"))
    fams = [f for f in rs.load_onboarded().get("families", []) if f.get("name") == "probe backend"]
    assert len(fams) == 1 and fams[0]["category"] == "Image"


def test_entry_without_a_name_is_refused(stores) -> None:
    rs, _ = stores
    assert rs.add_onboarded({"category": "LLM"}) is False


def test_drop_removes_and_reports(stores) -> None:
    rs, _ = stores
    rs.add_onboarded(_entry())
    assert rs.drop_onboarded("probe backend") is True
    assert rs.drop_onboarded("probe backend") is False


def test_missing_store_reads_as_empty(stores) -> None:
    rs, _ = stores
    assert rs.load_onboarded() == {}


# ─── visibility in the running process ───────────────────────────


def test_onboarded_entry_joins_the_registry(stores) -> None:
    rs, fb = stores
    before = len(fb.all_backends())
    rs.add_onboarded(_entry())
    fb.invalidate_overlay_cache()
    after = fb.all_backends()
    assert len(after) == before + 1
    assert any(b.name == "probe backend" for b in after)


def test_use_module_tree_survives_the_round_trip(stores) -> None:
    """Dropped on the way through, an onboarded walker silently becomes a
    sibling-copy backend."""
    rs, fb = stores
    rs.add_onboarded(_entry(use_module_tree=True))
    fb.invalidate_overlay_cache()
    b = next(x for x in fb.all_backends() if x.name == "probe backend")
    assert b.use_module_tree is True


def test_cache_is_stale_until_invalidated(stores) -> None:
    """Pins why invalidation is needed: the supplement layer memoises."""
    rs, fb = stores
    fb.all_backends()  # prime the memo
    rs.add_onboarded(_entry())
    assert not any(b.name == "probe backend" for b in fb.all_backends())
    fb.invalidate_overlay_cache()
    assert any(b.name == "probe backend" for b in fb.all_backends())


# ─── over-broad keys ─────────────────────────────────────────────


def test_family_wide_key_on_a_target_specific_folder_is_flagged() -> None:
    from scripts.tt_hw_planner.auto_onboard import keys_broader_than_demo

    target = "/tmp/components/some_model_9b_some_part"
    reason = keys_broader_than_demo(
        {"demo_path": "models/demos/some_model_9b_some_part", "model_type_keys": ["some_arch"]}, target
    )
    assert reason and "broader" in reason


def test_folder_scoped_to_its_key_is_not_flagged() -> None:
    from scripts.tt_hw_planner.auto_onboard import keys_broader_than_demo

    assert not keys_broader_than_demo(
        {"demo_path": "models/demos/some_arch", "model_type_keys": ["some_arch"]}, "org/Some-Arch-8B"
    )


def test_shared_backend_serving_another_family_is_not_flagged() -> None:
    """A shared demo legitimately serves families its folder is not named for."""
    from scripts.tt_hw_planner.auto_onboard import keys_broader_than_demo

    assert not keys_broader_than_demo(
        {"demo_path": "models/tt_transformers", "model_type_keys": ["some_arch"]}, "org/Some-Model"
    )


def test_no_target_means_no_verdict() -> None:
    from scripts.tt_hw_planner.auto_onboard import keys_broader_than_demo

    assert keys_broader_than_demo({"demo_path": "models/demos/x_y", "model_type_keys": ["a"]}) == ""
