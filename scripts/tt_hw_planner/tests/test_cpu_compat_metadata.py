"""Accelerator stand-ins must be self-consistent.

A stand-in makes ``find_spec(pkg)`` succeed, so availability probes conclude the
package is installed and then look it up in distribution metadata. Faking one and
not the other raises ``KeyError`` deep inside an unrelated framework import.
"""

from __future__ import annotations


def test_stand_ins_are_visible_to_metadata_not_just_to_imports() -> None:
    """A stand-in makes find_spec() succeed, so availability probes conclude the
    package is installed and then look it up in distribution metadata. If we fake
    one and not the other, the probe raises KeyError deep inside the framework --
    seen as `KeyError: 'flash_attn'` while importing a diffusion model."""
    import importlib.metadata as md

    from scripts.tt_hw_planner import cpu_compat as cc

    names = cc._stubbed_package_names()
    if not names:  # every accelerator package genuinely installed here
        return
    cc._publish_stub_distributions(names)
    mapping = md.packages_distributions()
    for name in names:
        assert name in mapping, f"{name} has a stand-in but no distribution metadata"
        assert isinstance(mapping[name], list) and mapping[name]


def test_snapshotted_mappings_are_repaired(monkeypatch) -> None:
    """Frameworks snapshot packages_distributions() at import time, before our
    stand-ins exist. Those snapshots must be repaired too, or the mismatch
    survives in the module that already read it."""
    import sys
    import types

    from scripts.tt_hw_planner import cpu_compat as cc

    names = cc._stubbed_package_names()
    if not names:
        return
    victim = types.ModuleType("_tt_fake_framework")
    # a realistic snapshot: import-name -> [distribution-name], no stand-ins in it
    victim.SOME_MAPPING = {f"real_pkg_{i}": [f"real-pkg-{i}"] for i in range(12)}
    monkeypatch.setitem(sys.modules, "_tt_fake_framework", victim)

    cc._publish_stub_distributions(names)

    for name in names:
        assert name in victim.SOME_MAPPING, f"snapshot not repaired for {name}"


def test_publishing_is_a_no_op_without_stand_ins() -> None:
    from scripts.tt_hw_planner import cpu_compat as cc

    cc._publish_stub_distributions([])  # must not raise
