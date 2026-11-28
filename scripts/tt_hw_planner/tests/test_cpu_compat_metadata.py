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


def test_a_lookalike_snapshot_is_not_repaired(monkeypatch) -> None:
    """Superseded shape-only rule: a dict of invented import->distribution names
    has the right SHAPE but describes nothing real, and mutating it corrupted a
    framework's lazy-import structure. Only a dict that agrees with the real
    mapping qualifies -- see test_a_real_distributions_snapshot_is_still_repaired
    for the genuine case."""
    import sys
    import types

    from scripts.tt_hw_planner import cpu_compat as cc

    names = cc._stubbed_package_names()
    if not names:
        return
    victim = types.ModuleType("_tt_fake_framework")
    victim.SOME_MAPPING = {f"invented_pkg_{i}": [f"invented-pkg-{i}"] for i in range(12)}
    before = {k: list(v) for k, v in victim.SOME_MAPPING.items()}
    monkeypatch.setitem(sys.modules, "_tt_fake_framework", victim)

    cc._publish_stub_distributions(names)

    assert victim.SOME_MAPPING == before, "a dict that describes no real package must be left alone"


def test_publishing_is_a_no_op_without_stand_ins() -> None:
    from scripts.tt_hw_planner import cpu_compat as cc

    cc._publish_stub_distributions([])  # must not raise


# ─── the snapshot repair must not touch look-alike dicts ───


def test_lazy_import_structure_is_not_mutated(monkeypatch) -> None:
    """A framework's lazy-import structure is also ``str -> list[str]``: module
    name -> class names. Injecting package names into one makes the framework look
    for classes that do not exist -- seen as
    ``ValueError: Could not find <SomeModel> ... nor in transformers``, from a
    model that loads fine otherwise. Shape must not be enough to qualify."""
    import sys
    import types

    from scripts.tt_hw_planner import cpu_compat as cc

    names = cc._stubbed_package_names()
    if not names:
        return
    victim = types.ModuleType("_tt_fake_lazy_framework")
    # same shape as a distributions map, entirely different meaning
    victim._import_structure = {f"submod_{i}": [f"SomeClass{i}", f"OtherClass{i}"] for i in range(30)}
    before = {k: list(v) for k, v in victim._import_structure.items()}
    monkeypatch.setitem(sys.modules, "_tt_fake_lazy_framework", victim)

    cc._publish_stub_distributions(names)

    assert victim._import_structure == before, "a lazy-import structure must be left alone"


def test_a_real_distributions_snapshot_is_still_repaired(monkeypatch) -> None:
    """The genuine case must keep working: a copy of packages_distributions()."""
    import importlib.metadata as md
    import sys
    import types

    from scripts.tt_hw_planner import cpu_compat as cc

    names = cc._stubbed_package_names()
    if not names:
        return
    real = {k: v for k, v in md.packages_distributions().items() if isinstance(v, list) and v}
    if len(real) < cc._SNAPSHOT_MIN_AGREEMENT:
        return
    victim = types.ModuleType("_tt_fake_consumer")
    victim.PACKAGE_DISTRIBUTION_MAPPING = dict(real)
    monkeypatch.setitem(sys.modules, "_tt_fake_consumer", victim)

    cc._publish_stub_distributions(names)

    for name in names:
        assert name in victim.PACKAGE_DISTRIBUTION_MAPPING, f"real snapshot not repaired for {name}"


def test_hostile_mapping_subclass_does_not_raise(monkeypatch) -> None:
    """Registries are often mapping subclasses with their own lookup semantics --
    probing one raised ``TypeError: get() missing 1 required positional argument``.
    Only plain dicts may be probed."""
    import sys
    import types

    from scripts.tt_hw_planner import cpu_compat as cc

    class _Hostile(dict):
        def get(self, key, default):  # deliberately not dict.get's signature
            return super().get(key, default)

    victim = types.ModuleType("_tt_fake_hostile")
    victim.REGISTRY = _Hostile({f"k{i}": [f"v{i}"] for i in range(30)})
    monkeypatch.setitem(sys.modules, "_tt_fake_hostile", victim)

    cc._publish_stub_distributions(cc._stubbed_package_names())  # must not raise
