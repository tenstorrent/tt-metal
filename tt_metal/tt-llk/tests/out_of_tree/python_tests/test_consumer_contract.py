# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the out-of-tree testing contract.

Module name: this file must not be called ``test_out_of_tree_contract.py``. The
wrapper that launches this suite already owns that basename in
``tests/python_tests/``, and this suite puts ``tests/python_tests`` on
``sys.path`` to reach the harness. Under pytest's default prepend import mode a
test module is imported by bare basename, so two modules sharing one can race in
``sys.modules``. Keeping the names distinct removes the possibility.

Every assertion here stands in for something a consumer outside this repository
depends on. If one of these fails, an external suite is broken — treat it as an
API break, not as a fixture to be adjusted until it passes again.

The contract itself is documented in ``docs/tests/getting_started.md`` §9.

Two layers:

* Host-only tests, which need no toolchain and no device.
* ``test_out_of_tree_driver_compiles``, which drives a real compile. The C++
  ``#error`` / ``static_assert`` checks in the driver verify search-dir wiring
  and precedence, so this suite never has to assert on ``TestConfig`` internals
  to know that the ``-I`` flags actually reached the compiler.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fixture_paths import (
    FIXTURE_ROOT,
    cpp_source,
    helpers_tree,
    llk_python_tests,
    shadowed_include_dirs,
)
from goldens.oot_golden import OutOfTreeGolden
from tt_llk_harness import (
    DataFormat,
    DestAccumulation,
    StimuliConfig,
    TestConfig,
    Tilize,
    generate_stimuli,
    get_golden_generator,
    golden_registry,
    goldens,
    input_output_formats,
    params,
    register_golden,
)

INPUT_DIMENSIONS = [32, 32]
NUM_FACES_PER_TILE = 4


# --------------------------------------------------------------------------- #
# The facade: what a consumer is allowed to import.
# --------------------------------------------------------------------------- #


# The exported surface, pinned *per version*. Keying by version is what ties the
# two together: a surface that grows has nowhere to be recorded except under a
# version that does not exist yet, so adding a name means adding an entry and
# moving HARNESS_API_VERSION onto it. A flat pin checked alongside a separate
# version equality would let the surface grow at (1, 0) — and then
# require_version(1, 1) would still reject a harness that already has the name,
# which is precisely the mismatch that gate exists to prevent.
CONTRACT_SURFACES = {
    (1, 0): frozenset(
        {
            "BlocksCalculationAlgorithm",
            "ChipArchitecture",
            "DataFormat",
            "DestAccumulation",
            "DestSync",
            "ELEMENTS_PER_TILE",
            "HARNESS_API_VERSION",
            "L1Accumulation",
            "PYTEST_PLUGIN",
            "StimuliConfig",
            "TILE_DIM",
            "TestConfig",
            "Tilize",
            "VectorMode",
            "blackhole_only",
            "format_dict",
            "generate_stimuli",
            "get_chip_architecture",
            "get_golden_generator",
            "get_num_blocks_and_num_tiles_in_block",
            "golden_registry",
            "goldens",
            "input_output_formats",
            "parametrize",
            "params",
            "passed_test",
            "quasar_only",
            "register_golden",
            "require_version",
            "round_to_dest_width",
            "skip_for_blackhole",
            "skip_for_coverage",
            "skip_for_quasar",
            "skip_for_wormhole",
            "tilize_block",
            "untilize_block",
            "wormhole_only",
        }
    ),
}


def test_exported_surface_matches_the_pinned_contract():
    """Changing the contract must be deliberate, and must move the version."""
    import tt_llk_harness

    version = tt_llk_harness.HARNESS_API_VERSION
    assert version in CONTRACT_SURFACES, (
        f"HARNESS_API_VERSION is {version}, which has no pinned surface. Add a "
        "CONTRACT_SURFACES entry for it recording exactly what that version "
        "exports."
    )

    expected = CONTRACT_SURFACES[version]
    exported = set(tt_llk_harness.__all__)
    added = sorted(exported - expected)
    removed = sorted(expected - exported)
    major, minor = version

    assert not removed, (
        f"removed from the contract: {removed}\n"
        "This breaks every external suite using those names. Land a "
        "deprecation first (both spellings working, migration note in "
        "docs/tests/getting_started.md §9); only then add a "
        f"CONTRACT_SURFACES[({major + 1}, 0)] entry without them and move "
        "HARNESS_API_VERSION onto it."
    )
    assert not added, (
        f"added to the contract: {added}\n"
        "Additions are fine — they are a promise to keep supporting these. Add "
        f"a CONTRACT_SURFACES[({major}, {minor + 1})] entry including them and "
        "move HARNESS_API_VERSION onto it, so a consumer can gate on the new "
        f"names with require_version({major}, {minor + 1}). Editing the "
        f"({major}, {minor}) entry in place would let the surface grow while "
        "the version stands still."
    )


def test_catalogue_aliases_import_both_ways():
    """In-tree spelling has to keep working when a test is lifted out of tree.

    ``params.DEST_INDEX(0)`` and ``from tt_llk_harness.params import
    DEST_INDEX`` must both resolve, and to the same object — otherwise copying
    an in-tree test into an external suite means rewriting every parameter
    reference, and people reach for ``helpers`` instead.
    """
    import tt_llk_harness
    from tt_llk_harness.goldens import DataCopyGolden
    from tt_llk_harness.params import DEST_INDEX

    assert DEST_INDEX is tt_llk_harness.params.DEST_INDEX
    assert DataCopyGolden is tt_llk_harness.goldens.DataCopyGolden


def test_names_the_harness_rebinds_are_resolved_at_call_time():
    """The facade must not freeze a name the harness swaps during configure.

    ``TestConfig.setup_mode`` replaces
    ``helpers.golden_generators.get_golden_generator`` in ``pytest_configure``
    — a stand-in under ``--compile-producer``, a caching proxy under
    ``--stimuli-only``. The facade is imported before that, when the consumer's
    conftest loads the plugin, so an eager ``from ... import`` would keep the
    pre-swap function and the two spellings this module presents as equivalent
    would silently disagree: ``goldens.get_golden_generator`` would follow the
    swap, the flat name would not.

    The visible damage is an out-of-tree suite computing real goldens under
    ``--compile-producer`` instead of the stand-in.
    """
    import helpers.golden_generators as implementation
    import tt_llk_harness

    original = implementation.get_golden_generator

    def stand_in(*args, **kwargs):  # what setup_mode installs
        return "stand-in"

    # The spelling that matters is the one ``__all__`` advertises and that this
    # module itself uses at the top: ``from tt_llk_harness import
    # get_golden_generator``. It resolves once, at import, and binds for good —
    # so a module ``__getattr__`` would not save it, and a conftest importing
    # this name would freeze the pre-swap generator. Hence the call-through
    # assertion below rather than an identity check on the attribute.
    from tt_llk_harness import get_golden_generator as imported_by_value

    implementation.get_golden_generator = stand_in
    try:
        assert imported_by_value(OutOfTreeGolden) == "stand-in", (
            "a name bound by `from tt_llk_harness import ...` did not follow "
            "the swap; the export must forward on every call, not re-export a "
            "reference captured at import time"
        )
        assert tt_llk_harness.get_golden_generator(OutOfTreeGolden) == "stand-in"
        assert tt_llk_harness.goldens.get_golden_generator is stand_in
    finally:
        implementation.get_golden_generator = original


def test_every_exported_name_resolves():
    """``__all__`` is the contract; nothing in it may be a dangling re-export.

    A rename under ``helpers`` that forgets this module would otherwise only
    surface in whichever consumer happened to use that one name.
    """
    import tt_llk_harness

    missing = [n for n in tt_llk_harness.__all__ if not hasattr(tt_llk_harness, n)]
    assert not missing, f"exported but not importable: {missing}"


def test_plugin_name_constant_matches_the_module_that_exists():
    """``PYTEST_PLUGIN`` is what a suite puts in ``pytest_plugins``; it has to
    name a real module, and the same one this fixture's conftest loaded."""
    import importlib

    import tt_llk_harness

    assert tt_llk_harness.PYTEST_PLUGIN == "tt_llk_harness.plugin"
    assert importlib.import_module(tt_llk_harness.PYTEST_PLUGIN) is not None


def test_version_gate_accepts_compatible_and_rejects_the_rest(monkeypatch):
    """Compatible means same major, minor no newer than the harness.

    The interesting case is a *newer* harness major. Plain tuple ordering would
    call that compatible, so a suite written for 1.x would sail through the gate
    against 2.x — which only exists because something breaking changed — and
    fail later somewhere less obvious.
    """
    import tt_llk_harness

    major, minor = tt_llk_harness.HARNESS_API_VERSION

    tt_llk_harness.require_version(major, minor)  # exact
    tt_llk_harness.require_version(major)  # minor defaults to 0
    if minor > 0:
        tt_llk_harness.require_version(major, minor - 1)  # older minor is fine

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError, match="harness API"
    ):
        tt_llk_harness.require_version(major, minor + 1)  # needs a newer minor
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError, match="harness API"
    ):
        tt_llk_harness.require_version(major + 1)  # needs a newer major

    # And a harness that has moved on past this suite's major must be rejected.
    monkeypatch.setattr(tt_llk_harness, "HARNESS_API_VERSION", (major + 1, 0))
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError, match="breaking change"
    ):
        tt_llk_harness.require_version(major, minor)


# The only places in this fixture allowed to name the implementation, and why.
# Everything here asserts *about* ``helpers`` — that it resolves to the harness,
# that ``register_golden`` genuinely comes from it — which is contract checking,
# not consumer usage. Consumer code reaches the harness through the facade.
SANCTIONED_IMPLEMENTATION_IMPORTS = {
    "python_tests/test_consumer_contract.py": {
        "import helpers",
        "import helpers.golden_generators as goldens_mod",
        "import helpers.golden_generators as implementation",
    },
}


def test_no_consumer_module_reaches_past_the_facade():
    """Scan the whole fixture, not just this file.

    An earlier version of this test read only its own source, which let the
    fixture's own golden module import ``helpers.golden_generators`` directly
    without anything noticing — the representative consumer was quietly
    depending on the private namespace the fixture exists to avoid. Walking the
    tree closes that, and an explicit allow-list keyed by file says which
    exceptions are intentional instead of pinning a count.
    """
    offenders: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}

    for path in sorted(FIXTURE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(FIXTURE_ROOT).as_posix()
        allowed = SANCTIONED_IMPLEMENTATION_IMPORTS.get(rel, set())
        found = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip().startswith(("import helpers", "from helpers"))
        ]
        seen[rel] = set(found)
        unsanctioned = [line for line in found if line not in allowed]
        if unsanctioned:
            offenders[rel] = unsanctioned

    assert not offenders, (
        "these consumer modules import the private `helpers` namespace:\n"
        + "\n".join(f"  {rel}: {lines}" for rel, lines in offenders.items())
        + "\nEither reach it through tt_llk_harness (adding the name to "
        "__all__ if missing), or add a justified entry to "
        "SANCTIONED_IMPLEMENTATION_IMPORTS."
    )

    # An exemption that no longer matches anything is worse than none: it sits
    # there pre-authorising an import nobody reviewed, in the file most likely
    # to be edited next. Make going stale a failure rather than a slow leak.
    stale = {
        rel: sorted(allowed - seen.get(rel, set()))
        for rel, allowed in SANCTIONED_IMPLEMENTATION_IMPORTS.items()
        if allowed - seen.get(rel, set())
    }
    assert not stale, (
        "SANCTIONED_IMPLEMENTATION_IMPORTS lists exemptions that match nothing "
        f"any more: {stale}. Remove them."
    )


# --------------------------------------------------------------------------- #
# Host-only: the Python side of the contract.
# --------------------------------------------------------------------------- #


def test_facade_plugin_registers_the_harness_hooks_exactly_once(pytestconfig):
    """``tt_llk_harness.plugin`` must forward, and forward once.

    This suite names only the facade in ``pytest_plugins``; the implementation
    module has to end up registered anyway, because it owns the hooks. Two
    things can go wrong and neither shows up as an import error:

    * the forwarding stops working, and the consumer silently runs with no
      ``--compile-producer``, no arch markers and no artefact wiring;
    * the facade starts re-exporting the hook functions instead of forwarding
      to the module, so pytest registers the same hooks under two plugin names
      and every hook runs twice.

    Checking the plugin manager catches both. An earlier version of this test
    asserted that the arch markers existed on the implementation module, which
    could never fail: the facade imports those same names eagerly, so a missing
    one aborts collection long before any test runs.
    """
    manager = pytestconfig.pluginmanager

    implementation = manager.get_plugin("helpers.llk_pytest_plugin")
    assert implementation is not None, (
        "tt_llk_harness.plugin did not forward to the harness hooks; a consumer "
        "naming only the facade would run without them"
    )

    # Count hook *implementations*, not registered module objects: a re-export
    # puts the same function on a second module, which an identity check over
    # registered plugins would miss, but ``__module__`` still points here.
    #
    # For ``pytest_addoption`` specifically pytest gets there first — a
    # duplicate aborts the session with "option names ... already added" — so
    # this assertion is the backstop for re-exported hooks that do not collide
    # that loudly, ``pytest_configure`` among them.
    implementations = [
        impl
        for impl in manager.hook.pytest_addoption.get_hookimpls()
        if getattr(impl.function, "__module__", "") == "helpers.llk_pytest_plugin"
    ]
    assert len(implementations) == 1, (
        f"the harness contributes {len(implementations)} pytest_addoption "
        "implementations; hooks registered twice run twice. The facade must "
        "forward via pytest_plugins, not re-export the hook functions."
    )

    # Proves pytest_addoption actually ran through that registration: this
    # option exists only because the harness plugin added it.
    assert pytestconfig.getoption("--compile-producer") is not None


def test_suite_local_package_imports_without_sys_path_edits():
    """A consumer's test modules must not have to touch ``sys.path`` themselves.

    This module imported ``goldens.oot_golden`` and ``fixture_paths`` at the
    top without doing so, which is the property the contract promises.

    On the mechanism, deliberately not asserted here: pytest's prepend import
    mode puts ``<rootdir>/python_tests`` on the path when it collects this
    module, and this suite's ``pytest.ini`` sets ``pythonpath`` as well, exactly
    as blaze's does. ``_ensure_suite_pythonpath`` in the plugin is a third
    fallback for a consumer that sets neither. Neutering it changes nothing
    here, so this test does not guard that function — it guards the outcome,
    which is what a consumer actually depends on.
    """
    import fixture_paths
    import goldens

    assert Path(goldens.__file__).resolve().is_relative_to(FIXTURE_ROOT)
    assert Path(fixture_paths.__file__).resolve().is_relative_to(FIXTURE_ROOT)


def test_helpers_resolves_to_the_harness_not_a_local_shadow():
    """``helpers`` must be the harness package even though the fixture ships
    its own C++ ``helpers/`` tree next door."""
    import helpers
    import helpers.golden_generators as goldens_mod

    assert Path(helpers.__file__).resolve().is_relative_to(llk_python_tests())
    assert goldens_mod.golden_registry is golden_registry


def test_out_of_tree_golden_registers():
    assert OutOfTreeGolden in golden_registry
    registered = golden_registry[OutOfTreeGolden]
    assert type(registered) is OutOfTreeGolden
    assert registered() == OutOfTreeGolden.MARKER

    # The registry assertions above are the substance, and they run everywhere.
    #
    # What follows does not run in CI, by design: the wrapper always launches
    # this suite with ``--compile-producer``, and ``setup_mode`` then swaps in a
    # stand-in generator that ignores ``cls`` entirely — so the lookup can only
    # return a DummyGoldenGenerator and the branch below is dead. It is kept for
    # a developer running this suite directly without that flag, where it does
    # check that the lookup resolves a real out-of-tree golden.
    generate = get_golden_generator(OutOfTreeGolden)
    if type(generate) is goldens.DummyGoldenGenerator:
        return
    assert generate is registered
    assert generate() == OutOfTreeGolden.MARKER


def test_register_golden_is_the_harness_decorator():
    """A consumer's goldens must go through the harness registry, so in-tree
    and out-of-tree goldens stay interchangeable."""
    assert register_golden.__module__ == "helpers.golden_generators"
    assert (
        Path(register_golden.__code__.co_filename)
        .resolve()
        .is_relative_to(llk_python_tests() / "helpers")
    )
    assert goldens.DataCopyGolden in golden_registry, "in-tree goldens still resolve"


def test_absolute_driver_path_is_accepted_and_keyed_safely():
    """An absolute ``test_name`` is the driver; the artefact key must stay a
    tame ``sources/<basename>`` path so the harness never mkdirs through a
    ``.cpp`` file."""
    driver = cpp_source("out_of_tree_contract_test.cpp")
    assert Path(driver).is_absolute()
    assert (
        not Path(driver).resolve().is_relative_to(llk_python_tests().parent / "sources")
    )

    configuration = TestConfig(driver, _formats(), skip_build_header=True)
    assert configuration.test_source_path == driver
    assert configuration.test_name == "sources/out_of_tree_contract_test.cpp"


def test_llk_tree_include_roots_expands_one_arch_tree():
    """``-I`` is not recursive; consumers rely on this to point at a
    proprietary ``tt_llk_<arch>`` tree with one call."""
    roots = TestConfig.llk_tree_include_roots(Path("/nonexistent/tt_llk_blackhole"))
    assert [Path(r).name for r in roots] == ["llk_lib", "hal", "inc", "sfpu"]
    assert [Path(r).parent.name for r in roots] == [
        "tt_llk_blackhole",
        "llk_lib",
        "common",
        "inc",
    ]


def test_per_variant_search_dirs_are_accepted():
    """The constructor keywords are part of the contract alongside the class
    methods, for suites that need one variant to differ."""
    low, high = shadowed_include_dirs()
    configuration = TestConfig(
        cpp_source("out_of_tree_contract_test.cpp"),
        _formats(),
        include_dirs=[high],
        src_include_dirs=[helpers_tree() / "src"],
        helpers_trees=[helpers_tree()],
        skip_build_header=True,
    )
    assert high.resolve() in configuration.include_dirs
    assert (helpers_tree() / "include").resolve() in configuration.include_dirs
    assert (helpers_tree() / "src").resolve() in configuration.src_include_dirs


# --------------------------------------------------------------------------- #
# Compile: proves the registered dirs reach the compiler, in the right order.
# --------------------------------------------------------------------------- #


def test_out_of_tree_driver_compiles():
    """Compile an out-of-tree driver that depends on every registered dir.

    The driver ``#error``s if ``oot_probe.h`` is missing or resolves to the
    shadowed copy, and ``static_assert``s on the helpers tree. So a green
    compile here is the contract holding end to end:

    * absolute driver path outside ``tests/sources/``
    * ``add_include_dirs`` reaching the compile, prepended (later call wins)
    * ``add_helpers_tree`` wiring both ``include/`` and ``src/``

    Under ``--compile-producer`` the harness compiles and then skips, which is
    what makes this runnable in CI with no silicon attached.
    """
    configuration = _compilable_variant()

    # The compile is the assertion, so this calls ``prepare()`` rather than
    # ``run()``: it builds and stops, with no device step and no producer-mode
    # ``pytest.skip`` to reason around. This fixture exists to prove the
    # out-of-tree *contract* — configuration, search-dir precedence, and that
    # the flags reach the compiler — not to re-check LLK numerics, which the
    # in-tree suites already do on silicon.
    configuration.prepare()


# --------------------------------------------------------------------------- #
# Negative: the contract's failure modes, and proof this suite is not vacuous.
# --------------------------------------------------------------------------- #


def test_per_variant_include_dirs_override_suite_wide_ones():
    """Invert precedence on one variant and require the build to notice.

    This is the anti-vacuity check for the whole fixture. The positive test
    above passes when the driver's ``#error`` directives stay quiet, which is
    also what happens if those directives silently stop being reachable — if
    someone drops the ``#include "oot_probe.h"``, or the probe headers go
    missing, or precedence stops being observable at all. So here we force the
    low-priority header to win and require the compile to fail *for the stated
    reason*. If this test ever passes-by-not-failing, the positive one has
    stopped proving anything.

    It doubles as the only coverage of a documented guarantee: a per-variant
    ``include_dirs=`` beats a suite-wide ``add_include_dirs``. Note it needs no
    mutation of ``TestConfig`` class state, so it cannot leak into other tests.
    """
    low, _high = shadowed_include_dirs()

    # Same stimuli as the positive test, so this variant is compilable in every
    # respect except the inverted include precedence. Without them
    # ``generate_runtime_args_struct`` omits buffer_A/buffer_Res, the driver's
    # TUs fail on a missing member whichever probe header wins, and the compile
    # would break for a reason that has nothing to do with what this test
    # claims to prove — leaving DID NOT RAISE unreachable.
    configuration = _compilable_variant(include_dirs=[low])

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError
    ) as excinfo:
        # ``prepare()``, not ``run()``: in PRODUCE mode ``run()`` ends in
        # ``pytest.skip``, and ``Skipped`` derives from BaseException, so it
        # would sail straight through this ``raises`` and the plugin would
        # rewrite the skip to *passed*. This test would then be green having
        # asserted nothing — in exactly the case it exists to catch, a probe
        # that stopped firing. ``prepare()`` compiles without that skip, so an
        # unexpectedly successful build surfaces as DID NOT RAISE.
        configuration.prepare()

    message = str(excinfo.value)
    assert "OOT_PROBE_SHADOWED" in message, (
        "expected the low-priority oot_probe.h to win for this variant; "
        "per-variant include_dirs no longer take precedence, or the driver's "
        f"probe went dead:\n{message}"
    )
    # A consumer debugging its own kernel needs the compiler's diagnostics, not
    # a bare non-zero exit. Match the *shape* of a gcc diagnostic rather than
    # the bare filename: RuntimeError from run_shell_command echoes the whole
    # argv, which already contains ``-I<artefact dir>/out_of_tree_contract_test.cpp/...``,
    # so a substring check on the name would pass on the echo alone and prove
    # nothing about stderr.
    assert re.search(r"out_of_tree_contract_test\.cpp:\d+:\d+: error:", message), (
        "the compiler's own diagnostic (file:line:col: error:) is not in the "
        f"failure; only the echoed command would be:\n{message}"
    )


def test_missing_test_name_is_rejected():
    """Documented guardrail: the driver is not optional."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError, match="test_name"
    ):
        TestConfig(None, _formats())


def test_driver_path_that_breaks_the_include_directive_is_rejected():
    """An absolute driver path is emitted as ``#include "<path>"``. A path
    containing a quote, backslash, or newline is not representable there, so
    the harness must reject it up front rather than emit broken C++."""
    for bad in ('/tmp/we"ird.cpp', "/tmp/back\\slash.cpp", "/tmp/new\nline.cpp"):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError
        ):
            TestConfig(bad, _formats())


def _compilable_variant(**overrides) -> TestConfig:
    """A variant of the fixture driver that compiles cleanly.

    Shared by the positive and negative compile tests so they differ in exactly
    one thing — the include precedence the negative test inverts. In particular
    the stimuli are not optional: without them ``generate_runtime_args_struct``
    omits ``buffer_A``/``buffer_Res``, which the driver's UNPACK and PACK
    translation units dereference unconditionally, and every compile would fail
    on a missing member regardless of which probe header won.
    """
    formats = _formats()
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )

    return TestConfig(
        cpp_source("out_of_tree_contract_test.cpp"),
        formats,
        templates=[
            params.generate_input_dim(INPUT_DIMENSIONS, INPUT_DIMENSIONS),
            params.TILIZE(Tilize.No),
        ],
        runtimes=[
            params.DEST_INDEX(0),
            params.TILE_COUNT(tile_cnt_A),
            params.NUM_FACES(NUM_FACES_PER_TILE),
            params.NUM_BLOCKS(1),
            params.NUM_TILES_IN_BLOCK(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=NUM_FACES_PER_TILE,
        ),
        dest_acc=DestAccumulation.No,
        **overrides,
    )


def _formats():
    return input_output_formats([DataFormat.Float16_b], same=True)[0]
