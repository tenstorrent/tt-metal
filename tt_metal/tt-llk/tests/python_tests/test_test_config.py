# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``helpers/test_config.py`` — the harness's own behaviour.

This is the home for host-side tests of ``TestConfig`` itself, as opposed to the
kernel tests that use it. New ones belong here rather than in another one-off
file.

What it covers today is the two properties whose failures are silent:

* **Variant keying.** The variant id decides which ELF a test loads —
  ``prepare`` does not rebuild in CONSUME mode, it trusts the id to find what
  the producer pass built. An id that ignores a compilation input means a test
  runs a binary built from different flags, and nothing reports it.
* **Parameter ownership.** Configs must not mutate a caller's lists or a shared
  default, or one variant silently changes what later variants compile.

Neither failure mode raises anything on its own, so there is nothing to notice
unless something asserts on it directly.

Host-only: no toolchain, no device. Every test restores the process-wide
``TestConfig`` state it touches, because that state is shared with every other
test in the same xdist worker.
"""

from __future__ import annotations

import pytest
from helpers.test_config import TestConfig

SEARCH_DIR_STATE = (
    "EXTRA_INCLUDE_PREPEND",
    "EXTRA_INCLUDE_APPEND",
    "EXTRA_SRC_INCLUDE_PREPEND",
    "EXTRA_SRC_INCLUDE_APPEND",
)

DRIVER = "sources/eltwise_unary_datacopy_test.cpp"

# Both regimes matter: before ``setup_build`` header extras live in
# ``EXTRA_INCLUDE_*`` and the hash's group fences separate them; after it they
# are folded into ``INCLUDES`` by order and the fences are empty.
_IN_TREE_REGIMES = pytest.mark.parametrize(
    "in_tree",
    [
        pytest.param(True, id="after-setup-build"),
        pytest.param(False, id="before-setup-build"),
    ],
)


@pytest.fixture
def isolated_search_dirs():
    """Snapshot and restore the class-level search-dir registries."""
    saved = {name: list(getattr(TestConfig, name)) for name in SEARCH_DIR_STATE}
    saved_includes = list(TestConfig.INCLUDES)
    try:
        yield
    finally:
        for name, value in saved.items():
            getattr(TestConfig, name)[:] = value
        TestConfig.INCLUDES = saved_includes


@pytest.fixture
def speed_of_light():
    saved = TestConfig.SPEED_OF_LIGHT
    TestConfig.SPEED_OF_LIGHT = True
    try:
        yield
    finally:
        TestConfig.SPEED_OF_LIGHT = saved


def variant_id() -> str:
    configuration = TestConfig(DRIVER, skip_build_header=True)
    configuration.generate_variant_hash()
    return configuration.variant_id


# Stand-ins for the in-tree ``-I`` flags ``setup_compilation_options`` installs.
IN_TREE_INCLUDES = ["-I/in/tree/first", "-I/in/tree/second"]


def clear_search_dirs(in_tree: bool = False) -> None:
    """Reset the registries, optionally as a session that has run ``setup_build``.

    The distinction is load-bearing. ``add_include_dirs`` folds header extras
    into ``INCLUDES`` (``prepend + rest + append``) only when ``INCLUDES`` is
    already populated, which in a real session it always is by the time a test
    runs. Clearing it to ``[]`` skips that fold, so a test written that way
    exercises a path production never takes.
    """
    for name in SEARCH_DIR_STATE:
        getattr(TestConfig, name).clear()
    TestConfig.INCLUDES = list(IN_TREE_INCLUDES) if in_tree else []


# --------------------------------------------------------------------------- #
# The variant hash must cover every compilation input, roles included.
# --------------------------------------------------------------------------- #


@_IN_TREE_REGIMES
def test_registered_search_dirs_change_the_variant_id(isolated_search_dirs, in_tree):
    """Search dirs live in class state, so ``self.__dict__`` cannot see them.

    ``prepare`` does not rebuild in CONSUME mode — it trusts the variant id to
    locate the ELF the producer pass built. Two configurations that compile
    against different headers must not share an id.

    Run in both regimes: ``add_helpers_tree`` here is what a real consumer's
    conftest calls, and after ``setup_build`` its header half lands in the
    merged ``INCLUDES`` rather than in the extras.
    """
    clear_search_dirs(in_tree)
    ids = [variant_id()]

    for register in (
        lambda: TestConfig.add_include_dirs("/probe/headers-one"),
        lambda: TestConfig.add_include_dirs("/probe/headers-two"),
        lambda: TestConfig.add_src_include_dirs("/probe/src"),
        lambda: TestConfig.add_helpers_tree("/probe/helpers-tree"),
    ):
        register()
        ids.append(variant_id())

    assert len(set(ids)) == len(ids), f"variant ids collided: {ids}"


@_IN_TREE_REGIMES
def test_search_dir_precedence_changes_the_variant_id(isolated_search_dirs, in_tree):
    """Registration order is a compilation input: it decides which copy wins."""
    clear_search_dirs(in_tree)
    TestConfig.add_include_dirs("/probe/low")
    TestConfig.add_include_dirs("/probe/high")
    high_wins = variant_id()

    clear_search_dirs(in_tree)
    TestConfig.add_include_dirs("/probe/high")
    TestConfig.add_include_dirs("/probe/low")
    low_wins = variant_id()

    assert high_wins != low_wins


@pytest.mark.parametrize(
    "register",
    [
        pytest.param(TestConfig.add_include_dirs, id="header-dirs"),
        pytest.param(TestConfig.add_src_include_dirs, id="src-dirs"),
    ],
)
@_IN_TREE_REGIMES
def test_search_dir_role_changes_the_variant_id(
    isolated_search_dirs, register, in_tree
):
    """The same dir, prepended vs appended, is not the same configuration.

    ``prepend`` decides whether the dir shadows the in-tree copy — for src dirs,
    whether a consumer's ``trisc.cpp`` wins over ``tests/helpers/src``. Hashing
    a flat concatenation of the groups lost that distinction: the token sequence
    was identical either way, so both roles shared one variant id and one cached
    ELF.

    Both regimes are covered because two different mechanisms carry the role.
    Before ``setup_build``, header extras sit in ``EXTRA_INCLUDE_*`` and the
    hash's group fences distinguish them. After it — which is every real session
    — they are folded into ``INCLUDES`` by order, the fences are empty, and the
    ordering is what the hash has to notice. A test that only ran the first
    regime would pass while the fold silently stopped honouring ``prepend``.
    """
    clear_search_dirs(in_tree)
    register("/probe/role", prepend=True)
    prepended = variant_id()

    clear_search_dirs(in_tree)
    register("/probe/role", prepend=False)
    appended = variant_id()

    assert prepended != appended, (
        "prepend and append hash identically, so one cached ELF now serves two "
        "different include precedences"
    )


def test_variant_id_is_stable_for_an_unchanged_configuration(isolated_search_dirs):
    """The flip side: no spurious cache invalidation."""
    clear_search_dirs()
    TestConfig.add_include_dirs("/probe/stable")
    assert variant_id() == variant_id()


# --------------------------------------------------------------------------- #
# Parameter lists belong to the instance, not to the caller or the default.
# --------------------------------------------------------------------------- #


def test_omitted_lists_do_not_accumulate_across_variants(speed_of_light):
    """A variant built without ``templates`` must not inherit the last one's.

    With a shared ``[]`` default and an in-place fold, the first
    speed-of-light variant wrote its runtimes into the default object and every
    later variant silently picked them up as templates.
    """
    first = TestConfig("/tmp/first.cpp", runtimes=["RUNTIME_A"], skip_build_header=True)
    second = TestConfig("/tmp/second.cpp", skip_build_header=True)

    assert first.templates == ["RUNTIME_A"], "speed-of-light should fold runtimes in"
    assert (
        second.templates == []
    ), f"leaked from the previous variant: {second.templates}"


def test_caller_lists_are_not_mutated(speed_of_light):
    """Constructing a config must not modify lists the caller still holds."""
    templates = ["TEMPLATE_A"]
    runtimes = ["RUNTIME_A"]

    TestConfig(
        "/tmp/x.cpp", templates=templates, runtimes=runtimes, skip_build_header=True
    )

    assert templates == ["TEMPLATE_A"]
    assert runtimes == ["RUNTIME_A"]


def test_variants_do_not_share_list_objects():
    """Two configs built from one list must not alias each other's parameters."""
    shared = ["TEMPLATE_A"]
    first = TestConfig("/tmp/a.cpp", templates=shared, skip_build_header=True)
    second = TestConfig("/tmp/b.cpp", templates=shared, skip_build_header=True)

    first.templates.append("TEMPLATE_B")

    assert second.templates == ["TEMPLATE_A"]
    assert shared == ["TEMPLATE_A"]


@pytest.fixture
def isolated_layout():
    from helpers import device
    from helpers.stimuli_config import StimuliConfig

    saved = {name: value for name, value in vars(TestConfig).items() if name.isupper()}
    old_coverage, old_mailboxes = StimuliConfig.WITH_COVERAGE, device.Mailboxes
    try:
        TestConfig.MEMORY_LAYOUT_LD_SCRIPT = None
        TestConfig.OPTIONS_COMPILE = None
        TestConfig.NON_COVERAGE_OPTIONS_COMPILE = None
        TestConfig.DEVICE_PRINT_ENABLED = False
        yield
    finally:
        for name in list(vars(TestConfig)):
            if name.isupper() and name not in saved:
                delattr(TestConfig, name)
        for name, value in saved.items():
            setattr(TestConfig, name, value)
        StimuliConfig.WITH_COVERAGE, device.Mailboxes = old_coverage, old_mailboxes


@pytest.mark.parametrize(
    "coverage,large", [(False, False), (True, True), (False, True)]
)
def test_memory_layout_is_independent_of_instrumentation(
    isolated_layout, coverage, large
):
    from helpers import device
    from helpers.stimuli_config import StimuliConfig
    from helpers.test_config import CoverageBuild, ProfilerBuild

    debug_script = (
        TestConfig.LINKER_SCRIPTS / f"memory.{TestConfig.ARCH.value}.debug.ld"
    )
    if large and not debug_script.is_file():
        with pytest.raises(  # allow-pytest.raises: LLK has no expect_error
            ValueError, match="match the target"
        ):
            TestConfig.setup_build(
                TestConfig.LLK_ROOT,
                with_coverage=coverage,
                memory_layout="debug" if not coverage else None,
            )
        return
    TestConfig.setup_build(
        TestConfig.LLK_ROOT,
        with_coverage=coverage,
        memory_layout="debug" if large and not coverage else None,
    )
    configuration = TestConfig(DRIVER, skip_build_header=True)
    flags, layout, uninstrumented = configuration.resolve_compile_options()
    assert layout.endswith(".debug.ld") is large
    assert ("-fprofile-arcs" in flags) is coverage
    assert ("-DCOVERAGE" in flags) is coverage
    assert "-fprofile-arcs" not in uninstrumented
    assert TestConfig.runtime_address() == (0x6E000 if large else 0x20000)
    assert device.Mailboxes.Unpacker.value == TestConfig.runtime_address() - 0x48
    assert configuration.coverage_build == (
        CoverageBuild.Yes if coverage else CoverageBuild.No
    )
    assert configuration.profiler_build == ProfilerBuild.No

    # Exercise the allocator itself with ordinary unary-format defaults.
    stimuli = object.__new__(StimuliConfig)
    from helpers.llk_params import DataFormat

    stimuli.stimuli_A_format = stimuli.stimuli_B_format = stimuli.stimuli_res_format = (
        DataFormat.Float16_b
    )
    stimuli.tile_dimensions = [32, 32]
    stimuli.tile_count_A = stimuli.tile_count_B = 1
    stimuli.operand_res_tile_size = None
    stimuli._srcs_32bit_mode = False
    stimuli._operand_use_srcs = lambda name: False
    stimuli._active_optional_operands = lambda: []
    stimuli._calculate_tile_sizes()
    assert stimuli.buf_a_addr == (0x70000 if large else 0x21000)
    assert stimuli.buf_res_addr > stimuli.buf_a_addr


def test_memory_layout_changes_variant_and_shared_identity(isolated_layout):
    root = TestConfig.LLK_ROOT
    identities = []
    debug_available = (
        TestConfig.LINKER_SCRIPTS / f"memory.{TestConfig.ARCH.value}.debug.ld"
    ).is_file()
    for coverage, large in [(False, False), (False, True), (True, True)]:
        if large and not debug_available:
            with pytest.raises(  # allow-pytest.raises: LLK has no expect_error
                ValueError, match="match the target"
            ):
                TestConfig.setup_build(
                    root, with_coverage=coverage, memory_layout="debug"
                )
            continue
        TestConfig.setup_build(
            root,
            with_coverage=coverage,
            memory_layout="debug" if large else None,
        )
        identities.append((variant_id(), TestConfig.SHARED_DIR))
    expected_count = 3 if debug_available else 1
    assert len({row[0] for row in identities}) == expected_count
    assert len({row[1] for row in identities}) == expected_count
    if not debug_available:
        TestConfig.setup_build(root)
    assert variant_id() == identities[-1][0]


def test_explicit_normal_layout_refuses_coverage(isolated_layout):
    with pytest.raises(  # allow-pytest.raises: LLK has no expect_error
        ValueError, match="support coverage"
    ):
        TestConfig.setup_build(
            TestConfig.LLK_ROOT, with_coverage=True, memory_layout="normal"
        )


def test_device_print_follows_layout_and_restores_normal(isolated_layout):
    for layout, base, size in (
        ("normal", 0x15000, 0x4000),
        ("debug", 0x6A000, 0x2000),
        ("normal", 0x15000, 0x4000),
    ):
        if (
            layout == "debug"
            and not (
                TestConfig.LINKER_SCRIPTS / f"memory.{TestConfig.ARCH.value}.debug.ld"
            ).is_file()
        ):
            with pytest.raises(  # allow-pytest.raises: LLK has no expect_error
                ValueError, match="match the target"
            ):
                TestConfig.setup_build(TestConfig.LLK_ROOT, memory_layout=layout)
            continue
        TestConfig.setup_build(TestConfig.LLK_ROOT, memory_layout=layout)
        configuration = TestConfig(DRIVER, skip_build_header=True)
        configuration.requires_device_print = True
        flags, _, _ = configuration.resolve_compile_options()
        assert "-DCOVERAGE" not in flags
        assert TestConfig.DEVICE_PRINT_BUFFER_BASE == base
        assert TestConfig.DEVICE_PRINT_BUFFER_SIZE == size
        expected_buffers = [(base, size, TestConfig.PROCESSOR_COUNT)]
        if TestConfig.ARCH.value == "quasar":
            expected_buffers = [(base, size, 16), (base + size, 0x2000, 8)]
        assert TestConfig.device_print_buffers() == expected_buffers
        assert base + size <= TestConfig.runtime_address() - 0x48


def test_layout_reconfiguration_resets_build_and_device_state(isolated_layout):
    from pathlib import Path

    from helpers import device

    identities = []
    debug_available = (
        TestConfig.LINKER_SCRIPTS / f"memory.{TestConfig.ARCH.value}.debug.ld"
    ).is_file()
    for layout in ("normal", "debug", "normal"):
        if layout == "debug" and not debug_available:
            with pytest.raises(  # allow-pytest.raises: LLK has no expect_error
                ValueError, match="match the target"
            ):
                TestConfig.setup_build(TestConfig.LLK_ROOT, memory_layout=layout)
            continue
        TestConfig.SHARED_ARTEFACTS_AVAILABLE = True
        TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE = True
        TestConfig._BUILD_DIRS_CREATED = True
        TestConfig.BRISC_ELF_LOADED = True
        TestConfig.LAST_LOADED_ELFS = Path("/previous-layout")
        TestConfig.CURRENT_LOADED_CONFIG = "previous-layout"
        TestConfig.setup_build(TestConfig.LLK_ROOT, memory_layout=layout)
        assert not TestConfig.SHARED_ARTEFACTS_AVAILABLE
        assert not TestConfig.PROFILER_SHARED_ARTEFACTS_AVAILABLE
        assert not TestConfig._BUILD_DIRS_CREATED
        assert not TestConfig.BRISC_ELF_LOADED
        assert TestConfig.LAST_LOADED_ELFS == Path()
        assert TestConfig.CURRENT_LOADED_CONFIG == "uninitialised"
        assert not TestConfig.WITH_COVERAGE
        assert device.Mailboxes.Unpacker.value == TestConfig.runtime_address() - 0x48
        identities.append(
            (variant_id(), TestConfig.SHARED_DIR, TestConfig.runtime_address())
        )
    assert identities[0] == identities[-1]
    if debug_available:
        assert identities[0] != identities[1]
        TestConfig.setup_build(TestConfig.LLK_ROOT, memory_layout="debug")
    # An omitted selector resets a previous explicit selection.
    TestConfig.setup_build(TestConfig.LLK_ROOT)
    assert (
        variant_id(),
        TestConfig.SHARED_DIR,
        TestConfig.runtime_address(),
    ) == identities[0]


@pytest.mark.parametrize("speed_of_light", [False, True])
def test_default_layout_and_perf_flags_are_unchanged(isolated_layout, speed_of_light):
    TestConfig.setup_build(TestConfig.LLK_ROOT, speed_of_light=speed_of_light)
    configuration = TestConfig(DRIVER, skip_build_header=True)
    flags, layout, _ = configuration.resolve_compile_options()
    assert not layout.endswith(".debug.ld")
    assert ("-DSPEED_OF_LIGHT" in flags) is speed_of_light
    assert "-fprofile-arcs" not in flags
    assert "-DCOVERAGE" not in flags


@pytest.mark.parametrize("layout", [None, "normal", "debug"])
def test_pytest_selects_layout_before_build_setup(isolated_layout, monkeypatch, layout):
    from types import SimpleNamespace
    from helpers import llk_pytest_plugin as plugin

    options = {}

    class Parser:
        def addoption(self, name, *args, **kwargs):
            options[name] = kwargs

    plugin.pytest_addoption(Parser())
    assert options["--memory-layout"]["choices"] == ("normal", "debug")
    assert options["--memory-layout"]["default"] is None
    values = {
        name: value.get("default")
        for name, value in options.items()
        if "default" in value
    }
    values["--memory-layout"] = layout
    values["--logging-level"] = "INFO"
    config = SimpleNamespace(
        rootpath=TestConfig.LLK_ROOT / "tests",
        option=SimpleNamespace(),
        getoption=lambda name, default=None: values.get(name, default),
    )
    called = []

    class SetupReached(Exception):
        pass

    def setup(*args, **kwargs):
        called.append(kwargs["memory_layout"])
        raise SetupReached

    monkeypatch.setattr(plugin, "_ensure_suite_pythonpath", lambda _: None)
    monkeypatch.setattr(plugin, "configure_logger", lambda **_: None)
    monkeypatch.setattr(TestConfig, "perf_run_tag", lambda: "host-test")
    monkeypatch.setattr(TestConfig, "setup_build", setup)
    monkeypatch.setenv("LLK_HOME", str(TestConfig.LLK_ROOT))
    with pytest.raises(SetupReached):  # allow-pytest.raises: LLK has no expect_error
        plugin.pytest_configure(config)
    assert called == [layout]
