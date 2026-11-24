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
* **Command assembly.** Compile commands must reach the child as argv. A shell
  splits a build root containing a space, and the compiler then names a
  fragment of it as a missing input — which reads as a broken checkout.

None of these raise anything on their own, so there is nothing to notice unless
something asserts on them directly.

Host-only: no toolchain, no device. Every test restores the process-wide
``TestConfig`` state it touches, because that state is shared with every other
test in the same xdist worker.
"""

from __future__ import annotations

import re

import pytest
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import COMPRESSED_META, LOOP_FACTOR, TILE_COUNT
from helpers.utils import run_shell_command

SEARCH_DIR_STATE = (
    "EXTRA_INCLUDE_PREPEND",
    "EXTRA_INCLUDE_APPEND",
    "EXTRA_SRC_INCLUDE_PREPEND",
    "EXTRA_SRC_INCLUDE_APPEND",
    "EXTRA_LINKER_SCRIPT_DIRS",
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


def test_compressed_metadata_words_change_the_variant_id():
    def compressed_variant_id(packed_meta):
        configuration = TestConfig(
            DRIVER,
            templates=[COMPRESSED_META("mixed", packed_meta)],
            skip_build_header=True,
        )
        configuration.generate_variant_hash()
        return configuration.variant_id

    assert compressed_variant_id(bytes.fromhex("0100000002000000")) != (
        compressed_variant_id(bytes.fromhex("0300000004000000"))
    )


def test_runtime_parameter_order_changes_the_variant_id():
    """build.h compiles the RuntimeParams layout in, and the host packs values in that order.

    Same runtimes in another order is another ELF: sharing one makes the kernel read each
    field from its neighbour's slot (test_hadamard read TILE_CNT=8 as LOOP_FACTOR).
    """

    def runtime_variant_id(runtimes):
        configuration = TestConfig(DRIVER, runtimes=runtimes, skip_build_header=True)
        configuration.generate_variant_hash()
        return configuration.variant_id

    assert runtime_variant_id([LOOP_FACTOR(1), TILE_COUNT(8)]) != runtime_variant_id(
        [TILE_COUNT(8), LOOP_FACTOR(1)]
    )
    assert runtime_variant_id([TILE_COUNT(8), LOOP_FACTOR(1)]) == runtime_variant_id(
        [TILE_COUNT(1), LOOP_FACTOR(4)]
    ), "runtime values alone must keep sharing one ELF"


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


def test_llk_assert_setting_changes_the_variant_id(isolated_search_dirs, monkeypatch):
    """Asserts-on and asserts-off ELFs must never share a CONSUME directory."""
    clear_search_dirs()

    monkeypatch.delenv("TT_LLK_DISABLE_ASSERTS", raising=False)
    asserts_on_default = variant_id()

    monkeypatch.setenv("TT_LLK_DISABLE_ASSERTS", "1")
    asserts_off = variant_id()

    monkeypatch.setenv("TT_LLK_DISABLE_ASSERTS", "0")
    asserts_on_explicit = variant_id()

    assert asserts_off != asserts_on_default
    assert asserts_on_explicit == asserts_on_default


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


def test_a_helpers_tree_replaces_the_linker_scripts_it_has(
    isolated_search_dirs, tmp_path
):
    """A suite with its own memory map supplies it this way; the rest stay the harness's."""
    (tmp_path / "ld").mkdir()
    (tmp_path / "ld" / "brisc.ld").touch()
    TestConfig.add_helpers_tree(tmp_path)

    assert TestConfig.linker_script("brisc.ld") == tmp_path / "ld" / "brisc.ld"
    assert (
        TestConfig.linker_script("sections.ld")
        == TestConfig.LINKER_SCRIPTS / "sections.ld"
    )


def test_a_relative_helpers_tree_gives_absolute_linker_scripts(
    isolated_search_dirs, tmp_path, monkeypatch
):
    """The link runs in ``$LLK_HOME/tests``, where a path relative to the suite points somewhere else."""
    (tmp_path / "helpers" / "ld").mkdir(parents=True)
    (tmp_path / "helpers" / "ld" / "brisc.ld").touch()
    monkeypatch.chdir(tmp_path)
    TestConfig.add_helpers_tree("helpers")

    assert TestConfig.linker_script("brisc.ld") == tmp_path / "helpers/ld/brisc.ld"


def test_linker_script_dirs_change_the_variant_id(isolated_search_dirs, tmp_path):
    """The same headers and sources with and without the tree's ``ld/`` link differently.

    CONSUME mode finds its ELF by the id, so the two must not share one.
    """
    TestConfig.add_include_dirs(tmp_path / "include")
    TestConfig.add_src_include_dirs(tmp_path / "src")
    without_linker_scripts = variant_id()

    clear_search_dirs(False)
    TestConfig.add_helpers_tree(tmp_path)

    assert variant_id() != without_linker_scripts


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


# --------------------------------------------------------------------------- #
# Which Tensix an xdist worker is bound to.
# --------------------------------------------------------------------------- #


@pytest.fixture
def worker_location(monkeypatch):
    """Record the core index resolve asks for, and restore the class state it writes."""
    asked = []

    def fake_location(worker_index, device_id=0):
        asked.append(worker_index)
        return f"core-{worker_index}"

    monkeypatch.setattr(
        "helpers.test_config.device_module.tensix_location_for_worker",
        fake_location,
    )
    saved = (
        TestConfig._PENDING_WORKER_INDEX,
        TestConfig.TENSIX_LOCATION,
        TestConfig.TEST_TARGET.run_simulator,
    )
    try:
        yield asked
    finally:
        (
            TestConfig._PENDING_WORKER_INDEX,
            TestConfig.TENSIX_LOCATION,
            TestConfig.TEST_TARGET.run_simulator,
        ) = saved


def test_ttsim_workers_share_the_first_core(monkeypatch, worker_location):
    """A ttsim worker past the chip's core count still gets the first core.

    Each xdist process dlopens its own simulator, so gw39 and a restarted
    gw32 both run on core 0. The descriptor size is not a pytest-worker cap.
    """
    monkeypatch.setenv("TT_METAL_SIMULATOR", "/sim/libttsim_qsr.so")
    TestConfig.TEST_TARGET.run_simulator = True
    TestConfig._PENDING_WORKER_INDEX = 39

    TestConfig.resolve_worker_tensix_location()

    assert worker_location == [0]
    assert TestConfig.TENSIX_LOCATION == "core-0"
    assert TestConfig._PENDING_WORKER_INDEX is None


def test_a_shared_device_keeps_one_core_per_worker(monkeypatch, worker_location):
    """Silicon and the RTL server still map the xdist index onto a distinct core."""
    monkeypatch.setenv("TT_METAL_SIMULATOR", "/sim/rtl_build")
    TestConfig.TEST_TARGET.run_simulator = True
    TestConfig._PENDING_WORKER_INDEX = 7

    TestConfig.resolve_worker_tensix_location()

    assert worker_location == [7]
    assert TestConfig.TENSIX_LOCATION == "core-7"

    monkeypatch.delenv("TT_METAL_SIMULATOR", raising=False)
    TestConfig._PENDING_WORKER_INDEX = 7
    TestConfig.TEST_TARGET.run_simulator = False

    TestConfig.resolve_worker_tensix_location()

    assert worker_location == [7, 7]
    assert TestConfig.TENSIX_LOCATION == "core-7"


# --------------------------------------------------------------------------- #
# Commands reach the compiler as argv, never as a shell string.
#
# The build root is whatever ``$RUNNER_TEMP``/``$TMPDIR`` holds, so it can carry
# a space. A command assembled as one string lets a shell split that path, and
# the compiler then reports a missing input file for a fragment of it — which
# reads as a broken checkout, not as a quoting bug.
# --------------------------------------------------------------------------- #

SPACED_ROOT = "a build root"


@pytest.fixture
def shared_build(monkeypatch, tmp_path):
    """Drive ``build_shared_artefacts`` with a spaced build root, capturing argv.

    ``run_shell_command`` is replaced, so no compiler runs and this stays
    host-only. The directories are real because the build touches a marker in
    them, and the class-level "already built" flag is restored on the way out.
    """
    obj_dir = tmp_path / SPACED_ROOT / "shared" / "obj"
    elf_dir = tmp_path / SPACED_ROOT / "shared" / "elf"
    obj_dir.mkdir(parents=True)
    elf_dir.mkdir(parents=True)

    commands = []
    monkeypatch.setattr(
        "helpers.test_config.run_shell_command",
        lambda command, *args, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(TestConfig, "SHARED_OBJ_DIR", obj_dir)
    monkeypatch.setattr(TestConfig, "SHARED_ELF_DIR", elf_dir)
    monkeypatch.setattr(TestConfig, "WITH_COVERAGE", True)
    monkeypatch.setattr(TestConfig, "SHARED_ARTEFACTS_AVAILABLE", False)

    TestConfig(DRIVER, skip_build_header=True).build_shared_artefacts()
    return commands, obj_dir, elf_dir


def output_of(command) -> str:
    """The ``-o`` value as the child would receive it: one element, or a fragment."""
    assert not isinstance(command, str), f"a shell string, not an argv: {command}"
    return command[command.index("-o") + 1]


def test_a_spaced_build_root_stays_one_argv_element(shared_build):
    """Every output path the shared build writes is a single argument.

    Equality against the whole path is the assertion that matters: a shell split
    yields ``a`` plus ``build root/...`` as separate elements, so an argv that
    merely *contains* the directory name is not enough.
    """
    commands, obj_dir, elf_dir = shared_build
    outputs = {output_of(argv) for argv in commands}

    assert outputs == {str(obj_dir / "coverage.o"), str(elf_dir / "brisc.elf")}
    for argv in commands:
        assert SPACED_ROOT not in argv, f"the build root was split into {argv}"


def test_linker_scripts_stay_glued_to_their_flag(shared_build):
    """``-T<path>`` is one token; splitting it makes the path a link input."""
    commands, _, elf_dir = shared_build
    link = next(argv for argv in commands if output_of(argv).endswith("brisc.elf"))

    scripts = [token for token in link if token.startswith("-T")]
    assert scripts, f"no linker script in {link}"
    assert all(len(token) > len("-T") for token in scripts)


def test_run_shell_command_rejects_a_string_before_running_it(monkeypatch):
    """A string must not reach subprocess at all, rather than run under a shell."""

    def unreachable(*args, **kwargs):
        raise AssertionError("subprocess.run reached with a string command")

    monkeypatch.setattr("helpers.utils.subprocess.run", unreachable)

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError, match="argv sequence"
    ):
        run_shell_command(f"echo {SPACED_ROOT}")


def test_the_variant_directory_carries_no_shell_metacharacters():
    """Why ``lcov --directory <variant>`` never sees a bracket.

    Variant dirs are ``<artefact key>/<variant id>``: the id is a sha256 digest,
    and a driver path that is not already a plain token is sanitized and
    disambiguated by a digest. Parametrised test ids never reach the path.
    """
    key = TestConfig._safe_artefact_key(
        "/drivers/eltwise[formats:Float32-dest:Yes].cpp"
    )

    assert re.fullmatch(r"sources/[A-Za-z0-9._+-]+", key), key
