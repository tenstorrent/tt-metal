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


@pytest.mark.parametrize(
    "in_tree",
    [
        pytest.param(True, id="after-setup-build"),
        pytest.param(False, id="before-setup-build"),
    ],
)
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


@pytest.mark.parametrize(
    "in_tree",
    [
        pytest.param(True, id="after-setup-build"),
        pytest.param(False, id="before-setup-build"),
    ],
)
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
@pytest.mark.parametrize(
    "in_tree",
    [
        pytest.param(True, id="after-setup-build"),
        pytest.param(False, id="before-setup-build"),
    ],
)
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
