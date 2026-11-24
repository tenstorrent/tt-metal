# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The SFPI toolchain is installed apart from the harness (``helpers/toolchain.py``).

What these pin: where the harness looks for it, that it reads the release from
the compiler rather than from a file an installer wrote, that it requires a
minimum and no more, and that a release other than the one ``LLK_HOME`` pins is
reported instead of going unnoticed.

Host-only. The toolchains here are stand-ins: a directory with a script for a
compiler that prints an SFPI banner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from helpers import toolchain
from helpers.test_config import BuildMode, TestConfig

LLK_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = LLK_ROOT / "tests" / "install_sfpi.sh"

MINIMUM = ".".join(str(number) for number in toolchain.MIN_SFPI_VERSION)
NEWER = f"{toolchain.MIN_SFPI_VERSION[0] + 1}.0.0"
OLDER = f"{toolchain.MIN_SFPI_VERSION[0] - 1}.99.0"


def _sfpi(root: Path, version: str = NEWER) -> Path:
    """A stand-in SFPI install at ``root`` whose compiler reports ``version``."""
    compiler = root / "compiler" / "bin" / "riscv-tt-elf-g++"
    compiler.parent.mkdir(parents=True)
    compiler.write_text(
        f"#!/bin/sh\necho 'riscv-tt-elf-g++ (tenstorrent/sfpi:{version}) 15.1.0'\n"
    )
    compiler.chmod(0o755)
    return root


def _llk_home(root: Path, pinned: str | None = None) -> Path:
    """A stand-in ``LLK_HOME``, with ``tests/sfpi-version`` pinning ``pinned``."""
    (root / "tests").mkdir(parents=True)
    if pinned is not None:
        (root / "tests" / "sfpi-version").write_text(
            f"# sfpi release {pinned}\n\nsfpi_repo='https://example.invalid/sfpi'\nsfpi_version='{pinned}'\n"
        )
    return root


class _Warnings:
    """Stands in for the logger: the messages ``warning`` was called with, formatted."""

    def __init__(self):
        self.messages = []

    def warning(self, message, *arguments):
        self.messages.append(message.format(*arguments))


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    recorder = _Warnings()
    monkeypatch.setattr(toolchain, "logger", recorder)
    return recorder.messages


@pytest.fixture(autouse=True)
def no_toolchain_of_this_machine(monkeypatch, tmp_path):
    """Lookups see only what a test builds: no variables of the outer run, no system install."""
    monkeypatch.delenv(toolchain.SFPI_ROOT_ENV, raising=False)
    monkeypatch.delenv(toolchain.SFPI_STRICT_ENV, raising=False)
    monkeypatch.setattr(toolchain, "SYSTEM_SFPI_ROOT", tmp_path / "no-system-sfpi")


# --- Where the toolchain is ---------------------------------------------------


def test_in_tree_toolchain_is_found_first(monkeypatch, tmp_path):
    llk_home = _llk_home(tmp_path / "llk")
    in_tree = _sfpi(llk_home / "tests" / "sfpi")
    monkeypatch.setattr(toolchain, "SYSTEM_SFPI_ROOT", _sfpi(tmp_path / "system"))

    assert toolchain.find_sfpi_root(llk_home) == in_tree


def test_system_toolchain_is_the_fallback(monkeypatch, tmp_path):
    llk_home = _llk_home(tmp_path / "llk")
    system = _sfpi(tmp_path / "system")
    monkeypatch.setattr(toolchain, "SYSTEM_SFPI_ROOT", system)

    assert toolchain.find_sfpi_root(llk_home) == system


def test_named_toolchain_wins_over_the_ones_that_are_found(monkeypatch, tmp_path):
    llk_home = _llk_home(tmp_path / "llk")
    _sfpi(llk_home / "tests" / "sfpi")
    named = _sfpi(tmp_path / "named")
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, str(named))

    assert toolchain.find_sfpi_root(llk_home) == named


def test_named_toolchain_that_is_missing_is_not_replaced(monkeypatch, tmp_path):
    """A wrong ``LLK_SFPI_ROOT`` must fail, not quietly build with the in-tree toolchain."""
    llk_home = _llk_home(tmp_path / "llk")
    _sfpi(llk_home / "tests" / "sfpi")
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, str(tmp_path / "typo"))

    assert toolchain.find_sfpi_root(llk_home) is None
    assert str(tmp_path / "typo") in toolchain.missing_sfpi_message(llk_home)


def test_named_toolchain_is_absolute(monkeypatch, tmp_path):
    """The compile runs in ``$LLK_HOME/tests``, so a relative root would point somewhere else there."""
    _sfpi(tmp_path / "named")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, "named")

    assert toolchain.find_sfpi_root(tmp_path / "llk") == tmp_path / "named"


def test_a_session_keeps_a_named_root_that_is_missing(monkeypatch, tmp_path):
    """With an in-tree toolchain present, the session must still not build with it."""
    llk_home = _llk_home(tmp_path / "llk")
    in_tree = _sfpi(llk_home / "tests" / "sfpi")
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, str(tmp_path / "typo"))

    assert toolchain.session_sfpi_root(llk_home) == tmp_path / "typo"

    monkeypatch.delenv(toolchain.SFPI_ROOT_ENV)
    assert toolchain.session_sfpi_root(llk_home) == in_tree


def test_missing_toolchain_names_every_place_looked_in(tmp_path):
    llk_home = _llk_home(tmp_path / "llk")

    message = toolchain.missing_sfpi_message(llk_home)

    assert str(llk_home / "tests" / "sfpi") in message
    assert str(toolchain.SYSTEM_SFPI_ROOT) in message
    assert toolchain.SFPI_ROOT_ENV in message


# --- Which release it is ------------------------------------------------------


def test_release_is_read_from_the_compiler_banner():
    version = toolchain.parse_sfpi_banner(
        "riscv-tt-elf-g++ (tenstorrent/sfpi:7.95.0[1101]) 15.1.0\nCopyright (C) 2025\n"
    )

    assert version.release == (7, 95, 0)
    assert str(version) == "7.95.0[1101]"


def test_a_trial_build_has_no_release_to_order():
    version = toolchain.parse_sfpi_banner(
        "riscv-tt-elf-g++ (tenstorrent/sfpi:7.96.0-rc1[7]) 15.1.0\n"
    )

    assert version.release is None
    assert str(version) == "7.96.0-rc1[7]"


def test_a_compiler_that_is_not_sfpi_is_rejected():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        toolchain.parse_sfpi_banner("riscv64-unknown-elf-g++ (GCC) 15.1.0\n")


def test_a_compiler_that_is_not_sfpi_is_named(tmp_path):
    """A RuntimeError, which the plugin turns into a usage error, not a traceback."""
    root = _sfpi(tmp_path / "other")
    (root / "compiler/bin/riscv-tt-elf-g++").write_text(
        "#!/bin/sh\necho 'riscv64-unknown-elf-g++ (GCC) 15.1.0'\n"
    )

    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="is not an SFPI compiler"
    ):
        toolchain.installed_sfpi_version(root)


def test_installed_release_is_asked_of_the_compiler(tmp_path):
    root = _sfpi(tmp_path / "sfpi", "8.1.0[1118]")

    assert toolchain.installed_sfpi_version(root).release == (8, 1, 0)


# --- What the harness requires of it ------------------------------------------


@pytest.mark.parametrize("version", [MINIMUM, NEWER])
def test_the_minimum_and_anything_newer_is_accepted(tmp_path, warnings, version):
    root = _sfpi(tmp_path / "sfpi", version)

    assert toolchain.check_sfpi(root, _llk_home(tmp_path / "llk")).release
    assert not warnings


def test_a_release_older_than_the_minimum_is_refused(tmp_path):
    root = _sfpi(tmp_path / "sfpi", OLDER)

    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="older than"
    ):
        toolchain.check_sfpi(root, _llk_home(tmp_path / "llk"))


def test_a_trial_build_is_used_with_a_warning(tmp_path, warnings):
    root = _sfpi(tmp_path / "sfpi", "0.0.0-trial")

    toolchain.check_sfpi(root, _llk_home(tmp_path / "llk", pinned=NEWER))

    assert len(warnings) == 1 and "not a numbered release" in warnings[0]


def test_the_pinned_release_is_silent(tmp_path, warnings):
    root = _sfpi(tmp_path / "sfpi", f"{NEWER}[5]")

    toolchain.check_sfpi(root, _llk_home(tmp_path / "llk", pinned=NEWER))

    assert not warnings


def test_a_release_other_than_the_pinned_one_is_reported(tmp_path, warnings):
    """Reported, not refused: a toolchain newer than the pin is why it is installed separately."""
    root = _sfpi(tmp_path / "sfpi", NEWER)

    toolchain.check_sfpi(root, _llk_home(tmp_path / "llk", pinned=MINIMUM))

    assert len(warnings) == 1
    assert NEWER in warnings[0] and MINIMUM in warnings[0]


def test_strict_mode_refuses_a_release_other_than_the_pinned_one(monkeypatch, tmp_path):
    root = _sfpi(tmp_path / "sfpi", NEWER)
    monkeypatch.setenv(toolchain.SFPI_STRICT_ENV, "1")

    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="not the release LLK_HOME pins"
    ):
        toolchain.check_sfpi(root, _llk_home(tmp_path / "llk", pinned=MINIMUM))


def test_this_tree_pins_a_release_the_harness_accepts():
    """The harness's minimum must never overtake the release its own tree builds with."""
    pinned = toolchain.pinned_sfpi_release(LLK_ROOT)

    assert pinned is not None, "tests/sfpi-version has no sfpi_version"
    assert pinned >= toolchain.MIN_SFPI_VERSION


# --- TestConfig ---------------------------------------------------------------


@pytest.fixture
def session(monkeypatch, tmp_path):
    """``TestConfig`` as a session with ``LLK_HOME`` at ``tmp_path/llk`` and an in-tree toolchain there."""
    llk_home = _llk_home(tmp_path / "llk", pinned=NEWER)
    root = _sfpi(llk_home / "tests" / "sfpi", NEWER)
    (tmp_path / "artefacts").mkdir()
    monkeypatch.setattr(TestConfig, "LLK_ROOT", llk_home, raising=False)
    monkeypatch.setattr(TestConfig, "SFPI_ROOT", root, raising=False)
    monkeypatch.setattr(
        TestConfig, "GXX", str(root / "compiler/bin/riscv-tt-elf-g++"), raising=False
    )
    monkeypatch.setattr(
        TestConfig, "ARTEFACTS_DIR", tmp_path / "artefacts", raising=False
    )
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.PRODUCE)
    return tmp_path


def test_in_tree_toolchain_keeps_the_relative_include(session):
    """The flag is hashed into every variant id, so the in-tree spelling must not move."""
    assert TestConfig.sfpi_include_flag() == "-Isfpi/include"


def test_a_toolchain_elsewhere_is_included_by_its_own_path(session, monkeypatch):
    elsewhere = _sfpi(session / "elsewhere")
    monkeypatch.setattr(TestConfig, "SFPI_ROOT", elsewhere)

    assert TestConfig.sfpi_include_flag() == f"-I{elsewhere}/include"


def test_compiling_without_a_toolchain_says_where_it_looked(session, monkeypatch):
    monkeypatch.setattr(TestConfig, "GXX", str(session / "absent" / "g++"))

    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="No SFPI toolchain found"
    ):
        TestConfig.require_toolchain()


def test_a_session_without_a_toolchain_is_not_checked(session, monkeypatch):
    """Host-only sessions never compile; they must still start."""
    monkeypatch.setattr(TestConfig, "GXX", str(session / "absent" / "g++"))

    TestConfig.check_toolchain()

    assert not (session / "artefacts" / "sfpi.version").exists()


def test_a_producer_records_the_release_it_builds_with(session):
    TestConfig.check_toolchain()

    assert (session / "artefacts" / "sfpi.version").read_text().strip() == NEWER


def test_a_consumer_reports_elfs_of_another_release(session, monkeypatch, warnings):
    """Consuming never rebuilds, so nothing else would notice ELFs left by an earlier toolchain."""
    (session / "artefacts" / "sfpi.version").write_text(f"{MINIMUM}\n")
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)

    TestConfig.check_toolchain()

    assert len(warnings) == 1 and "were built with SFPI" in warnings[0]
    # The record is the producer's; a consumer must not rewrite it to match itself.
    assert (session / "artefacts" / "sfpi.version").read_text().strip() == MINIMUM


def test_a_consumer_of_its_own_release_is_silent(session, monkeypatch, warnings):
    (session / "artefacts" / "sfpi.version").write_text(f"{NEWER}\n")
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)

    TestConfig.check_toolchain()

    assert not warnings


# --- tests/install_sfpi.sh ----------------------------------------------------
# Only the paths that stop before the download.


def _install(*arguments) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments], capture_output=True, text=True
    )


def test_installer_does_not_replace_a_directory_that_is_not_sfpi(tmp_path):
    """``--dest`` is replaced as a whole; a mistyped one must not cost somebody a directory."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "notes.txt").write_text("keep me")

    result = _install("--dest", str(tmp_path / "work"))

    assert result.returncode != 0
    assert "not an SFPI install" in result.stderr
    assert (tmp_path / "work" / "notes.txt").read_text() == "keep me"


def test_installer_leaves_the_pinned_release_alone(tmp_path):
    pinned = ".".join(str(number) for number in toolchain.pinned_sfpi_release(LLK_ROOT))
    (tmp_path / "sfpi").mkdir()
    (tmp_path / "sfpi" / "sfpi.version").write_text(f"{pinned}\n")

    result = _install("--dest", str(tmp_path / "sfpi"))

    assert result.returncode == 0, result.stderr
    assert "already at the correct version" in result.stdout
