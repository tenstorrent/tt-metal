# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The SFPI toolchain is installed apart from the harness (``helpers/toolchain.py``).

Where it is looked for, which release it is, and what happens when that is not
the release ``LLK_HOME`` pins. Host-only: the toolchains are stand-ins, a script
for a compiler that prints a banner.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from helpers import toolchain
from helpers.test_config import BuildMode, TestConfig

LLK_ROOT = Path(__file__).resolve().parents[3]
MINIMUM = ".".join(map(str, toolchain.MIN_SFPI_VERSION))
NEWER = f"{toolchain.MIN_SFPI_VERSION[0] + 1}.0.0"
OLDER = f"{toolchain.MIN_SFPI_VERSION[0] - 1}.99.0"


def _sfpi(root: Path, version: str = NEWER, mark: str = "tenstorrent/sfpi:") -> Path:
    """A stand-in SFPI install at ``root`` whose compiler reports ``version``."""
    compiler = root / "compiler/bin/riscv-tt-elf-g++"
    compiler.parent.mkdir(parents=True)
    compiler.write_text(
        f"#!/bin/sh\necho 'riscv-tt-elf-g++ ({mark}{version}) 15.1.0'\n"
    )
    compiler.chmod(0o755)
    return root


def _llk_home(root: Path, pinned: str | None = None) -> Path:
    (root / "tests").mkdir(parents=True)
    if pinned:
        (root / "tests/sfpi-version").write_text(f"sfpi_version='{pinned}'\n")
    return root


@pytest.fixture(autouse=True)
def warnings(monkeypatch, tmp_path) -> list[str]:
    """The warnings of a test, which sees no toolchain or variable of this machine."""
    monkeypatch.delenv(toolchain.SFPI_ROOT_ENV, raising=False)
    monkeypatch.delenv(toolchain.SFPI_STRICT_ENV, raising=False)
    monkeypatch.setattr(toolchain, "SYSTEM_SFPI_ROOT", tmp_path / "no-system-sfpi")
    messages = []

    class Recorder:
        @staticmethod
        def warning(message, *arguments):
            messages.append(message.format(*arguments))

    monkeypatch.setattr(toolchain, "logger", Recorder)
    return messages


def test_where_the_toolchain_is_taken_from(monkeypatch, tmp_path):
    """The system install, before it the one in ``LLK_HOME``, before both the named one."""
    llk_home = _llk_home(tmp_path / "llk")
    system = _sfpi(tmp_path / "system")
    monkeypatch.setattr(toolchain, "SYSTEM_SFPI_ROOT", system)
    assert toolchain.find_sfpi_root(llk_home) == system

    in_tree = _sfpi(llk_home / "tests/sfpi")
    assert toolchain.find_sfpi_root(llk_home) == in_tree

    # Relative, and made absolute: the compile runs in $LLK_HOME/tests.
    _sfpi(tmp_path / "named")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, "named")
    assert toolchain.find_sfpi_root(llk_home) == tmp_path / "named"


def test_a_named_toolchain_that_is_missing_is_not_replaced(monkeypatch, tmp_path):
    """A wrong ``LLK_SFPI_ROOT`` must fail, not quietly build with the in-tree toolchain."""
    llk_home = _llk_home(tmp_path / "llk")
    _sfpi(llk_home / "tests/sfpi")
    monkeypatch.setenv(toolchain.SFPI_ROOT_ENV, str(tmp_path / "typo"))

    assert toolchain.find_sfpi_root(llk_home) is None
    assert toolchain.session_sfpi_root(llk_home) == tmp_path / "typo"
    assert str(tmp_path / "typo") in toolchain.missing_sfpi_message(llk_home)


@pytest.mark.parametrize(
    "version, release", [("7.95.0[1101]", (7, 95, 0)), ("7.96.0-rc1[7]", None)]
)
def test_the_release_is_asked_of_the_compiler(tmp_path, version, release):
    found = toolchain.installed_sfpi_version(_sfpi(tmp_path, version))

    assert (str(found), found.release) == (version, release)


def test_a_compiler_that_is_not_sfpi_is_named(tmp_path):
    """A RuntimeError, which the plugin turns into a usage error, not a traceback."""
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="is not an SFPI compiler"
    ):
        toolchain.installed_sfpi_version(_sfpi(tmp_path, mark="GCC "))


@pytest.mark.parametrize(
    "version, pinned, strict, warned, refused",
    [
        (MINIMUM, None, False, 0, None),
        (f"{NEWER}[5]", NEWER, False, 0, None),
        # Reported, not refused: a toolchain newer than the pin is why it is installed separately.
        (NEWER, MINIMUM, False, 1, None),
        ("0.0.0-trial", NEWER, False, 1, None),
        (NEWER, MINIMUM, True, 0, "not the release LLK_HOME pins"),
        (OLDER, None, False, 0, "older than"),
    ],
)
def test_what_the_harness_requires_of_a_release(
    monkeypatch, tmp_path, warnings, version, pinned, strict, warned, refused
):
    root = _sfpi(tmp_path / "sfpi", version)
    llk_home = _llk_home(tmp_path / "llk", pinned)
    if strict:
        monkeypatch.setenv(toolchain.SFPI_STRICT_ENV, "1")

    if refused:
        with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
            RuntimeError, match=refused
        ):
            toolchain.check_sfpi(root, llk_home)
    else:
        toolchain.check_sfpi(root, llk_home)
    assert len(warnings) == warned


def test_this_tree_pins_a_release_the_harness_accepts():
    """The harness's minimum must never overtake the release its own tree builds with."""
    assert toolchain.pinned_sfpi_release(LLK_ROOT) >= toolchain.MIN_SFPI_VERSION


@pytest.fixture
def session(monkeypatch, tmp_path) -> Path:
    """``TestConfig`` as a producer session with ``LLK_HOME`` at ``tmp_path/llk`` and a toolchain in it."""
    llk_home = _llk_home(tmp_path / "llk", pinned=NEWER)
    root = _sfpi(llk_home / "tests/sfpi")
    (tmp_path / "artefacts").mkdir()
    for name, value in {
        "LLK_ROOT": llk_home,
        "SFPI_ROOT": root,
        "GXX": str(root / "compiler/bin/riscv-tt-elf-g++"),
        "ARTEFACTS_DIR": tmp_path / "artefacts",
        "BUILD_MODE": BuildMode.PRODUCE,
    }.items():
        monkeypatch.setattr(TestConfig, name, value, raising=False)
    return tmp_path


def test_the_include_flag_follows_the_toolchain(session, monkeypatch):
    """Relative for the in-tree toolchain: the flag is hashed into every variant id and must not move."""
    assert TestConfig.sfpi_include_flag() == "-Isfpi/include"

    monkeypatch.setattr(TestConfig, "SFPI_ROOT", session / "elsewhere")
    assert TestConfig.sfpi_include_flag() == f"-I{session}/elsewhere/include"


def test_a_session_without_a_toolchain_fails_only_where_it_compiles(
    session, monkeypatch
):
    """Host-only sessions never compile; they must still start."""
    monkeypatch.setattr(TestConfig, "GXX", str(session / "absent/g++"))

    TestConfig.check_toolchain()
    assert not (session / "artefacts/sfpi.version").exists()
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="No SFPI toolchain found"
    ):
        TestConfig.require_toolchain()


def test_a_consumer_reports_elfs_of_another_release(session, monkeypatch, warnings):
    """Consuming never rebuilds, so nothing else would notice ELFs left by an earlier toolchain."""
    built_with = session / "artefacts/sfpi.version"
    TestConfig.check_toolchain()
    assert built_with.read_text().strip() == NEWER

    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME)
    TestConfig.check_toolchain()
    assert not warnings

    built_with.write_text(f"{MINIMUM}\n")
    TestConfig.check_toolchain()
    assert len(warnings) == 1 and "were built with SFPI" in warnings[0]
    # The record is the producer's; a consumer must not rewrite it to match itself.
    assert built_with.read_text().strip() == MINIMUM


def test_installer_does_not_replace_a_directory_that_is_not_sfpi(tmp_path):
    """``--dest`` is replaced as a whole; a mistyped one must not cost somebody a directory."""
    (tmp_path / "install_sfpi.sh").symlink_to(LLK_ROOT / "tests/install_sfpi.sh")
    (tmp_path / "work").mkdir()
    (tmp_path / "work/notes.txt").write_text("keep me")

    result = subprocess.run(
        ["bash", "install_sfpi.sh", "--dest", "work"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0 and "not an SFPI install" in result.stderr
    assert (tmp_path / "work/notes.txt").read_text() == "keep me"
