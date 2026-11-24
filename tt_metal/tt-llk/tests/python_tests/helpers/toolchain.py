# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The SFPI toolchain a session compiles with: where it is and which release it is.

SFPI is released far more often than the harness changes, so the harness does
not carry or pin it. It takes the toolchain from ``LLK_SFPI_ROOT`` or finds it,
reads the release from the compiler and requires only ``MIN_SFPI_VERSION``. A
release other than the one ``LLK_HOME`` pins (``tests/sfpi-version``) is
reported, and is an error only with ``LLK_SFPI_STRICT=1``.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

from .logger import logger

SFPI_ROOT_ENV = "LLK_SFPI_ROOT"
SFPI_STRICT_ENV = "LLK_SFPI_STRICT"

# Where the SFPI deb and rpm packages install; tt-metal's JIT falls back to it as well (jit_build/build.cpp).
SYSTEM_SFPI_ROOT = Path("/opt/tenstorrent/sfpi")

# The release tt-metal pinned when the newest toolchain flag the harness passes was added (Quasar's
# -march=..._xttzbkb, TestConfig.QUASAR_VECTOR_MARCH). What an LLK tree's own sources need is its pin, often newer.
MIN_SFPI_VERSION = (7, 84, 0)

_COMPILER = Path("compiler") / "bin" / "riscv-tt-elf-g++"
# ``riscv-tt-elf-g++ --version`` starts ``riscv-tt-elf-g++ (tenstorrent/sfpi:7.95.0[1101]) 15.1.0``.
_BANNER_MARK = "tenstorrent/sfpi:"


class SfpiVersion(NamedTuple):
    """An SFPI release as its compiler reports it."""

    text: str  # as printed, e.g. "7.95.0[1101]"
    # None for a build that is not MAJOR.MINOR.MICRO: a trial toolchain, which cannot be ordered.
    release: Optional[tuple[int, ...]]

    def __str__(self) -> str:
        return self.text


def in_tree_sfpi_root(llk_home: Path) -> Path:
    """Where ``tests/setup_testing_env.sh`` of a tt-llk tree installs SFPI."""
    return Path(llk_home) / "tests" / "sfpi"


def candidate_sfpi_roots(llk_home: Path) -> list[Path]:
    """The roots a toolchain is looked for in, in order.

    ``LLK_SFPI_ROOT`` alone when it is set: a named toolchain that is missing
    must not be replaced by another one silently.
    """
    explicit = os.environ.get(SFPI_ROOT_ENV)
    if explicit:
        # Absolute: the compile runs in $LLK_HOME/tests, not in the directory the variable was written for.
        return [Path(explicit).expanduser().absolute()]
    return [in_tree_sfpi_root(llk_home), SYSTEM_SFPI_ROOT]


def find_sfpi_root(llk_home: Path) -> Optional[Path]:
    """The first candidate root that holds the compiler, or None."""
    for root in candidate_sfpi_roots(llk_home):
        if (root / _COMPILER).is_file():
            return root
    return None


def session_sfpi_root(llk_home: Path) -> Path:
    """The root found, else the first place looked in: a named root that is missing is not replaced."""
    return find_sfpi_root(llk_home) or candidate_sfpi_roots(llk_home)[0]


def missing_sfpi_message(llk_home: Path) -> str:
    tried = "\n".join(
        f"  {root / _COMPILER}" for root in candidate_sfpi_roots(llk_home)
    )
    return (
        f"No SFPI toolchain found. Looked for:\n{tried}\n"
        f"Install the release LLK_HOME pins (tests/sfpi-version) and set {SFPI_ROOT_ENV} to its root, "
        "the directory that holds compiler/ and include/."
    )


def parse_sfpi_banner(banner: str) -> SfpiVersion:
    """The release in the first line of ``riscv-tt-elf-g++ --version``."""
    _, mark, rest = banner.partition(_BANNER_MARK)
    text, closed, _ = rest.partition(")")
    if not (mark and closed and text):
        raise ValueError(f"not an SFPI compiler banner: {banner.splitlines()[:1]}")
    numbers = text.partition("[")[0].split(".")
    release = (
        tuple(int(number) for number in numbers)
        if len(numbers) == 3 and all(number.isdecimal() for number in numbers)
        else None
    )
    return SfpiVersion(text=text, release=release)


@lru_cache(maxsize=None)
def installed_sfpi_version(root: Path) -> SfpiVersion:
    """The release of the toolchain at ``root``, asked of its compiler."""
    compiler = root / _COMPILER
    try:
        banner = subprocess.run(
            ["riscv-tt-elf-g++", "--version"],
            executable=compiler,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        # A toolchain built for another machine lands here (an x86_64 SFPI on an aarch64 host, say).
        raise RuntimeError(
            f"Cannot run the SFPI compiler {compiler}: {error}"
        ) from error
    try:
        return parse_sfpi_banner(banner)
    except ValueError as error:
        raise RuntimeError(f"{compiler} is not an SFPI compiler: {error}") from error


def pinned_sfpi_release(llk_home: Path) -> Optional[tuple[int, ...]]:
    """The release ``LLK_HOME``'s ``tests/sfpi-version`` pins, or None when the tree has no readable pin."""
    pin = Path(llk_home) / "tests" / "sfpi-version"
    if not pin.is_file():
        return None
    # A file of shell assignments, one of them sfpi_version='7.96.0'.
    for line in pin.read_text().splitlines():
        name, assigned, value = line.partition("=")
        if assigned and name.strip() == "sfpi_version":
            numbers = value.strip().strip("'\"").split(".")
            if all(number.isdecimal() for number in numbers):
                return tuple(int(number) for number in numbers)
    return None


def report_sfpi_drift(message: str) -> None:
    """A toolchain that is not the expected one: a warning, or an error under ``LLK_SFPI_STRICT=1``."""
    if os.environ.get(SFPI_STRICT_ENV) == "1":
        raise RuntimeError(f"{message} ({SFPI_STRICT_ENV}=1 makes this an error.)")
    logger.warning("{}", message)


def check_sfpi(root: Path, llk_home: Path) -> SfpiVersion:
    """The toolchain at ``root`` is new enough for the harness; say so when it is not the one ``llk_home`` pins."""
    version = installed_sfpi_version(root)
    if version.release is None:
        logger.warning(
            "SFPI {} at {} is not a numbered release; it is not checked against the minimum or the pin.",
            version,
            root,
        )
        return version
    if version.release < MIN_SFPI_VERSION:
        minimum = ".".join(str(number) for number in MIN_SFPI_VERSION)
        raise RuntimeError(
            f"SFPI {version} at {root} is older than {minimum}, the oldest release this harness builds with. "
            f"Install a newer one, or point {SFPI_ROOT_ENV} at one."
        )
    pinned = pinned_sfpi_release(llk_home)
    if pinned is not None and pinned != version.release:
        pin = ".".join(str(number) for number in pinned)
        report_sfpi_drift(
            f"SFPI {version} at {root} is not the release LLK_HOME pins ({pin}, tests/sfpi-version)."
        )
    return version
