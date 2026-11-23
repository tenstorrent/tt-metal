#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the wheel of the tt-llk Python test harness.

    tests/harness_wheel/make_wheel.py [--out-dir DIR] [--version VERSION]

The wheel is the harness an out-of-tree suite attaches to through
``tt_llk_harness`` (docs/tests/getting_started.md, "The supported surface"),
packaged so that suite installs it instead of putting a tt-llk checkout on
``sys.path``. It holds

- ``helpers``: build and run of a test variant, stimuli generation, golden
  generators, packing and unpacking of L1 data, comparison, the pytest plugin;
- ``tt_llk_harness``: the supported interface to it;
- ``tt_llk_perf`` and ``ttsim_error``, which ``helpers`` imports;
- under ``tt_llk_harness/llk_home``, in tt-llk layout, the tt-llk files the
  harness's own Python reads and the headers its generated ``build.h``
  includes (``LLK_HOME_FILES`` below; ``tt_llk_harness.harness_files_root()``
  returns that directory);
- ``tt_llk_harness/requirements.txt``, the pinned Python requirements, which
  are also the wheel's ``pinned`` extra.

It holds no LLK, kernel, test driver, firmware source, linker script or
toolchain: the suite that installs it supplies those through ``LLK_HOME`` and
``TestConfig.add_include_dirs`` / ``add_helpers_tree``.

The version is ``<API major>.<API minor>.<commit date>+g<commit>``, the API
being ``tt_llk_harness.HARNESS_API_VERSION``, so wheels order by API and then by
day, and an installed wheel names the commit it was built from.

The wheel is written directly (a pure-Python wheel is a zip archive with three
metadata files, PEP 427), so building it needs only the standard library and
the same commit always gives the same bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

LLK_ROOT = Path(__file__).resolve().parents[2]

DISTRIBUTION = "tt_llk_harness"

# Python packages: directory in tt-llk -> package in the wheel.
PACKAGES = {
    "tests/python_tests/helpers": "helpers",
    "tests/python_tests/tt_llk_harness": "tt_llk_harness",
    "tools/python/tt_llk_perf": "tt_llk_perf",
}
# Top-level modules: file in tt-llk -> module in the wheel.
MODULES = {
    "tests/python_tests/ttsim_error.py": "ttsim_error.py",
}
# tt-llk files and directories the harness needs beside its Python, kept at these same paths under LLK_HOME_PREFIX.
LLK_HOME_FILES = (
    # parsed by helpers/counters.py when it is imported
    "tests/helpers/include/counters.h",
    # included by the build.h the harness generates for every test variant
    "tests/helpers/include/operand.h",
    # the counter name tables tt_llk_perf parses, and that counters.h includes
    "tools/include/perf_counters",
    # the perf split's cost tables (helpers/llk_pytest_plugin.py)
    "tests/perf_split_costs.json",
    "tests/perf_split_costs_sol.json",
)
LLK_HOME_PREFIX = "tt_llk_harness/llk_home"
REQUIREMENTS = "tests/requirements.txt"
REQUIREMENTS_IN_WHEEL = "tt_llk_harness/requirements.txt"

_IGNORED_DIRS = {"__pycache__"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}
# Zip's earliest timestamp: a fixed one keeps the archive a function of its contents.
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _files_under(source: Path):
    """Every file of a package or data directory, sorted, as paths relative to it."""
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix in _IGNORED_SUFFIXES:
            continue
        if _IGNORED_DIRS.intersection(path.relative_to(source).parts):
            continue
        yield path.relative_to(source)


def wheel_contents(llk_root: Path = LLK_ROOT) -> dict[str, Path]:
    """Path in the wheel -> file in tt-llk, for everything but the metadata."""
    contents: dict[str, Path] = {}

    def add(target: str, source: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(
                f"{source} is listed in {Path(__file__).name} but does not exist"
            )
        contents[target] = source

    for source, package in PACKAGES.items():
        root = llk_root / source
        if not (root / "__init__.py").is_file():
            raise FileNotFoundError(f"{root} is not a Python package")
        for relative in _files_under(root):
            add(f"{package}/{relative.as_posix()}", root / relative)
    for source, module in MODULES.items():
        add(module, llk_root / source)
    for source in LLK_HOME_FILES:
        path = llk_root / source
        if path.is_dir():
            for relative in _files_under(path):
                add(
                    f"{LLK_HOME_PREFIX}/{source}/{relative.as_posix()}", path / relative
                )
        else:
            add(f"{LLK_HOME_PREFIX}/{source}", path)
    add(REQUIREMENTS_IN_WHEEL, llk_root / REQUIREMENTS)
    return contents


def api_version(llk_root: Path = LLK_ROOT) -> tuple[int, int]:
    """``HARNESS_API_VERSION``, read from the source: importing the harness needs its requirements installed."""
    text = (llk_root / "tests/python_tests/tt_llk_harness/__init__.py").read_text()
    match = re.search(
        r"^HARNESS_API_VERSION\s*=\s*\((\d+),\s*(\d+)\)", text, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(
            "HARNESS_API_VERSION not found in tt_llk_harness/__init__.py"
        )
    return int(match.group(1)), int(match.group(2))


def _git(llk_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(llk_root), *args],
        check=True,
        capture_output=True,
        text=True,
        env={"TZ": "UTC", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    ).stdout.strip()


def default_version(llk_root: Path = LLK_ROOT) -> str:
    """``<API major>.<API minor>.<commit date, UTC>+g<commit>``, with ``.dirty`` for uncommitted harness changes."""
    major, minor = api_version(llk_root)
    try:
        commit = _git(llk_root, "rev-parse", "--short=12", "HEAD")
        date = _git(llk_root, "log", "-1", "--format=%cd", "--date=format-local:%Y%m%d")
        sources = [*PACKAGES, *MODULES, *LLK_HOME_FILES, REQUIREMENTS]
        dirty = _git(llk_root, "status", "--porcelain", "--", *sources)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Cannot derive the wheel version from git; pass --version."
        ) from error
    return f"{major}.{minor}.{date}+g{commit}{'.dirty' if dirty else ''}"


def pinned_requirements(llk_root: Path = LLK_ROOT) -> list[str]:
    """The ``name==version`` lines of tests/requirements.txt."""
    pins = []
    for line in (llk_root / REQUIREMENTS).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            pins.append(line)
    return pins


def _metadata(version: str, llk_root: Path) -> str:
    lines = [
        "Metadata-Version: 2.1",
        "Name: tt-llk-harness",
        f"Version: {version}",
        "Summary: tt-llk Python test harness: build and run of a test variant, stimuli, goldens, comparison",
        "Home-page: https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk",
        "License: Apache-2.0",
        "Requires-Python: >=3.10",
        "Provides-Extra: pinned",
    ]
    lines += [
        f'Requires-Dist: {pin} ; extra == "pinned"'
        for pin in pinned_requirements(llk_root)
    ]
    return "\n".join(lines) + "\n"


_WHEEL = """Wheel-Version: 1.0
Generator: tt-llk tests/harness_wheel/make_wheel.py
Root-Is-Purelib: true
Tag: py3-none-any
"""


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def build_wheel(
    out_dir: Path, version: str | None = None, llk_root: Path = LLK_ROOT
) -> Path:
    """Write the wheel into ``out_dir`` and return its path."""
    version = version or default_version(llk_root)
    if not re.fullmatch(r"\d+(\.\d+)*(\+[A-Za-z0-9]+(\.[A-Za-z0-9]+)*)?", version):
        raise ValueError(f"not a wheel version: {version!r}")

    entries = {
        name: path.read_bytes() for name, path in wheel_contents(llk_root).items()
    }
    dist_info = f"{DISTRIBUTION}-{version}.dist-info"
    entries[f"{dist_info}/METADATA"] = _metadata(version, llk_root).encode()
    entries[f"{dist_info}/WHEEL"] = _WHEEL.encode()
    entries[f"{dist_info}/top_level.txt"] = "".join(
        f"{name}\n"
        for name in sorted(
            [*PACKAGES.values(), *(Path(m).stem for m in MODULES.values())]
        )
    ).encode()
    record = [_record_line(name, data) for name, data in sorted(entries.items())]
    record.append(f"{dist_info}/RECORD,,")
    entries[f"{dist_info}/RECORD"] = ("\n".join(record) + "\n").encode()

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"{DISTRIBUTION}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return wheel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the wheel of the tt-llk Python test harness."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd() / "dist",
        help="directory the wheel is written to (default: ./dist)",
    )
    parser.add_argument(
        "--version",
        help="wheel version (default: <API major>.<API minor>.<commit date>+g<commit>)",
    )
    args = parser.parse_args()
    print(build_wheel(args.out_dir, args.version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
