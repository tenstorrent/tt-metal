#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the wheel of the tt-llk Python test harness.

    tests/harness_wheel/make_wheel.py [--out-dir DIR]

The wheel is the harness an out-of-tree suite attaches to through
``tt_llk_harness`` (docs/tests/getting_started.md, "The supported surface"),
packaged so that suite installs it instead of putting a tt-llk checkout on
``sys.path``. It holds

- ``helpers``: build and run of a test variant, stimuli generation, golden
  generators, packing and unpacking of L1 data, comparison, the pytest plugin;
- ``tt_llk_harness``: the supported interface to it;
- ``ttsim_error``, which ``helpers`` imports;
- under ``tt_llk_harness/llk_home``, in tt-llk layout, the tt-llk files the
  harness's own Python reads, the helper tree every test variant is built with
  (``tests/helpers``: firmware sources, linker scripts, headers) and
  ``tt_llk_perf`` (``LLK_HOME_FILES`` below);
- ``tt_llk_harness/requirements.txt``, the Python requirements the harness is
  tested with. The wheel requires what the harness imports
  (``RUNTIME_REQUIREMENTS``), unpinned; installing that file first pins them.

It holds no LLK, kernel, test driver or toolchain: the suite that installs it
supplies those through ``LLK_HOME`` and ``TestConfig.add_include_dirs`` /
``add_helpers_tree``.

The version is ``<API major>.<API minor>.<commit time>+g<commit>``, the API
being ``tt_llk_harness.HARNESS_API_VERSION`` and the time ``YYYYMMDDHHMM`` in
UTC, so wheels order by API and then by commit time, and an installed wheel
names the commit it was built from.

The wheel is written directly (a pure-Python wheel is a zip archive with three
metadata files, PEP 427), uncompressed, so building it needs only git and the
standard library, and the same commit always gives the same bytes. It holds the
files git tracks, so nothing untracked or ignored.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import time
import zipfile
from email.message import Message
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import NamedTuple

LLK_ROOT = Path(__file__).resolve().parents[2]

DISTRIBUTION = "tt_llk_harness"

# Python packages: directory in tt-llk -> package in the wheel.
PACKAGES = {
    "tests/python_tests/helpers": PurePosixPath("helpers"),
    "tests/python_tests/tt_llk_harness": PurePosixPath("tt_llk_harness"),
}
# Top-level modules: file in tt-llk -> module in the wheel.
MODULES = {
    "tests/python_tests/ttsim_error.py": PurePosixPath("ttsim_error.py"),
}
# tt-llk files and directories the harness needs beside its Python, kept at these same paths under LLK_HOME_PREFIX.
LLK_HOME_FILES = (
    # the firmware sources, linker scripts and headers every test variant is built with; the harness's Python parses
    # some (counters.h) and generates against others (operand.h), so they travel with it
    "tests/helpers",
    # the counter name tables tt_llk_perf parses, and that counters.h includes
    "tools/include/perf_counters",
    # the perf split's cost tables (helpers/llk_pytest_plugin.py)
    "tests/perf_split_costs.json",
    "tests/perf_split_costs_sol.json",
    # imported by helpers; the pytest plugin puts this directory on sys.path. Not a top-level package of this wheel:
    # the ttnn wheel installs one of that name.
    "tools/python/tt_llk_perf",
)
LLK_HOME_PREFIX = PurePosixPath("tt_llk_harness/llk_home")
REQUIREMENTS = "tests/requirements.txt"
REQUIREMENTS_IN_WHEEL = PurePosixPath("tt_llk_harness/requirements.txt")
# What the harness imports.
RUNTIME_REQUIREMENTS = (
    "filelock",
    "loguru",
    "ml_dtypes",
    "mpmath",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "pyyaml",
    "torch",
    "tt-exalens",
    "typing-extensions",
)

# Zip's earliest timestamp: a fixed one keeps the archive a function of its contents.
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _in(directory: Path) -> dict:
    """``subprocess.run`` options for a git command, each a literal argument list, run in ``directory``."""
    # Run from a git hook, these variables point at the hook's index and work tree.
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
    }
    return dict(cwd=directory, check=True, stdout=subprocess.PIPE, text=True, env=env)


def _files_under(source: Path) -> list[Path]:
    """The files of a directory that git tracks, sorted, as paths relative to it."""
    # -z: one NUL-terminated name per file, exactly as stored, with no quoting to undo.
    listed = subprocess.run(["git", "ls-files", "-z"], **_in(source)).stdout
    return sorted(
        Path(name) for name in listed.split("\0") if (source / name).is_file()
    )


def wheel_contents(llk_root: Path = LLK_ROOT) -> dict[PurePosixPath, Path]:
    """Path in the wheel -> file in tt-llk, for everything but the metadata."""
    contents: dict[PurePosixPath, Path] = {}

    def add(target: PurePosixPath, source: Path) -> None:
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
            add(package / relative, root / relative)
    for source, module in MODULES.items():
        add(module, llk_root / source)
    for source in LLK_HOME_FILES:
        path = llk_root / source
        if path.is_dir():
            for relative in _files_under(path):
                add(LLK_HOME_PREFIX / source / relative, path / relative)
        else:
            add(LLK_HOME_PREFIX / source, path)
    add(REQUIREMENTS_IN_WHEEL, llk_root / REQUIREMENTS)
    return contents


def api_version(llk_root: Path = LLK_ROOT) -> tuple[int, int]:
    """``HARNESS_API_VERSION``, read from the parsed source: importing the harness needs its requirements installed."""
    source = llk_root / "tests/python_tests/tt_llk_harness/__init__.py"
    for statement in ast.parse(source.read_text()).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HARNESS_API_VERSION"
            for target in statement.targets
        ):
            major, minor = ast.literal_eval(statement.value)
            return major, minor
    raise RuntimeError(f"{source} does not assign HARNESS_API_VERSION")


class Version(NamedTuple):
    """The wheel's version, ``<API major>.<API minor>.<commit time>+g<commit>``, by its parts."""

    api: tuple[int, int]
    committed: int  # commit time, seconds since the epoch
    commit: int  # the commit's SHA-1
    dirty: bool = False  # built with uncommitted changes to tracked tt-llk files

    @property
    def release(self) -> str:
        major, minor = self.api
        # To the minute: the ``+g<commit>`` part orders lexically, so it cannot order two wheels of one day.
        stamp = time.strftime("%Y%m%d%H%M", time.gmtime(self.committed))
        return f"{major}.{minor}.{stamp}"

    @property
    def local(self) -> str:
        # The first 12 of the SHA-1's 40 hex digits.
        commit = self.commit >> 4 * (40 - 12)
        return f"g{commit:012x}{'.dirty' if self.dirty else ''}"

    def __str__(self) -> str:
        return f"{self.release}+{self.local}"

    @property
    def tag(self) -> str:
        """The git tag of a release of this wheel; a tag cannot hold the ``+``."""
        return f"v{self.release}-{self.local}"


def default_version(llk_root: Path = LLK_ROOT) -> Version:
    """The version of the commit ``llk_root`` is at."""
    here = _in(llk_root)
    committed = subprocess.run(["git", "log", "-1", "--format=%ct"], **here).stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], **here).stdout
    # Changes to tracked files under llk_root; untracked files are not in the wheel.
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "."], **here
    ).stdout
    return Version(
        api=api_version(llk_root),
        committed=int(committed),
        commit=int(commit, 16),
        dirty=bool(changes),
    )


def _headers(fields: list[tuple[str, str]]) -> bytes:
    """A ``Name: value`` metadata file (METADATA and WHEEL are both in the email header format)."""
    message = Message(policy=compat32.clone(max_line_length=0))
    for name, value in fields:
        message[name] = value
    return message.as_bytes()


def _metadata(version: Version) -> bytes:
    return _headers(
        [
            ("Metadata-Version", "2.1"),
            ("Name", "tt-llk-harness"),
            ("Version", str(version)),
            (
                "Summary",
                "tt-llk Python test harness: build and run of a test variant, stimuli, goldens, comparison",
            ),
            (
                "Home-page",
                "https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk",
            ),
            ("License", "Apache-2.0"),
            ("Requires-Python", ">=3.10"),
            *(("Requires-Dist", name) for name in RUNTIME_REQUIREMENTS),
        ]
    )


_WHEEL = _headers(
    [
        ("Wheel-Version", "1.0"),
        ("Generator", "tt-llk tests/harness_wheel/make_wheel.py"),
        ("Root-Is-Purelib", "true"),
        ("Tag", "py3-none-any"),
    ]
)


def _record(entries: dict[PurePosixPath, bytes], record: PurePosixPath) -> bytes:
    """RECORD: a CSV row of path, hash and size per file, and one without hash and size for itself."""
    out = io.StringIO()
    rows = csv.writer(out, lineterminator="\n")
    for name, data in sorted(entries.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.writerow([name, f"sha256={digest.rstrip(b'=').decode()}", len(data)])
    rows.writerow([record, "", ""])
    return out.getvalue().encode()


def build_wheel(out_dir: Path, version: Version, llk_root: Path = LLK_ROOT) -> Path:
    """Write the wheel into ``out_dir`` and return its path."""

    entries = {
        name: path.read_bytes() for name, path in wheel_contents(llk_root).items()
    }
    dist_info = PurePosixPath(f"{DISTRIBUTION}-{version}.dist-info")
    entries[dist_info / "METADATA"] = _metadata(version)
    entries[dist_info / "WHEEL"] = _WHEEL
    entries[dist_info / "licenses" / "LICENSE"] = (llk_root / "LICENSE").read_bytes()
    entries[dist_info / "RECORD"] = _record(entries, dist_info / "RECORD")

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"{DISTRIBUTION}-{version}-py3-none-any.whl"
    # Stored, not deflated: the bytes then do not depend on the zlib that built them.
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(str(name), date_time=_ZIP_TIMESTAMP)
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
    args = parser.parse_args()
    version = default_version()
    print(build_wheel(args.out_dir, version))
    # In a GitHub Actions step, the step's outputs.
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as outputs:
            print(f"version={version}", f"tag={version.tag}", sep="\n", file=outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
