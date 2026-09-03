#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build the `tt-metal-models` wheel (and optionally an sdist) from this repository.

Why a staging step: `models/` lives at the repository root, next to the `ttnn`
distribution's own pyproject.toml. A PEP 517 build cannot reach outside its project
directory, and the root pyproject.toml already belongs to `ttnn`. So this script copies
the packaging files and a pruned `models/` tree into a build root and builds there.

Usage:
    python packaging/tt-metal-models/build_wheel.py --output-dir dist
    python packaging/tt-metal-models/build_wheel.py --version 0.77.0 --sdist

The version defaults to the same setuptools-scm derivation the `ttnn` wheel uses, so an
artifact built from a given commit matches the `ttnn` wheel built from that commit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_DIR.parent.parent

# Copied verbatim into the staged build root.
PACKAGING_FILES = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md")

# Directory names pruned wherever they appear under models/.
#
# `tests` is the stable, one-line rule from the packaging plan: it removes the test
# suites without needing a per-release import-graph computation. `test` and `unit_tests`
# are the two places test code lives under a differently-spelled directory
# (models/experimental/petr/test, models/experimental/bert_large_performant/unit_tests).
PRUNED_DIR_NAMES = frozenset({"tests", "test", "unit_tests", "__pycache__"})

# The one git submodule under models/. It is not initialized in a normal clone, and two
# vLLM entry points import from it.
#
# Populating it before building makes both of those entry points work -- but the
# submodule is Meta's llama repository, licensed under the Llama 2 Community License,
# not Apache-2.0. Including it in a wheel that declares Apache-2.0 and is published to a
# package index is a redistribution decision with licence consequences, so this script
# never populates it on its own: it builds without it and warns. See README.md.
SUBMODULE_PATH = "models/demos/t3000/llama2_70b/reference/llama"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    # Resolve the executable before invoking it: a missing tool becomes one clear error
    # rather than a FileNotFoundError from deep inside subprocess, and only a verified
    # executable path ever reaches the invocation.
    executable = shutil.which(cmd[0])
    if executable is None:
        raise SystemExit(f"error: `{cmd[0]}` was not found on PATH.")
    cmd = [executable, *cmd[1:]]
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, shell=False, **kwargs)


def derive_version(repo_root: Path) -> str:
    """Derive the version exactly as the `ttnn` wheel does.

    The scm configuration (tag_regex, git_describe_command) is read from the repository's
    own pyproject.toml rather than restated here, so the two artifacts cannot drift
    apart when that configuration changes. The version/local schemes mirror
    `get_metal_main_version_scheme` / `get_metal_local_version_scheme` in the repo root
    setup.py -- keep them in sync if that file changes.
    """
    try:
        from setuptools_scm import get_version
        from setuptools_scm.version import guess_next_dev_version
    except ImportError:
        raise SystemExit(
            "setuptools-scm is required to derive the version.\n"
            "Install it (`pip install setuptools-scm`) or pass --version explicitly."
        )

    # Kept separate from the setuptools-scm import above so the two failures cannot be
    # reported as each other. tomllib is stdlib only from Python 3.11; `tomli` is the
    # same library under its pre-stdlib name.
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise SystemExit(
                f"reading {repo_root / 'pyproject.toml'} needs a TOML parser, which this "
                f"interpreter (Python {sys.version_info.major}.{sys.version_info.minor}) "
                "does not provide.\n"
                "Use Python 3.11 or newer, `pip install tomli`, or pass --version explicitly."
            )

    # setuptools_scm.get_version() builds its configuration from the keyword arguments
    # it is given; it does not read pyproject.toml when called this way. So forward the
    # repository's own [tool.setuptools_scm] table rather than restating it here. That
    # table matters: `git_describe_command` excludes `*-dev*` tags, without which a
    # nightly tag like v0.77.0-dev20260804 makes the bump fail outright.
    with open(repo_root / "pyproject.toml", "rb") as f:
        scm_config = tomllib.load(f).get("tool", {}).get("setuptools_scm", {})

    def version_scheme(version):
        if version is None:
            return "0.0.0.dev0"
        if getattr(version, "exact", False):
            return version.format_with("{tag}")
        return guess_next_dev_version(version)

    def local_scheme(version):
        return f"+g{version.node}" if version.dirty else ""

    return get_version(
        root=str(repo_root),
        version_scheme=version_scheme,
        local_scheme=local_scheme,
        **scm_config,
    )


def submodule_is_populated(repo_root: Path) -> bool:
    path = repo_root / SUBMODULE_PATH
    return path.is_dir() and any(path.iterdir())


def stage(repo_root: Path, build_root: Path, prune: bool) -> None:
    """Copy the packaging files and the `models/` tree into a clean build root."""
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    for name in PACKAGING_FILES:
        shutil.copy2(PACKAGING_DIR / name, build_root / name)
    shutil.copy2(repo_root / "LICENSE", build_root / "LICENSE")

    def ignore(directory: str, names: list[str]) -> set[str]:
        if not prune:
            return {n for n in names if n == "__pycache__"}
        return {n for n in names if n in PRUNED_DIR_NAMES and (Path(directory) / n).is_dir()}

    shutil.copytree(repo_root / "models", build_root / "models", ignore=ignore, symlinks=False)


def build(build_root: Path, output_dir: Path, version: str, sdist: bool) -> None:
    env = {**os.environ, "TT_METAL_MODELS_VERSION": version}
    targets = ["--wheel"] + (["--sdist"] if sdist else [])
    _run(
        [sys.executable, "-m", "build", "--no-isolation", *targets, "--outdir", str(output_dir), str(build_root)],
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version",
        help="Version string for the artifact. Defaults to the same setuptools-scm derivation the ttnn wheel uses.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dist", help="Where to write the artifacts.")
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_ROOT / "build_tt_metal_models",
        help="Staging directory. Recreated from scratch on every run.",
    )
    parser.add_argument("--sdist", action="store_true", help="Also build an sdist.")
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Keep test directories in the staged tree. For debugging the build only.",
    )
    parser.add_argument(
        "--print-requirements",
        action="store_true",
        help=(
            "Print the runtime dependencies, one per line, and exit without building. "
            "Excludes the exact `ttnn` pin, which is not on any index until the ttnn wheel "
            "from the same commit is published. Used by CI to provision the import-matrix venv."
        ),
    )
    parser.add_argument(
        "--require-submodule",
        action="store_true",
        help=(
            "Fail if the llama reference submodule is not populated. Two vLLM entry points "
            "import from it; without it they are unimportable from the built wheel."
        ),
    )
    args = parser.parse_args()

    if args.print_requirements:
        # setup.py guards its setup() call behind __main__, so importing it here reads
        # the dependency list without triggering a build.
        sys.path.insert(0, str(PACKAGING_DIR))
        import setup as models_setup

        print("\n".join(models_setup.INSTALL_REQUIRES))
        return 0

    version = args.version or derive_version(REPO_ROOT)
    print(f"tt-metal-models version: {version}")

    if not submodule_is_populated(REPO_ROOT):
        message = (
            f"the git submodule at {SUBMODULE_PATH} is not populated, so\n"
            "  models.demos.t3000.llama2_70b.tt.generator_vllm and\n"
            "  models.demos.llama3_70b_galaxy.tt.generator_vllm\n"
            "will not be importable from the resulting wheel. Populate it with:\n"
            f"  git submodule update --init {SUBMODULE_PATH}\n"
            "but note that it is licensed under the Llama 2 Community License, not\n"
            "Apache-2.0: shipping it changes the licensing of the published artifact."
        )
        if args.require_submodule:
            raise SystemExit(f"error: {message}")
        print(f"warning: {message}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage(REPO_ROOT, args.build_root, prune=not args.no_prune)
    build(args.build_root, args.output_dir, version, args.sdist)

    print(f"\nArtifacts in {args.output_dir}:")
    for path in sorted(args.output_dir.glob("tt_metal_models-*")):
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
