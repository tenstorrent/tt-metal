#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Import every vLLM entry point in an installed `tt-metal-models` against the wheel.

This is the package's real contract. A wheel that is missing a `package_data` file, or
that has been pruned too far, still builds and still imports at the top level -- it fails
later, at model construction, which is the worst shape of bug. Importing every entry
point catches both immediately.

Two properties make the result trustworthy, and both are easy to lose:

* **Each import runs in its own subprocess, from a neutral working directory.** `models`
  is a PEP 420 namespace package, so a `models/` directory in the current directory --
  a tt-metal checkout, most obviously -- silently shadows the installed one, and the
  matrix would then be testing the checkout rather than the wheel.
* **The environment must contain nothing but the wheel and its declared dependencies.**
  A run inside an environment that already has extra packages under-reports missing
  dependencies. A stray `pytest` is exactly how this class of bug reaches users, so it
  is checked for explicitly.

Usage:
    # Against the current environment (CI: install the wheel first)
    python import_matrix.py

    # Build a clean venv, install the wheel into it, and run the matrix there
    python import_matrix.py --wheel dist/tt_metal_models-*.whl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Packages that a consumer of this one supplies, not this one. The `generator_vllm`
# entry points import `vllm` because they exist to be loaded *by* vLLM; depending on it
# here would invert the relationship (the Tenstorrent vLLM plugin is what depends on
# `models`). Their absence is reported as SKIP rather than failure, so the matrix stays
# runnable in a CI job that has no vLLM fork installed. Use --strict to demand them.
CONSUMER_SUPPLIED = frozenset({"vllm"})

# Entry points that cannot import unless the `models/demos/t3000/llama2_70b/reference/llama`
# git submodule was populated when the wheel was built. Reported as expected failures
# when the submodule is absent from the installed tree, so that the common case (a wheel
# built from a plain clone) does not drown the real signal -- but still counted as
# failures for a release build, which must populate the submodule. See README.md.
SUBMODULE_DEPENDENT = {
    "models.demos.t3000.llama2_70b.tt.generator_vllm",
    "models.demos.llama3_70b_galaxy.tt.generator_vllm",
}
SUBMODULE_MARKER = "models/demos/t3000/llama2_70b/reference/llama"

# Entry points that are already broken in the source tree, for reasons that have nothing
# to do with packaging. Keyed by the *specific* module whose absence is expected, so that
# any other failure in the same entry point is still reported: this allowlist cannot
# silently absorb a regression, and it stops being satisfied the moment upstream fixes
# the defect.
# This entry is arch-dependent, and will not fire in CI -- do not delete it as dead.
# The branch is chosen by is_blackhole(), i.e. ttnn.get_arch_name(), which returns
# 'invalid' on a machine with no Tenstorrent device (verified). CI runners therefore
# take the Wormhole branch and this entry point passes there; it fails only on an actual
# Blackhole host, which is where the entry earns its keep.
KNOWN_BROKEN = {
    "models.demos.wormhole.bge_large_en.demo.generator_vllm": (
        "models.demos.blackhole.bge_large_en",
        "models/demos/bge_large_en/runner/performant_runner_infra.py takes a Blackhole "
        "branch importing models.demos.blackhole.bge_large_en, which does not exist in "
        "the repository. Unimportable on any Blackhole host, packaged or not.",
    ),
}

# Imported in addition to the discovered entry points. `model_config` is the module that
# reads the per-model parameter JSON and the prefetcher YAML, so it is the one that fails
# when the non-Python payload is missing from the wheel.
EXTRA_SENTINELS = (
    "models.tt_transformers.tt.model_config",
    "models.demos.utils.model_targets",
    "models.demos.utils.trace_region_sizes",
)


# The probe embeds the module name into Python source (see _PROBE_SNIPPET), so this is
# an injection boundary: anything that is not a plain dotted identifier is refused
# before it gets near the interpolation. Names arrive from filesystem discovery, and a
# hostile filename in the tree under test must not become code.
_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


def checked_module(module: str) -> str:
    if not _MODULE_NAME.match(module):
        raise SystemExit(f"error: {module!r} is not a dotted module name; refusing to probe it.")
    return module


def checked_interpreter(python: str) -> str:
    """Resolve `python` to a concrete executable file, or fail with one clear error.

    Interpreter paths arrive from the command line (--python) or from a venv this script
    just created. Resolving them here keeps unvetted strings out of every subprocess
    invocation below, and turns a typo into an immediate error instead of a confusing
    per-probe failure.
    """
    resolved = shutil.which(python)
    if resolved is None:
        raise SystemExit(f"error: {python!r} is not an executable Python interpreter.")
    # Absolutize WITHOUT following symlinks: a venv's bin/python is a symlink to the
    # base interpreter, and resolving it would silently escape the venv -- the matrix
    # would then probe an environment that does not contain the wheel under test.
    return os.path.abspath(resolved)


def find_models_root(python: str, cwd: str) -> Path:
    """Locate the installed `models` namespace package, without importing it.

    Runs from a neutral directory for the same reason the probes do: from a tt-metal
    checkout, `models` resolves to the checkout and every result below would describe
    the source tree rather than the wheel under test.
    """
    code = (
        "import importlib.machinery as m, json;"
        "s = m.PathFinder.find_spec('models');"
        "print(json.dumps(list(s.submodule_search_locations) if s else []))"
    )
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True, cwd=cwd, shell=False)
    locations = json.loads(out.stdout)
    if not locations:
        raise SystemExit("error: no `models` package is importable in the target environment.")
    return Path(locations[0])


def discover_entry_points(models_root: Path) -> list[str]:
    """Every `generator_vllm` module in the installed tree.

    Discovered from the installed files rather than from a hardcoded list, so that a
    model family added upstream is covered by this gate without anyone remembering to
    update it.
    """
    modules = []
    for path in sorted(models_root.rglob("generator_vllm.py")):
        rel = path.relative_to(models_root.parent).with_suffix("")
        modules.append(str(rel).replace(os.sep, "."))
    return modules


# Reporting the failure cannot rely on reading the last line of stderr: importing ttnn
# installs a nanobind leak checker that prints at interpreter shutdown, so the last line
# is reliably teardown noise rather than the exception. Catch the exception in-process
# and emit it as JSON on a marker line instead.
_MARKER = "@@PROBE@@"
_PROBE_SNIPPET = textwrap.dedent(
    f"""
    import json, sys, traceback
    try:
        import %(module)s
    except BaseException as exc:
        frames = traceback.extract_tb(sys.exc_info()[2])
        sys.stdout.write("{_MARKER}" + json.dumps({{
            "error": type(exc).__name__,
            "message": str(exc),
            "missing": getattr(exc, "name", None),
            "filename": getattr(exc, "filename", None),
            "origin": frames[-1].filename if frames else None,
        }}) + "\\n")
        sys.stdout.flush()
        sys.exit(17)
    """
)


def probe(python: str, module: str, cwd: str) -> dict | None:
    """Import one module in a subprocess. Returns None on success, else failure detail."""
    result = subprocess.run(
        [python, "-c", _PROBE_SNIPPET % {"module": checked_module(module)}],
        capture_output=True,
        text=True,
        cwd=cwd,
        shell=False,
    )
    if result.returncode == 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER) :])
    return {"error": "ProbeFailed", "message": f"exited {result.returncode}", "missing": None}


def classify(failure: dict) -> tuple[str, str]:
    """Map a failure onto the packaging defect it represents.

    The distinction is the useful part of this gate. A missing `models.*` module means
    the prune went too far; a missing third-party module means the wheel under-declares
    its dependencies; a FileNotFoundError means a data file never made it into the
    wheel at all. Those are three different fixes.
    """
    error, missing = failure.get("error"), failure.get("missing")
    message = failure.get("message") or ""

    if error == "ModuleNotFoundError" and missing:
        root = missing.split(".", 1)[0]
        if root in CONSUMER_SUPPLIED:
            return "SKIP", f"needs `{missing}`, which the consumer supplies (not a dependency of this package)"
        if root == "models":
            return "FAIL", f"missing module `{missing}` -- the tree was pruned too far"
        if root in ("tests", "tools"):
            return "FAIL", f"imports `{missing}` from the repository, which is not part of this distribution"
        return "FAIL", f"missing dependency `{missing}` -- the wheel does not declare it"

    if error == "FileNotFoundError":
        return "FAIL", f"missing data file {failure.get('filename') or message} -- not included in the wheel"

    return "FAIL", f"{error}: {message}"


def check_ttnn_present(python: str) -> None:
    """Fail loudly if `ttnn` is missing, rather than blaming every entry point on it.

    `ttnn` is a declared dependency, but it is routinely installed separately (the
    matching build is not on any index until it is published), so it is easy to end up
    with an environment that has everything *except* it. Without this check, every model
    import fails with `No module named 'ttnn'` and gets classified as an undeclared
    dependency -- a whole matrix of misleading failures with one real cause.
    """
    failure = probe(python, "ttnn", cwd=tempfile.gettempdir())
    if failure is None:
        return
    raise SystemExit(
        "error: `ttnn` is not importable in the target environment, so no model can be\n"
        "imported and the matrix would report every entry point as broken.\n"
        f"  {failure.get('error')}: {failure.get('message')}\n"
        "Install a ttnn matching this build, or the nearest published release, and re-run."
    )


def check_environment_is_clean(python: str) -> list[str]:
    """Warn about packages whose presence would make the matrix over-report success."""
    warnings = []
    for package in ("pytest",):
        if probe(python, package, cwd=tempfile.gettempdir()) is None:
            warnings.append(
                f"`{package}` is installed in the target environment. It is not a declared "
                f"dependency of tt-metal-models, so any entry point that needs it will "
                f"pass here and fail for users. Re-run in a clean environment."
            )
    return warnings


def make_venv(wheel: Path, workdir: Path) -> str:
    """Create a venv containing only the wheel and its declared dependencies."""
    wheel = wheel.resolve(strict=True)
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise SystemExit(f"error: {wheel} is not a wheel file.")
    venv = workdir / "venv"
    print(f"Creating a clean virtual environment at {venv}", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, shell=False)
    python = checked_interpreter(str(venv / "bin" / "python"))
    subprocess.run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True, shell=False)
    print(f"Installing {wheel.name} and its declared dependencies", flush=True)
    subprocess.run([python, "-m", "pip", "install", "--quiet", str(wheel)], check=True, shell=False)
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--wheel",
        type=Path,
        help=(
            "Build a clean venv and install this wheel into it. Omit to test the current environment. "
            "Note that this resolves the wheel's exact `ttnn==<version>` pin, so it only works once the "
            "ttnn wheel from the same commit has been published; before that, install the wheel yourself "
            "with --no-deps and point --python at that environment."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter whose environment to test. Ignored when --wheel is given.",
    )
    parser.add_argument(
        "--strict-submodule",
        action="store_true",
        help="Count the submodule-dependent entry points as failures even when the submodule is absent.",
    )
    parser.add_argument(
        "--strict-known-broken",
        action="store_true",
        help="Count the KNOWN_BROKEN entry points as failures. Use to check whether upstream has fixed them.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Count consumer-supplied packages as required. Use when the Tenstorrent vLLM fork is "
            "installed, to cover the entry points that are otherwise skipped."
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tt-metal-models-matrix-") as tmp:
        workdir = Path(tmp)
        python = make_venv(args.wheel, workdir) if args.wheel else checked_interpreter(args.python)

        # A neutral cwd: importing from the repository root would resolve `models` to the
        # checkout instead of the installed package, and the matrix would prove nothing.
        neutral_cwd = str(workdir)

        models_root = find_models_root(python, cwd=neutral_cwd)
        print(f"Testing `models` at {models_root}\n")

        check_ttnn_present(python)

        for warning in check_environment_is_clean(python):
            print(f"warning: {warning}\n", file=sys.stderr)

        have_submodule = (models_root.parent / SUBMODULE_MARKER).is_dir() and any(
            (models_root.parent / SUBMODULE_MARKER).iterdir()
        )

        modules = discover_entry_points(models_root) + list(EXTRA_SENTINELS)

        passed, skipped, failures, expected_failures, known_broken = [], [], [], [], []
        width = max(len(m) for m in modules)
        for module in modules:
            failure = probe(python, module, cwd=neutral_cwd)
            if failure is None:
                status, detail = "PASS", ""
                passed.append(module)
            else:
                status, detail = classify(failure)
                known = KNOWN_BROKEN.get(module)
                if status == "SKIP" and args.strict:
                    status = "FAIL"
                if status == "SKIP":
                    skipped.append(module)
                elif known and failure.get("missing") == known[0] and not args.strict_known_broken:
                    status = "KNOWN"
                    detail = known[1]
                    known_broken.append(module)
                elif module in SUBMODULE_DEPENDENT and not have_submodule and not args.strict_submodule:
                    status = "XFAIL"
                    expected_failures.append(module)
                else:
                    failures.append((module, detail))
            print(f"  {status:<6} {module:<{width}}  {detail}".rstrip(), flush=True)

        print(
            f"\n{len(passed)} passed, {len(skipped)} skipped, {len(known_broken)} known broken, "
            f"{len(expected_failures)} expected failures, {len(failures)} failed "
            f"(of {len(modules)})."
        )

        if known_broken:
            print(
                f"\n{len(known_broken)} entry point(s) are broken upstream, independently of "
                f"packaging. These are tracked in KNOWN_BROKEN in this script and should be "
                f"removed from it as they are fixed."
            )

        if skipped:
            print(
                f"\n{len(skipped)} entry point(s) were skipped because a consumer-supplied package "
                f"({', '.join(sorted(CONSUMER_SUPPLIED))}) is not installed. To cover them, install the "
                f"Tenstorrent vLLM fork into the same environment and re-run with --strict."
            )

        if expected_failures:
            print(
                f"\n{len(expected_failures)} expected failure(s): the llama reference submodule was "
                f"not populated when this wheel was built.\n"
                f"  Release builds must populate it: git submodule update --init {SUBMODULE_MARKER}"
            )

        if failures:
            print(f"\n{len(failures)} entry point(s) failed to import:", file=sys.stderr)
            for module, detail in failures:
                print(f"  {module}\n      {detail}", file=sys.stderr)
            return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
