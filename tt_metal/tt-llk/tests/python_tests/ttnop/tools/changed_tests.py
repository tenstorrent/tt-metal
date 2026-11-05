#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Map changed LLK files to the pytest files that exercise them."""

import argparse
import io
import json
import logging
import os
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
TEST_GLOB = "test_*.py"
# Every TTNOP sweep runs on this fixed number of shards.
SHARDS = 15

logger = logging.getLogger(__name__)


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(
        args,
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _changed_paths(repo: Path, base: str) -> list[str]:
    output = _run(
        repo, "git", "diff", "--name-only", "--diff-filter=ACMRTD", base, "HEAD"
    )
    return [line for line in output.splitlines() if line]


def _kernel_roots(arch: str) -> tuple[str, ...]:
    if arch == "wormhole":
        return (
            "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/",
            "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/",
        )
    return (
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/",
    )


def _kernel_include_names(path: str, arch: str) -> set[str]:
    name = Path(path).name
    if path.startswith("tt_metal/hw/inc/api/compute/experimental/"):
        stem = Path(name).stem
        return {
            f"llk_unpack_{stem}.h",
            f"llk_math_{stem}.h",
            f"llk_pack_{stem}.h",
        }
    if (
        path.startswith(_kernel_roots(arch))
        and name.endswith(".h")
        and name.startswith(("llk_unpack_", "llk_math_", "llk_pack"))
        and "_common" not in name
        and name not in {"llk_unpack_A.h", "llk_pack.h"}
    ):
        return {name.replace("_api.h", ".h")}
    return set()


def _is_kernel_header(path: str, arch: str) -> bool:
    return bool(_kernel_include_names(path, arch))


def _includes(text: str) -> set[str]:
    return {Path(match).name for match in INCLUDE_RE.findall(text)}


def _kernel_sources(repo: Path, changed: str, arch: str) -> set[Path]:
    include_names = _kernel_include_names(changed, arch)

    sources = set()
    source_root = repo / "tt_metal/tt-llk/tests/sources"
    for source in source_root.rglob("*.cpp"):
        if "quasar" in source.parts:
            continue
        if _includes(source.read_text(errors="ignore")) & include_names:
            sources.add(source)
    return sources


def _pytest_files(repo: Path) -> list[Path]:
    root = repo / "tt_metal/tt-llk/tests/python_tests"
    # helpers/ files match TEST_GLOB but define no tests.
    return [
        path
        for path in root.rglob(TEST_GLOB)
        if "quasar" not in path.parts and "helpers" not in path.parts
    ]


def _source_tests(repo: Path, source: Path, pytest_files: list[Path]) -> set[str]:
    tests_root = repo / "tt_metal/tt-llk/tests"
    source_ref = source.relative_to(tests_root).as_posix()
    matches = {
        path.relative_to(repo).as_posix()
        for path in pytest_files
        if source_ref in path.read_text(errors="ignore")
    }
    if source_ref.startswith("sources/fused_tests/"):
        fused_test = tests_root / "python_tests/test_fused.py"
        matches.add(f"{fused_test.relative_to(repo)}::test_fuser[{source.stem}]")
    return matches


def select(repo: Path, arch: str, changed_paths: list[str]) -> dict:
    pytest_files = _pytest_files(repo)
    reasons: dict[str, set[str]] = defaultdict(set)

    for changed in changed_paths:
        if "/quasar/" in changed:
            continue
        changed_path = repo / changed
        if (
            changed.startswith("tt_metal/tt-llk/tests/python_tests/")
            and changed_path.match(TEST_GLOB)
            and "helpers" not in changed_path.parts
            and changed_path.is_file()
        ):
            reasons[changed].add(f"{changed} changed")
            continue

        sources = set()
        if (
            changed.startswith("tt_metal/tt-llk/tests/sources/")
            and changed_path.suffix == ".cpp"
        ):
            sources.add(changed_path)
        elif _is_kernel_header(changed, arch):
            sources = _kernel_sources(repo, changed, arch)

        for source in sources:
            for test in _source_tests(repo, source, pytest_files):
                reasons[test].add(f"{changed} is used by {source.relative_to(repo)}")

    tests = sorted(reasons)
    explanation = [
        f"Run {test} on {arch}: {reason}"
        for test in tests
        for reason in sorted(reasons[test])
    ]
    return {"tests": tests, "reasons": explanation}


def _github(url: str) -> bytes:
    request = urllib.request.Request(url)
    # Unredirected: the artifact download redirects to storage that rejects it.
    request.add_unredirected_header("Authorization", f"Bearer {os.environ['GH_TOKEN']}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _last_failed_tests(repo: Path, base: str, arch: str) -> tuple[str, list[str]]:
    """A failed last scheduled run gives its commit and its failed test files."""
    api = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions"
    runs = json.loads(
        _github(
            f"{api}/workflows/ttnop-injector.yaml/runs"
            "?event=schedule&status=completed&per_page=1"
        )
    )["workflow_runs"]
    if not runs or runs[0]["conclusion"] == "success":
        return base, []
    artifacts = json.loads(
        _github(f"{api}/runs/{runs[0]['id']}/artifacts?per_page=100")
    )["artifacts"]
    artifacts = [
        artifact
        for artifact in artifacts
        if artifact["name"].startswith(f"ttnop-junit-{arch}-")
    ]
    if len(artifacts) < SHARDS:
        # A shard without junit never finished, so keep sweeping from base.
        logger.warning(
            f"::warning::ttnop run {runs[0]['id']} has {len(artifacts)} of "
            f"{SHARDS} {arch} junit artifacts; keeping base {base}"
        )
        return base, []
    files = set()
    for artifact in artifacts:
        with zipfile.ZipFile(
            io.BytesIO(_github(f"{api}/artifacts/{int(artifact['id'])}/zip"))
        ) as archive:
            for name in archive.namelist():
                if not name.endswith("junit.xml"):
                    continue
                for case in ET.fromstring(archive.read(name)).iter("testcase"):
                    if case.find("failure") is None and case.find("error") is None:
                        continue
                    path = case.get("classname").replace(".", "/") + ".py"
                    files.add(f"tt_metal/tt-llk/tests/python_tests/{path}")
    return runs[0]["head_sha"], sorted(
        path for path in files if (repo / path).is_file()
    )


def _weekly(repo: Path, base: str, arch: str) -> tuple[str, list[str]]:
    """A failed week is retried only by its failed files."""
    try:
        return _last_failed_tests(repo, base, arch)
    except Exception as error:
        # Without the failed list, sweep everything since the given base.
        logger.warning(f"::warning::ttnop retry lookup failed: {error}")
        return base, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--arch", choices=("wormhole", "blackhole"), required=True)
    parser.add_argument("--paths", action="store_true")
    # Comma-separated retry files that select-tests already resolved.
    parser.add_argument("--retry", default="")
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s")

    repo = args.repo.resolve()
    base, retry = args.base, [path for path in args.retry.split(",") if path]
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GH_TOKEN"):
        base, retry = _weekly(repo, base, args.arch)
    changed = _changed_paths(repo, base)
    result = select(repo, args.arch, changed)
    result["tests"] = sorted({*result["tests"], *retry})
    result["reasons"] += [
        f"Run {path} on {args.arch}: failed in the last scheduled run" for path in retry
    ]
    result["base"], result["retry"] = base, retry
    if args.paths:
        print("\n".join(result["tests"]))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
