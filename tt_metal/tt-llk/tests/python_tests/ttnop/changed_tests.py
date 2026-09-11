#!/usr/bin/env python3
"""Map changed LLK files to the pytest files that exercise them."""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
TEST_GLOB = "test_*.py"


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
    return [path for path in root.rglob(TEST_GLOB) if "quasar" not in path.parts]


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


def select(repo: Path, base: str, arch: str, changed_paths: list[str]) -> dict:
    pytest_files = _pytest_files(repo)
    reasons: dict[str, set[str]] = defaultdict(set)

    for changed in changed_paths:
        if "/quasar/" in changed:
            continue
        changed_path = repo / changed
        if (
            changed.startswith("tt_metal/tt-llk/tests/python_tests/")
            and changed_path.match(TEST_GLOB)
            and changed_path.is_file()
        ):
            reasons[changed].add(f"{changed} changed")
            continue

        sources = set()
        if (
            changed.startswith("tt_metal/tt-llk/tests/sources/")
            and changed_path.suffix == ".cpp"
        ):
            if changed_path.is_file():
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--arch", choices=("wormhole", "blackhole"), required=True)
    parser.add_argument("--paths", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    changed = _changed_paths(repo, args.base)
    result = select(repo, args.base, args.arch, changed)
    if args.paths:
        print("\n".join(result["tests"]))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
