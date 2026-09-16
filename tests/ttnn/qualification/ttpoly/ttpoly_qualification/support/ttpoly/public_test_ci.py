# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Paired upstream-base/complete-candidate CI for canonical accuracy tests.

This builds fresh roots; it does not infer release performance or CI success.
The upstream runner owns its allocated hardware exclusively for this job.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import re
import subprocess

from ttpoly.ttmetal_public_tests import MANIFEST_PATH, SUPPORT_ROOT, TEST_ROOT, resolve_test_entry

WORKFLOW_PATH = ".github/workflows/ttpoly-generated-qualification.yaml"


def add_ci_files(root: Path, package_files: dict[str, bytes]) -> dict[str, bytes]:
    """Add executable PR CI wiring; a build revision is an input, not evidence."""
    files = dict(package_files)
    manifest = json.loads(files[MANIFEST_PATH])
    if manifest["architectures"] != ["blackhole", "wormhole_b0"]:
        raise ValueError("paired upstream CI requires explicit BH and WH package targets")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("common base must be a full resolved revision")
    runner = SUPPORT_ROOT + "/ttpoly/public_test_ci.py"
    files[runner] = Path(__file__).read_bytes()
    files[WORKFLOW_PATH] = _workflow(sorted(set(files) | {WORKFLOW_PATH})).encode()
    manifest["ci"] = {
        "workflow": WORKFLOW_PATH,
        "common_base_revision": revision,
        "overlay_paths": sorted(files),
        "scope": "fresh_paired_exhaustive_accuracy_only",
        "release_runtime_gate": "separate_same_build_256_tile_20_replay_qualification_required",
    }
    files[MANIFEST_PATH] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    return files


def validate_ci_files(files: dict) -> None:
    """Check generated wiring, not whether CI or release qualification passed."""
    manifest = json.loads(files[MANIFEST_PATH])
    ci = manifest.get("ci", {})
    if (
        manifest.get("architectures") != ["blackhole", "wormhole_b0"]
        or ci.get("workflow") != WORKFLOW_PATH
        or ci.get("scope") != "fresh_paired_exhaustive_accuracy_only"
        or ci.get("release_runtime_gate") != "separate_same_build_256_tile_20_replay_qualification_required"
        or ci.get("overlay_paths") != sorted(files)
        or not isinstance(ci.get("common_base_revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", ci["common_base_revision"])
    ):
        raise ValueError("incomplete paired upstream CI wiring")
    expected = {
        WORKFLOW_PATH: _workflow(ci["overlay_paths"]).encode(),
        SUPPORT_ROOT + "/ttpoly/public_test_ci.py": Path(__file__).read_bytes(),
    }
    for path, content in expected.items():
        actual = files.get(path)
        if isinstance(actual, str):
            actual = actual.encode()
        if actual != content:
            raise ValueError("paired CI runner differs from generated implementation: " + path)


def _workflow(overlay_paths):
    return (
        """name: Generated paired exhaustive accuracy
on:
  pull_request:
    branches: [main]
    paths:
"""
        + "".join("      - " + json.dumps(path) + "\n" for path in overlay_paths)
        + """
  workflow_dispatch:
    inputs:
      base_revision:
        description: Full SHA of the comparison base, ancestor of the selected ref
        required: true
        type: string
permissions:
  contents: read
  packages: write
jobs:
  environment:
    uses: ./.github/workflows/build-docker-artifact.yaml
    with:
      platform: Ubuntu 22.04
      build-manylinux: false
  paired-accuracy:
    needs: environment
    permissions:
      contents: read
    strategy:
      fail-fast: false
      matrix:
        include:
          - architecture: blackhole
            runner: [cloud-virtual-machine, P100, in-service]
          - architecture: wormhole_b0
            runner: [tt-ubuntu-2204-N300-viommu-stable]
    runs-on: ${{ matrix.runner }}
    timeout-minutes: 240
    container:
      image: ${{ needs.environment.outputs.dev-tag }}
      options: --device /dev/tenstorrent -e TT_GH_CI_INFRA
      volumes:
        - /dev/hugepages-1G:/dev/hugepages-1G
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          submodules: recursive
      - name: Build matched pair and run canonical exhaustive tests
        env:
          TARGET_ARCHITECTURE: ${{ matrix.architecture }}
          BASE_REVISION: ${{ github.event.pull_request.base.sha || inputs.base_revision }}
        shell: bash
        run: |
          export PYTHONPATH="$PWD/"""
        + SUPPORT_ROOT
        + """"
          python3 -m ttpoly.public_test_ci --checkout "$PWD" \\
            --work "$RUNNER_TEMP/ttpoly-paired-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" \\
            --architecture "$TARGET_ARCHITECTURE" --visible-devices 0 \\
            --base-revision "$BASE_REVISION"
      - name: Retain raw accuracy and test reports
        if: always()
        uses: actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f
        with:
          name: generated-accuracy-${{ matrix.architecture }}
          path: |
            ${{ runner.temp }}/ttpoly-paired-${{ github.run_id }}-${{ github.run_attempt }}/captures
            ${{ runner.temp }}/ttpoly-paired-${{ github.run_id }}-${{ github.run_attempt }}/pytest
            ${{ runner.temp }}/ttpoly-paired-${{ github.run_id }}-${{ github.run_attempt }}/accuracy*.xml
"""
    )


def _run(command, *, cwd, env):
    print("CI:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=7200)


def _path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe CI overlay path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("CI overlay must name an installed file inside checkout")
    return path


def verify_pr_tree(checkout: Path, revision: str, overlay: dict[str, Path]) -> None:
    """Validate the complete committed candidate tree and installed package files."""
    git = ["git", "-C", str(checkout)]
    ancestor = subprocess.run([*git, "merge-base", "--is-ancestor", revision, "HEAD"], capture_output=True)
    if ancestor.returncode:
        raise ValueError("common base is not an ancestor of the actual PR checkout")
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("paired CI requires a clean committed PR checkout")
    # Installed package inputs must remain ordinary committed files.
    tree = subprocess.check_output([*git, "ls-tree", "-z", "HEAD", "--", *sorted(overlay)])
    entries = {}
    for record in tree.split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            entries[path.decode()] = metadata.split()[0]
    if entries != {path: b"100644" for path in overlay}:
        raise ValueError("CI overlay requires committed ordinary non-executable files")
    for relative, path in overlay.items():
        if path.read_bytes() != subprocess.check_output([*git, "show", "HEAD:" + relative]):
            raise ValueError("installed overlay differs from committed PR tree: " + relative)


@contextmanager
def _device_lock(chip):
    import fcntl

    path = f"/tmp/ttpoly-ttmetal-package-device{chip}.lock"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def _build_reader(root):
    import fcntl

    directory = Path(
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "--absolute-git-dir"], text=True).strip()
    )
    fd = os.open(directory / "ttpoly-package-build.lock", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def run_ci(checkout: Path, work: Path, architecture: str, visible_devices: str, base_revision: str, chip: int = 0):
    checkout = checkout.resolve(strict=True)
    manifest = json.loads(_path(checkout, MANIFEST_PATH).read_text())
    if architecture not in manifest["architectures"] or not visible_devices.strip() or chip < 0:
        raise ValueError("explicit packaged target and device mapping required")
    ci = manifest["ci"]
    # The manifest records generation context, not a permanently pinned CI base.
    # PR events supply the actual base SHA; dispatch callers must choose explicitly.
    revision = base_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("invalid common-base build input")
    overlay = {relative: _path(checkout, relative) for relative in ci["overlay_paths"]}
    if MANIFEST_PATH not in overlay or SUPPORT_ROOT + "/ttpoly/public_test_ci.py" not in overlay:
        raise ValueError("incomplete CI overlay")
    operations = [
        operation
        for operation, entry in manifest["operations"].items()
        if "targets" not in entry or architecture in entry["targets"]
    ]
    if not operations:
        raise ValueError("no operations for requested architecture")
    for operation in operations:
        resolve_test_entry(manifest, operation, architecture)
    verify_pr_tree(checkout, revision, overlay)
    candidate_revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_revision):
        raise ValueError("invalid committed candidate revision")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for name in (
        "TT_METAL_HOME",
        "TT_METAL_RUNTIME_ROOT",
        "TT_METAL_DEVICE_HOME",
        "TT_METAL_COMPILE_HOME",
        "TT_METAL_INSTALL_ROOT",
        "TT_METAL_BUILD_DIR",
        "TT_METAL_JIT_SERVER_CACHE_ROOT",
        "PYTHON_ENV_DIR",
        "PYTHONPATH",
    ):
        env.pop(name, None)
    roots = {arm: work / arm for arm in ("stock", "candidate")}
    for arm, root in roots.items():
        selected_revision = revision if arm == "stock" else candidate_revision
        _run(["git", "worktree", "add", "--detach", root, selected_revision], cwd=checkout, env=env)
        _run(["git", "submodule", "update", "--init", "--recursive"], cwd=root, env=env)
        build_env = dict(env, TT_METAL_HOME=str(root), TT_METAL_CACHE=str(work / (arm + "-build-cache")))
        _run(["bash", "build_metal.sh", "--release"], cwd=root, env=build_env)
        _run(["bash", "create_venv.sh", "--env-dir", root / "python_env"], cwd=root, env=build_env)
        _run(
            ["uv", "pip", "install", "--python", root / "python_env/bin/python", "mpmath==1.3.0"],
            cwd=root,
            env=build_env,
        )
    candidate, stock = roots["candidate"], roots["stock"]
    support = candidate / SUPPORT_ROOT
    env.update(
        TTPOLY_TEST_ARCH=architecture,
        TTPOLY_TEST_CHIP=str(chip),
        TT_VISIBLE_DEVICES=visible_devices,
        TTPOLY_TEST_STOCK_CAPTURE_ROOT=str(work / "captures"),
        TT_METAL_HOME=str(candidate),
        TT_METAL_RUNTIME_ROOT=str(candidate),
        TT_METAL_BUILD_DIR=str(candidate / "build_Release"),
        PYTHONPATH=f"{candidate}:{candidate / 'ttnn'}:{support}",
    )
    for arm, root in roots.items():
        env["TTPOLY_TEST_" + arm.upper() + "_PYTHON"] = str(root / "python_env/bin/python")
        env["TTPOLY_TEST_" + arm.upper() + "_BUILD_DIR"] = str(root / "build_Release")
    with ExitStack() as locks:
        for root in roots.values():
            locks.enter_context(_build_reader(root))
        locks.enter_context(_device_lock(chip))
        for operation in operations:
            _run(
                [
                    candidate / "python_env/bin/python",
                    "-m",
                    "ttpoly.public_test_baseline",
                    "--package-root",
                    candidate,
                    "--stock-root",
                    stock,
                    "--operation",
                    operation,
                    "--output-root",
                    work / "captures",
                    "--architecture",
                    architecture,
                    "--visible-devices",
                    visible_devices,
                    "--chip",
                    str(chip),
                    "--caller-holds-locks",
                    "--prepared-builds-verified",
                    "--upstream-base-revision",
                    revision,
                ],
                cwd=candidate,
                env=env,
            )
        tests = [candidate / resolve_test_entry(manifest, op, architecture)["test"] for op in operations]
        _run(
            [
                candidate / "python_env/bin/python",
                "-m",
                "pytest",
                "-q",
                *tests,
                "--basetemp",
                work / "pytest",
                "--junitxml",
                work / "accuracy.xml",
            ],
            cwd=candidate,
            env=env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--architecture", choices=("blackhole", "wormhole_b0"), required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--chip", type=int, default=0)
    run_ci(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
