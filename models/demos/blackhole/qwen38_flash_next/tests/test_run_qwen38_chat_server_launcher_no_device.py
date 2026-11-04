# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The public launcher without a device: ``--help`` and ``--validate-only`` behave the same from the model directory
and from an unrelated directory, the interpreter it starts runs from the repository root (so the model's own ``ttnn/``
package never shadows ttnn), relative path arguments are the caller's, and a tree without git history starts only
with its head declared."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.helpers.chat import skip_without_git_checkout

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
LAUNCHER = MODEL_DIR / "tools" / "run_qwen38_chat_server.sh"
# a stand-in for the venv interpreter: the extension probe (``-c``) answers with a file under the checkout, the server
# run reports its working directory and its arguments
STUB_PYTHON = f"""#!/usr/bin/env bash
if [[ "$1" == -c ]]; then printf '%s\\n' "{REPO_ROOT}/ttnn/ttnn/__init__.py"; exit 0; fi
printf 'cwd=%s\\n' "$PWD"
printf 'arg=%s\\n' "$@"
"""


def run_launcher(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "TT_METAL_HOME": str(REPO_ROOT)}  # the launcher refuses another checkout's TT_METAL_HOME
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )


@pytest.fixture
def stub_python(tmp_path) -> Path:
    """The stand-in interpreter (the launcher records the checkout's head: a git checkout only)."""

    skip_without_git_checkout(REPO_ROOT)
    stub = tmp_path / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)
    (tmp_path / "checkpoint").mkdir()
    return stub


def test_help_is_the_same_from_the_model_directory_and_from_elsewhere(tmp_path) -> None:
    runs = [run_launcher(cwd, "--help") for cwd in (MODEL_DIR, tmp_path)]
    for run in runs:
        assert run.returncode == 2 and run.stdout == ""
        assert run.stderr.startswith("Start the Qwen3.8-Flash-Next chat server")
        assert "--validate-only" in run.stderr and "set -euo pipefail" not in run.stderr
    assert runs[0].stderr == runs[1].stderr


def test_validate_only_runs_the_interpreter_from_the_repository_root_wherever_it_is_started(
    tmp_path, stub_python
) -> None:
    checkpoint, cache_root, elsewhere = tmp_path / "checkpoint", tmp_path / "cache", tmp_path / "elsewhere"
    elsewhere.mkdir()
    paths = {"--checkpoint": checkpoint, "--cache-root": cache_root, "--python": stub_python}
    reports = []
    for cwd in (MODEL_DIR, elsewhere):
        relative = [value for flag, path in paths.items() for value in (flag, os.path.relpath(path, cwd))]
        run = run_launcher(cwd, "--profile", "p150-line", *relative, "--validate-only")
        assert run.returncode == 0, run.stderr
        lines = run.stdout.splitlines()
        assert lines[0] == f"cwd={REPO_ROOT}", lines[0]
        args = [line[len("arg=") :] for line in lines[1:]]
        assert args[0] == str(MODEL_DIR / "tools" / "qwen38_chat_server.py") and args[-1] == "--validate-only"
        # path arguments are the caller's, resolved before the launcher changes directory; the run directory is
        # time-stamped and differs between the two runs
        assert Path(args[args.index("--checkpoint") + 1]).resolve() == checkpoint.resolve()
        assert Path(args[args.index("--component-cache-root") + 1]).resolve().is_relative_to(cache_root.resolve())
        reports.append(
            [
                "<run-dir>" if Path(value).resolve().is_relative_to(cache_root.resolve() / "runs") else value
                for value in (str(Path(value).resolve()) if value.startswith((".", "/")) else value for value in args)
            ]
        )
    assert reports[0] == reports[1]


def test_long_chunks_and_a_slab_combine_with_mtp(tmp_path, stub_python) -> None:
    """The MTP chain prefills in 128-row chunks and in slabs: both flag pairs reach the server. A slab above the
    2048-row cap is refused."""

    checkpoint, cache_root = tmp_path / "checkpoint", tmp_path / "cache"
    common = ("--profile", "p150-line", "--checkpoint", str(checkpoint), "--cache-root", str(cache_root))
    common += ("--python", str(stub_python))
    run = run_launcher(tmp_path, *common, "--long-chunks", "--mtp", "4", "--validate-only")
    assert run.returncode == 0, run.stderr
    args = [line[len("arg=") :] for line in run.stdout.splitlines()[1:]]
    assert "--long-chunks" in args and args[args.index("--mtp") + 1] == "4"
    slab = run_launcher(tmp_path, *common, "--prefill-slab", "256", "--mtp", "4", "--validate-only")
    assert slab.returncode == 0, slab.stderr
    args = [line[len("arg=") :] for line in slab.stdout.splitlines()[1:]]
    assert args[args.index("--prefill-slab") + 1] == "256" and args[args.index("--mtp") + 1] == "4"
    capped = run_launcher(tmp_path, *common, "--prefill-slab", "4096", "--validate-only")
    assert capped.returncode == 2 and "256..2048" in capped.stderr, (capped.returncode, capped.stderr)
    acceptance = run_launcher(tmp_path, *common, "--acceptance", "--acceptance-only", "--require-json-96")
    assert acceptance.returncode == 0, acceptance.stderr
    assert "arg=--acceptance-only" in acceptance.stdout and "arg=--acceptance-prompts" in acceptance.stdout
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "--long-chunks and --mtp are alternatives" not in launcher
    assert "--prefill-slab and --mtp are alternatives" not in launcher
    assert "accepted for compatibility: the 128-row chunks are the server's default (combines with --mtp)" in launcher


def test_a_tree_without_git_history_starts_only_with_its_declared_head(tmp_path) -> None:
    """A tree without git history (such as a CI image) takes its head from QWEN38_TT_METAL_SHA, as the server's
    runtime admission does; without a 40-hex value the launcher refuses before it probes the interpreter."""

    tree = tmp_path / "tree"
    launcher = tree / LAUNCHER.relative_to(REPO_ROOT)
    launcher.parent.mkdir(parents=True)
    launcher.write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    (tree / "ttnn" / "ttnn").mkdir(parents=True)
    (tree / "ttnn" / "ttnn" / "__init__.py").touch()
    stub = tmp_path / "python"
    stub.write_text(STUB_PYTHON.replace(str(REPO_ROOT), str(tree)))
    stub.chmod(0o755)
    (tmp_path / "checkpoint").mkdir()
    head = "0123456789abcdef0123456789abcdef01234567"
    runs = {}
    for declared in (None, head[:12], head):
        env = {name: value for name, value in os.environ.items() if name != "QWEN38_TT_METAL_SHA"}
        env["TT_METAL_HOME"] = str(tree)
        if declared is not None:
            env["QWEN38_TT_METAL_SHA"] = declared
        runs[declared] = subprocess.run(
            ["bash", str(launcher), "--profile", "p150-line", "--checkpoint", str(tmp_path / "checkpoint")]
            + ["--cache-root", str(tmp_path / "cache"), "--python", str(stub), "--validate-only"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    for declared in (None, head[:12]):
        run = runs[declared]
        assert run.returncode == 2 and "set QWEN38_TT_METAL_SHA" in run.stderr, (declared, run.returncode, run.stderr)
    run = runs[head]
    assert run.returncode == 0, run.stderr
    assert f"head {head}" in run.stderr and run.stdout.splitlines()[-1] == "arg=--validate-only", run.stdout


def test_the_launcher_changes_directory_before_the_interpreter_runs() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    change = launcher.index('cd "$REPO_ROOT"')
    assert launcher.index('[[ -d "$checkpoint" ]]') < change < launcher.index("import ttnn._ttnn")
    assert change < launcher.index('exec "$python" "$SERVER"')
    assert "Start it from any directory, the model directory included" in launcher
