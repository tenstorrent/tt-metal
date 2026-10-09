# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run the tool revision's pytest harness against one side's device code.

The Python comes from the tool checkout; ``LLK_HOME`` points the harness at the
side's tree, so every C++ source, header and linker script it compiles is that
side's. ``RUNNER_TEMP`` gives each side a private artefact directory.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from overlay import LLK_RELPATH

#: The tool lives in ``.github/llk_sfpu_report/`` of the tool checkout; the harness it
#: drives is that checkout's tt-llk.
TOOL_ROOT = Path(__file__).resolve().parents[2]
TOOL_LLK = TOOL_ROOT / LLK_RELPATH

#: Run on ttsim ($TT_METAL_SIMULATOR) instead of a device: tool development only.
SIMULATOR = False
PYTHON_TESTS = TOOL_LLK / "tests" / "python_tests"


@dataclass
class Side:
    name: str  # "base" | "head"
    sha: str
    tree: Path  # worktree root
    build: Path  # RUNNER_TEMP for this side

    @property
    def llk(self):
        return self.tree / LLK_RELPATH

    @property
    def artefacts(self):
        return self.build / "tt-llk-build"


def at_slot(side, slot_root):
    """``side`` seen through fixed symlinks: ``slot_root/tree`` and ``slot_root/build``.

    Profiler zone ids hash ``__FILE__`` (helpers/include/profiler.h), so the same
    code compiled at two paths differs in its ``.text``. Compiling both sides
    through one path -- one side at a time -- makes identical code hash alike.
    """
    slot_root = Path(slot_root)
    slot_root.mkdir(parents=True, exist_ok=True)
    side.build.mkdir(parents=True, exist_ok=True)
    for name, target in (("tree", side.tree), ("build", side.build)):
        link = slot_root / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(target).resolve())
    return Side(side.name, side.sha, slot_root / "tree", slot_root / "build")


def side_env(side, arch, extra=None):
    env = dict(os.environ)
    env.update(
        LLK_HOME=str(side.llk),
        RUNNER_TEMP=str(side.build),
        CHIP_ARCH=arch,
        TT_LLK_DISABLE_ASSERTS="1",
    )
    if not SIMULATOR:  # ttsim does not model SFPLOADMACRO; keep the caller's setting
        env.pop("TT_METAL_DISABLE_SFPLOADMACRO", None)
    env.update(extra or {})
    return env


def pytest(side, arch, args, *, env=None, log=None, check=True):
    """One pytest invocation from the tool's python_tests dir."""
    side.build.mkdir(parents=True, exist_ok=True)
    if SIMULATOR and "--compile-producer" not in args:
        args = ["--run-simulator", *args]
    cmd = [sys.executable, "-m", "pytest", "-q", "--override-ini=log_cli=false", *args]
    with open(log, "a") if log else open(os.devnull, "w") as out:
        out.write(f"\n$ ({side.name}) {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.run(
            cmd,
            cwd=PYTHON_TESTS,
            env=side_env(side, arch, env),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if check and proc.returncode not in (0, 5):  # 5: nothing collected
        raise RuntimeError(f"pytest failed on the {side.name} side (exit {proc.returncode}); see {log}")
    return proc.returncode


def produce_consume(
    side,
    arch,
    test_args,
    *,
    env=None,
    log=None,
    producer_jobs=8,
    consumer_jobs=1,
    check=True,
):
    """The harness's two-phase flow: compile everything, then run on the device."""
    pytest(
        side,
        arch,
        ["--compile-producer", "-n", str(producer_jobs), *test_args],
        env=env,
        log=log,
        check=check,
    )
    return pytest(
        side,
        arch,
        ["--compile-consumer", "-n", str(consumer_jobs), *test_args],
        env=env,
        log=log,
        check=check,
    )
