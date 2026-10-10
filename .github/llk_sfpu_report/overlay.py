# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build the two trees an SFPU report measures: base and head.

Both sides are sparse worktrees of the *tool* revision (the trusted checkout this
file runs from, main in CI). The base side is that revision unchanged. The head
side is the same revision with the PR's diff to device-side C++ applied on top
(``git apply --3way``), i.e. the kernel change as it would land if merged now.

Everything the host executes -- the Python harness, the pytest plugins, this tool
-- comes from the tool revision on both sides, so a PR can change what runs on
the Tensix, but never what runs on the runner. Both sides share one harness, so
the only difference between the two measurements is the PR's kernel change.

Copying whole files from the merge-base instead does not work: the harness C++
moves on (a kernel removed on main is still called by an old sfpu_operations.h).
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LLK_RELPATH = "tt_metal/tt-llk"

#: What a kernel build reads from outside tt-llk (see TestConfig.INCLUDES), plus
#: tt-llk itself. Same set as perf_compare_commits.sh.
SPARSE_PATHS = (
    LLK_RELPATH,
    "tt_metal/hw",
    "tt_metal/hostdevcommon",
    "ttnn/cpp/ttnn/operations/experimental",
)

#: A changed file is device code -- and taken from the PR -- only under these
#: prefixes. The harness C++ (tests/helpers, tests/sources) is included: a PR that
#: changes a kernel signature has to update its call site there too.
DEVICE_PREFIXES = (
    f"{LLK_RELPATH}/tt_llk_",
    f"{LLK_RELPATH}/common/",
    f"{LLK_RELPATH}/tests/helpers/",
    f"{LLK_RELPATH}/tests/sources/",
    "tt_metal/hw/",
    "tt_metal/hostdevcommon/",
    "ttnn/cpp/ttnn/operations/experimental/",
)
DEVICE_SUFFIXES = (".h", ".hpp", ".c", ".cc", ".cpp", ".S", ".ld", ".inc")

#: Changed files that affect the measurement but that the tool does not take from
#: the PR. They are reported, so a reviewer knows the numbers do not include them.
REPORTED_NOT_APPLIED = (
    f"{LLK_RELPATH}/tests/python_tests/",
    f"{LLK_RELPATH}/tests/sfpi-version",
)


def git(repo, *args, check=True):
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def is_device_file(path):
    return path.startswith(DEVICE_PREFIXES) and path.endswith(DEVICE_SUFFIXES)


@dataclass
class Plan:
    """What the two sides are made of."""

    repo: Path
    tool_sha: str
    base_sha: str
    head_sha: str
    changed: list = field(default_factory=list)
    applied: list = field(default_factory=list)
    not_applied: list = field(default_factory=list)


def make_plan(repo, head, base=None, main_ref="origin/main"):
    repo = Path(repo).resolve()
    head_sha = git(repo, "rev-parse", f"{head}^{{commit}}")
    base_sha = git(repo, "rev-parse", f"{base}^{{commit}}") if base else git(repo, "merge-base", main_ref, head_sha)
    changed = git(repo, "diff", "--name-only", base_sha, head_sha).splitlines()
    return Plan(
        repo=repo,
        tool_sha=git(repo, "rev-parse", "HEAD"),
        base_sha=base_sha,
        head_sha=head_sha,
        changed=changed,
        applied=[p for p in changed if is_device_file(p)],
        not_applied=[p for p in changed if p.startswith(REPORTED_NOT_APPLIED)],
    )


def device_diff(plan):
    """The PR's change to device files, as a patch against the merge-base."""
    if not plan.applied:
        return b""
    return subprocess.run(
        [
            "git",
            "-C",
            str(plan.repo),
            "diff",
            "--binary",
            plan.base_sha,
            plan.head_sha,
            "--",
            *plan.applied,
        ],
        check=True,
        capture_output=True,
    ).stdout


class PatchConflict(RuntimeError):
    """The PR's kernel diff does not apply to the tool revision."""


def build_side(plan, dest, patch=b"", at=None):
    """A sparse worktree of ``at`` (default: the tool revision), ``patch`` applied."""
    at = at or plan.tool_sha
    dest = Path(dest)
    if dest.exists():
        git(plan.repo, "worktree", "remove", "--force", str(dest), check=False)
        shutil.rmtree(dest, ignore_errors=True)
    git(plan.repo, "worktree", "prune")
    git(plan.repo, "worktree", "add", "--no-checkout", "--detach", str(dest), at)
    git(dest, "sparse-checkout", "set", "--cone", *SPARSE_PATHS)
    git(dest, "checkout", "--detach", at)

    if patch:
        # --3way falls back to a merge when main has moved under the PR's hunks;
        # it needs the PR's blobs, which the fetch of the PR ref brought in.
        proc = subprocess.run(
            ["git", "-C", str(dest), "apply", "--3way", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise PatchConflict(proc.stderr.decode(errors="replace").strip())

    # tests/sfpi is downloaded, not tracked: share the tool checkout's toolchain,
    # so both sides compile with the same compiler.
    sfpi = plan.repo / LLK_RELPATH / "tests" / "sfpi"
    if not sfpi.is_dir():
        raise SystemExit(f"no sfpi toolchain at {sfpi}; run tests/setup_testing_env.sh")
    link = dest / LLK_RELPATH / "tests" / "sfpi"
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)
    os.symlink(sfpi, link)
    return dest


def remove_side(plan, dest):
    git(plan.repo, "worktree", "remove", "--force", str(dest), check=False)
    shutil.rmtree(dest, ignore_errors=True)
