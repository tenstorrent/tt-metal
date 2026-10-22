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
import re
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


#: The architectures the report measures, in report order. Quasar is not one of them.
MEASURED_ARCHS = ("wormhole", "blackhole")

#: Directory names that put a device file in one architecture's tree: tt_llk_<arch>,
#: hw/ckernels/<arch>, the tt-1xx/<arch> firmware and headers, tt-2xx for Quasar.
_ARCH_DIRS = {
    "wormhole": {"wormhole", "wormhole_b0", "tt_llk_wormhole_b0"},
    "blackhole": {"blackhole", "tt_llk_blackhole"},
    "quasar": {"quasar", "tt_llk_quasar", "tt-2xx"},
}


def archs_built(paths):
    """The measured architectures whose device code these changed paths touch.

    A device file in one architecture's tree touches that architecture; any other
    device file is shared and touches both. Host files touch neither: they run from
    the tool revision, so the PR's never reach the measurement. Neither do Quasar's.
    """
    touched = set()
    for path in paths:
        if not is_device_file(path):
            continue
        dirs = set(path.split("/")[:-1])
        own = [arch for arch, names in _ARCH_DIRS.items() if dirs & names]
        touched.update(own or MEASURED_ARCHS)
    return [arch for arch in MEASURED_ARCHS if arch in touched]


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


def commit_sha(repo, ref):
    """The commit ``ref`` names, as a full SHA.

    Refs come from the command line; past this point git only sees SHAs, so no
    ref can be read as an option (``--output=...``).
    """
    sha = "" if ref.startswith("-") else git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise SystemExit(f"not a commit in {repo}: {ref!r}")
    return sha


def make_plan(repo, head, base=None, main_ref="origin/main"):
    repo = Path(repo).resolve()
    head_sha = commit_sha(repo, head)
    if base:
        base_sha = commit_sha(repo, base)
    else:
        base_sha = commit_sha(repo, git(repo, "merge-base", commit_sha(repo, main_ref), head_sha, check=False))
    # No rename detection: a moved header is its old path deleted and its new path added,
    # so the old copy cannot stay behind on the head side and shadow the moved one.
    changed = git(repo, "diff", "--no-renames", "--name-only", base_sha, head_sha).splitlines()
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
            "--no-renames",
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

    @property
    def files(self):
        """The files ``git apply --3way`` left conflicted (its ``U <path>`` lines)."""
        return [line[2:] for line in str(self).splitlines() if line.startswith("U ")]


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


if __name__ == "__main__":
    # The plan job's call: changed paths on stdin, the architectures to run on stdout.
    import sys

    if sys.argv[1:] != ["archs"]:
        raise SystemExit("usage: overlay.py archs < changed-paths")
    print(",".join(archs_built(line for line in sys.stdin.read().splitlines() if line)))
