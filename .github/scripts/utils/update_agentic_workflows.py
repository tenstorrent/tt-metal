#!/usr/bin/env python3

"""Pre-commit hook: recompile gh-aw workflow lock files from their source markdown.

Takes one or more paths under .github/workflows/ (as passed by pre-commit for files
matching `^\\.github/workflows/[^/]+\\.(md|lock\\.yml)$`), maps each to its workflow name
(the shared basename of <name>.md and <name>.lock.yml), and runs `gh aw compile <name>`
for every unique workflow touched.

Agentic workflows can each be pinned to their own gh-aw compiler version — one
workflow doesn't have to track another's. The single source of truth for which
version a given workflow uses is .github/aw/workflow-versions.json (one entry per
workflow name, maintained by hand). That map deliberately lives in its own file
rather than inside actions-lock.json: gh-aw's own `compile` rewrites
actions-lock.json's `entries` block — and drops any key it doesn't recognize — as a
side effect whenever the compiling version differs from what's currently recorded
there, which would silently destroy a version map stored inside it. A workflow not
yet listed in workflow-versions.json (e.g. a brand new one, before someone
deliberately pins it) falls back to actions-lock.json's `entries` block's
github/gh-aw-actions/setup version.

The gh-aw CLI itself is never installed globally: for each workflow, this script
downloads that workflow's pinned release binary for the current OS/arch into a
temporary directory, uses it to compile, and deletes it immediately afterward — before
moving on to the next workflow, which may be pinned to a different version entirely.
Nothing persists on disk (in the repo or otherwise) beyond each individual compile, and
there's nothing to uninstall/downgrade on a contributor's machine.

This intentionally does not commit or push anything. Pre-commit's own tracked-file
modification detection is the enforcement mechanism for an existing lock file falling
out of sync; a newly created lock file (first compile of a new workflow) is called out
explicitly below since pre-commit can't detect changes to a file it doesn't know about
yet.

Runs the same way on a merge_group event as on a real pull_request — no special-casing.
GitHub's merge queue tests the exact commit it will merge (the merge-group branch's
HEAD *is* what lands on the target branch if checks pass), so if two queued PRs touch
the same workflow and the second one's checked-in lock file no longer matches what
compiling produces once combined with the first, that's a genuine problem: merging as
scheduled would land a stale lock file. Failing here is correct — it dequeues the PR so
its author can pull latest, let this hook recompile locally, and re-push.
"""

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import tempfile

WORKFLOWS_DIR = os.path.join(".github", "workflows")
ACTIONS_LOCK_PATH = os.path.join(".github", "aw", "actions-lock.json")
WORKFLOW_VERSIONS_PATH = os.path.join(".github", "aw", "workflow-versions.json")
GH_AW_REPO = "github/gh-aw"


def default_pinned_version():
    """Fallback version, used only when a workflow has no entry in workflow-versions.json."""
    with open(ACTIONS_LOCK_PATH) as f:
        entries = json.load(f)["entries"]
    for entry in entries.values():
        if entry["repo"] == "github/gh-aw-actions/setup":
            return entry["version"]
    print(f"::error::No github/gh-aw-actions/setup entry found in {ACTIONS_LOCK_PATH}", file=sys.stderr)
    sys.exit(1)


def resolve_pinned_version(name):
    """The single source of truth for a workflow's gh-aw version: workflow-versions.json,
    keyed by workflow name."""
    with open(WORKFLOW_VERSIONS_PATH) as f:
        data = json.load(f)
    return data.get(name) or default_pinned_version()


def platform_asset_name():
    system = platform.system().lower()
    machine = platform.machine().lower()

    system_map = {"linux": "linux", "darwin": "darwin", "windows": "windows", "freebsd": "freebsd"}
    machine_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "i386": "386",
        "i686": "386",
        "armv7l": "arm",
    }

    if system not in system_map or machine not in machine_map:
        print(f"::error::Unsupported platform for gh-aw: {platform.system()}/{platform.machine()}", file=sys.stderr)
        sys.exit(1)

    asset = f"{system_map[system]}-{machine_map[machine]}"
    if system_map[system] == "windows":
        asset += ".exe"
    return asset


@contextlib.contextmanager
def gh_aw_binary(version):
    """Download the pinned gh-aw binary into a temp dir; delete it on exit either way."""
    asset = platform_asset_name()
    with tempfile.TemporaryDirectory() as tmp:
        binary_path = os.path.join(tmp, "gh-aw")
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                version,
                "--repo",
                GH_AW_REPO,
                "--pattern",
                asset,
                "--output",
                binary_path,
                "--clobber",
            ],
            check=True,
        )
        os.chmod(binary_path, 0o755)
        yield binary_path


def workflow_name_for(path):
    rel = os.path.relpath(path, WORKFLOWS_DIR)
    if rel.endswith(".lock.yml"):
        return rel[: -len(".lock.yml")]
    if rel.endswith(".md"):
        return rel[: -len(".md")]
    return None


def is_tracked(path):
    result = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True)
    return result.returncode == 0


@contextlib.contextmanager
def preserved_actions_lock():
    """gh-aw's own `compile` rewrites actions-lock.json's `entries` block to match
    whichever version actually ran, as a side effect of every compile — so touching two
    workflows pinned to different versions in one invocation would otherwise leave that
    shared file bouncing to reflect whichever ran last, an incidental diff unrelated to
    either workflow's own change. Snapshot it before compiling and restore it after,
    so each workflow's compile only ever touches its own lock file."""
    with open(ACTIONS_LOCK_PATH, "rb") as f:
        before = f.read()
    try:
        yield
    finally:
        with open(ACTIONS_LOCK_PATH, "wb") as f:
            f.write(before)


def compile_workflow(name):
    md_path = os.path.join(WORKFLOWS_DIR, f"{name}.md")
    if not os.path.isfile(md_path):
        print(f"::error::{md_path} does not exist; cannot compile an orphan lock file for '{name}'.", file=sys.stderr)
        return False

    lock_path = os.path.join(WORKFLOWS_DIR, f"{name}.lock.yml")
    lock_existed_before = os.path.isfile(lock_path)

    version = resolve_pinned_version(name)
    with preserved_actions_lock(), gh_aw_binary(version) as binary_path:
        result = subprocess.run([binary_path, "compile", name])

    if result.returncode != 0:
        return False

    if not lock_existed_before or not is_tracked(lock_path):
        print(f"Created {lock_path} for the first time — `git add` it and commit again.", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help=".github/workflows/<name>.md and/or <name>.lock.yml paths")
    args = parser.parse_args()

    names = sorted({workflow_name_for(f) for f in args.files if workflow_name_for(f)})
    if not names:
        return

    ok = True
    for name in names:
        if not compile_workflow(name):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
