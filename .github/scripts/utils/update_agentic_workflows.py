#!/usr/bin/env python3

"""Recompile a gh-aw workflow and report whether its lock file changed.

Runs `gh aw compile <workflow>` (the gh-aw CLI must already be installed and on PATH,
pinned to the version the repo's .lock.yml files were compiled with — see
.github/workflows/copilot-setup-steps.yml) against the current checkout, then checks
whether the resulting .github/workflows/<workflow>.lock.yml differs from what's
already committed.

On success, prints and (when running under GitHub Actions) writes to GITHUB_OUTPUT:
  lock-file-changed=true|false
  lock-file=.github/workflows/<workflow>.lock.yml

A failed `gh aw compile` (e.g. a validation error in the source markdown) propagates
as a non-zero exit from this script; gh-aw itself leaves the lock file untouched in
that case, so there is nothing to detect or commit.
"""

import argparse
import os
import subprocess
import sys

WORKFLOWS_DIR = os.path.join(".github", "workflows")


def lock_file_path(workflow_name):
    return os.path.join(WORKFLOWS_DIR, f"{workflow_name}.lock.yml")


def compile_workflow(workflow_name):
    subprocess.run(["gh", "aw", "compile", workflow_name], check=True)


def lock_file_changed(path):
    # `git diff --quiet` exits 1 when there are differences, 0 when there are none.
    result = subprocess.run(["git", "diff", "--quiet", "--", path])
    return result.returncode != 0


def write_output(name, value):
    print(f"{name}={value}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}={value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", required=True, help="gh-aw workflow name, e.g. 'test-command'")
    args = parser.parse_args()

    lock_path = lock_file_path(args.workflow)
    if not os.path.isfile(lock_path):
        print(f"::error::{lock_path} does not exist; nothing to compare against.", file=sys.stderr)
        sys.exit(1)

    compile_workflow(args.workflow)

    changed = lock_file_changed(lock_path)
    write_output("lock-file-changed", "true" if changed else "false")
    write_output("lock-file", lock_path)


if __name__ == "__main__":
    main()
