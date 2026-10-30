# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Migrate checkout consumers to the published cachew action and verify its pins.

Run with --write to migrate, --check to detect unmigrated consumers, or
--verify-pins to check that every pin points at code matching this tree (add
--require-main after merge to also require the pins to be reachable from
origin/main). With no paths, inspect all tracked action/workflow files.

Only checkouts of the exact version the wrapper forwards to (actions/checkout
v6.0.2, or a v6 tag) are migrated, so no consumer changes checkout version.
Excluded on purpose:
  * the wrapper's internal checkout (replacing it would recurse);
  * gh-aw generated lock files and their .md sources (generated, and they pin
    their own checkout version; they also run on GitHub-hosted runners, where
    cachew is a no-op);
  * the vendored tt-llk tree, whose workflows run in tenstorrent/tt-llk.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


ACTION_DIR = Path(".github/actions/cachew-checkout")
ACTION_REPO = "tenstorrent/tt-metal/.github/actions/cachew-checkout"
ACTION_SHA = "73f53fe5d7091229a6d0203ccd7c85fcfda52d8c"
ACTION = f"{ACTION_REPO}@{ACTION_SHA}"
UPSTREAM_WRAPPER = ACTION_DIR / "action.yml"
FORWARDED_CHECKOUT = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
MIGRATABLE_REFS = (FORWARDED_CHECKOUT, "v6", "v6.0.2")
EXCLUDED_PREFIXES = ("tt_metal/tt-llk/",)
CHECKOUT = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?:-[ \t]+)?uses:[ \t]*)(?P<quote>['\"]?)"
    r"(?:actions/checkout@(?:"
    + "|".join(map(re.escape, MIGRATABLE_REFS))
    + r")|"
    + re.escape(ACTION_REPO)
    + r"@[0-9a-f]{40})"
    r"(?P=quote)(?:[ \t]*#[^\n]*)?[ \t]*$"
)
ANY_PIN = re.compile(re.escape(ACTION_REPO) + r"(?P<subpath>/[\w./-]+)?@(?P<sha>[0-9a-f]{40})")


def repository_root():
    # Independent of the caller's working directory.
    here = Path(__file__).resolve().parent
    return Path(subprocess.check_output(["git", "-C", str(here), "rev-parse", "--show-toplevel"], text=True).strip())


def excluded(path):
    relative = path.as_posix()
    if relative.startswith(EXCLUDED_PREFIXES) or relative.endswith(".lock.yml"):
        return True
    return path.suffix == ".md"


def rewrite(text):
    # A trailing "cachew: keep" comment opts a checkout out (e.g. a bootstrap
    # checkout that must test the working tree's action).
    return CHECKOUT.sub(
        lambda match: match[0] if "cachew: keep" in match[0] else f"{match['prefix']}{ACTION} # cachew checkout", text
    )


def migrate_file(path, write=False, root=None):
    root = root or repository_root()
    path = Path(path)
    absolute = path if path.is_absolute() else root / path
    relative = absolute.resolve().relative_to(root.resolve())
    if relative == UPSTREAM_WRAPPER or excluded(relative):
        return False
    before = absolute.read_text()
    after = rewrite(before)
    if before == after:
        return False
    if write:
        absolute.write_text(after)
    return True


def tracked_consumers(root):
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for value in paths:
        path = Path(value)
        parts = path.parts
        if ".github" not in parts:
            continue
        github = parts.index(".github")
        if len(parts) <= github + 1 or parts[github + 1] not in ("actions", "workflows"):
            continue
        if path.suffix in (".yml", ".yaml"):
            yield path


def git_ok(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True).returncode == 0


def verify_pins(root, require_main):
    """The consumers' pin to this action is the only pin. Fail if any consumer
    pins something else, or if the pin is unknown, differs from this tree, or
    (after merge) is not on main."""
    problems = []
    pins = {}
    for path in tracked_consumers(root):
        for match in ANY_PIN.finditer((root / path).read_text()):
            if match["subpath"]:
                problems.append(f"{path} references {match[0]}; reference the action itself, not a subdirectory")
            pins.setdefault(match["sha"], []).append(path.as_posix())
    if set(pins) - {ACTION_SHA}:
        problems.append(f"consumers pin {sorted(pins)}, expected only ACTION_SHA {ACTION_SHA}")
    for sha, files in sorted(pins.items()):
        if not git_ok(root, "cat-file", "-e", f"{sha}^{{commit}}"):
            problems.append(f"pin {sha} is not a known commit (fetch it, or it was never pushed)")
            continue
        if not git_ok(root, "diff", "--quiet", sha, "--", str(ACTION_DIR)):
            problems.append(f"pin {sha} differs from this tree in {ACTION_DIR} (used by {files[0]})")
        if require_main and not git_ok(root, "merge-base", "--is-ancestor", sha, "origin/main"):
            problems.append(f"pin {sha} is not reachable from origin/main and may be garbage-collected")
    for problem in problems:
        print(problem)
    print("Pins verified." if not problems else f"{len(problems)} pin problem(s).")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--verify-pins", action="store_true")
    parser.add_argument("--require-main", action="store_true", help="with --verify-pins, require pins on origin/main")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = repository_root()
    if args.verify_pins:
        return verify_pins(root, args.require_main)
    changed = [path for path in (args.paths or tracked_consumers(root)) if migrate_file(path, args.write, root)]
    for path in changed:
        print(f"{'Migrated' if args.write else 'Needs cachew migration:'} {path}")
    print(f"{len(changed)} file(s) {'updated' if args.write else 'need migration'}.")
    return 1 if changed and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
