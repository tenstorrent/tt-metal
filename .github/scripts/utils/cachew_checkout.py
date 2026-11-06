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
  * jobs whose runs-on is a literal GitHub-hosted label (ubuntu-*, windows-*,
    macos-*): cachew is a no-op there and the remote action only adds an archive
    download, so --write turns their cachew pins back into actions/checkout;
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
# Consumers run a published copy of ACTION_DIR from the small tenstorrent/.github
# repository. A remote `uses:` makes the runner download the whole host repository
# for every job, so hosting it in tt-metal cost a full tt-metal tarball per job.
# TEST ONLY (MINFRA-2477): hosted on a personal fork of tenstorrent/tt-github-actions to
# prove the fix before the upstream PR lands. Switch to tenstorrent/tt-github-actions
# before merging.
HOST_REPO = "asattaurTT/tt-github-actions"
HOST_URL = f"https://github.com/{HOST_REPO}"
HOST_DIR = ".github/actions/cachew-checkout"
ACTION_REPO = f"{HOST_REPO}/{HOST_DIR}"
ACTION_SHA = "66fab509e2c11f6d518f82d16d3409041432877c"
ACTION = f"{ACTION_REPO}@{ACTION_SHA}"
# Earlier pins hosted the action in tt-metal itself; --write moves them to ACTION.
LEGACY_ACTION_REPOS = (
    "tenstorrent/tt-metal/.github/actions/cachew-checkout",
    "tenstorrent/.github/actions/cachew-checkout",
)
PIN_REPOS = (ACTION_REPO, *LEGACY_ACTION_REPOS)
UPSTREAM_WRAPPER = ACTION_DIR / "action.yml"
FORWARDED_CHECKOUT = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
MIGRATABLE_REFS = (FORWARDED_CHECKOUT, "v6", "v6.0.2")
EXCLUDED_PREFIXES = ("tt_metal/tt-llk/",)
CHECKOUT = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?:-[ \t]+)?uses:[ \t]*)(?P<quote>['\"]?)"
    r"(?:actions/checkout@(?:"
    + "|".join(map(re.escape, MIGRATABLE_REFS))
    + r")|(?:"
    + "|".join(map(re.escape, PIN_REPOS))
    + r")@[0-9a-f]{40})"
    r"(?P=quote)(?:[ \t]*#[^\n]*)?[ \t]*$"
)
ANY_PIN = re.compile(
    r"(?P<repo>" + "|".join(map(re.escape, PIN_REPOS)) + r")(?P<subpath>/[\w./-]+)?@(?P<sha>[0-9a-f]{40})"
)
PLAIN_CHECKOUT = f"actions/checkout@{FORWARDED_CHECKOUT} # v6.0.2"
HOSTED_LABEL = re.compile(r"(?:ubuntu|windows|macos)-[\w.-]+")
TOP_LEVEL = re.compile(r"^(?P<key>[\w-]+):")
JOB = re.compile(r"^  [\w-]+:[ \t]*(?:#.*)?$")
RUNS_ON = re.compile(r"^    runs-on:[ \t]*(?P<quote>['\"]?)(?P<label>[^'\"#\s]+)(?P=quote)[ \t]*(?:#.*)?$")
# Only the files cached checkouts execute; docs and tests can change without a re-pin.
RUNTIME_FILES = ("action.yml", "checkout.py", "git_wrapper.py", "session", "trusted_caches.json")


def repository_root():
    # Independent of the caller's working directory.
    here = Path(__file__).resolve().parent
    return Path(subprocess.check_output(["git", "-C", str(here), "rev-parse", "--show-toplevel"], text=True).strip())


def excluded(path):
    relative = path.as_posix()
    if relative.startswith(EXCLUDED_PREFIXES) or relative.endswith(".lock.yml"):
        return True
    return path.suffix == ".md"


def hosted_jobs(text):
    """Character ranges of the workflow jobs whose runs-on is a literal GitHub-hosted label."""
    spans, start, hosted, in_jobs, offset = [], None, False, False, 0

    def close(end):
        if start is not None and hosted:
            spans.append((start, end))

    for line in text.splitlines(keepends=True):
        top = TOP_LEVEL.match(line)
        if top:
            close(offset)
            start, hosted, in_jobs = None, False, top["key"] == "jobs"
        elif in_jobs and JOB.match(line):
            close(offset)
            start, hosted = offset, False
        elif start is not None:
            runs_on = RUNS_ON.match(line)
            if runs_on:
                hosted = bool(HOSTED_LABEL.fullmatch(runs_on["label"]))
        offset += len(line)
    close(offset)
    return spans


def rewrite(text):
    # A trailing "cachew: keep" comment opts a checkout out (e.g. a bootstrap
    # checkout that must test the working tree's action).
    hosted = hosted_jobs(text)

    def replace(match):
        if "cachew: keep" in match[0]:
            return match[0]
        if any(start <= match.start() < end for start, end in hosted):
            pinned = any(repo in match[0] for repo in PIN_REPOS)
            return f"{match['prefix']}{PLAIN_CHECKOUT}" if pinned else match[0]
        return f"{match['prefix']}{ACTION} # cachew checkout"

    return CHECKOUT.sub(replace, text)


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
        if value.startswith(EXCLUDED_PREFIXES):
            continue
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


def git_out(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)


def runtime_blobs(root, sha=None):
    """{runtime path relative to the action: blob id}, from the published action
    at sha, or from this tree's ACTION_DIR (working tree contents) without one."""
    if sha:
        listing = git_out(root, "ls-tree", "-r", sha, "--", f"{HOST_DIR}/").stdout.splitlines()
        blobs = {}
        for line in listing:
            meta, path = line.split("\t", 1)
            blobs[path[len(HOST_DIR) + 1 :]] = meta.split()[2]
    else:
        listing = git_out(root, "ls-files", "--", str(ACTION_DIR)).stdout.splitlines()
        paths = [path[len(ACTION_DIR.as_posix()) + 1 :] for path in listing]
        blobs = {path: git_out(root, "hash-object", str(ACTION_DIR / path)).stdout.strip() for path in paths}
    return {path: blob for path, blob in blobs.items() if path.split("/")[0] in RUNTIME_FILES}


def fetch_from_host(root, *refspecs):
    return git_ok(root, "fetch", "--quiet", "--no-tags", HOST_URL, *refspecs)


def verify_pins(root, require_main):
    """The consumers' pin to this action is the only pin. Fail if any consumer
    pins something else (including the legacy tt-metal-hosted path), or if the
    pin is unknown in HOST_REPO, its runtime files differ from this tree's
    ACTION_DIR, or (with require_main) it is not on HOST_REPO's main."""
    problems = []
    pins = {}
    for path in tracked_consumers(root):
        for match in ANY_PIN.finditer((root / path).read_text()):
            if match["subpath"]:
                problems.append(f"{path} references {match[0]}; reference the action itself, not a subdirectory")
            if match["repo"] != ACTION_REPO:
                problems.append(f"{path} pins legacy {match['repo']}; run --write to move it to {ACTION_REPO}")
            pins.setdefault(match["sha"], []).append(path.as_posix())
    if set(pins) - {ACTION_SHA}:
        problems.append(f"consumers pin {sorted(pins)}, expected only ACTION_SHA {ACTION_SHA}")
    if require_main and pins and not fetch_from_host(root, "+refs/heads/main:refs/cachew-host/main"):
        problems.append(f"could not fetch {HOST_REPO} main")
    local = runtime_blobs(root)
    for sha, files in sorted(pins.items()):
        if not git_ok(root, "cat-file", "-e", f"{sha}^{{commit}}") and not fetch_from_host(root, "--depth=1", sha):
            problems.append(f"pin {sha} is not a known {HOST_REPO} commit (was it pushed?)")
            continue
        if runtime_blobs(root, sha) != local:
            problems.append(
                f"pin {sha} {HOST_REPO}/{HOST_DIR} differs from this tree's {ACTION_DIR} runtime files (used by {files[0]})"
            )
        if require_main and not git_ok(root, "merge-base", "--is-ancestor", sha, "refs/cachew-host/main"):
            problems.append(f"pin {sha} is not reachable from {HOST_REPO} main and may be garbage-collected")
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
