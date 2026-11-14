# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Migrate checkout consumers to the published cachew action and verify its pins.

The action lives in tenstorrent/.github (actions/cachew-checkout); every
consumer here pins it at ACTION_SHA. Run with --write to migrate (including
pins to its old tt-metal location), --check to detect unmigrated consumers, or
--verify-pins to check that every consumer pins exactly ACTION_SHA in
tenstorrent/.github. With no paths, inspect all tracked action/workflow files.

Only checkouts of the exact version the wrapper forwards to (actions/checkout
v6.0.2, or a v6 tag) are migrated, so no consumer changes checkout version.
Excluded on purpose:
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


ACTION_REPO = "tenstorrent/.github/actions/cachew-checkout"
# A full commit of tenstorrent/.github; its actions/cachew-checkout is what runs.
ACTION_SHA = "e0302d6af3504b1f3d8b64ad6fba7d88276a23f4"
ACTION = f"{ACTION_REPO}@{ACTION_SHA}"
# Where the action lived before it moved to tenstorrent/.github. --write repins
# these; --verify-pins rejects them.
LEGACY_REPO = "tenstorrent/tt-metal/.github/actions/cachew-checkout"
FORWARDED_CHECKOUT = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
MIGRATABLE_REFS = (FORWARDED_CHECKOUT, "v6", "v6.0.2")
EXCLUDED_PREFIXES = ("tt_metal/tt-llk/",)
CHECKOUT = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?:-[ \t]+)?uses:[ \t]*)(?P<quote>['\"]?)"
    r"(?:actions/checkout@(?:"
    + "|".join(map(re.escape, MIGRATABLE_REFS))
    + r")|(?:"
    + re.escape(ACTION_REPO)
    + "|"
    + re.escape(LEGACY_REPO)
    + r")@[0-9a-f]{40})"
    r"(?P=quote)(?:[ \t]*#[^\n]*)?[ \t]*$"
)
ANY_PIN = re.compile(
    r"(?P<repo>" + re.escape(ACTION_REPO) + "|" + re.escape(LEGACY_REPO) + r")(?P<subpath>/[\w./-]+)?@(?P<sha>[0-9a-f]{40})"
)
PLAIN_CHECKOUT = f"actions/checkout@{FORWARDED_CHECKOUT} # v6.0.2"
HOSTED_LABEL = re.compile(r"(?:ubuntu|windows|macos)-[\w.-]+")
TOP_LEVEL = re.compile(r"^(?P<key>[\w-]+):")
JOB = re.compile(r"^  [\w-]+:[ \t]*(?:#.*)?$")
RUNS_ON = re.compile(r"^    runs-on:[ \t]*(?P<quote>['\"]?)(?P<label>[^'\"#\s]+)(?P=quote)[ \t]*(?:#.*)?$")


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
            return f"{match['prefix']}{PLAIN_CHECKOUT}" if "cachew-checkout@" in match[0] else match[0]
        return f"{match['prefix']}{ACTION} # cachew checkout"

    return CHECKOUT.sub(replace, text)


def migrate_file(path, write=False, root=None):
    root = root or repository_root()
    path = Path(path)
    absolute = path if path.is_absolute() else root / path
    relative = absolute.resolve().relative_to(root.resolve())
    if excluded(relative):
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


def verify_pins(root):
    """The consumers' pin to the action is the only pin. Fail if any consumer
    pins another commit, the action's old tt-metal location, or a subdirectory.
    Whether ACTION_SHA exists in tenstorrent/.github (and is on its main) is
    checked by the test-cachew-checkout workflow, which can reach GitHub."""
    problems = []
    pins = {}
    for path in tracked_consumers(root):
        for match in ANY_PIN.finditer((root / path).read_text()):
            if match["repo"] == LEGACY_REPO:
                problems.append(f"{path} references {match[0]}; the action moved to {ACTION_REPO} (run --write)")
                continue
            if match["subpath"]:
                problems.append(f"{path} references {match[0]}; reference the action itself, not a subdirectory")
            pins.setdefault(match["sha"], []).append(path.as_posix())
    if set(pins) - {ACTION_SHA}:
        problems.append(f"consumers pin {sorted(pins)}, expected only ACTION_SHA {ACTION_SHA}")
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
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = repository_root()
    if args.verify_pins:
        return verify_pins(root)
    changed = [path for path in (args.paths or tracked_consumers(root)) if migrate_file(path, args.write, root)]
    for path in changed:
        print(f"{'Migrated' if args.write else 'Needs cachew migration:'} {path}")
    print(f"{len(changed)} file(s) {'updated' if args.write else 'need migration'}.")
    return 1 if changed and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
