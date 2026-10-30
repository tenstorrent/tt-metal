# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Migrate checkout consumers, including gh-aw output, to the published cachew action.

Run with --write to migrate, or --check to detect drift. With no paths, inspect all
tracked action/workflow files. The wrapper's internal checkout is intentionally
excluded: replacing it would recurse instead of providing upstream fallback.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ACTION_REPO = "tenstorrent/tt-metal/.github/actions/cachew-checkout"
ACTION_SHA = "b4bd422967a6297fd5586d8aacd8b72ea17433cd"
ACTION = f"{ACTION_REPO}@{ACTION_SHA}"
UPSTREAM_WRAPPER = Path(".github/actions/cachew-checkout/action.yml")
CHECKOUT = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?:-[ \t]+)?uses:[ \t]*)(?P<quote>['\"]?)"
    r"(?:actions/checkout|" + re.escape(ACTION_REPO) + r")@[^\s'\"#]+(?P=quote)(?:[ \t]*#[^\n]*)?[ \t]*$"
)
ACTION_COMMENT = re.compile(r"(?m)^#   - (?:actions/checkout|" + re.escape(ACTION_REPO) + r")@[^\n]+$")
REGEN_COMMAND = "python3 .github/scripts/utils/cachew_checkout.py --write"


def rewrite(text):
    text = CHECKOUT.sub(lambda match: f"{match['prefix']}{ACTION} # cachew checkout", text)
    text = ACTION_COMMENT.sub(f"#   - {ACTION} # cachew checkout", text)
    if text.startswith("# gh-aw-metadata:"):
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.startswith("# gh-aw-manifest: "):
                manifest = json.loads(line.removeprefix("# gh-aw-manifest: "))
                actions = []
                for action in manifest["actions"]:
                    if action["repo"] in ("actions/checkout", ACTION_REPO):
                        action = {"repo": ACTION_REPO, "sha": ACTION_SHA, "version": ACTION_SHA}
                    if action not in actions:
                        actions.append(action)
                manifest["actions"] = actions
                lines[index] = "# gh-aw-manifest: " + json.dumps(manifest, separators=(",", ":")) + "\n"
        text = "".join(lines)
        text = text.replace("#   gh aw compile\n", f"#   gh aw compile && {REGEN_COMMAND}\n")
        # Different upstream checkout versions can produce duplicate action comments.
        comment = f"#   - {ACTION} # cachew checkout\n"
        first = text.find(comment)
        if first >= 0:
            text = text[: first + len(comment)] + text[first + len(comment) :].replace(comment, "")
    return text


def migrate_file(path, write=False):
    path = Path(path)
    if path.resolve() == UPSTREAM_WRAPPER.resolve():
        return False
    before = path.read_text()
    after = rewrite(before)
    if before == after:
        return False
    if write:
        path.write_text(after)
    return True


def tracked_consumers():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    for value in paths:
        path = Path(value)
        parts = path.parts
        if ".github" not in parts:
            continue
        github = parts.index(".github")
        if len(parts) <= github + 1 or parts[github + 1] not in ("actions", "workflows"):
            continue
        if path.suffix in (".yml", ".yaml") or (path.suffix == ".md" and parts[github + 1] == "workflows"):
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    changed = [path for path in (args.paths or tracked_consumers()) if migrate_file(path, args.write)]
    for path in changed:
        print(f"{'Migrated' if args.write else 'Needs cachew migration:'} {path}")
    print(f"{len(changed)} file(s) {'updated' if args.write else 'need migration'}.")
    return 1 if changed and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
