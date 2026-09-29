#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Print a frozen issue snapshot, or seed an existing worktree from snapshot/GitHub.

The default snapshot-print interface is unchanged. --seed-state writes the
router's existing bootstrap fields in one locked state update, without passing
issue text through generated shell commands. It never creates a worktree.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ARCH_ALIASES = {"bh": "blackhole", "wh": "wormhole", "qsr": "quasar"}
ARCHES = {"blackhole", "wormhole", "quasar"}


def _validate_issue(payload: str, number: int) -> dict:
    issue = json.loads(payload)
    if (
        not isinstance(issue, dict)
        or type(issue.get("number")) is not int
        or issue["number"] != number
    ):
        raise ValueError(f"snapshot issue number does not match {number}")
    return issue


def load_issue(number: int, snapshot: str | None = None) -> str:
    """Preserve the original snapshot-only, verbatim print interface."""
    snapshot = snapshot or os.environ.get("CODEGEN_ISSUE_SNAPSHOT")
    if not snapshot:
        raise ValueError("CODEGEN_ISSUE_SNAPSHOT is not set")
    payload = Path(snapshot).read_text()
    _validate_issue(payload, number)
    return payload


def _arches(raw: str) -> list[str]:
    values = json.loads(raw) if raw.lstrip().startswith("[") else raw.split(",")
    if not isinstance(values, list) or not values:
        raise ValueError("--arches must name at least one architecture")
    targets = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("architecture entries must be strings")
        arch = ARCH_ALIASES.get(value.strip().lower(), value.strip().lower())
        if arch not in ARCHES:
            raise ValueError(f"unknown target arch: {value}")
        if arch not in targets:
            targets.append(arch)
    return targets


def _simulator_inputs(args, targets: list[str]) -> dict:
    if args.test_backend != "ttsim":
        if args.ttsim_so_path or args.ttsim_so_paths:
            raise ValueError("simulator paths require --test-backend ttsim")
        return {}
    # Explicit CLI input wins over ambient environment; do not merge stale maps.
    one, many = args.ttsim_so_path, args.ttsim_so_paths
    if not one and not many:
        if len(targets) == 1:
            one = os.environ.get("TTSIM_SO_PATH")
        many = None if one else os.environ.get("TTSIM_SO_PATHS")
    if many:
        paths = json.loads(many)
        if not isinstance(paths, dict):
            raise ValueError("--ttsim-so-paths must be a JSON object")
        normalized = {}
        for arch, path in paths.items():
            names = _arches(arch)
            if len(names) != 1 or names[0] in normalized:
                raise ValueError(
                    "simulator map must have one unique architecture per key"
                )
            normalized[names[0]] = path
        if set(normalized) != set(targets):
            raise ValueError(
                "simulator map must cover exactly the requested architectures"
            )
    elif one and len(targets) == 1:
        normalized = {targets[0]: one}
    else:
        raise ValueError(
            "ttsim requires an explicit library path for every target architecture"
        )
    for arch, path in normalized.items():
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not Path(path).is_file()
        ):
            raise ValueError(
                f"simulator path for {arch} must be an existing absolute file"
            )
    if len(targets) == 1:
        return {"TTSIM_SO_PATH": normalized[targets[0]]}
    return {"TTSIM_SO_PATHS": json.dumps(normalized, ensure_ascii=False)}


def seed_issue_state(args) -> dict:
    """Use the existing state writer; retain setup admission/resume metadata."""
    import state

    if args.number <= 0:
        raise ValueError("issue number must be positive")
    for flag in (
        "worktree_dir",
        "worktree_branch",
        "arches",
        "test_backend",
        "create_local_branch",
        "create_pr",
    ):
        if not getattr(args, flag):
            raise ValueError("--seed-state requires --" + flag.replace("_", "-"))
    targets = _arches(args.arches)
    simulator = _simulator_inputs(args, targets)
    worktree = Path(args.worktree_dir).resolve()
    llk = worktree / "tt_metal" / "tt-llk"
    if not llk.is_dir():
        raise ValueError(
            "--worktree-dir must be an existing launcher-created tt-metal worktree"
        )
    branch = subprocess.run(
        ["git", "-C", str(worktree), "branch", "--show-current"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    git_dirs = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            "--git-common-dir",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if len(git_dirs) != 2 or Path(git_dirs[0]).resolve() == Path(git_dirs[1]).resolve():
        raise ValueError(
            "--worktree-dir must be a linked worktree, not the main checkout"
        )
    if branch != args.worktree_branch:
        raise ValueError(
            "--worktree-branch does not match the existing worktree branch"
        )
    if args.snapshot or os.environ.get("CODEGEN_ISSUE_SNAPSHOT"):
        issue = _validate_issue(load_issue(args.number, args.snapshot), args.number)
    else:
        command = [
            "gh",
            "issue",
            "view",
            str(args.number),
            "--json",
            "number,title,body,labels,comments,url",
        ]
        if args.repo:
            command.extend(["--repo", args.repo])
        result = subprocess.run(
            command, cwd=worktree, check=True, text=True, capture_output=True
        )
        issue = _validate_issue(result.stdout, args.number)
    for field in ("title", "body"):
        if not isinstance(issue.get(field), str):
            raise ValueError(f"issue {field} must be a string")
    labels = issue.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("issue labels must be a list")
    names = [
        label.get("name") if isinstance(label, dict) else label for label in labels
    ]
    if not all(isinstance(name, str) for name in names):
        raise ValueError("issue labels must be names or objects with a name")
    comments = issue.get("comments", [])
    if not isinstance(comments, list) or not all(
        isinstance(comment, dict) and isinstance(comment.get("body"), str)
        for comment in comments
    ):
        raise ValueError(
            "issue comments must be full comment objects with string bodies"
        )
    url = issue.get("url", "")
    if not isinstance(url, str):
        raise ValueError("issue url must be a string")
    no_push = os.environ.get("CODEGEN_NO_PUSH") == "1"
    patch = {
        "RUN_KIND": "issue",
        "RUN_MODE": "single" if len(targets) == 1 else "multi",
        "ISSUE_NUMBER": str(args.number),
        "ISSUE_TITLE": issue["title"],
        "ISSUE_BODY": issue["body"],
        "ISSUE_LABELS": ",".join(names),
        # Keep the legacy display string, but do not split comma-containing
        # GitHub label names when creating run.json later.
        "ISSUE_LABELS_JSON": names,
        # Preserve comment author/id/timestamps and any future metadata, not
        # only comment body text. state.py get returns this as lossless JSON.
        "ISSUE_COMMENTS": json.dumps(comments, ensure_ascii=False),
        "ISSUE_URL": url,
        "WORKTREE_BRANCH": branch,
        "TEST_BACKEND": args.test_backend,
        "CREATE_LOCAL_BRANCH": "yes" if no_push else args.create_local_branch,
        "CREATE_PR": "no" if no_push else args.create_pr,
        **simulator,
    }
    patch["TARGET_ARCH" if len(targets) == 1 else "TARGET_ARCHES"] = (
        targets[0] if len(targets) == 1 else json.dumps(targets)
    )
    path = state._resolve_path(None, None, str(worktree))

    def update(store):
        if store.get("RUN_ID") or store.get("LOG_DIR"):
            raise ValueError("bootstrap is already bound to a run; refuse to reseed it")
        if store.get("ISSUE_NUMBER") and str(store["ISSUE_NUMBER"]) != str(args.number):
            raise ValueError("existing bootstrap issue number does not match")
        if store.get("RUN_KIND") not in (None, "", "issue"):
            raise ValueError("existing bootstrap belongs to a different run kind")
        for key in ("TARGET_ARCH", "TARGET_ARCHES", "TTSIM_SO_PATH", "TTSIM_SO_PATHS"):
            store.pop(key, None)
        store.update(patch)

    state._locked_update(path, update)
    return {
        "state_file": str(path),
        "issue_number": args.number,
        "run_mode": patch["RUN_MODE"],
        "target_arches": targets,
        "test_backend": args.test_backend,
        "worktree_branch": branch,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument(
        "--snapshot", help="explicit snapshot instead of CODEGEN_ISSUE_SNAPSHOT"
    )
    parser.add_argument("--seed-state", action="store_true")
    parser.add_argument("--worktree-dir")
    parser.add_argument("--worktree-branch")
    parser.add_argument(
        "--arches", help="explicit target list as JSON or comma-separated names"
    )
    parser.add_argument("--test-backend", choices=("local", "ttsim"))
    parser.add_argument("--create-local-branch", choices=("yes", "no"))
    parser.add_argument("--create-pr", choices=("yes", "no"))
    parser.add_argument("--repo", help="GitHub owner/repo for live fetching only")
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--ttsim-so-path")
    paths.add_argument(
        "--ttsim-so-paths", help="JSON mapping from architecture to library path"
    )
    args = parser.parse_args(argv)
    try:
        if args.seed_state:
            print(json.dumps(seed_issue_state(args)))
            return 0
        payload = load_issue(args.number, args.snapshot)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"failed to load issue {args.number}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(payload)
    if not payload.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
