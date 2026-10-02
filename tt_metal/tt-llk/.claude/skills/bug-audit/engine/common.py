"""Shared helpers for the bug-audit engine.

Every engine script works on one RUN DIRECTORY, which holds all state for one audit: its config, batch
manifest, per-batch findings and verdicts, and the derived reports. Pick the run with `--run DIR`, or the
BUG_AUDIT_RUN environment variable, or by running the script from inside the run directory.
"""

import hashlib
import json
import os
import sys

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}

# Headless audit sessions run unattended in auto mode, and their agents (workflow agents inherit the session's
# permission rules) must never build, run tests, touch a card, or change the audited tree: the audit is static.
# Deny rules are checked before auto mode's classifier, and match any
# subcommand of a compound command. They match the command text only, so this is a guard, not a sandbox.
_DENY_CMDS = [
    "make",
    "cmake",
    "ninja",
    "./build_metal.sh",
    "build_metal.sh",
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "ctest",
    "tt-smi",
    "tt-exalens",
    "rm",
    "sed -i",
    "sed -E -i",
    "sed -n -i",
]
_DENY_GIT = [
    "checkout",
    "switch",
    "restore",
    "reset",
    "clean",
    "commit",
    "push",
    "stash",
    "rebase",
    "merge",
    "cherry-pick",
    "am",
    "apply",
]
STATIC_DENY = (
    [f"Bash({c} *)" for c in _DENY_CMDS]
    + [f"Bash(git {g} *)" for g in _DENY_GIT]
    + [f"Bash(git -C * {g} *)" for g in _DENY_GIT]
)


def blocked_actions(session_id, config_dir=None):
    """How many tool calls the deny rules refused in a headless session's workflow agents (information only).

    Workflow agents' transcripts live under <config>/projects/<cwd>/<session>/subagents/; a refused call's result reads
    "Permission to use <tool> with command <cmd> has been denied."
    """
    import glob

    root = (
        config_dir
        or os.environ.get("CLAUDE_CONFIG_DIR")
        or os.path.expanduser("~/.claude")
    )
    n = 0
    for f in glob.glob(
        os.path.join(root, "projects", "*", session_id, "subagents", "**", "*.jsonl"),
        recursive=True,
    ):
        with open(f, errors="replace") as fh:
            n += sum(
                1
                for ln in fh
                if "Permission to use " in ln and "has been denied." in ln
            )
    return n


def headless_flags():
    """claude -p flags shared by every headless driver: auto mode, plus the static-hunt deny rules."""
    return ["--permission-mode", "auto", "--disallowedTools", *STATIC_DENY]


def run_dir(argv=None):
    argv = sys.argv if argv is None else argv
    if "--run" in argv:
        i = argv.index("--run")
        d = argv[i + 1]
        del argv[i : i + 2]
    else:
        d = os.environ.get("BUG_AUDIT_RUN", os.getcwd())
    d = os.path.abspath(d)
    if not os.path.exists(os.path.join(d, "state.json")):
        sys.exit(
            f"{d} is not a bug-audit run directory (no state.json); create one with init_run.py"
        )
    return d


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)  # atomic: a crash mid-write cannot truncate the real file


def state(out):
    return load(os.path.join(out, "state.json"))


def manifest(out):
    return {
        b["batch"]: b for b in load(os.path.join(out, "batches", "manifest.json"), [])
    }


def marker_set(out, sub, ext):
    d = os.path.join(out, sub)
    if not os.path.isdir(d):
        return set()
    return {f[: -len(ext)] for f in os.listdir(d) if f.endswith(ext)}


def findings_of(out, batch):
    """Candidate list for a batch, or None when its findings file is missing or unreadable."""
    try:
        return load(os.path.join(out, "findings", f"{batch}.json"))
    except (OSError, ValueError):
        return None


def key_of(f):
    return f"{f['file']}:{f['line']}"


def seeded_order(items, seed, key):
    """A reproducible shuffle: the same seed and ids always give the same order, a new seed a new one."""
    return sorted(
        items, key=lambda x: hashlib.sha256(f"{seed}:{key(x)}".encode()).hexdigest()
    )
