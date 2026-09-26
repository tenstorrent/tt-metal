"""Shared helpers for the bug-audit engine.

Every engine script works on one RUN DIRECTORY, which holds all state for one audit: its config, batch
manifest, per-batch findings and verdicts, and the derived reports. Pick the run with `--run DIR`, or the
BUG_AUDIT_RUN environment variable, or by running the script from inside the run directory.
"""

import json
import os
import sys

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


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
