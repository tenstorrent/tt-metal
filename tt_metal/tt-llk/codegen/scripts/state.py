# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
JSON-backed key-value store for passing values between separate shell calls.

Backing file resolution (highest precedence first):
    --file <path>           explicit path — edge cases only, prefer the modes below
    --log-dir <dir>         <dir>/state.json
    $LOG_DIR                $LOG_DIR/state.json
    --worktree-dir <dir>    <dir>/tt_metal/tt-llk/.codegen_run_state.json
    $WORKTREE_DIR           $WORKTREE_DIR/tt_metal/tt-llk/.codegen_run_state.json

Subcommands:
    set K V         Store string value V under key K.
    set K V --json  Store V parsed as JSON (typed: numbers, bools, objects).
    get K           Print value of K (raw for strings, JSON otherwise).
                    Missing key -> print --default (empty string) and exit 0.
    del K           Remove key K (no error if absent).
    keys            Print all keys, one per line.
    dump            Print the whole store as pretty JSON.

Usage:
    python state.py --log-dir "$DIR" set RUN_ID "$RUN_ID"
    VALUE=$(python state.py --log-dir "$DIR" get RUN_ID --default unknown)
    python state.py --worktree-dir "$WORKTREE_DIR" set KERNEL_NAME "gelu"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Backing-file resolution
# --------------------------------------------------------------------------


def _resolve_path(
    file_arg: str | None, log_dir_arg: str | None, worktree_dir_arg: str | None
) -> Path:
    # Explicit flags always win over ambient env vars — an explicit
    # --worktree-dir must not be silently redirected by a $LOG_DIR that
    # happens to already be exported in the same shell (e.g. orchestrator
    # Step 0, which exports LOG_DIR before writing WORKTREE_DIR-scoped keys).
    if file_arg:
        return Path(file_arg)
    if log_dir_arg:
        return Path(log_dir_arg) / "state.json"
    if worktree_dir_arg:
        return (
            Path(worktree_dir_arg) / "tt_metal" / "tt-llk" / ".codegen_run_state.json"
        )
    log_dir = os.environ.get("LOG_DIR")
    if log_dir:
        return Path(log_dir) / "state.json"
    worktree_dir = os.environ.get("WORKTREE_DIR")
    if worktree_dir:
        return Path(worktree_dir) / "tt_metal" / "tt-llk" / ".codegen_run_state.json"
    raise SystemExit(
        "state.py: no backing file — pass --file, --log-dir, --worktree-dir, "
        "or set $LOG_DIR/$WORKTREE_DIR"
    )


# --------------------------------------------------------------------------
# Low-level IO (mirrors run_json_writer.py)
# --------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state.json.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(store, f, indent=2)
            f.write("\n")
        # Match run_json_writer.py: mkstemp is 0o600, which locks out the shared
        # group; relax to 0o664 so the same readers can see this store.
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _emit(value: Any) -> None:
    """Print a value for shell capture: strings raw, everything else as JSON."""
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value))


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def _cmd_set(path: Path, key: str, value: str, as_json: bool) -> int:
    if as_json:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"state.py: --json value is not valid JSON: {exc}")
        store = _load(path)
        store[key] = parsed
    else:
        store = _load(path)
        store[key] = value
    _atomic_write(path, store)
    return 0


def _cmd_get(path: Path, key: str, default: str) -> int:
    store = _load(path)
    if key in store:
        _emit(store[key])
    else:
        print(default)
    return 0


def _cmd_del(path: Path, key: str) -> int:
    store = _load(path)
    if key in store:
        del store[key]
        _atomic_write(path, store)
    return 0


def _cmd_keys(path: Path) -> int:
    for key in _load(path):
        print(key)
    return 0


def _cmd_dump(path: Path) -> int:
    print(json.dumps(_load(path), indent=2))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--file",
        default=None,
        help="Explicit path to the state file. Edge cases only — prefer "
        "--log-dir or --worktree-dir so every caller resolves the same file.",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help="Run LOG_DIR; state lives at <log-dir>/state.json (default: $LOG_DIR).",
    )
    ap.add_argument(
        "--worktree-dir",
        default=None,
        help="Worktree root; state lives at <worktree-dir>/tt_metal/tt-llk/"
        ".codegen_run_state.json (default: $WORKTREE_DIR).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Store a value under a key.")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Parse VALUE as JSON (store typed: number, bool, object, array).",
    )

    p_get = sub.add_parser("get", help="Print the value of a key.")
    p_get.add_argument("key")
    p_get.add_argument(
        "--default",
        default="",
        help="Printed when the key is absent (default: empty string).",
    )

    p_del = sub.add_parser("del", help="Remove a key.")
    p_del.add_argument("key")

    sub.add_parser("keys", help="Print all keys, one per line.")
    sub.add_parser("dump", help="Print the whole store as pretty JSON.")

    args = ap.parse_args(argv)
    path = _resolve_path(args.file, args.log_dir, args.worktree_dir)

    if args.cmd == "set":
        return _cmd_set(path, args.key, args.value, args.as_json)
    if args.cmd == "get":
        return _cmd_get(path, args.key, args.default)
    if args.cmd == "del":
        return _cmd_del(path, args.key)
    if args.cmd == "keys":
        return _cmd_keys(path)
    if args.cmd == "dump":
        return _cmd_dump(path)
    return 2


if __name__ == "__main__":
    sys.exit(main())
