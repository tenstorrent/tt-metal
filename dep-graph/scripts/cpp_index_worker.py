"""Worker: parse one C++ TU with libclang, write JSONL shards.

Designed to be called either directly from a shell (for debugging) or
via multiprocessing.Pool from `cpp_index_driver.py` (the normal path).

Output layout per TU:
    <shard_root>/<tu-hash>/
        manifest.json       — TU identity, mtimes, counts
        nodes.jsonl
        edges.jsonl
        bindings.jsonl
        diagnostics.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from pathlib import Path

# Allow `from _cpp_lib import …` when run via subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cpp_lib import (  # noqa: E402
    Indexer,
    TUEntry,
    load_db,
    tu_cache_key,
    write_jsonl,
)


def _mtime_ns(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def index_one(
    tu_path: str,
    db: dict[str, TUEntry],
    shard_root: Path,
) -> dict:
    """Parse `tu_path`, write its shard, return a small summary dict."""
    entry = db.get(os.path.normpath(tu_path))
    if entry is None:
        return {"tu": tu_path, "status": "skipped", "reason": "not-in-db"}

    cache_key = tu_cache_key(entry.file, entry.argv)
    shard_dir = shard_root / cache_key

    indexer = Indexer(db)
    try:
        indexer.index_tu(tu_path)
    except Exception as e:  # libclang occasionally surfaces hard errors
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "manifest.json").write_text(json.dumps({
            "tu": tu_path,
            "hash": cache_key,
            "status": "failed",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }, indent=2))
        return {"tu": tu_path, "status": "failed", "error": str(e), "hash": cache_key}

    nodes_n  = write_jsonl(shard_dir / "nodes.jsonl",   indexer.nodes.values())
    edges_n  = write_jsonl(shard_dir / "edges.jsonl",   indexer.edges)
    binds_n  = write_jsonl(shard_dir / "bindings.jsonl", indexer.bindings)
    diags_n  = write_jsonl(shard_dir / "diagnostics.jsonl", indexer.diagnostics)

    tu_mtime = _mtime_ns(entry.file)
    seen_inc: set[str] = set()
    included = []
    for inc in indexer.included_files:
        if inc in seen_inc:
            continue
        seen_inc.add(inc)
        mt = _mtime_ns(inc)
        if mt is not None:
            included.append({"path": inc, "mtime_ns": mt})

    manifest = {
        "tu": entry.file,
        "hash": cache_key,
        "argv": list(entry.argv),
        "directory": entry.directory,
        "tu_mtime_ns": tu_mtime,
        "included_files": included,
        "parsed_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "counts": {
            "nodes": nodes_n,
            "edges": edges_n,
            "bindings": binds_n,
            "diagnostics": diags_n,
        },
        "status": "ok",
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"tu": tu_path, "status": "ok", "hash": cache_key, **manifest["counts"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu", required=True, help="Absolute path of the TU to index")
    ap.add_argument("--db", required=True, help="compile_commands.json path")
    ap.add_argument(
        "--shard-root",
        default="/workspace/dep-graph/cache/tu_shards",
        help="Directory under which per-TU shard subdirs live",
    )
    args = ap.parse_args()
    db = load_db(Path(args.db))
    summary = index_one(args.tu, db, Path(args.shard_root))
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
