"""Driver: parallel C++ indexer.

Walks `compile_commands.json`, filters TUs by §1 scope, checks the per-TU
shard cache for freshness, and farms the rest out to a worker pool. Each
worker is a separate process running `cpp_index_worker.index_one`.

Cache freshness: a shard is considered valid iff
    - manifest.json exists with status="ok"
    - the TU's mtime_ns matches the recorded value
    - every previously-included header still exists and its mtime_ns matches

Usage:
    python cpp_index_driver.py \
        --db /workspace/build_Release/compile_commands.json \
        --shard-root /workspace/dep-graph/cache/tu_shards \
        --workers 0          # 0 == os.cpu_count()
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Allow `from _cpp_lib import …` when the driver runs both sides (main + workers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cpp_lib import TUEntry, load_db, in_scope, tu_cache_key  # noqa: E402


# ─── cache freshness ───────────────────────────────────────────────────────


def shard_is_fresh(shard_dir: Path, entry: TUEntry) -> bool:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if m.get("status") != "ok":
        return False
    try:
        cur_tu_mtime = os.stat(entry.file).st_mtime_ns
    except OSError:
        return False
    if m.get("tu_mtime_ns") != cur_tu_mtime:
        return False
    for inc in m.get("included_files", []):
        try:
            if os.stat(inc["path"]).st_mtime_ns != inc["mtime_ns"]:
                return False
        except OSError:
            return False
    return True


# ─── worker entry point (top-level so multiprocessing can pickle it) ──────

# The worker module is imported lazily *inside* the function so the master
# process does not have to import libclang.

def _worker_entry(arg: tuple[str, str, str]) -> dict:
    tu_path, db_path, shard_root = arg
    from cpp_index_worker import index_one  # local import
    from _cpp_lib import load_db
    db = load_db(Path(db_path))
    return index_one(tu_path, db, Path(shard_root))


# ─── main ──────────────────────────────────────────────────────────────────


@dataclass
class Plan:
    fresh: list[str]
    to_parse: list[str]
    out_of_scope_skipped: int


def build_plan(
    db: dict[str, TUEntry],
    shard_root: Path,
    only_tus: set[str] | None = None,
) -> Plan:
    fresh: list[str] = []
    to_parse: list[str] = []
    out_of_scope = 0
    for tu, entry in db.items():
        if only_tus is not None and tu not in only_tus:
            continue
        if not in_scope(tu):
            out_of_scope += 1
            continue
        cache_key = tu_cache_key(entry.file, entry.argv)
        shard_dir = shard_root / cache_key
        if shard_is_fresh(shard_dir, entry):
            fresh.append(tu)
        else:
            to_parse.append(tu)
    fresh.sort()
    to_parse.sort()
    return Plan(fresh=fresh, to_parse=to_parse, out_of_scope_skipped=out_of_scope)


def run_pool(work: Iterable[tuple[str, str, str]], workers: int, total: int) -> dict:
    """Spawn workers, stream results, report progress to stderr."""
    counts = {"ok": 0, "failed": 0, "skipped": 0, "started_at": time.time()}
    if total == 0:
        return counts

    # maxtasksperchild=1 → each worker process indexes exactly one TU then
    # exits. This bounds memory growth (libclang holds the AST in process
    # heap) at the cost of ~50–100 ms spawn time per TU. Acceptable.
    with mp.Pool(processes=workers, maxtasksperchild=1) as pool:
        start = time.time()
        done = 0
        for result in pool.imap_unordered(_worker_entry, work, chunksize=1):
            done += 1
            counts[result.get("status", "skipped")] = counts.get(result.get("status", "skipped"), 0) + 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                eta = max(total - done, 0) / max(rate, 1e-6)
                print(
                    f"[driver] {done}/{total}  rate={rate:.1f} TU/s  "
                    f"eta={eta:5.1f}s  ok={counts['ok']} fail={counts['failed']}",
                    file=sys.stderr,
                )
            if result.get("status") == "failed":
                print(f"[driver] FAILED {result['tu']}: {result.get('error')}", file=sys.stderr)
    counts["finished_at"] = time.time()
    counts["wall_seconds"] = counts["finished_at"] - counts["started_at"]
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--shard-root", default="/workspace/dep-graph/cache/tu_shards")
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--limit", type=int, default=0, help="Index only the first N TUs (debug)")
    ap.add_argument("--force", action="store_true", help="Ignore cache; reparse all in-scope TUs")
    ap.add_argument(
        "--tu-list-file",
        default="",
        help="File with one absolute TU path per line; restrict indexing to these",
    )
    ap.add_argument("--manifest-out", default="/workspace/dep-graph/cache/tu_shards/_run_manifest.json")
    args = ap.parse_args()

    db = load_db(Path(args.db))
    shard_root = Path(args.shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    only_tus: set[str] | None = None
    if args.tu_list_file:
        only_tus = set()
        for line in Path(args.tu_list_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                only_tus.add(os.path.normpath(line))
        print(f"[driver] restricting to {len(only_tus)} TUs from {args.tu_list_file}", file=sys.stderr)

    plan = build_plan(db, shard_root, only_tus=only_tus)
    if args.force:
        plan.to_parse = plan.fresh + plan.to_parse
        plan.fresh = []

    if args.limit:
        plan.to_parse = plan.to_parse[: args.limit]

    workers = args.workers or (os.cpu_count() or 1)
    print(
        f"[driver] in_scope={len(plan.fresh) + len(plan.to_parse)} "
        f"out_of_scope_skipped={plan.out_of_scope_skipped} "
        f"cache_hits={len(plan.fresh)} to_parse={len(plan.to_parse)} "
        f"workers={workers}",
        file=sys.stderr,
    )

    work = [(tu, args.db, args.shard_root) for tu in plan.to_parse]

    # Ignore SIGINT in workers so the master can shut down cleanly on Ctrl-C
    def _sig_handler(*_a):  # noqa: ARG001
        print("\n[driver] interrupted — terminating worker pool", file=sys.stderr)
        os._exit(130)
    signal.signal(signal.SIGINT, _sig_handler)

    counts = run_pool(work, workers, total=len(work))

    run_manifest = {
        "db": args.db,
        "shard_root": args.shard_root,
        "workers": workers,
        "cache_hits": len(plan.fresh),
        "out_of_scope_skipped": plan.out_of_scope_skipped,
        "parsed": counts,
        "in_scope_total": len(plan.fresh) + len(plan.to_parse),
    }
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(json.dumps(run_manifest, indent=2))
    print(f"[driver] done. run-manifest: {args.manifest_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
