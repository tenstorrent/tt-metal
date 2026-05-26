"""Merger: collapse per-TU shards into a single C++ index.

Inputs:  /workspace/dep-graph/cache/tu_shards/<hash>/{nodes,edges,bindings,diagnostics}.jsonl
Outputs: /workspace/dep-graph/cache/cpp_index/{nodes,edges,bindings,diagnostics}.jsonl
         /workspace/dep-graph/cache/cpp_index/manifest.json

Dedup policy:
    - nodes:    USR-keyed. A definition record beats a declaration record.
                discovered_in is unioned.
    - edges:    deduped on (src, dst, kind, site_file, site_line).
    - bindings: deduped on (python_name, cpp_node_id, site_file, site_line).
    - diagnostics: passed through (no dedup).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cpp_lib import read_jsonl  # noqa: E402

# Use orjson when available — at the scale of 1454 shards × thousands of
# lines each, this is the difference between ~8 min and ~1 min for the merge.
try:
    import orjson
    def _loads(b: bytes | str):  # type: ignore[no-redef]
        return orjson.loads(b)
    def _dumps(obj) -> bytes:
        return orjson.dumps(obj)
except ImportError:
    def _loads(b):
        return json.loads(b)
    def _dumps(obj) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode()


def _stream_jsonl_bytes(path: Path):
    """Yield raw byte lines from a JSONL file, skipping blanks."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def merge(shard_root: Path, out_dir: Path, only_tus: set[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes: dict[str, dict] = {}
    edges_seen: set[tuple] = set()
    bindings_seen: set[tuple] = set()
    diag_count = 0
    edge_count = 0
    binding_count = 0
    shard_count = 0
    skipped_filter = 0
    start = time.time()

    # Stream-write edges/bindings/diagnostics; nodes need dedup so build in memory.
    edges_path = out_dir / "edges.jsonl"
    bindings_path = out_dir / "bindings.jsonl"
    diag_path = out_dir / "diagnostics.jsonl"

    with open(edges_path, "wb") as fe, open(bindings_path, "wb") as fb, open(diag_path, "wb") as fd:
        for shard in sorted(shard_root.iterdir()):
            if not shard.is_dir():
                continue
            manifest_path = shard / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                m = _loads(manifest_path.read_bytes())
            except Exception:
                continue
            if m.get("status") != "ok":
                continue
            if only_tus is not None and os.path.normpath(m.get("tu", "")) not in only_tus:
                skipped_filter += 1
                continue
            shard_count += 1

            for raw in _stream_jsonl_bytes(shard / "nodes.jsonl"):
                n = _loads(raw)
                nid = n["id"]
                existing = nodes.get(nid)
                if existing is None:
                    nodes[nid] = n
                    continue
                # Prefer a record marked as definition.
                if n.get("is_definition") and not existing.get("is_definition"):
                    di = existing.get("discovered_in", [])
                    n["discovered_in"] = sorted(set(di + n.get("discovered_in", [])))
                    if n.get("is_binding_target") or existing.get("is_binding_target"):
                        n["is_binding_target"] = True
                    nodes[nid] = n
                else:
                    existing["discovered_in"] = sorted(set(
                        existing.get("discovered_in", []) + n.get("discovered_in", [])
                    ))
                    if n.get("is_binding_target"):
                        existing["is_binding_target"] = True

            for raw in _stream_jsonl_bytes(shard / "edges.jsonl"):
                e = _loads(raw)
                k = (e.get("src"), e.get("dst"), e.get("kind"),
                     e.get("site_file"), e.get("site_line"))
                if k in edges_seen:
                    continue
                edges_seen.add(k)
                fe.write(_dumps(e))
                fe.write(b"\n")
                edge_count += 1

            for raw in _stream_jsonl_bytes(shard / "bindings.jsonl"):
                b = _loads(raw)
                k = (b.get("python_name"), b.get("cpp_node_id"),
                     b.get("site_file"), b.get("site_line"))
                if k in bindings_seen:
                    continue
                bindings_seen.add(k)
                fb.write(_dumps(b))
                fb.write(b"\n")
                binding_count += 1

            for raw in _stream_jsonl_bytes(shard / "diagnostics.jsonl"):
                d = _loads(raw)
                fd.write(_dumps(d))
                fd.write(b"\n")
                diag_count += 1

    with open(out_dir / "nodes.jsonl", "wb") as fn:
        for n in nodes.values():
            fn.write(_dumps(n))
            fn.write(b"\n")

    elapsed = time.time() - start
    manifest = {
        "shard_root": str(shard_root),
        "shards_merged": shard_count,
        "counts": {
            "nodes": len(nodes),
            "edges": edge_count,
            "bindings": binding_count,
            "diagnostics": diag_count,
        },
        "wall_seconds": round(elapsed, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", default="/workspace/dep-graph/cache/tu_shards")
    ap.add_argument("--out", default="/workspace/dep-graph/cache/cpp_index")
    ap.add_argument(
        "--tu-list-file",
        default="",
        help="Optional file with TU paths to include (one per line); excludes others",
    )
    args = ap.parse_args()
    only_tus: set[str] | None = None
    if args.tu_list_file:
        only_tus = {
            os.path.normpath(line.strip())
            for line in Path(args.tu_list_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    manifest = merge(Path(args.shard_root), Path(args.out), only_tus=only_tus)
    print(f"[merger] {manifest['shards_merged']} shards merged -> {args.out}", file=sys.stderr)
    print(f"  nodes={manifest['counts']['nodes']} edges={manifest['counts']['edges']} "
          f"bindings={manifest['counts']['bindings']} diagnostics={manifest['counts']['diagnostics']}",
          file=sys.stderr)
    print(f"  wall_seconds={manifest['wall_seconds']}", file=sys.stderr)


if __name__ == "__main__":
    main()
