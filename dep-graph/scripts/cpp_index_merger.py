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

    with open(edges_path, "w") as fe, open(bindings_path, "w") as fb, open(diag_path, "w") as fd:
        for shard in sorted(shard_root.iterdir()):
            if not shard.is_dir():
                continue
            manifest_path = shard / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                m = json.loads(manifest_path.read_text())
            except Exception:
                continue
            if m.get("status") != "ok":
                continue
            if only_tus is not None and os.path.normpath(m.get("tu", "")) not in only_tus:
                skipped_filter += 1
                continue
            shard_count += 1

            for n in read_jsonl(shard / "nodes.jsonl"):
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

            for e in read_jsonl(shard / "edges.jsonl"):
                k = (e.get("src"), e.get("dst"), e.get("kind"),
                     e.get("site_file"), e.get("site_line"))
                if k in edges_seen:
                    continue
                edges_seen.add(k)
                fe.write(json.dumps(e, separators=(",", ":")) + "\n")
                edge_count += 1

            for b in read_jsonl(shard / "bindings.jsonl"):
                k = (b.get("python_name"), b.get("cpp_node_id"),
                     b.get("site_file"), b.get("site_line"))
                if k in bindings_seen:
                    continue
                bindings_seen.add(k)
                fb.write(json.dumps(b, separators=(",", ":")) + "\n")
                binding_count += 1

            for d in read_jsonl(shard / "diagnostics.jsonl"):
                fd.write(json.dumps(d, separators=(",", ":")) + "\n")
                diag_count += 1

    with open(out_dir / "nodes.jsonl", "w") as fn:
        for n in nodes.values():
            fn.write(json.dumps(n, separators=(",", ":")) + "\n")

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
