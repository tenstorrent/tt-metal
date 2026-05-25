"""Stitch the C++ and Python indexes into a single dependency graph JSON.

For each Python ref whose target chain is `["ttnn", "<name>"]` (or
`["ttnn", "<sub>", "<name>"]`), look up the trailing `<name>` in the C++
binding table. If found, emit a `binds`-kind edge with `crosses_language=True`
pointing from the Python caller to the C++ symbol(s) bound under that name.

The graph schema is the §3 schema in opus-instructions.md.

Usage:
    python stitch.py --cpp dep-graph/cache/cpp_index.json \
                     --py  dep-graph/cache/py_index.json \
                     --out dep-graph/out/dependency_graph.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cpp = json.loads(Path(args.cpp).read_text())
    py = json.loads(Path(args.py).read_text())

    # ── build binding lookup: python_name -> list[cpp_node_id]
    by_name: dict[str, list[dict]] = defaultdict(list)
    for b in cpp["bindings"]:
        by_name[b["python_name"]].append(b)

    # ── merge nodes
    nodes: dict[str, dict] = {}
    for n in cpp["nodes"]:
        nodes[n["id"]] = n
    for n in py["nodes"]:
        nodes[n["id"]] = n

    # ── merge edges (already-resolved intra-language call edges)
    edges: list[dict] = []
    for e in cpp["edges"]:
        edges.append({
            "src": e["src"],
            "dst": e["dst"],
            "kind": e["kind"],
            "site_file": e["site_file"],
            "site_line": e["site_line"],
            "crosses_language": False,
            "via_decorator": None,
        })
    for e in py["edges"]:
        edges.append({
            "src": e["src"],
            "dst": e["dst"],
            "kind": e["kind"],
            "site_file": e["site_file"],
            "site_line": e["site_line"],
            "crosses_language": False,
            "via_decorator": None,
        })

    # ── resolve Python refs against the C++ binding map
    cross_edges: list[dict] = []
    unresolved: list[dict] = []
    for r in py["refs"]:
        chain = r["target_chain"]
        if not chain or chain[0] != "ttnn" or len(chain) < 2:
            unresolved.append(r)
            continue
        candidate_name = chain[-1]
        matches = by_name.get(candidate_name, [])
        if not matches:
            unresolved.append(r)
            continue
        for m in matches:
            cross_edges.append({
                "src": r["src"],
                "dst": m["cpp_node_id"],
                "kind": "binds" if r["kind"] == "attr_access" else "calls",
                "site_file": r["site_file"],
                "site_line": r["site_line"],
                "crosses_language": True,
                "via_decorator": None,
                "via_helper": m["helper"],
                "python_chain": ".".join(chain),
            })
    edges.extend(cross_edges)

    graph = {
        "schema_version": "0.1",
        "nodes": list(nodes.values()),
        "edges": edges,
        "bindings": cpp["bindings"],
        "stitched": {
            "cross_language_edges": len(cross_edges),
            "unresolved_refs": len(unresolved),
            "binding_names": sorted(by_name.keys()),
        },
        "diagnostics": {
            "cpp": cpp.get("diagnostics", []),
            "py": py.get("diagnostics", []),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(graph, indent=2))
    print(
        f"[stitch] {len(graph['nodes'])} nodes, {len(graph['edges'])} edges "
        f"({len(cross_edges)} cross-language), {len(unresolved)} unresolved refs -> {args.out}"
    )


if __name__ == "__main__":
    main()
