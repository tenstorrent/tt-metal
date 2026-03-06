#!/usr/bin/env python3
"""
Analyze fabric adjacency graph from topology_mapper debug log output.

Parses lines that contain (anywhere in the line):
  Node <id> connected to: [<id>, <id>, ...]
  Host_name = <name>

Example log format (prefix/suffix allowed):
  [1,0]<stdout>: 2026-03-06 18:14:41.850 | debug | Fabric |     Node 231145077570810177 connected to: [519375453722521921, ...] (topology_mapper.cpp:1571)
  [1,0]<stdout>: 2026-03-06 18:14:41.850 | debug | Fabric |     Host_name = UF-MN-B3-GWH02

Reads from stdin or a file (streaming for long logs). Reports:
- Degree histogram (degree = unique neighbor count, no double-counting)
- Connectivity (is the graph connected?)
- Mesh-shape checks (2D mesh has degree 2/3/4 for corners/edges/interior)
- Optional: per-host subgraphs

Usage:
  python analyze_adjacency_graph.py < fabric.log
  python analyze_adjacency_graph.py fabric.log
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import permutations
from typing import Optional, Tuple


# Log line may have prefix (e.g. "[1,0]<stdout>: ... | Fabric | ") and suffix (e.g. " (topology_mapper.cpp:1571)")
_RE_NODE = re.compile(
    r"Node\s+(\d+)\s+connected to:\s+\[([^\]]*)\]"
)
_RE_HOST = re.compile(
    r"Host_name\s*=\s*(\S+)"
)


def parse_line(line: str) -> Optional[tuple[str, list[str], Optional[str]]]:
    """Parse a line. Returns (node_id, list of neighbor ids, host_name or None)."""
    # Node ID and neighbors (anywhere in line)
    node_match = _RE_NODE.search(line)
    if node_match:
        node_id = node_match.group(1)
        rest = node_match.group(2).strip()
        if rest:
            neighbor_strs = [s.strip() for s in rest.split(",") if s.strip()]
            return (node_id, neighbor_strs, None)
        return (node_id, [], None)

    # Host line: "Host_name = UF-MN-B3-GWH02" (anywhere in line)
    host_match = _RE_HOST.search(line)
    if host_match:
        return ("__host__", [], host_match.group(1).strip())

    return None


def build_graph(
    lines,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Build adjacency graph from parsed lines (iterable, e.g. file handle).
    Returns (adjacency map: node_id -> set of unique neighbor ids, node_id -> host_name).
    """
    adj: dict[str, set[str]] = defaultdict(set)
    host_for_node: dict[str, str] = {}
    last_node: Optional[str] = None

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        node_id, neighbor_list, host_name = parsed
        if node_id == "__host__":
            if last_node is not None and host_name:
                host_for_node[last_node] = host_name
            continue
        last_node = node_id
        for n in neighbor_list:
            if n:
                adj[node_id].add(n)
    return dict(adj), host_for_node


def degree_histogram(adj: dict[str, set[str]]) -> dict[int, int]:
    """Degree = number of unique neighbors. Returns {degree: count}."""
    hist: dict[int, int] = defaultdict(int)
    for neighbors in adj.values():
        hist[len(neighbors)] += 1
    return dict(sorted(hist.items()))


def is_connected(adj: dict[str, set[str]]) -> bool:
    """BFS to check if graph is connected."""
    if not adj:
        return True
    start = next(iter(adj))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, set()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


# Grid directions for 2D mesh: (dr, dc)
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _embed_dfs(
    adj: dict[str, set[str]],
    u: str,
    coords: dict[str, Tuple[int, int]],
    coord_to_node: dict[Tuple[int, int], str],
) -> bool:
    """Backtracking DFS: assign coords for unvisited neighbors and recurse. Returns True if valid."""
    r, c = coords[u]
    neighbors = sorted(adj[u])
    visited_n = [v for v in neighbors if v in coords]
    unvisited_n = [v for v in neighbors if v not in coords]
    for v in visited_n:
        vr, vc = coords[v]
        if abs(r - vr) + abs(c - vc) != 1:
            return False
    if not unvisited_n:
        return True
    available_dirs = [
        (dr, dc) for dr, dc in _DIRS
        if (r + dr, c + dc) not in coord_to_node
    ]
    if len(available_dirs) < len(unvisited_n):
        return False
    # Try each way to assign unvisited nodes to distinct directions (permutations of available_dirs)
    for dir_tuple in permutations(available_dirs, len(unvisited_n)):
        backup: list[Tuple[Tuple[int, int], str]] = []
        for i, v in enumerate(unvisited_n):
            dr, dc = dir_tuple[i]
            nr, nc = r + dr, c + dc
            coords[v] = (nr, nc)
            coord_to_node[(nr, nc)] = v
            backup.append(((nr, nc), v))
        all_ok = True
        for v in unvisited_n:
            if not _embed_dfs(adj, v, coords, coord_to_node):
                all_ok = False
                break
        if all_ok:
            return True
        for (nr, nc), v in backup:
            del coords[v]
            del coord_to_node[(nr, nc)]
    return False


def check_uniform_mesh_dfs(
    adj: dict[str, set[str]]
) -> Tuple[bool, Optional[dict[str, Tuple[int, int]]], str]:
    """
    Use DFS with backtracking to assign (row, col) to each node. Check that every
    edge connects Manhattan-adjacent cells. If so, the graph is a uniform 2D mesh.
    Returns (success, coords_or_none, message).
    """
    if not adj:
        return True, {}, "empty graph"
    if not is_connected(adj):
        return False, None, "graph is not connected"
    # Start from a degree-2 node (corner) if any, else arbitrary
    start = next(iter(adj))
    for node in sorted(adj.keys()):
        if len(adj[node]) == 2:
            start = node
            break
    coords: dict[str, Tuple[int, int]] = {start: (0, 0)}
    coord_to_node: dict[Tuple[int, int], str] = {(0, 0): start}
    if _embed_dfs(adj, start, coords, coord_to_node):
        return True, coords, "DFS embedding: every edge is grid-adjacent (uniform 2D mesh)"
    return False, None, "no valid 2D grid embedding found (DFS backtracking exhausted)"


def infer_mesh_dimensions(hist: dict[int, int], num_nodes: int) -> Optional[tuple[int, int]]:
    """
    If degree distribution matches a 2D rectangular mesh (no wrap), return (rows, cols).
    Mesh: 4 corners (deg 2), 2*(W-2)+2*(H-2) edges (deg 3), (W-2)(H-2) interior (deg 4). N = W*H.
    """
    deg_2 = hist.get(2, 0)
    deg_3 = hist.get(3, 0)
    deg_4 = hist.get(4, 0)
    if set(hist.keys()) - {2, 3, 4}:
        return None
    if deg_2 != 4 or num_nodes != 4 + deg_3 + deg_4:
        return None
    # W + H = (deg_3 + 8) / 2, W*H = N
    sum_wh = (deg_3 + 8) // 2
    if (deg_3 + 8) % 2 != 0:
        return None
    # t^2 - sum_wh*t + num_nodes = 0  =>  t = (sum_wh ± sqrt(sum_wh^2 - 4*N)) / 2
    disc = sum_wh * sum_wh - 4 * num_nodes
    if disc < 0:
        return None
    sqrt_d = int(disc ** 0.5)
    if sqrt_d * sqrt_d != disc:
        return None
    w = (sum_wh + sqrt_d) // 2
    h = (sum_wh - sqrt_d) // 2
    if w * h != num_nodes or w < 2 or h < 2:
        return None
    return (h, w)  # (rows, cols), with rows <= cols by convention


def mesh_shape_report(hist: dict[int, int], num_nodes: int) -> list[str]:
    """
    Heuristic check for 2D rectangular mesh.
    - 2D mesh: degree 2 = corner, 3 = edge, 4 = interior.
    - Torus: all nodes degree 4 (or 3/4 with wraparound).
    """
    report = []
    deg_2 = hist.get(2, 0)
    deg_3 = hist.get(3, 0)
    deg_4 = hist.get(4, 0)
    other_degrees = {d: c for d, c in hist.items() if d not in (2, 3, 4)}

    if not other_degrees:
        report.append("All degrees are 2, 3, or 4 (consistent with 2D mesh/torus).")
        # 2D MxN mesh (no wrap): 4 corners, 2(M-2)+2(N-2) edges, (M-2)(N-2) interior
        if deg_2 == 4 and deg_4 > 0:
            report.append("  -> Looks like a 2D rectangular mesh (4 corners).")
        elif deg_2 == 0 and deg_3 == 0 and deg_4 == num_nodes:
            report.append("  -> All degree 4: torus or closed 2D grid.")
        elif deg_4 > 0 or deg_3 > 0:
            report.append("  -> Mix of degrees 2/3/4: plausible 2D mesh.")
    else:
        report.append(f"Degrees outside 2/3/4: {other_degrees}")
        report.append("  -> Not a standard 2D rectangular mesh (or has extra links).")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze fabric adjacency graph from topology_mapper debug logs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input file (default: stdin)",
    )
    parser.add_argument(
        "--by-host",
        action="store_true",
        help="Also report degree histogram per host",
    )
    args = parser.parse_args()

    with args.input as f:
        adj, host_for_node = build_graph(f)
    if not adj:
        print("No adjacency data found. Expect lines like:")
        print("  Node <id> connected to: [<id>, ...]")
        return 1

    num_nodes = len(adj)
    num_edges = sum(len(ns) for ns in adj.values())
    # Each edge counted from both ends
    unique_edges = num_edges // 2

    hist = degree_histogram(adj)
    connected = is_connected(adj)
    mesh_reports = mesh_shape_report(hist, num_nodes)
    dims = infer_mesh_dimensions(hist, num_nodes)
    uniform_ok, uniform_coords, uniform_msg = check_uniform_mesh_dfs(adj)

    print("=== Adjacency graph summary ===")
    print(f"Mesh size: {num_nodes} nodes")
    if dims is not None:
        print(f"Mesh dimensions (inferred): {dims[0]} x {dims[1]} (rows x cols)")
    print(f"Unique edges (undirected): {unique_edges}")
    print(f"Connected: {'yes' if connected else 'NO'}")
    print()
    print("Uniform mesh check (DFS):")
    if uniform_ok:
        print(f"  Yes — {uniform_msg}")
        if uniform_coords and len(uniform_coords) <= 20:
            print("  Coordinates (row, col) per node:")
            for node in sorted(uniform_coords.keys(), key=lambda n: (uniform_coords[n][0], uniform_coords[n][1])):
                print(f"    {node}: {uniform_coords[node]}")
        elif uniform_coords:
            print(f"  (Coordinates assigned for {len(uniform_coords)} nodes; omit listing for large graphs)")
    else:
        print(f"  No — {uniform_msg}")
    print()
    print("Degree histogram (degree = unique neighbor count):")
    for d, c in sorted(hist.items()):
        print(f"  degree {d}: {c} node(s)")
    print()
    print("Mesh shape check:")
    for line in mesh_reports:
        print(f"  {line}")

    if args.by_host and host_for_node:
        print()
        print("=== Per-host degree histogram (induced subgraph: same-host neighbors only) ===")
        by_host: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for node, neighbors in adj.items():
            host = host_for_node.get(node, "unknown")
            by_host[host][node]  # ensure node is in subgraph
            for n in neighbors:
                if host_for_node.get(n) == host:
                    by_host[host][node].add(n)
        for host in sorted(by_host.keys()):
            host_adj = dict(by_host[host])
            host_hist = degree_histogram(host_adj)
            print(f"  Host {host}: {len(host_adj)} nodes")
            for d, c in sorted(host_hist.items()):
                print(f"    degree {d}: {c} node(s)")

    return 0 if connected else 1


if __name__ == "__main__":
    sys.exit(main())
