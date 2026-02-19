#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Parse a cluster cabling descriptor (protobuf text format) and build
a graph of physical connections between hosts.

The input format follows the ClusterDescriptor proto schema defined in
tools/scaleout/cabling_descriptor/schemas/cluster_config.proto.

graph_templates maps template names to GraphTemplate messages, each of which
contains:
  - children: list of ChildInstance (node_ref -> leaf host, or graph_ref ->
    nested template)
  - internal_connections: map from connection type (e.g. "QSFP_DD") to
    PortConnections, listing Port-to-Port cables

Port.path is a repeated string giving the hierarchical address of a node
(e.g. ["hostname"] for a flat topology, or ["rack", "node"] for nested).

Usage:
    python parse_cluster_config.py <input_file> [options]

    --summary              Print hosts and all connections (default)
    --neighbors HOST       List hosts connected to HOST with link counts
    --connections A B      List every cable between host A and host B
    --adjacency-matrix     Print N x N connection-count matrix
    --dot                  Emit a Graphviz DOT file to stdout
"""

import re
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Port:
    """A physical port addressed by hierarchical path + tray/port IDs.

    path is a tuple of strings matching proto's `repeated string path`.
    For flat topologies this is a 1-tuple, e.g. ("bh-glx-b02u02",).
    For hierarchical setups it traces the template nesting, e.g.
    ("superpod1", "pod2", "node1").
    """

    path: tuple
    tray_id: int
    port_id: int

    @property
    def host(self) -> str:
        """Leaf node name – the last element of the path."""
        return self.path[-1] if self.path else ""

    @property
    def path_str(self) -> str:
        return "/".join(self.path)

    def __str__(self) -> str:
        return f"{self.path_str}[tray={self.tray_id}, port={self.port_id}]"

    def __hash__(self):
        return hash((self.path, self.tray_id, self.port_id))

    def __eq__(self, other):
        return (self.path, self.tray_id, self.port_id) == (other.path, other.tray_id, other.port_id)


@dataclass
class Connection:
    """One physical cable between port_a and port_b."""

    port_a: Port
    port_b: Port
    connection_type: str  # e.g. "QSFP_DD"

    def other_end(self, host_path: tuple) -> Optional[Port]:
        """Return the port on the far end as seen from host_path."""
        if self.port_a.path == host_path:
            return self.port_b
        if self.port_b.path == host_path:
            return self.port_a
        return None

    def __str__(self) -> str:
        return f"[{self.connection_type}] {self.port_a} <-> {self.port_b}"


@dataclass
class Host:
    """A leaf node (physical machine) defined as a child in a graph template."""

    name: str
    node_descriptor: str  # e.g. "BH_GALAXY"

    def __str__(self) -> str:
        return f"{self.name} ({self.node_descriptor})"


class ClusterTopology:
    """
    Physical cabling topology parsed from a ClusterDescriptor textproto.

    Public attributes
    -----------------
    hosts        dict[str, Host]            host name -> Host
    connections  list[Connection]           all cables in the file
    """

    def __init__(self):
        self.hosts: dict = {}
        self.connections: list = []
        # Keyed by full path tuple for hierarchical support.
        self._adjacency: dict = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutation helpers (used by the parser)
    # ------------------------------------------------------------------

    def add_host(self, host: Host) -> None:
        self.hosts[host.name] = host

    def add_connection(self, conn: Connection) -> None:
        self.connections.append(conn)
        self._adjacency[conn.port_a.path].append(conn)
        if conn.port_b.path != conn.port_a.path:
            self._adjacency[conn.port_b.path].append(conn)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def _resolve_path(self, name: str) -> Optional[tuple]:
        """Convert a host name or slash-separated path string to a path tuple."""
        if name in self.hosts:
            return (name,)
        parts = tuple(name.split("/"))
        if parts in self._adjacency:
            return parts
        return None

    def get_connections_for(self, host: str) -> list:
        """All cables that involve *host*."""
        path = self._resolve_path(host)
        return list(self._adjacency.get(path, [])) if path else []

    def get_connections_between(self, host_a: str, host_b: str) -> list:
        """All cables that run between host_a and host_b."""
        path_a = self._resolve_path(host_a)
        path_b = self._resolve_path(host_b)
        if path_a is None or path_b is None:
            return []
        return [c for c in self._adjacency.get(path_a, []) if c.port_a.path == path_b or c.port_b.path == path_b]

    def get_neighbors(self, host: str) -> dict:
        """Return {neighbor_name: link_count} for all directly connected hosts."""
        path = self._resolve_path(host)
        if path is None:
            return {}
        counts: dict = defaultdict(int)
        for conn in self._adjacency.get(path, []):
            other = conn.other_end(path)
            if other is not None:
                counts["/".join(other.path)] += 1
        return dict(counts)

    def adjacency_matrix(self) -> tuple:
        """Return (hosts, matrix) where matrix[i][j] is the link count."""
        hosts = sorted(self.hosts)
        idx = {h: i for i, h in enumerate(hosts)}
        n = len(hosts)
        mat = [[0] * n for _ in range(n)]
        for conn in self.connections:
            ha = conn.port_a.host
            hb = conn.port_b.host
            if ha in idx and hb in idx and ha != hb:
                mat[idx[ha]][idx[hb]] += 1
                mat[idx[hb]][idx[ha]] += 1
        return hosts, mat

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"Hosts ({len(self.hosts)}):"]
        for h in self.hosts.values():
            lines.append(f"  {h}")
        lines.append(f"\nConnections ({len(self.connections)}):")
        for conn in self.connections:
            lines.append(f"  {conn}")
        lines.append("\nAdjacency:")
        for name in sorted(self.hosts):
            neighbors = self.get_neighbors(name)
            if neighbors:
                parts = ", ".join(f"{n} x{c}" for n, c in sorted(neighbors.items()))
                lines.append(f"  {name} -> {parts}")
            else:
                lines.append(f"  {name} -> (no outgoing connections)")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Return a Graphviz DOT graph of the host-level topology."""
        from collections import Counter

        lines = [
            "graph cluster_topology {",
            "    rankdir=LR;",
            '    node [shape=box fontname="monospace"];',
        ]
        for name, host in sorted(self.hosts.items()):
            label = f"{name}\\n{host.node_descriptor}"
            lines.append(f'    "{name}" [label="{label}"];')

        # One edge per unordered host pair, with link count as label.
        pair_conns: dict = defaultdict(list)
        for conn in self.connections:
            ha, hb = conn.port_a.host, conn.port_b.host
            if ha == hb:
                continue
            pair_conns[frozenset([ha, hb])].append(conn)

        for pair, conns in sorted(pair_conns.items(), key=lambda x: sorted(x[0])):
            ha, hb = sorted(pair)
            type_counts = Counter(c.connection_type for c in conns)
            label = ", ".join(f"{t} x{n}" for t, n in sorted(type_counts.items()))
            lines.append(f'    "{ha}" -- "{hb}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def print_adjacency_matrix(self) -> None:
        hosts, mat = self.adjacency_matrix()
        if not hosts:
            print("(no hosts)")
            return
        col_w = max(len(h) for h in hosts)
        header = " " * (col_w + 2) + "  ".join(h.ljust(4)[:4] for h in hosts)
        print(header)
        for i, ha in enumerate(hosts):
            row = f"{ha:<{col_w}}  "
            row += "  ".join(str(mat[i][j]).center(4) for j in range(len(hosts)))
            print(row)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _extract_block(content: str, brace_pos: int) -> tuple:
    """Return (inner_content, end_pos) for the {} block whose '{' is at brace_pos."""
    assert content[brace_pos] == "{"
    depth = 0
    j = brace_pos
    while j < len(content):
        if content[j] == "{":
            depth += 1
        elif content[j] == "}":
            depth -= 1
            if depth == 0:
                return content[brace_pos + 1 : j], j
        j += 1
    raise ValueError(f"Unmatched '{{' at offset {brace_pos}")


def _find_blocks(content: str, keyword: str) -> list:
    """Return a list of inner-content strings for every `keyword { ... }` block."""
    blocks = []
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\s*\{")
    for m in pattern.finditer(content):
        inner, _ = _extract_block(content, m.end() - 1)
        blocks.append(inner)
    return blocks


def _str_field(content: str, field_name: str) -> Optional[str]:
    m = re.search(r"\b" + re.escape(field_name) + r'\s*:\s*"([^"]*)"', content)
    return m.group(1) if m else None


def _all_str_fields(content: str, field_name: str) -> list:
    """Extract all string values for a repeated field, in file order.

    Handles both proto text formats:
      field: "value"           # single scalar form
      field: ["v1", "v2"]      # array / repeated form
    """
    results = []
    base = r"\b" + re.escape(field_name) + r"\s*:\s*"
    for m in re.finditer(base + r'("(?:[^"\\]|\\.)*"|\[[^\]]*\])', content):
        chunk = m.group(1)
        if chunk.startswith("["):
            # Repeated array syntax: extract every quoted element inside
            results.extend(re.findall(r'"([^"]*)"', chunk))
        else:
            # Single quoted value
            results.append(chunk[1:-1])
    return results


def _int_field(content: str, field_name: str, default: int = 0) -> int:
    m = re.search(r"\b" + re.escape(field_name) + r"\s*:\s*(\d+)", content)
    return int(m.group(1)) if m else default


def _parse_port(block: str) -> Port:
    path = tuple(_all_str_fields(block, "path"))
    tray_id = _int_field(block, "tray_id")
    port_id = _int_field(block, "port_id")
    return Port(path=path, tray_id=tray_id, port_id=port_id)


def parse_file(filepath: str) -> ClusterTopology:
    """
    Parse a ClusterDescriptor textproto file and return a ClusterTopology.

    The parser walks graph_templates blocks, extracts children (hosts) and
    internal_connections (cables), and builds the adjacency graph.
    """
    with open(filepath) as f:
        content = f.read()

    topology = ClusterTopology()

    for gt_block in _find_blocks(content, "graph_templates"):
        for value_block in _find_blocks(gt_block, "value"):
            # --- children (host definitions) ---
            for child_block in _find_blocks(value_block, "children"):
                name = _str_field(child_block, "name")
                descriptor = None
                for node_ref_block in _find_blocks(child_block, "node_ref"):
                    descriptor = _str_field(node_ref_block, "node_descriptor")
                if name:
                    topology.add_host(Host(name=name, node_descriptor=descriptor or ""))

            # --- internal_connections ---
            for ic_block in _find_blocks(value_block, "internal_connections"):
                conn_type = _str_field(ic_block, "key") or "UNKNOWN"

                for ic_value in _find_blocks(ic_block, "value"):
                    for conn_block in _find_blocks(ic_value, "connections"):
                        port_a_blocks = _find_blocks(conn_block, "port_a")
                        port_b_blocks = _find_blocks(conn_block, "port_b")
                        if port_a_blocks and port_b_blocks:
                            topology.add_connection(
                                Connection(
                                    port_a=_parse_port(port_a_blocks[0]),
                                    port_b=_parse_port(port_b_blocks[0]),
                                    connection_type=conn_type,
                                )
                            )

    return topology


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse a ClusterDescriptor textproto and inspect the cabling topology."
    )
    parser.add_argument("input_file", help="Path to the cabling descriptor file")
    parser.add_argument("--summary", action="store_true", help="Print full topology summary (default)")
    parser.add_argument("--neighbors", metavar="HOST", help="List hosts connected to HOST with link counts")
    parser.add_argument(
        "--connections",
        nargs=2,
        metavar=("HOST_A", "HOST_B"),
        help="List all cables between HOST_A and HOST_B",
    )
    parser.add_argument("--adjacency-matrix", action="store_true", help="Print host-to-host link count matrix")
    parser.add_argument("--dot", action="store_true", help="Output a Graphviz DOT representation")
    args = parser.parse_args()

    topology = parse_file(args.input_file)

    any_explicit = args.neighbors or args.connections or args.adjacency_matrix or args.dot
    if args.summary or not any_explicit:
        print(topology.summary())

    if args.neighbors:
        neighbors = topology.get_neighbors(args.neighbors)
        if not neighbors:
            print(f"No connections found for host '{args.neighbors}'", file=sys.stderr)
        else:
            print(f"Neighbors of {args.neighbors}:")
            for name, count in sorted(neighbors.items()):
                print(f"  {name}  ({count} link{'s' if count != 1 else ''})")

    if args.connections:
        ha, hb = args.connections
        conns = topology.get_connections_between(ha, hb)
        if not conns:
            print(f"No connections found between '{ha}' and '{hb}'", file=sys.stderr)
        else:
            print(f"Connections between {ha} and {hb} ({len(conns)} total):")
            for c in conns:
                print(f"  {c}")

    if args.adjacency_matrix:
        print("\nAdjacency matrix (link counts):")
        topology.print_adjacency_matrix()

    if args.dot:
        print(topology.to_dot())


if __name__ == "__main__":
    main()
