#!/usr/bin/env python3
"""
Script to identify and report on clusters in a cabling_descriptor.textproto file.
A cluster is defined here as a group of nodes that are connected through one or more connections.

USAGE:
    Basic usage (human-readable output):
        python3 cabling_descriptor_analysis.py <path_to_cabling_descriptor.textproto>

    Verbose output (shows all connections):
        python3 cabling_descriptor_analysis.py -v <path_to_cabling_descriptor.textproto>
        python3 cabling_descriptor_analysis.py --verbose <path_to_cabling_descriptor.textproto>

    JSON output (for use in other scripts):
        python3 cabling_descriptor_analysis.py --json <path_to_cabling_descriptor.textproto>

    JSON format:
        {
          "total_clusters": 5,
          "total_hosts": 20,
          "total_connections": 240,
          "clusters": [
            {
              "cluster_id": 1,
              "num_hosts": 4,
              "num_connections": 48,
              "hostnames": ["node1", "node2", "node3", "node4"]
            },
            ...
          ]
        }
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Set, Dict, List, Tuple


class UnionFind:
    """Union-Find data structure for finding connected components."""

    def __init__(self, nodes: List[str]):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node: str) -> str:
        """Find the root parent of a node with path compression."""
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: str, node2: str) -> None:
        """Union two nodes by rank."""
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def get_components(self) -> Dict[str, Set[str]]:
        """Get all connected components grouped by their root."""
        components = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            components[root].add(node)
        return components


def parse_cabling_descriptor(filepath: str) -> Tuple[Set[str], List[Tuple[str, str]], List[Dict]]:
    """
    Parse a cabling_descriptor.textproto file and extract nodes and connections.

    Returns:
        Tuple of (set of node names, list of connection tuples, list of detailed connections)
    """
    nodes = set()
    connections = []
    detailed_connections = []

    with open(filepath, "r") as f:
        content = f.read()

    # Extract node names from children blocks
    children_pattern = r'children\s*\{\s*name:\s*"([^"]+)"'
    for match in re.finditer(children_pattern, content):
        node_name = match.group(1)
        nodes.add(node_name)

    # Extract connections from internal_connections with full details
    # Pattern to match: connections { port_a { path: "node" tray_id: X port_id: Y } port_b { path: "node" tray_id: X port_id: Y } }
    connection_pattern = r'connections\s*\{\s*port_a\s*\{\s*path:\s*"([^"]+)"\s*tray_id:\s*(\d+)\s*port_id:\s*(\d+)\s*\}\s*port_b\s*\{\s*path:\s*"([^"]+)"\s*tray_id:\s*(\d+)\s*port_id:\s*(\d+)\s*\}\s*\}'

    for match in re.finditer(connection_pattern, content):
        node_a = match.group(1)
        tray_a = int(match.group(2))
        port_a = int(match.group(3))
        node_b = match.group(4)
        tray_b = int(match.group(5))
        port_b = int(match.group(6))

        connections.append((node_a, node_b))
        detailed_connections.append(
            {"node_a": node_a, "tray_a": tray_a, "port_a": port_a, "node_b": node_b, "tray_b": tray_b, "port_b": port_b}
        )

    return nodes, connections, detailed_connections


def analyze_clusters(
    nodes: Set[str], connections: List[Tuple[str, str]], detailed_connections: List[Dict]
) -> List[Dict]:
    """
    Analyze the graph to find clusters and their properties.

    Args:
        nodes: Set of all node names
        connections: List of (node_a, node_b) tuples
        detailed_connections: List of dicts with full connection details (tray_id, port_id)

    Returns:
        List of dictionaries containing cluster information
    """
    if not nodes:
        return []

    # Build Union-Find structure
    uf = UnionFind(list(nodes))

    # Union all connected nodes
    for node_a, node_b in connections:
        if node_a in nodes and node_b in nodes:
            uf.union(node_a, node_b)

    # Get connected components
    components = uf.get_components()

    # Build cluster information
    clusters = []
    for cluster_nodes in components.values():
        # Count connections within this cluster
        cluster_connections = []
        cluster_detailed_connections = []
        for idx, (node_a, node_b) in enumerate(connections):
            if node_a in cluster_nodes and node_b in cluster_nodes:
                cluster_connections.append((node_a, node_b))
                cluster_detailed_connections.append(detailed_connections[idx])

        cluster_info = {
            "nodes": sorted(cluster_nodes),
            "num_hosts": len(cluster_nodes),
            "num_connections": len(cluster_connections),
            "connections": cluster_connections,
            "detailed_connections": cluster_detailed_connections,
        }
        clusters.append(cluster_info)

    # Sort clusters by size (descending) for better readability
    clusters.sort(key=lambda x: x["num_hosts"], reverse=True)

    return clusters


def print_cluster_report(clusters: List[Dict], verbose: bool = False) -> None:
    """Print a formatted report of the clusters."""

    print(f"\n{'='*80}")
    print(f"CLUSTER ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"\nTotal clusters found: {len(clusters)}")
    print(f"Total hosts across all clusters: {sum(c['num_hosts'] for c in clusters)}")
    print(f"Total connections: {sum(c['num_connections'] for c in clusters)}")

    for idx, cluster in enumerate(clusters, 1):
        # Determine cluster type for descriptive naming
        num_hosts = cluster["num_hosts"]

        if num_hosts == 4:
            cluster_type = "QUAD"
        elif num_hosts % 4 == 0:
            # Multiple of 4 nodes = multiple QUADs, call it SC#
            num_quads = num_hosts // 4
            cluster_type = f"SC{num_quads}"
        else:
            cluster_type = "CLUSTER"

        print(f"\n{'-'*80}")
        print(f"{cluster_type} #{idx}")
        print(f"{'-'*80}")
        print(f"Number of hosts: {cluster['num_hosts']}")
        print(f"Number of connections: {cluster['num_connections']}")
        print(f"\nHostnames:")
        for node in cluster["nodes"]:
            print(f"  - {node}")

        # Count connections between each pair of nodes
        connection_counts = defaultdict(int)
        for node_a, node_b in cluster["connections"]:
            # Normalize the pair so (A, B) and (B, A) are treated the same
            pair = tuple(sorted([node_a, node_b]))
            connection_counts[pair] += 1

        # Display connection counts between nodes
        if connection_counts:
            print(f"\nConnection counts between nodes:")
            # Sort by node pairs for consistent output
            for (node_a, node_b), count in sorted(connection_counts.items()):
                print(f"  {node_a} <-> {node_b}: {count} connection(s)")

        if verbose and cluster["detailed_connections"]:
            print(f"\nDetailed connections:")
            for conn in cluster["detailed_connections"]:
                print(
                    f"  {conn['node_a']} [tray {conn['tray_a']}, port {conn['port_a']}] <-> {conn['node_b']} [tray {conn['tray_b']}, port {conn['port_b']}]"
                )

    print(f"\n{'='*80}\n")


def print_json_output(clusters: List[Dict]) -> None:
    """Print cluster information in JSON format for programmatic consumption."""
    output = {
        "total_clusters": len(clusters),
        "total_hosts": sum(c["num_hosts"] for c in clusters),
        "total_connections": sum(c["num_connections"] for c in clusters),
        "clusters": [],
    }

    for idx, cluster in enumerate(clusters, 1):
        cluster_info = {
            "cluster_id": idx,
            "num_hosts": cluster["num_hosts"],
            "num_connections": cluster["num_connections"],
            "hostnames": cluster["nodes"],
        }
        output["clusters"].append(cluster_info)

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cabling_descriptor.textproto to identify clusters of connected nodes.",
        epilog="""
examples:
  # Basic human-readable output
  %(prog)s /path/to/cabling_descriptor.textproto

  # Verbose output with connection details
  %(prog)s -v /path/to/cabling_descriptor.textproto

  # JSON output for use in other scripts
  %(prog)s --json /path/to/cabling_descriptor.textproto

  # Use with jq to extract hostnames
  %(prog)s --json file.textproto | jq '.clusters[].hostnames'
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("filepath", help="Path to the cabling_descriptor.textproto file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed connection information for each cluster"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format for programmatic consumption"
    )

    args = parser.parse_args()

    try:
        # Parse the file
        if not args.json:
            print(f"Reading cabling descriptor from: {args.filepath}")
        nodes, connections, detailed_connections = parse_cabling_descriptor(args.filepath)

        if not args.json:
            print(f"Found {len(nodes)} nodes and {len(connections)} connections")

        # Analyze clusters
        clusters = analyze_clusters(nodes, connections, detailed_connections)

        # Print report in requested format
        if args.json:
            print_json_output(clusters)
        else:
            print_cluster_report(clusters, verbose=args.verbose)

    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
