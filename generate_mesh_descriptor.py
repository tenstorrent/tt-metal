#!/usr/bin/env python3
"""
Generate Mesh Graph Descriptor from Physical Topology

This tool analyzes a physical cluster descriptor and generates
a corresponding mesh graph descriptor for a given mesh shape.
"""

import yaml
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def load_cluster_descriptor(yaml_path):
    """Load physical cluster descriptor"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def analyze_ethernet_connectivity(cluster_desc):
    """Analyze ethernet connections to determine link counts"""
    conn_count = defaultdict(lambda: defaultdict(int))

    if "ethernet_connections" in cluster_desc:
        for conn in cluster_desc["ethernet_connections"]:
            if len(conn) == 2:
                src = conn[0]["chip"]
                dst = conn[1]["chip"]

                # Count bidirectional links
                key = tuple(sorted([src, dst]))
                conn_count[key]["count"] += 1

    return conn_count


def infer_channels_per_neighbor(cluster_desc, mesh_shape):
    """Infer how many channels per neighbor from physical topology"""
    rows, cols = mesh_shape
    total_chips = rows * cols

    conn_count = analyze_ethernet_connectivity(cluster_desc)

    # Count links per chip
    links_per_chip = defaultdict(int)
    for (src, dst), info in conn_count.items():
        links_per_chip[src] += info["count"]
        links_per_chip[dst] += info["count"]

    if links_per_chip:
        # Average links per chip
        avg_links = sum(links_per_chip.values()) / len(links_per_chip)

        # In a 2D mesh, most chips have 4 neighbors (edge chips have less)
        # Assume bidirectional links
        channels = int(avg_links / 4 + 0.5)  # Round to nearest int
        channels = max(1, min(channels, 4))  # Clamp to reasonable range

        return channels

    return 2  # Default fallback


def detect_arch(cluster_desc):
    """Detect chip architecture"""
    if "arch" in cluster_desc:
        arches = set(cluster_desc["arch"].values())
        if len(arches) == 1:
            arch = list(arches)[0]
            return arch
        else:
            print(f"⚠️  Warning: Multiple architectures detected: {arches}")
            print(f"   Using first one: {list(arches)[0]}")
            return list(arches)[0]

    return "wormhole_b0"  # Default


def detect_board_type(cluster_desc):
    """Detect board type from cluster descriptor"""
    if "chip_to_boardtype" in cluster_desc:
        board_types = set(cluster_desc["chip_to_boardtype"].values())
        if len(board_types) == 1:
            return list(board_types)[0].upper()

    # Infer from chip count
    if "arch" in cluster_desc:
        num_chips = len(cluster_desc["arch"])
        if num_chips == 1:
            return "N150"
        elif num_chips == 2:
            return "N300"
        elif num_chips == 8:
            return "T3K"
        elif num_chips == 32:
            return "GALAXY"

    return "CUSTOM"


def count_ethernet_ports(cluster_desc):
    """Count ethernet ports per direction"""
    # For now, use standard counts based on arch
    arch = detect_arch(cluster_desc)

    if "blackhole" in arch.lower():
        return 2  # Blackhole has 2 per direction
    else:
        return 4  # Wormhole B0 has 4 per direction (but T3K uses 2)

    # Could be enhanced to count actual ports from ethernet_connections


def generate_mesh_graph_descriptor(cluster_desc, mesh_shape, output_path=None):
    """Generate mesh graph descriptor from cluster descriptor and shape"""

    rows, cols = mesh_shape
    arch = detect_arch(cluster_desc)
    board_type = detect_board_type(cluster_desc)
    channels = infer_channels_per_neighbor(cluster_desc, mesh_shape)
    eth_ports = count_ethernet_ports(cluster_desc)

    # For T3K-like configurations, use 2 ports per direction
    if rows * cols == 8 and (rows, cols) == (2, 4):
        eth_ports = 2
        board_type = "T3K"

    print(f"\n📊 Inferred Configuration:")
    print(f"   Architecture: {arch}")
    print(f"   Board Type: {board_type}")
    print(f"   Mesh Shape: {rows}×{cols} ({rows * cols} chips)")
    print(f"   Channels per neighbor: {channels}")
    print(f"   Ethernet ports per direction: {eth_ports}")

    # Generate YAML content
    descriptor = {
        "ChipSpec": {
            "arch": arch,
            "ethernet_ports": {
                "N": eth_ports,
                "E": eth_ports,
                "S": eth_ports,
                "W": eth_ports,
            },
        },
        "Board": [{"name": board_type, "type": "Mesh", "topology": [rows, cols]}],
        "Mesh": [
            {
                "id": 0,
                "board": board_type,
                "device_topology": [rows, cols],
                "host_topology": [1, 1],
            }
        ],
        "Graph": [],
    }

    # Save to file
    if output_path:
        with open(output_path, "w") as f:
            yaml.dump(descriptor, f, default_flow_style=False, sort_keys=False)
        print(f"\n✅ Mesh graph descriptor saved to: {output_path}")

    return descriptor


def generate_comparison_report(cluster_desc, mesh_desc):
    """Generate a comparison report between physical and logical topology"""

    print("\n" + "=" * 60)
    print("PHYSICAL vs LOGICAL TOPOLOGY COMPARISON")
    print("=" * 60)

    # Physical topology info
    num_chips = len(cluster_desc.get("arch", {}))
    num_connections = len(cluster_desc.get("ethernet_connections", []))

    print(f"\n📍 Physical Topology (Cluster Descriptor):")
    print(f"   Total Chips: {num_chips}")
    print(f"   Total Ethernet Links: {num_connections}")
    print(f"   Board Type: {detect_board_type(cluster_desc)}")

    # Logical topology info
    if "Mesh" in mesh_desc and mesh_desc["Mesh"]:
        mesh = mesh_desc["Mesh"][0]
        dev_topo = mesh.get("device_topology", [])
        print(f"\n🔲 Logical Topology (Mesh Graph Descriptor):")
        print(f"   Mesh Shape: {dev_topo[0]}×{dev_topo[1]}")
        print(f"   Total Devices: {dev_topo[0] * dev_topo[1]}")
        print(f"   Board: {mesh.get('board', 'unknown')}")

        # Check consistency
        logical_chips = dev_topo[0] * dev_topo[1]
        if num_chips == logical_chips:
            print(f"\n✅ Chip count matches: {num_chips} chips")
        else:
            print(f"\n⚠️  WARNING: Chip count mismatch!")
            print(f"   Physical: {num_chips} chips")
            print(f"   Logical: {logical_chips} chips")


def extract_physical_connections(cluster_desc):
    """Extract bidirectional physical ethernet connections"""
    connections = set()

    if "ethernet_connections" in cluster_desc:
        for conn in cluster_desc["ethernet_connections"]:
            if len(conn) == 2:
                chip_a = conn[0]["chip"]
                chip_b = conn[1]["chip"]
                connections.add((min(chip_a, chip_b), max(chip_a, chip_b)))

    return connections


def is_valid_mesh(chips, connections, rows, cols, preserve_order=False):
    """Check if chips can form a valid mesh of given shape

    Args:
        chips: tuple/list of chip IDs
        connections: set of (chip_a, chip_b) tuples
        rows: number of rows in mesh
        cols: number of columns in mesh
        preserve_order: if True, use chips in given order; if False, sort them first
    """
    if len(chips) != rows * cols:
        return False

    # Create a mapping of logical position to chip ID
    chip_list = list(chips) if preserve_order else sorted(chips)

    # Check required connectivity for mesh
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            chip = chip_list[idx]

            # Check horizontal connection (to right neighbor)
            if c < cols - 1:
                right_chip = chip_list[idx + 1]
                conn = (min(chip, right_chip), max(chip, right_chip))
                if conn not in connections:
                    return False

            # Check vertical connection (to bottom neighbor)
            if r < rows - 1:
                bottom_chip = chip_list[idx + cols]
                conn = (min(chip, bottom_chip), max(chip, bottom_chip))
                if conn not in connections:
                    return False

    return True


def analyze_mesh_options(cluster_desc, target_shape=None):
    """Analyze and display available mesh configurations"""
    num_chips = len(cluster_desc.get("arch", {}))
    connections = extract_physical_connections(cluster_desc)

    print(f"\n📊 Physical Topology Analysis")
    print(f"=" * 60)
    print(f"Total Chips: {num_chips}")
    print(f"Ethernet Links: {len(connections)}")

    # Display connections
    print(f"\nPhysical Connections:")
    for chip_a, chip_b in sorted(connections):
        print(f"  • Chip {chip_a} ↔ Chip {chip_b}")

    # If target shape specified, analyze only that
    if target_shape:
        rows, cols = target_shape
        print(f"\n🔍 Analyzing {rows}×{cols} Mesh Configurations:")
        print("=" * 60)

        valid_configs = []
        for chips in combinations(range(num_chips), rows * cols):
            if is_valid_mesh(chips, connections, rows, cols):
                valid_configs.append(chips)

        if valid_configs:
            print(f"✓ Found {len(valid_configs)} valid {rows}×{cols} configuration(s):\n")
            for i, chips in enumerate(valid_configs, 1):
                print(f"  Option {i}: Chips {', '.join(map(str, chips))}")
                chip_list = sorted(chips)
                for r in range(rows):
                    row_chips = chip_list[r * cols : (r + 1) * cols]
                    print(f"    Row {r}: {' ↔ '.join(map(str, row_chips))}")
                print()
        else:
            print(f"❌ No valid {rows}×{cols} configurations found!")
            print(f"   The physical connectivity doesn't support this shape.")

        return valid_configs

    # Otherwise, analyze common shapes
    print(f"\n🔍 Available Mesh Configurations:")
    print("=" * 60)

    common_shapes = [(1, 2), (2, 2), (1, 4), (2, 4), (4, 4), (4, 8)]
    found_any = False

    for rows, cols in common_shapes:
        if rows * cols > num_chips:
            continue

        valid_configs = []
        for chips in combinations(range(num_chips), rows * cols):
            if is_valid_mesh(chips, connections, rows, cols):
                valid_configs.append(chips)

        if valid_configs:
            found_any = True
            print(f"\n✓ {rows}×{cols} Mesh: {len(valid_configs)} configuration(s)")
            for i, chips in enumerate(valid_configs[:3], 1):  # Show max 3
                print(f"  Option {i}: Chips {', '.join(map(str, chips))}")
            if len(valid_configs) > 3:
                print(f"  ... and {len(valid_configs) - 3} more")

    if not found_any:
        print("\n⚠️  No standard mesh configurations found.")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mesh Graph Descriptor from Physical Cluster Topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze available mesh configurations
  ./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml --analyze

  # Analyze specific mesh shape
  ./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml --analyze --shape 2,2

  # Generate mesh descriptor for 2x4 mesh
  ./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml --shape 2,4 \\
      --output t3k_mesh_graph.yaml

  # Generate with specific chips
  ./generate_mesh_descriptor.py --cluster t3k_phys_topology.yaml --shape 2,2 \\
      --chips 0,1,4,5 --output t3k_mesh_graph_2x2.yaml

  # Auto-infer shape from chip count
  ./generate_mesh_descriptor.py --cluster my_cluster.yaml --auto \\
      --output my_mesh.yaml
        """,
    )

    parser.add_argument("--cluster", required=True, help="Path to cluster descriptor YAML (physical topology)")
    parser.add_argument("--shape", help="Mesh shape as rows,cols (e.g., 2,4)")
    parser.add_argument("--chips", help="Specific chips to use (e.g., 0,1,4,5)")
    parser.add_argument("--auto", action="store_true", help="Auto-infer shape from chip count")
    parser.add_argument("--analyze", action="store_true", help="Analyze and display available mesh configurations")
    parser.add_argument("--output", help="Output path for mesh graph descriptor YAML")
    parser.add_argument("--compare", action="store_true", help="Generate comparison report")

    args = parser.parse_args()

    # Load cluster descriptor
    print(f"📂 Loading cluster descriptor: {args.cluster}")
    try:
        cluster_desc = load_cluster_descriptor(args.cluster)
        num_chips = len(cluster_desc.get("arch", {}))
        print(f"   Found {num_chips} chips")
    except Exception as e:
        print(f"❌ Error loading cluster descriptor: {e}")
        sys.exit(1)

    # If analyze mode, show available configurations and exit
    if args.analyze:
        target_shape = None
        if args.shape:
            try:
                rows, cols = map(int, args.shape.split(","))
                target_shape = (rows, cols)
            except:
                print(f"❌ Error: Invalid shape format. Use: rows,cols (e.g., 2,2)")
                sys.exit(1)

        analyze_mesh_options(cluster_desc, target_shape)
        sys.exit(0)

    # Require output for generation mode
    if not args.output:
        print(f"❌ Error: --output is required (unless using --analyze)")
        sys.exit(1)

    # Determine mesh shape
    if args.shape:
        try:
            rows, cols = map(int, args.shape.split(","))
            mesh_shape = (rows, cols)
        except:
            print(f"❌ Error: Invalid shape format. Use: rows,cols (e.g., 2,4)")
            sys.exit(1)
    elif args.auto:
        # Auto-infer shape from chip count
        if num_chips == 1:
            mesh_shape = (1, 1)
        elif num_chips == 2:
            mesh_shape = (1, 2)
        elif num_chips == 4:
            mesh_shape = (2, 2)
        elif num_chips == 8:
            mesh_shape = (2, 4)
        elif num_chips == 32:
            mesh_shape = (4, 8)
        else:
            print(f"❌ Error: Cannot auto-infer shape for {num_chips} chips")
            print(f"   Please specify --shape rows,cols")
            sys.exit(1)
        print(f"   Auto-inferred shape: {mesh_shape[0]}×{mesh_shape[1]}")
    else:
        print(f"❌ Error: Must specify either --shape or --auto")
        sys.exit(1)

    # Handle specific chips parameter
    if args.chips:
        try:
            specified_chips = tuple(map(int, args.chips.split(",")))
            required_chips = mesh_shape[0] * mesh_shape[1]

            if len(specified_chips) != required_chips:
                print(
                    f"❌ Error: Specified {len(specified_chips)} chips but {mesh_shape[0]}×{mesh_shape[1]} mesh needs {required_chips}"
                )
                sys.exit(1)

            # Validate connectivity - use chips in the order specified!
            connections = extract_physical_connections(cluster_desc)
            if not is_valid_mesh(specified_chips, connections, mesh_shape[0], mesh_shape[1], preserve_order=True):
                print(f"❌ Error: Chips {args.chips} cannot form a valid {mesh_shape[0]}×{mesh_shape[1]} mesh!")
                print(f"   The physical ethernet connectivity doesn't support this configuration.")
                print(f"   Chips are checked in the order you specified (row-major).")
                print(f"\n💡 Tip: Run with --analyze --shape {mesh_shape[0]},{mesh_shape[1]} to see valid options")
                sys.exit(1)

            print(f"✓ Validated chips {args.chips} for {mesh_shape[0]}×{mesh_shape[1]} mesh")

        except ValueError as e:
            print(f"❌ Error: Invalid chips format. Use comma-separated list (e.g., 0,1,4,5)")
            sys.exit(1)

    # INTELLIGENT VALIDATION: Check if the requested mesh is physically possible
    connections = extract_physical_connections(cluster_desc)

    if not args.chips:
        # If no specific chips provided, check if ANY valid configuration exists
        print(f"\n🔍 Validating {mesh_shape[0]}×{mesh_shape[1]} mesh against physical topology...")

        valid_configs = []
        for chips in combinations(range(num_chips), mesh_shape[0] * mesh_shape[1]):
            if is_valid_mesh(chips, connections, mesh_shape[0], mesh_shape[1]):
                valid_configs.append(chips)

        if not valid_configs:
            print(f"❌ ERROR: No valid {mesh_shape[0]}×{mesh_shape[1]} mesh configuration possible!")
            print(f"   Your physical topology doesn't have the required ethernet connectivity.")
            print(f"\n💡 Available options:")
            print(f"   1. Run: python3 {sys.argv[0]} --cluster {args.cluster} --analyze")
            print(f"      to see all supported mesh configurations")
            print(f"   2. Choose a different mesh shape that's physically supported")
            sys.exit(1)

        if len(valid_configs) == 1:
            print(f"✓ Found 1 valid configuration: Chips {', '.join(map(str, valid_configs[0]))}")
            print(f"  This will be used for the mesh descriptor.")
        else:
            print(f"✓ Found {len(valid_configs)} valid configurations for {mesh_shape[0]}×{mesh_shape[1]} mesh")
            print(f"\n  Available options:")
            for i, chips in enumerate(valid_configs, 1):
                print(f"    Option {i}: Chips {', '.join(map(str, chips))}")

            print(f"\n💡 Tip: Use --chips to specify which configuration to use, e.g.:")
            print(f"   python3 {sys.argv[0]} --cluster {args.cluster} --shape {mesh_shape[0]},{mesh_shape[1]} \\")
            print(f"       --chips {','.join(map(str, valid_configs[0]))} --output {args.output}")
            print(f"\n⚠️  Generating with first option by default: Chips {', '.join(map(str, valid_configs[0]))}")

    # Validate shape
    if mesh_shape[0] * mesh_shape[1] != num_chips:
        if not args.chips:  # Only warn if not using specific chips
            print(f"\n⚠️  Note: Mesh uses {mesh_shape[0] * mesh_shape[1]} chips out of {num_chips} available")

    # Generate mesh graph descriptor
    print(f"\n🔧 Generating mesh graph descriptor...")
    mesh_desc = generate_mesh_graph_descriptor(cluster_desc, mesh_shape, args.output)

    # Generate comparison report if requested
    if args.compare:
        generate_comparison_report(cluster_desc, mesh_desc)

    print(f"\n🎉 Done! Use with:")
    print(f"   export TT_METAL_CLUSTER_DESC_PATH={args.cluster}")
    print(f"   export TT_MESH_GRAPH_DESC_PATH={args.output}")
    print(f"   # Then run your TT-Metal application")


if __name__ == "__main__":
    main()
