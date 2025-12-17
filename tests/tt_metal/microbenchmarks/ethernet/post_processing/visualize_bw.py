# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
import argparse
import sys


def load_data(topology_file, bandwidth_file):
    """Load and validate the CSV files."""
    try:
        # Load topology data
        topology_df = pd.read_csv(topology_file)
        required_topo_cols = ["MeshRow", "MeshCol", "ConnectedMeshRow", "ConnectedMeshCol", "EthernetChannel"]
        if not all(col in topology_df.columns for col in required_topo_cols):
            raise ValueError(f"Topology file missing required columns: {required_topo_cols}")

        # Load bandwidth data
        bandwidth_df = pd.read_csv(bandwidth_file)
        required_bw_cols = ["MeshRow", "MeshCol", "EthernetChannel", "BandwidthGB_s", "PacketsPerSecond"]
        if not all(col in bandwidth_df.columns for col in required_bw_cols):
            raise ValueError(f"Bandwidth file missing required columns: {required_bw_cols}")

        return topology_df, bandwidth_df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def get_mesh_extent(topology_df):
    """Determine the mesh grid dimensions."""
    all_rows = pd.concat([topology_df["MeshRow"], topology_df["ConnectedMeshRow"]])
    all_cols = pd.concat([topology_df["MeshCol"], topology_df["ConnectedMeshCol"]])

    # min_row, max_row = all_rows.min(), all_rows.max()
    # min_col, max_col = all_cols.min(), all_cols.max()
    min_row, max_row = 0, all_rows.max()
    min_col, max_col = 0, all_cols.max()

    return min_row, max_row, min_col, max_col


def create_bandwidth_lookup(bandwidth_df):
    """Create a lookup dictionary for bandwidth data."""
    lookup = {}
    for _, row in bandwidth_df.iterrows():
        key = (row["MeshRow"], row["MeshCol"], row["EthernetChannel"])
        lookup[key] = {"bandwidth": row["BandwidthGB_s"], "packets": row["PacketsPerSecond"]}
    return lookup


def get_connection_bandwidth(topology_row, bw_lookup):
    """Get bandwidth for a specific connection."""
    key = (topology_row["MeshRow"], topology_row["MeshCol"], topology_row["EthernetChannel"])
    if key in bw_lookup:
        return bw_lookup[key]["bandwidth"]
    return None


def visualize_mesh_topology(topology_df, bandwidth_df, output_file=None):
    """Create the main visualization."""
    # Get mesh dimensions
    min_row, max_row, min_col, max_col = get_mesh_extent(topology_df)

    # Create bandwidth lookup
    bw_lookup = create_bandwidth_lookup(bandwidth_df)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create grid
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1

    # Draw mesh nodes
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            # Position on plot (flip y-axis for intuitive row/col display)
            x, y = c, max_row - r

            # Draw node as a square
            rect = FancyBboxPatch(
                (x - 0.3, y - 0.3),
                0.6,
                0.6,
                boxstyle="round,pad=0.05",
                facecolor="lightblue",
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Add node label
            ax.text(x, y, f"({r},{c})", ha="center", va="center", fontsize=8, fontweight="bold")

    # Process connections and collect bandwidth data
    connections = []
    bandwidths = []

    # Group connections by source-destination pairs to calculate offsets for multiple channels
    connection_groups = {}

    for _, row in topology_df.iterrows():
        src_r, src_c = row["MeshRow"], row["MeshCol"]
        dst_r, dst_c = row["ConnectedMeshRow"], row["ConnectedMeshCol"]
        channel = row["EthernetChannel"]

        # Get bandwidth
        bandwidth = get_connection_bandwidth(row, bw_lookup)

        # Group key for counting multiple channels between same nodes (directional)
        key = (src_r, src_c, dst_r, dst_c)
        if key not in connection_groups:
            connection_groups[key] = []

        connection_groups[key].append({"channel": channel, "bandwidth": bandwidth, "row_data": row})

    # Detect bidirectional connections for offset calculation
    bidirectional_pairs = set()
    for src_r, src_c, dst_r, dst_c in connection_groups.keys():
        reverse_key = (dst_r, dst_c, src_r, src_c)
        if reverse_key in connection_groups:
            # Create a normalized pair key (smaller coordinates first)
            pair_key = tuple(sorted([(src_r, src_c), (dst_r, dst_c)]))
            bidirectional_pairs.add(pair_key)

    # Draw individual connections with offsets for multiple channels
    for (src_r, src_c, dst_r, dst_c), channels in connection_groups.items():
        # Calculate base positions (flip y for display)
        x1, y1 = src_c, max_row - src_r
        x2, y2 = dst_c, max_row - dst_r

        # Check if this connection is bidirectional
        pair_key = tuple(sorted([(src_r, src_c), (dst_r, dst_c)]))
        is_bidirectional = pair_key in bidirectional_pairs

        # Calculate line direction vector
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2) if dx != 0 or dy != 0 else 1

        # Perpendicular unit vector for offsets
        perp_x, perp_y = -dy / length, dx / length

        # Apply bidirectional offset if needed
        bidirectional_offset = 0
        if is_bidirectional:
            # Determine which direction this is (use coordinate comparison for consistency)
            is_forward = (src_r, src_c) < (dst_r, dst_c)
            bidirectional_offset = 0.02 if is_forward else -0.08

        # Calculate offsets for multiple channels within the same direction
        num_channels = len(channels)
        # Always apply channel spacing to avoid overlaps
        channel_spacing = 0.04
        # Start from a base offset to ensure separation from the center line
        channel_base_offset = 0.02

        # Draw each channel as a separate line
        for i, ch in enumerate(channels):
            # Calculate offset positions for multiple channels
            if ch["bandwidth"] is None:
                continue

            # Combine bidirectional offset with multi-channel offset
            total_offset = bidirectional_offset

            # Apply channel offset (starts from base offset to avoid center line)
            channel_offset = channel_base_offset + (i * channel_spacing)
            # If bidirectional, apply channel offset in the appropriate direction
            if is_bidirectional:
                # Forward direction: positive offset, Backward direction: negative offset
                is_forward = (src_r, src_c) < (dst_r, dst_c)
                channel_offset = channel_offset if is_forward else -channel_offset

            total_offset += channel_offset

            # Apply the combined offset
            line_x1 = x1 + total_offset * perp_x
            line_y1 = y1 + total_offset * perp_y
            line_x2 = x2 + total_offset * perp_x
            line_y2 = y2 + total_offset * perp_y

            # Store connection for line collection
            connections.append([(line_x1, line_y1), (line_x2, line_y2)])

            # Store bandwidth for heatmap coloring
            bandwidths.append(ch["bandwidth"])

            # Add direction arrow for bidirectional connections
            if is_bidirectional:
                # Calculate arrow position (closer to destination)
                arrow_ratio = 0.75
                arrow_x = line_x1 + arrow_ratio * (line_x2 - line_x1)
                arrow_y = line_y1 + arrow_ratio * (line_y2 - line_y1)

                # Arrow direction
                arrow_dx = (line_x2 - line_x1) * 0.1
                arrow_dy = (line_y2 - line_y1) * 0.1

                ax.annotate(
                    "",
                    xy=(arrow_x + arrow_dx, arrow_y + arrow_dy),
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                )

            # Create individual annotation for this channel
            label = f"Ch{ch['channel']}: {ch['bandwidth']:.2f}GB/s"

            # Position annotation at midpoint of this specific line
            mid_x = (line_x1 + line_x2) / 2
            mid_y = (line_y1 + line_y2) / 2

            # Check if this is a vertical or horizontal line for special handling
            is_vertical = abs(x2 - x1) < 0.1
            is_horizontal = abs(y2 - y1) < 0.1

            offset_x = 0
            offset_y = 0
            rotation = 0
            # Calculate annotation positioning and rotation
            if is_vertical:
                # For vertical lines, spread annotations vertically to avoid overlap
                vertical_spacing = 0.4  # Increased spacing for better readability
                # Center the annotations around the midpoint
                total_vertical_span = (num_channels - 1) * vertical_spacing
                vertical_offset = (i * vertical_spacing) - (total_vertical_span / 2)

                # Add horizontal offset based on direction for bidirectional connections
                # if is_bidirectional:
                is_forward = (src_r, src_c) < (dst_r, dst_c)
                horizontal_offset = -0.05 if is_forward else 0.05
                # else:
                #     horizontal_offset = 0.2

                offset_x = 0  # horizontal_offset
                offset_y = vertical_offset + horizontal_offset
            else:
                offset_y = i * 0.1

            ax.annotate(
                label,
                (mid_x + offset_x, mid_y + offset_y),
                fontsize=7,
                ha="center",
                va="center",
                rotation=rotation,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    # Apply heatmap coloring to connections
    if bandwidths and max(bandwidths) > 0:
        # Normalize bandwidths for colormap
        norm_bw = np.array(bandwidths)
        norm_bw = norm_bw / max(norm_bw) if max(norm_bw) > 0 else norm_bw

        # Create line collection with heatmap colors
        lc = LineCollection(connections, linewidths=2, cmap="viridis", alpha=0.7)
        lc.set_array(norm_bw)
        line_collection = ax.add_collection(lc)

        # Add colorbar
        cbar = plt.colorbar(line_collection, ax=ax, shrink=0.8)
        cbar.set_label("Normalized Bandwidth (GB/s)", rotation=270, labelpad=15)
    else:
        # Draw connections without heatmap if no bandwidth data
        for connection in connections:
            ax.plot(
                [connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], "k-", linewidth=2, alpha=0.5
            )

    # Set plot properties
    ax.set_xlim(min_col - 0.5, max_col + 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Mesh Column")
    ax.set_ylabel("Mesh Row (flipped for display)")
    ax.set_title("Mesh Topology with Bandwidth Visualization")

    # Set integer ticks
    ax.set_xticks(range(min_col, max_col + 1))
    ax.set_yticks(range(rows))
    ax.set_yticklabels(range(max_row, min_row - 1, -1))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_file}")

    plt.show()


def print_summary_stats(topology_df, bandwidth_df):
    """Print summary statistics about the data."""
    print("\n=== Data Summary ===")
    print(f"Topology connections: {len(topology_df)}")
    print(f"Bandwidth measurements: {len(bandwidth_df)}")

    # Bandwidth stats
    if not bandwidth_df.empty:
        bw_stats = bandwidth_df["BandwidthGB_s"].describe()
        print(f"\nBandwidth Statistics (GB/s):")
        print(f"  Min: {bw_stats['min']:.2f}")
        print(f"  Max: {bw_stats['max']:.2f}")
        print(f"  Mean: {bw_stats['mean']:.2f}")
        print(f"  Std: {bw_stats['std']:.2f}")

    # Coverage analysis
    bw_lookup = create_bandwidth_lookup(bandwidth_df)
    covered_connections = 0
    total_connections = len(topology_df)

    for _, row in topology_df.iterrows():
        key = (row["MeshRow"], row["MeshCol"], row["EthernetChannel"])
        if key in bw_lookup:
            covered_connections += 1

    coverage_pct = (covered_connections / total_connections) * 100 if total_connections > 0 else 0
    print(f"\nBandwidth Coverage: {covered_connections}/{total_connections} ({coverage_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize mesh topology with bandwidth data")
    parser.add_argument("topology_file", help="CSV file with mesh topology connections")
    parser.add_argument("bandwidth_file", help="CSV file with bandwidth measurements")
    parser.add_argument("-o", "--output", help="Output file for saving the plot (optional)")

    args = parser.parse_args()

    # Load data
    topology_df, bandwidth_df = load_data(args.topology_file, args.bandwidth_file)

    # Print summary
    print_summary_stats(topology_df, bandwidth_df)

    # Create visualization
    visualize_mesh_topology(topology_df, bandwidth_df, args.output)


if __name__ == "__main__":
    main()
