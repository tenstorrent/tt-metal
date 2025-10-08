#!/usr/bin/env python3
"""
Script to generate mock cluster descriptors for 4x4 dual mesh setup.

Run this on each host/rank to capture the cluster descriptor:

Rank 0:
    python generate_4x4_dual_mesh_descriptors.py --rank 0 --output galaxy_4x4_dual_mesh_cluster_desc_rank_0.yaml

Rank 1:
    python generate_4x4_dual_mesh_descriptors.py --rank 1 --output galaxy_4x4_dual_mesh_cluster_desc_rank_1.yaml

Or use tt-run with rank bindings to run on both:
    tt-run --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_strict_connection_rank_bindings.yaml \
           python generate_4x4_dual_mesh_descriptors.py
"""

import argparse
import os
import sys

import ttnn


def main():
    parser = argparse.ArgumentParser(description="Generate cluster descriptor for 4x4 dual mesh")
    parser.add_argument("--rank", type=int, default=None, help="Rank ID (0 or 1)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (defaults to galaxy_4x4_dual_mesh_cluster_desc_rank_<rank>.yaml)",
    )
    args = parser.parse_args()

    # Get rank from environment if not provided
    if args.rank is None:
        # Try TTNN_DEVICE_MESH_RANK first
        if "TTNN_DEVICE_MESH_RANK" in os.environ:
            args.rank = int(os.environ["TTNN_DEVICE_MESH_RANK"])
        # Fall back to detecting from TT_VISIBLE_DEVICES
        elif "TT_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["TT_VISIBLE_DEVICES"]
            # If first device is 0-15, it's rank 0; if 16-31, it's rank 1
            first_device = int(visible_devices.split(",")[0])
            args.rank = 0 if first_device < 16 else 1
            print(f"DEBUG: Detected rank {args.rank} from TT_VISIBLE_DEVICES (first device: {first_device})")
        else:
            args.rank = 0

    # Debug: Print all relevant environment variables
    print(f"DEBUG: TTNN_DEVICE_MESH_RANK = {os.environ.get('TTNN_DEVICE_MESH_RANK', 'NOT SET')}")
    print(f"DEBUG: TT_VISIBLE_DEVICES = {os.environ.get('TT_VISIBLE_DEVICES', 'NOT SET')}")

    # Default output filename - put in custom_mock_cluster_descriptors directory
    if args.output is None:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(repo_root, "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors")
        args.output = os.path.join(output_dir, f"galaxy_4x4_dual_mesh_cluster_desc_rank_{args.rank}.yaml")

    print(f"Generating cluster descriptor for Rank {args.rank}")
    print(f"Output file: {args.output}")

    # Serialize the cluster descriptor using ttnn API
    import shutil

    temp_path = ttnn._ttnn.cluster.serialize_cluster_descriptor()

    if not temp_path:
        print("ERROR: Failed to serialize cluster descriptor")
        return 1

    # Copy from temp location to desired output path
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    shutil.copy(temp_path, args.output)

    print(f"Successfully generated cluster descriptor at: {args.output}")

    # Try to get cluster info if available
    try:
        device_ids = ttnn.get_device_ids()
        print(f"\nCluster info:")
        print(f"  Available devices: {len(device_ids)}")
        print(f"  Device IDs: {sorted(device_ids)}")
    except Exception as e:
        print(f"\n(Could not get detailed cluster info: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
