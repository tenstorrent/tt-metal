#!/usr/bin/env python3
"""
Create a test copy of ttnn_operations_master.json with mock placement values.

This script:
1. Reads the original JSON
2. Adds mock placement values to existing tensor_placements
3. For some configs, adds additional mesh variants (2x4, 4x8)
4. Saves to a new file for testing the database round-trip

Usage:
    python create_test_master_json.py
"""

import json
import random
import copy
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "ttnn_operations_master.json"
OUTPUT_FILE = Path(__file__).parent / "ttnn_operations_master_test.json"

# Mock placement configurations to add
MOCK_MESH_CONFIGS = [
    {"mesh_device_shape": "[1, 1]", "placement": "[PlacementReplicate]", "distribution_shape": "[1]"},
    {"mesh_device_shape": "[2, 4]", "placement": "[PlacementShard(0)]", "distribution_shape": "[8]"},
    {"mesh_device_shape": "[2, 4]", "placement": "[PlacementShard(1)]", "distribution_shape": "[8]"},
    {"mesh_device_shape": "[2, 4]", "placement": "[PlacementShard(2)]", "distribution_shape": "[8]"},
    {"mesh_device_shape": "[2, 4]", "placement": "[PlacementShard(3)]", "distribution_shape": "[8]"},
    {"mesh_device_shape": "[2, 4]", "placement": "[PlacementReplicate]", "distribution_shape": "[8]"},
    {"mesh_device_shape": "[4, 8]", "placement": "[PlacementShard(0)]", "distribution_shape": "[32]"},
    {"mesh_device_shape": "[4, 8]", "placement": "[PlacementShard(3)]", "distribution_shape": "[32]"},
    {"mesh_device_shape": "[4, 8]", "placement": "[PlacementReplicate]", "distribution_shape": "[32]"},
]


def enrich_tensor_placement(placement):
    """Add mock placement and distribution_shape to an existing tensor_placement."""
    mesh_shape = placement.get("mesh_device_shape", "[1, 1]")

    # Parse mesh shape to determine appropriate mock values
    try:
        # Handle string format "[2, 4]"
        if isinstance(mesh_shape, str):
            dims = [int(x.strip()) for x in mesh_shape.strip("[]").split(",")]
        else:
            dims = mesh_shape

        total_devices = dims[0] * dims[1] if len(dims) == 2 else 1
    except:
        total_devices = 1

    # Add placement if not present
    if "placement" not in placement:
        # Randomly choose shard or replicate
        if random.random() < 0.7:  # 70% shard, 30% replicate
            shard_dim = random.choice([0, 1, 2, 3])
            placement["placement"] = f"[PlacementShard({shard_dim})]"
        else:
            placement["placement"] = "[PlacementReplicate]"

    # Add distribution_shape if not present
    if "distribution_shape" not in placement:
        placement["distribution_shape"] = f"[{total_devices}]"

    return placement


def add_mesh_variants(config, op_name):
    """Add additional mesh variant configs for testing.

    Returns a list of configs: the original plus any new mesh variants.
    """
    configs = [config]

    # Only add variants to some configs (20% chance for multi-device)
    if random.random() > 0.2:
        return configs

    machine_info = config.get("machine_info", [{}])
    if not machine_info:
        return configs

    base_machine = machine_info[0] if machine_info else {}

    # Check if this config already has multi-device tensor_placements
    existing_placements = base_machine.get("tensor_placements", [])
    existing_meshes = set()
    for p in existing_placements:
        mesh = p.get("mesh_device_shape", "[1, 1]")
        existing_meshes.add(mesh)

    # Add 2x4 variant if not already present
    if "[2, 4]" not in existing_meshes and random.random() < 0.5:
        new_config = copy.deepcopy(config)
        new_machine = new_config.get("machine_info", [{}])[0]
        new_machine["tensor_placements"] = [
            {
                "mesh_device_shape": "[2, 4]",
                "placement": f"[PlacementShard({random.choice([0, 1, 2, 3])})]",
                "distribution_shape": "[8]",
            }
        ]
        new_machine["card_count"] = 4
        configs.append(new_config)

    # Add 4x8 variant if not already present
    if "[4, 8]" not in existing_meshes and random.random() < 0.3:
        new_config = copy.deepcopy(config)
        new_machine = new_config.get("machine_info", [{}])[0]
        new_machine["tensor_placements"] = [
            {
                "mesh_device_shape": "[4, 8]",
                "placement": f"[PlacementShard({random.choice([0, 3])})]",
                "distribution_shape": "[32]",
            }
        ]
        new_machine["card_count"] = 32
        configs.append(new_config)

    return configs


def process_json():
    """Main processing function."""
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    operations = data.get("operations", {})
    print(f"Found {len(operations)} operations")

    stats = {
        "ops_processed": 0,
        "configs_original": 0,
        "configs_with_placements_enriched": 0,
        "configs_with_new_variants": 0,
        "total_configs_output": 0,
    }

    # Set random seed for reproducibility
    random.seed(42)

    new_operations = {}

    for op_name, op_data in operations.items():
        configurations = op_data.get("configurations", [])
        stats["configs_original"] += len(configurations)

        new_configs = []

        for config in configurations:
            machine_info = config.get("machine_info", [])

            if machine_info and len(machine_info) > 0:
                machine = machine_info[0]
                tensor_placements = machine.get("tensor_placements", [])

                # Enrich existing tensor_placements with mock values
                if tensor_placements:
                    for placement in tensor_placements:
                        enrich_tensor_placement(placement)
                    stats["configs_with_placements_enriched"] += 1
                else:
                    # Add default 1x1 placement for configs without any
                    machine["tensor_placements"] = [
                        {
                            "mesh_device_shape": "[1, 1]",
                            "placement": "[PlacementReplicate]",
                            "distribution_shape": "[1]",
                        }
                    ]

            # Add mesh variants (creates additional configs)
            variants = add_mesh_variants(config, op_name)
            if len(variants) > 1:
                stats["configs_with_new_variants"] += len(variants) - 1
            new_configs.extend(variants)

        new_operations[op_name] = {"configurations": new_configs}
        stats["total_configs_output"] += len(new_configs)
        stats["ops_processed"] += 1

        if stats["ops_processed"] % 10 == 0:
            print(f"  Processed {stats['ops_processed']} operations...")

    # Create output
    output_data = {"operations": new_operations}

    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Operations: {stats['ops_processed']}")
    print(f"   Original configs: {stats['configs_original']}")
    print(f"   Configs with placements enriched: {stats['configs_with_placements_enriched']}")
    print(f"   New mesh variant configs added: {stats['configs_with_new_variants']}")
    print(f"   Total output configs: {stats['total_configs_output']}")
    print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_json()
